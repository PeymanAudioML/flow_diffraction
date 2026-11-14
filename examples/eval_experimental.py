#!/usr/bin/env python
"""Score the experimental fine-tuning runs (V, V_exp, V_mix, V_mix2) on ONE common test set, and finish ft_V_mix2.

ft_V_mix2 was stopped by a machine shutdown in epoch 27 of 30, before finetune_experimental.py reached its end steps.
Its best checkpoint (best.pt, lowest validation loss) exists, but without a tuned threshold, without the simulation
threshold and without test scores. This script does those steps, in the same way as the end of training:

1. experimental threshold: F1-optimal threshold on its own validation set (3 val acquisitions x 500 patterns),
   grid 0.30-0.70 (spotiflow's optimize_threshold range). Spotiflow tunes on random 128-px crops; here the
   whole patterns are scored with the test scoring (3 px, direct beam ignored), so the value can differ slightly.
   Written to ft_V_mix2/model/thresholds.yaml (prob_thresh_best).
2. simulation threshold: tune_sim_threshold's rule (lowest threshold at which the model is as precise as ft_V on
   the simulation materials it can be tuned on: ft_V_mix2 trained on ft_V's sim val materials too, so as in
   finetune_experimental.py the training simulations are used). Written to ft_V_mix2/model/threshold_sim.json.
3. all four models on the same test patterns: 6 test acquisitions x 500 patterns (= ft_V_mix2's test set; the
   section 7 numbers of V_exp / V_mix used 200 per acquisition) and ft_V's 45 simulation test patterns.

Each model predicts once per pattern at threshold 0.05; the points are then filtered by their probability (this gives
exactly the points of predict(prob_thresh=t), see eval_diffraction.py), so threshold sweeps cost one prediction.

Outputs (OUT, default diffraction_runs/eval_exp):
    thresholds.json            threshold per model (experimental, simulation) and how it was chosen
    exp_test.csv               per model x material: tp, n_pred, n_label, precision, recall, f1 (own threshold)
    exp_test_patterns.csv      per model x test pattern (own threshold)
    exp_sweep.csv              per model x split (val / test) x threshold: pooled precision / recall / f1
    sim_test.csv               per model x threshold rule (own / sim) x angle: bd.score pooled
    sim_test_patterns.csv      per model x rule x simulation test pattern

    python eval_experimental.py
    python eval_experimental.py --models V V_exp V_mix V_mix2 V_mix3 V_mix4 V_mix5 --out diffraction_runs/eval_exp2

A model without thresholds.yaml gets the whole-pattern val-F1 threshold (written to thresholds.yaml); a model without
threshold_sim.json gets the tune_sim_threshold rule on the training simulations (only if it was trained with
simulations, else its own threshold).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import benchmark_diffraction as bd
import finetune_experimental as fe
from experimental_data import load_experimental

HERE = Path(__file__).parent
RUNS = {"V": "ft_V", "V_exp": "ft_V_exp", "V_mix": "ft_V_mix", "V_mix2": "ft_V_mix2"}
NO_SIM = ("V", "V_exp")  # runs without simulations in their fine-tuning: one threshold for both kinds of data
EXP_GRID = np.round(np.arange(0.30, 0.701, 0.01), 2)   # experimental threshold for V_mix2 (optimize_threshold range)
SWEEP = np.round(np.arange(0.10, 0.901, 0.02), 2)      # reported sweep for every model
SIM_GRID = np.round(np.arange(0.30, 0.81, 0.02), 2)    # as fe.tune_sim_threshold


def predict_low(model, xs, device):
    """One prediction per image at threshold 0.05 -> list of (points, probabilities)."""
    out = []
    for x in xs:
        p, d = model.predict(x, prob_thresh=0.05, min_distance=3, subpix=True, verbose=False, device=device)
        out.append((np.asarray(p, float).reshape(-1, 2), np.asarray(d.prob, float)))
    return out


def at(low, thr):
    p, prob = low
    return p[prob >= thr]


def exp_rows(low, labels, meta, thr):
    """Per-pattern tp / n_pred / n_label against the r4 labels (as fe.score_experimental)."""
    from spotiflow.utils.matching import points_matching
    rows = []
    for lw, g, (_, m) in zip(low, labels, meta.iterrows()):
        p = at(lw, thr)
        p = p[bd.far_from(p, (m.beam_r, m.beam_c), bd.BEAM_R)]
        tp = len(points_matching(g, p, cutoff_distance=bd.CUTOFF).matched_pairs) if len(p) and len(g) else 0
        rows.append(dict(material=m.material, acquisition=m.acquisition, index=m["index"], tp=tp, n_pred=len(p),
                         n_label=len(g)))
    return pd.DataFrame(rows)


def pooled_exp(df, by="material"):
    out = df.groupby(by)[["tp", "n_pred", "n_label"]].sum()
    out.loc["all"] = out.sum()
    out["precision"] = out.tp / out.n_pred.clip(lower=1)
    out["recall"] = out.tp / out.n_label
    out["f1"] = 2 * out.tp / (out.n_pred + out.n_label)
    return out


def sim_rows(low, sims, thr):
    return pd.DataFrame([dict(material=it["material"], angle=it["angle"], thickness=it["thickness"],
                              **bd.score(it["g"], at(lw, thr))) for lw, it in zip(low, sims)])


def pooled_sim(df):
    out = df.groupby("angle")[["n_vis", "found_vis", "found_not_vis", "n_pred", "false", "truly_false"]].sum()
    out["recall_vis"] = out.found_vis / out.n_vis
    out["precision"] = 1 - out["false"] / out.n_pred.clip(lower=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/mnt/data/4D-ImageNet/converted")
    ap.add_argument("--split", default=str(HERE / "diffraction_runs/exp_split.json"))
    ap.add_argument("--runs", default=str(HERE / "diffraction_runs"))
    ap.add_argument("--out", default=str(HERE / "diffraction_runs/eval_exp"))
    ap.add_argument("--eval-per-acq", type=int, default=500)
    ap.add_argument("--min-conf", type=float, default=0.8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="+", default=list(RUNS), help="names; run directory = diffraction_runs/ft_<name>")
    args = ap.parse_args()
    runs_of = {k: RUNS.get(k, f"ft_{k}") for k in args.models}
    logging.getLogger("spotiflow").setLevel(logging.WARNING)

    from spotiflow.model import Spotiflow

    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())

    def load(role):
        imgs, pts, meta = load_experimental(args.data, split[role], args.min_conf, args.eval_per_acq)
        return [fe.log_input(im) for im in imgs], pts, meta

    xva, yva, mva = load("val")
    xte, yte, mte = load("test")
    sims = fe.load_simulation_test(runs / "ft_V")
    print(f"experimental: val {len(xva)}, test {len(xte)} patterns; simulation test {len(sims)} patterns", flush=True)

    models = {k: Spotiflow.from_folder(str(runs / r / "model"), map_location=args.device) for k, r in runs_of.items()}
    low = {k: dict(val=predict_low(m, xva, args.device), test=predict_low(m, xte, args.device),
                   sim=predict_low(m, [it["x"] for it in sims], args.device)) for k, m in models.items()}
    print("predictions done", flush=True)

    # --- 1. experimental thresholds (val F1 curve for every model; tuned only where thresholds.yaml is missing) ---
    thr, tune = {}, None
    for k, m in models.items():
        mdir = runs / runs_of[k] / "model"
        f1_val = {float(t): pooled_exp(exp_rows(low[k]["val"], yva, mva, t)).loc["all", "f1"] for t in EXP_GRID}
        if (mdir / "thresholds.yaml").exists():
            t_exp = float(m._prob_thresh[0])
            rule = ("F1 on the experimental validation patterns (whole patterns), set by eval_experimental.py"
                    if k == "V_mix2" else "spotiflow optimize_threshold at the end of training (thresholds.yaml)")
        else:
            t_exp = max(f1_val, key=f1_val.get)
            (mdir / "thresholds.yaml").write_text(yaml.safe_dump({"prob_thresh_best": [t_exp]}))
            m._prob_thresh = [t_exp]
            rule = "F1 on the experimental validation patterns (whole patterns), set by eval_experimental.py"
        thr[k] = dict(exp=t_exp, exp_rule=rule, val_f1_by_threshold={f"{t:.2f}": round(v, 4) for t, v in f1_val.items()})

        # --- 2. simulation threshold -----------------------------------------------------------------------------
        sim_file = mdir / "threshold_sim.json"
        if k in NO_SIM:
            thr[k].update(sim=t_exp, sim_rule="own threshold")
            continue
        if not sim_file.exists():  # tune_sim_threshold on the training simulations, as for ft_V_mix2
            if tune is None:
                tune = fe.load_simulation(runs / "ft_V", "train") + fe.load_simulation(runs / "ft_V", "val")
                lv = predict_low(models["V"] if "V" in models else
                                 Spotiflow.from_folder(str(runs / "ft_V/model"), map_location=args.device),
                                 [it["x"] for it in tune], args.device)
                prec = lambda lw, t: (lambda s: 1 - s["false"].sum() / s.n_pred.sum())(sim_rows(lw, tune, t))
                target = prec(lv, float(Spotiflow.from_folder(str(runs / "ft_V/model"), map_location="cpu")
                                        ._prob_thresh[0]))
            lm = predict_low(m, [it["x"] for it in tune], args.device)
            t_sim, p = float(SIM_GRID[-1]), np.nan
            for t in SIM_GRID:
                p = prec(lm, float(t))
                if p >= target:
                    t_sim = float(t)
                    break
            sim_file.write_text(json.dumps(dict(
                prob_thresh_sim=t_sim, rule="lowest threshold with ft_V's pooled precision on the simulation train "
                                            "materials (set by eval_experimental.py)",
                target_precision=round(float(target), 3), val_precision=round(float(p), 3)), indent=2))
        js = json.loads(sim_file.read_text())
        thr[k].update(sim=js["prob_thresh_sim"], sim_rule=js["rule"])
    (out / "thresholds.json").write_text(json.dumps(thr, indent=2))
    print("thresholds:", {k: (v["exp"], v["sim"]) for k, v in thr.items()}, flush=True)

    # --- 3. test scores ----------------------------------------------------------------------------------------
    pat, tab, sweep = [], [], []
    for k in models:
        df = exp_rows(low[k]["test"], yte, mte, thr[k]["exp"]).assign(model=k)
        pat.append(df)
        tab.append(pooled_exp(df).assign(model=k))
        for role, lw, ys, ms in (("val", low[k]["val"], yva, mva), ("test", low[k]["test"], yte, mte)):
            for t in SWEEP:
                s = pooled_exp(exp_rows(lw, ys, ms, float(t))).loc["all"]
                sweep.append(dict(model=k, split=role, threshold=float(t), **s.to_dict()))
    pd.concat(pat).to_csv(out / "exp_test_patterns.csv", index=False)
    pd.concat(tab).reset_index().set_index(["model", "material"]).round(4).to_csv(out / "exp_test.csv")
    pd.DataFrame(sweep).round(4).to_csv(out / "exp_sweep.csv", index=False)

    spat, stab = [], []
    for k in models:
        for rule in ("own", "sim"):
            t = thr[k]["exp" if rule == "own" else "sim"]
            df = sim_rows(low[k]["sim"], sims, t).assign(model=k, rule=rule, threshold=t)
            spat.append(df)
            stab.append(pooled_sim(df).assign(model=k, rule=rule, threshold=t))
    pd.concat(spat).to_csv(out / "sim_test_patterns.csv", index=False)
    pd.concat(stab).reset_index().set_index(["model", "rule", "angle"]).round(4).to_csv(out / "sim_test.csv")

    print("\n=== experimental test (own thresholds) ===")
    print(pd.read_csv(out / "exp_test.csv").pivot(index="material", columns="model", values="f1").round(3))
    print("\n=== simulation test ===")
    print(pd.read_csv(out / "sim_test.csv")[["model", "rule", "angle", "recall_vis", "precision"]].round(3).to_string())
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
