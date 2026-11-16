#!/usr/bin/env python
"""Section 6 methods of notebook 7 at a thickness that eval_sim93.py did not evaluate (default 60 A).

The settings are not re-tuned: they are the ones chosen on the validation materials with truly false x 2
(sim93_eval/fcunet_best_w2.csv, tradeoff_best_w2.csv). Patterns: the same materials, uids and angles as
sim93_eval/patterns.csv (93 new + 3 old test materials) at thickness T. Output in sim93_eval/t<T>/.

    python thick_sim93.py predict                         # spotiflow env: patterns.csv, V_mix2 predictions.npz
    SIM93_DIR=.../sim93_eval/t60 /mnt/data/envs/fcunet/bin/python fcunet_sim93.py    # FCU-Net -> t60/fcunet_*
    python thick_sim93.py score                           # spotiflow env: final_patterns.csv (+ intensity accuracy)

Thickness: environment THICK (default 60).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import benchmark_diffraction as bd
from intensity_accuracy import weighted_accuracy
from postproc_sim93 import COLS, E, R, apply, load
from score_fcunet_sim93 import near

T = float(os.environ.get("THICK", 60))
OUT = E / f"t{T:g}"
MODEL = "V_mix2"
VAL, TEST, OLD = "new validation (31)", "new test (62)", "old test"


def old_dirs():
    return {f.name: f for f in bd.find_material_folders(json.loads((R / "ft_V/split.json").read_text())["args"]["root"])}


def predict():
    import logging
    from eval_experimental import RUNS, predict_low
    from spotiflow.model import Spotiflow
    logging.getLogger("spotiflow").setLevel(logging.WARNING)
    S = json.loads((E / "settings.json").read_text())
    meta = (pd.read_csv(E / "patterns.csv").drop_duplicates(["dataset", "folder", "uid"])
            [["dataset", "folder", "uid", "material", "angle"]].assign(thickness=T).reset_index(drop=True))
    od = old_dirs()
    xs, n_vis = [], []
    for r in meta.itertuples():
        img, pots, probe = load(r, od)
        g = bd.ground_truth(img, pots, bd.disk_radius(probe), S["vis_box"])
        n_vis.append(int(g["visible"].sum()))
        xs.append(bd.spotiflow_input(img, probe, "log", log_scale=S["log_scale"]).astype(np.float32))
    meta["n_vis"] = n_vis
    OUT.mkdir(exist_ok=True)
    meta.to_csv(OUT / "patterns.csv", index_label="pattern")
    model = Spotiflow.from_folder(str(R / RUNS.get(MODEL, f"ft_{MODEL}") / "model"), map_location="cuda")
    low = predict_low(model, xs, "cuda")
    np.savez_compressed(OUT / "predictions.npz", **{f"{MODEL}_{i}_pts": p for i, (p, _) in enumerate(low)},
                        **{f"{MODEL}_{i}_prob": q for i, (_, q) in enumerate(low)})
    print(len(meta), "patterns predicted", flush=True)


def settings():
    fb = pd.read_csv(E / "fcunet_best_w2.csv", keep_default_na=False, na_values=[""], dtype={"setting": str})
    fb = fb[fb.part == TEST].set_index("method")
    tb = pd.read_csv(E / "tradeoff_best_w2.csv")
    t_lat = float(tb.loc[tb.rule.str.contains("goal rule"), "threshold"].iloc[0])
    return {m: (r.setting, float(r.cut) if pd.notna(r.cut) else np.nan, float(r.t) if pd.notna(r.t) else np.nan)
            for m, r in fb.iterrows()}, t_lat


def work(args):
    i, r, od, vis_box, st, t_lat = args
    img, pots, probe = load(r, od)
    rad = bd.disk_radius(probe)
    g = bd.ground_truth(img, pots, rad, vis_box)
    F, P = np.load(OUT / "fcunet_peaks.npz"), np.load(OUT / "predictions.npz")
    mp, mprob = P[f"{MODEL}_{i}_pts"], P[f"{MODEL}_{i}_prob"]
    out = {}
    for m, (k, c, t) in st.items():
        fc = F[f"{k}_{i}_pts"][F[f"{k}_{i}_rel"] >= c] if m != MODEL else None
        vm = mp[mprob >= t] if m != "fcunet" else None
        out[m] = (fc if m == "fcunet" else vm if m == MODEL else vm[near(vm, fc)] if "AND" in m
                  else np.vstack([vm, fc[~near(fc, vm)]]))
    keep = mprob >= t_lat
    out["lattice"] = apply("lattice2.5", mp[keep], mprob[keep], g["beam"], None, rad)[0]
    rows = []
    for m, p in out.items():
        s = bd.score(g, p)
        row = dict(pattern=i, method=m, **{q: s[q] for q in COLS})
        for name, im in (("acc_pots", pots), ("acc_img", img)):
            row[name] = weighted_accuracy(im, g["points"], p, g["beam"], gt_required=g["visible"])["accuracy"]
        rows.append(row)
    return rows


def score():
    S = json.loads((E / "settings.json").read_text())
    split = json.loads((E / "threshold_split.json").read_text())
    meta = pd.read_csv(OUT / "patterns.csv").set_index("pattern")
    meta["part"] = np.where(meta.material.isin(split["val"]), VAL, np.where(meta.material.isin(split["test"]), TEST, OLD))
    st, t_lat = settings()
    print("settings:", st, "lattice", t_lat, flush=True)
    od = old_dirs()
    jobs = [(i, r, od, S["vis_box"], st, t_lat) for i, r in meta.iterrows()]
    with Pool(4) as pool:
        rows = [x for part in pool.imap_unordered(work, jobs, chunksize=4) for x in part]
    d = pd.DataFrame(rows).join(meta[["part", "angle", "thickness"]], on="pattern")
    d.to_csv(OUT / "final_patterns.csv", index=False)
    s = d[d.part != VAL].groupby(["part", "method"])[list(COLS)].sum()
    s["recall"] = s.found_vis / s.n_vis
    s["score"] = (s.found_vis - 2 * s.truly_false) / s.n_vis
    s = s.join(d[d.part != VAL].groupby(["part", "method"])[["acc_pots", "acc_img"]].mean())
    print(s[["recall", "found_vis", "truly_false", "score", "acc_pots", "acc_img"]].round(3).to_string())


if __name__ == "__main__":
    {"predict": predict, "score": score}[sys.argv[1]]()
