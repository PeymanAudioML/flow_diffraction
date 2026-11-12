#!/usr/bin/env python
"""Threshold rule and brightness-band recall for fine-tuned runs of finetune_diffraction.py.

For each run folder (e.g. diffraction_runs/ft_V):

1. Threshold rule, fixed before looking at results: the LOWEST threshold whose pooled precision on the
   run's own validation materials (split.json, all thicknesses) is at least --min-precision (0.95),
   with the reference scoring (benchmark_diffraction.score). Thresholds 0.05, 0.06, ..., 0.95.
2. Test materials at --eval-thicknesses: scores at the run's own tuned threshold (thresholds.yaml, F1 on
   validation) and at the rule threshold, pooled by angle, plus recall of visible spots by brightness:
   the 3 x 3 maximum of dataMeas around the spot (the visibility test) in units of the floor (max * 1e-4),
   bands 1-10, 10-100, 100-1000, > 1000.

One prediction per pattern at threshold 0.05 is filtered by its probability for every threshold; this gives
exactly the points that predict(prob_thresh=t) gives (checked on 54 patterns).

    python eval_diffraction.py diffraction_runs/ft_V diffraction_runs/ft_V_log1e4 --out diffraction_runs/eval
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import maximum_filter

import benchmark_diffraction as bd

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
BANDS = [1, 10, 100, 1000, np.inf]
BAND_NAMES = ["1-10x", "10-100x", "100-1000x", ">1000x"]


def load(roots, materials, thicknesses=None):
    """(material, angle, thickness, img, probe, g) for every pattern of these materials."""
    import h5py

    out = []
    for folder, material, uid, angle, h5 in bd.iter_simulations(roots, materials):
        with h5py.File(h5, "r") as h:
            probe = h["dataProbe"][:]
            r = bd.disk_radius(probe)
            for i, t in enumerate(np.asarray(h["thicknesses"][:])):
                if thicknesses and not any(np.isclose(t, x) for x in thicknesses):
                    continue
                img = np.asarray(h["dataMeas"][..., i], np.float32)
                g = bd.ground_truth(img, h["dataPots"][..., i], r, 3)
                g["brightness"] = maximum_filter(img, size=3)[g["points"][:, 0], g["points"][:, 1]] / g["floor"]
                out.append((material, angle, float(t), img, probe, g))
    return out


def found_mask(g, pred):
    """Which ground-truth candidates are matched (same matching as bd.score)."""
    from spotiflow.utils.matching import points_matching

    pred = np.asarray(pred, float).reshape(-1, 2)
    pred = pred[bd.far_from(pred, g["beam"], bd.BEAM_R)]
    found = np.zeros(len(g["points"]), bool)
    if len(pred):
        found[[a for a, _ in points_matching(g["points"], pred, cutoff_distance=bd.CUTOFF).matched_pairs]] = True
    return found


def predict_low(model, pattern, log_scale):
    """One prediction at threshold 0.05: points and their probabilities."""
    x = bd.spotiflow_input(pattern[3], pattern[4], "log", log_scale=log_scale).astype(np.float32)
    p, d = model.predict(x, prob_thresh=0.05, min_distance=3, subpix=True, verbose=False, device="cuda")
    return p, d.prob


def pooled(rows):
    df = pd.DataFrame(rows)
    keys = [k for k in ("run", "rule", "threshold", "angle") if k in df]
    s = df.groupby(keys)[["n_vis", "found_vis", "found_not_vis", "n_pred", "false", "truly_false"]].sum()
    s["recall_vis"] = (s.found_vis / s.n_vis).round(3)
    s["precision"] = (1 - s["false"] / s.n_pred.replace(0, np.nan)).round(3)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run folders written by finetune_diffraction.py")
    p.add_argument("--out", required=True)
    p.add_argument("--min-precision", type=float, default=0.95)
    p.add_argument("--eval-thicknesses", nargs="*", type=float, default=[20, 40, 100, 200, 500])
    args = p.parse_args()
    logging.getLogger("spotiflow").setLevel(logging.WARNING)
    from spotiflow.model import Spotiflow

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    sweep_rows, test_rows, band_rows, chosen = [], [], [], {}
    for run in map(Path, args.runs):
        split = json.loads((run / "split.json").read_text())
        roots, log_scale = split["args"]["root"], float(split["args"]["log_scale"])
        tuned = yaml.safe_load((run / "model" / "thresholds.yaml").read_text())["prob_thresh_best"][0]
        model = Spotiflow.from_folder(str(run / "model"), map_location="cuda")
        key_val, key_test = ("val", tuple(split["val"])), ("test", tuple(split["test"]))
        if key_val not in cache:
            cache[key_val] = load(roots, split["val"])
        if key_test not in cache:
            cache[key_test] = load(roots, split["test"], args.eval_thicknesses)

        # 1. threshold rule on the validation materials
        val_preds = [predict_low(model, pt, log_scale) for pt in cache[key_val]]
        for t in THRESHOLDS:
            for pt, (pr, prob) in zip(cache[key_val], val_preds):
                sweep_rows.append(dict(run=run.name, threshold=t, angle=pt[1], **bd.score(pt[5], pr[prob >= t])))
        sw = pooled(sweep_rows).loc[run.name].groupby("threshold")[["n_pred", "false"]].sum()
        val_prec = 1 - sw["false"] / sw.n_pred
        ok = val_prec[val_prec >= args.min_precision]
        rule = float(ok.index.min()) if len(ok) else float("nan")
        chosen[run.name] = dict(tuned=float(tuned), rule=rule, val_precision_at_rule=round(float(val_prec.get(rule, np.nan)), 3),
                                val_precision_at_tuned=round(float(val_prec.loc[np.isclose(val_prec.index, round(tuned, 2))].iloc[0]), 3)
                                if np.isclose(val_prec.index, round(tuned, 2)).any() else None,
                                log_scale=log_scale, seed=split["args"]["seed"], val=split["val"], test=split["test"])
        print(f"{run.name}: tuned threshold {tuned:.3f}, rule threshold {rule} "
              f"(lowest with validation precision >= {args.min_precision})", flush=True)

        # 2. test materials at both thresholds, pooled by angle and by brightness band
        for pt in cache[key_test]:
            pr, prob = predict_low(model, pt, log_scale)
            g = pt[5]
            for name, t in (("tuned", tuned), ("rule", rule)):
                keep = pr[prob >= t]
                test_rows.append(dict(run=run.name, rule=name, threshold=t, angle=pt[1], **bd.score(g, keep)))
                f, vis, b = found_mask(g, keep), g["visible"], g["brightness"]
                for lo, hi, bn in zip(BANDS[:-1], BANDS[1:], BAND_NAMES):
                    sel = vis & (b >= lo) & (b < hi)
                    band_rows.append(dict(run=run.name, rule=name, angle=pt[1], band=bn, n=int(sel.sum()), found=int((f & sel).sum())))

    # zero-shot reference for the brightness bands (section 2 pipelines, same test patterns)
    base = Spotiflow.from_pretrained("general", map_location="cuda", verbose=False)
    from spotiflow.utils import normalize
    for key, pats in cache.items():
        if key[0] != "test":
            continue
        for material, angle, t, img, probe, g in pats:
            method, kw = bd.choose_pipeline("angle", angle, None)
            x0 = bd.spotiflow_input(img, probe, method, pad=False, **kw)
            p0, _ = base.predict(x0, min_distance=3, prob_thresh=kw["prob_thresh"], subpix=True, verbose=False,
                                 normalizer=lambda im: normalize(im, 1, 99.8), device="cuda")
            test_rows.append(dict(run="zero-shot", rule="-", threshold=kw["prob_thresh"], angle=angle, **bd.score(g, p0)))
            f, vis, b = found_mask(g, p0), g["visible"], g["brightness"]
            for lo, hi, bn in zip(BANDS[:-1], BANDS[1:], BAND_NAMES):
                sel = vis & (b >= lo) & (b < hi)
                band_rows.append(dict(run="zero-shot", rule="-", angle=angle, band=bn, n=int(sel.sum()), found=int((f & sel).sum())))
        break  # one test set

    pd.DataFrame(sweep_rows).to_csv(out / "val_sweep.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(chosen, indent=2))
    test = pooled(test_rows)
    test.to_csv(out / "test_by_angle.csv")
    bands = pd.DataFrame(band_rows).groupby(["run", "rule", "angle", "band"])[["n", "found"]].sum()
    bands["recall"] = (bands.found / bands.n.replace(0, np.nan)).round(3)
    bands.to_csv(out / "test_recall_by_brightness.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print("\n=== thresholds ===\n" + pd.DataFrame(chosen).T[["log_scale", "seed", "tuned", "rule",
                                                                      "val_precision_at_tuned", "val_precision_at_rule"]].to_string())
        print("\n=== test materials, pooled by angle ===\n" + test.to_string())
        print("\n=== test materials, visible recall by brightness (3x3 max / floor) ===\n"
              + bands.recall.unstack("band")[BAND_NAMES].to_string())


if __name__ == "__main__":
    main()
