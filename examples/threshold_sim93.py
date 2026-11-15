#!/usr/bin/env python
"""Fair threshold for the new simulated materials: re-score eval_sim93.py's stored predictions (threshold 0.05) over a
wider grid, split the 93 new materials into validation (1/3, threshold chosen there) and test (2/3, scored at that
threshold), by material, fixed seed.

    python threshold_sim93.py   # -> sim93_eval/sweep_wide.csv.gz, threshold_split.json (with the tuned thresholds),
                                #    at_threshold.csv (scores + intensity-weighted accuracy at the usual / tuned threshold)
Re-running with sweep_wide.csv.gz and threshold_split.json present only redoes at_threshold.csv.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import benchmark_diffraction as bd

R = Path(__file__).resolve().parent / "diffraction_runs"
E = R / "sim93_eval"
NEW = Path("/mnt/data/sim_diffraction_gdrive")
GRID = np.round(np.arange(0.05, 0.801, 0.01), 2)
COLS = ("n_vis", "n_pred", "found_vis", "found_not_vis", "missed_vis", "false", "false_near_spot", "false_on_weak_peak",
        "truly_false")


def pooled_f1(d):
    t = d[["n_vis", "found_vis", "n_pred", "false"]].sum()
    P, R_ = 1 - t["false"] / max(t.n_pred, 1), t.found_vis / max(t.n_vis, 1)
    return 2 * P * R_ / (P + R_) if P + R_ else 0.0


def net_found(d):
    """The goal: as many visible spots found as possible with few truly false detections. Score = (visible spots
    found - truly false detections) / visible spots: every truly false detection cancels one found spot. Detections
    on weaker real peaks or 3-5 px from a missed spot are not counted against it."""
    t = d[["n_vis", "found_vis", "truly_false"]].sum()
    return (t.found_vis - t.truly_false) / max(t.n_vis, 1)


def at_thresholds():
    """Per pattern and model, scores and intensity-weighted accuracy (image and dataPots weights) at the usual
    threshold and at two thresholds chosen on the new validation materials: 'goal' (best net_found) and 'f1' (best F1).
    -> at_threshold.csv; the thresholds in threshold_split.json."""
    from intensity_accuracy import weighted_accuracy
    S = json.loads((E / "settings.json").read_text())
    meta = pd.read_csv(E / "patterns.csv").set_index("pattern")
    split = json.loads((E / "threshold_split.json").read_text())
    wide = pd.read_csv(E / "sweep_wide.csv.gz").join(meta[["material"]], on="pattern")
    val = wide[wide.material.isin(split["val"])]
    rules = {"usual": S["thresholds"],
             "goal": {k: float(val[val.model == k].groupby("threshold").apply(net_found).idxmax()) for k in S["models"]},
             "f1": {k: float(val[val.model == k].groupby("threshold").apply(pooled_f1).idxmax()) for k in S["models"]}}
    split.pop("tuned_thresholds", None)
    split.update({f"{r}_thresholds": v for r, v in rules.items()})
    (E / "threshold_split.json").write_text(json.dumps(split, indent=1))
    P = np.load(E / "predictions.npz")
    old_dirs = {f.name: f for f in bd.find_material_folders(json.loads((R / "ft_V/split.json").read_text())["args"]["root"])}
    rows = []
    for i, r in meta.iterrows():
        folder = NEW / r.folder if r.dataset == "new" else old_dirs[r.folder]
        with h5py.File(folder / f"{r.uid}_training.h5", "r") as h:
            ti = int(np.where(h["thicknesses"][:] == r.thickness)[0][0])
            img, pots, probe = h["dataMeas"][..., ti], h["dataPots"][..., ti], h["dataProbe"][:]
        g = bd.ground_truth(img, pots, bd.disk_radius(probe), S["vis_box"])
        part = "val" if r.material in split["val"] else ("test" if r.material in split["test"] else "old test")
        for k in S["models"]:
            pts, prob = P[f"{k}_{i}_pts"], P[f"{k}_{i}_prob"]
            for rule, th in rules.items():
                t = th[k]
                pr = pts[prob >= t]
                s = bd.score(g, pr)
                rows.append(dict(model=k, pattern=i, part=part, rule=rule, threshold=t, **{c: s[c] for c in COLS},
                                 acc_img=weighted_accuracy(img, g["points"], pr, g["beam"], gt_required=g["visible"])["accuracy"],
                                 acc_pots=weighted_accuracy(pots, g["points"], pr, g["beam"], gt_required=g["visible"])["accuracy"]))
    pd.DataFrame(rows).to_csv(E / "at_threshold.csv", index=False)
    print({r: v for r, v in rules.items()})


def main():
    if (E / "sweep_wide.csv.gz").exists() and (E / "threshold_split.json").exists():
        at_thresholds()
        return
    S = json.loads((E / "settings.json").read_text())
    meta = pd.read_csv(E / "patterns.csv").set_index("pattern")
    P = np.load(E / "predictions.npz")
    old_dirs = {f.name: f for f in bd.find_material_folders(json.loads((R / "ft_V/split.json").read_text())["args"]["root"])}
    rows = []
    for i, r in meta.iterrows():
        folder = NEW / r.folder if r.dataset == "new" else old_dirs[r.folder]
        with h5py.File(folder / f"{r.uid}_training.h5", "r") as h:
            ti = int(np.where(h["thicknesses"][:] == r.thickness)[0][0])
            img, pots, probe = h["dataMeas"][..., ti], h["dataPots"][..., ti], h["dataProbe"][:]
        g = bd.ground_truth(img, pots, bd.disk_radius(probe), S["vis_box"])
        for k in S["models"]:
            pts, prob = P[f"{k}_{i}_pts"], P[f"{k}_{i}_prob"]
            for t in GRID:
                s = bd.score(g, pts[prob >= t])
                rows.append(dict(model=k, pattern=i, threshold=t, **{c: s[c] for c in COLS}))
        if i % 200 == 0:
            print(i, flush=True)
    pd.DataFrame(rows).to_csv(E / "sweep_wide.csv.gz", index=False)

    mats = sorted(meta[meta.dataset == "new"].material.unique())
    rng = np.random.default_rng(0)
    val = sorted(rng.choice(mats, size=len(mats) // 3, replace=False).tolist())
    split = dict(seed=0, val=val, test=[m for m in mats if m not in val])
    (E / "threshold_split.json").write_text(json.dumps(split, indent=1))
    print(len(split["val"]), "validation and", len(split["test"]), "test materials")
    at_thresholds()


if __name__ == "__main__":
    main()
