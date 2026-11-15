#!/usr/bin/env python
"""Can V_mix2 find more visible spots with fewer truly false ones than at its goal threshold (0.23)?

A threshold alone cannot: raising it only removes detections, so found spots and truly false detections fall
together. This script sweeps V_mix2 with and without the lattice filter of postproc_sim93.py (lattice2.5: keep
detections within 2.5 px of the 2D lattice fitted to the confident detections) over thresholds 0.05-0.45 in steps of
0.01 and chooses the lattice threshold on the 31 new validation materials by two rules:

    same-false   the lowest threshold whose truly false detections do not exceed V_mix2 at 0.23 (most found spots
                 for the same number of truly false ones)
    goal         the largest net found = (found - truly false) / visible

and scores them, with V_mix2 at 0.23, on the 62 new test materials and the old test (with intensity-weighted
accuracy).

    python tradeoff_vmix2_sim93.py   # -> sim93_eval/tradeoff_sweep.csv.gz, tradeoff_best.csv
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

import benchmark_diffraction as bd
from intensity_accuracy import weighted_accuracy
from postproc_sim93 import COLS, E, R, apply, load, pooled

MODEL, REF_T = "V_mix2", 0.23
TS = np.round(np.arange(0.05, 0.451, 0.01), 2)
VARIANTS = ["base", "lattice2.5"]
VAL, TEST, OLD = "new validation (31)", "new test (62)", "old test"


def work(args):
    i, r, od, vis_box, final = args
    img, pots, probe = load(r, od)
    rad = bd.disk_radius(probe)
    g = bd.ground_truth(img, pots, rad, vis_box)
    P = np.load(E / "predictions.npz")
    pts, prob = P[f"{MODEL}_{i}_pts"], P[f"{MODEL}_{i}_prob"]
    rows = []
    for v, ts in (final.items() if final else ((v, TS) for v in VARIANTS)):
        for t in ts:
            p, _ = apply(v, pts[prob >= t], prob[prob >= t], g["beam"], None, rad)
            s = bd.score(g, p)
            row = dict(pattern=i, variant=v, threshold=t, **{c: s[c] for c in COLS})
            if final:
                for name, im in (("acc_pots", pots), ("acc_img", img)):
                    row[name] = weighted_accuracy(im, g["points"], p, g["beam"], gt_required=g["visible"])["accuracy"]
            rows.append(row)
    return rows


def run(meta, od, vis, final=None):
    jobs = [(i, r, od, vis, final) for i, r in meta.iterrows()]
    with Pool(4) as pool:
        rows = [x for part in pool.imap_unordered(work, jobs, chunksize=4) for x in part]
    return pd.DataFrame(rows).join(meta[["part", "angle", "thickness"]], on="pattern")


def main():
    S = json.loads((E / "settings.json").read_text())
    split = json.loads((E / "threshold_split.json").read_text())
    meta = pd.read_csv(E / "patterns.csv").set_index("pattern")
    meta["part"] = np.where(meta.material.isin(split["val"]), VAL, np.where(meta.material.isin(split["test"]), TEST, OLD))
    od = {f.name: f for f in bd.find_material_folders(json.loads((R / "ft_V/split.json").read_text())["args"]["root"])}

    if (E / "tradeoff_sweep.csv.gz").exists():
        sw = pd.read_csv(E / "tradeoff_sweep.csv.gz")
    else:
        sw = run(meta, od, S["vis_box"])
        sw.to_csv(E / "tradeoff_sweep.csv.gz", index=False)
    v = sw[sw.part == VAL].groupby(["variant", "threshold"])[list(COLS)].sum()
    ref_false = v.loc[("base", REF_T), "truly_false"]
    lat = v.loc["lattice2.5"]
    same = float(lat[lat.truly_false <= ref_false].index.min())
    goal = float(((lat.found_vis - lat.truly_false) / lat.n_vis).idxmax())
    rules = {("base", REF_T): "V_mix2 (goal threshold)", ("lattice2.5", same): "V_mix2 + lattice, same-false rule",
             ("lattice2.5", goal): "V_mix2 + lattice, goal rule"}
    print("lattice thresholds: same-false", same, "goal", goal, flush=True)

    final = {}
    for (var, t) in rules:
        final.setdefault(var, []).append(t)
    fin = run(meta, od, S["vis_box"], final=final)
    out = []
    for (var, t), name in rules.items():
        d = fin[(fin.variant == var) & np.isclose(fin.threshold, t)]
        for part in (VAL, TEST, OLD):
            dp = d[d.part == part]
            out.append(dict(rule=name, variant=var, threshold=t, part=part, **pooled(dp).to_dict(),
                            acc_pots=dp.acc_pots.mean(), acc_img=dp.acc_img.mean()))
    b = pd.DataFrame(out)
    b.to_csv(E / "tradeoff_best.csv", index=False)
    pd.set_option("display.width", 250)
    print(b.set_index(["part", "rule"])[["threshold", "recall", "found", "truly_false", "truly_false_share", "net_found",
                                         "f1", "acc_pots", "acc_img"]].round(3).to_string())


if __name__ == "__main__":
    main()
