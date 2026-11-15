#!/usr/bin/env python
"""Score py4DSTEM's FCU-Net peaks (fcunet_sim93.py) like the Spotiflow models, alone and combined with V_mix2.

Every method's settings are chosen on the 31 new validation materials by the goal score
(net found = (visible spots found - truly false detections) / visible spots) and scored on the 62 new test
materials and the 3 old test materials, against the same py4DSTEM ground truth (benchmark_diffraction.ground_truth).

Methods
    fcunet         FCU-Net peaks; tuned: peak setting (sigma, minSpacing) and relative-intensity cutoff
    V_mix2         V_mix2 at its goal threshold (base, from threshold_sim93.py)
    V_mix2 AND fc  V_mix2 detections at threshold t kept only if an FCU-Net peak (best setting, cutoff c) lies within
                   3 px; tuned: t, c
    V_mix2 OR fc   V_mix2 at threshold t plus FCU-Net peaks (cutoff c) more than 3 px from every V_mix2 detection;
                   tuned: t, c

    python score_fcunet_sim93.py   # -> sim93_eval/fcunet_sweep.csv.gz, fcunet_best.csv
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import benchmark_diffraction as bd
from intensity_accuracy import weighted_accuracy
from postproc_sim93 import COLS, E, R, load, pooled

CUTS = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12]
TS = np.round(np.arange(0.10, 0.651, 0.05), 2)
MODEL = "V_mix2"
SETS = [f"s{s}_m{m}" for s in (0, 0.5) for m in (2, 4, 6)]
VAL, TEST, OLD = "new validation (31)", "new test (62)", "old test"


def near(a, b, d=3.0):
    """Mask over a: within d px of a point of b."""
    if not len(a) or not len(b):
        return np.zeros(len(a), bool)
    return cKDTree(b).query(a, k=1)[0] <= d


def _key(m, k, c, t):
    """Hashable key with NaN replaced (NaN != NaN after pickling to the workers)."""
    return (m, k, -1.0 if pd.isna(c) else round(float(c), 4), -1.0 if pd.isna(t) else round(float(t), 4))


def methods(i, F, P, best_set=None):
    """{(method, setting, cut, t): points}; with best_set only the combinations."""
    out = {}
    mp, mprob = P[f"{MODEL}_{i}_pts"], P[f"{MODEL}_{i}_prob"]
    if best_set is None:
        for k in SETS:
            pts, rel = F[f"{k}_{i}_pts"], F[f"{k}_{i}_rel"]
            for c in CUTS:
                out[("fcunet", k, c, np.nan)] = pts[rel >= c]
        for t in TS:
            out[(MODEL, "", np.nan, t)] = mp[mprob >= t]
        return out
    pts, rel = F[f"{best_set}_{i}_pts"], F[f"{best_set}_{i}_rel"]
    for c in CUTS:
        fc = pts[rel >= c]
        for t in TS:
            m = mp[mprob >= t]
            out[(f"{MODEL} AND fc", best_set, c, t)] = m[near(m, fc)]
            out[(f"{MODEL} OR fc", best_set, c, t)] = np.vstack([m, fc[~near(fc, m)]])
    return out


def work(args):
    i, r, od, vis_box, best_set, final = args
    img, pots, probe = load(r, od)
    g = bd.ground_truth(img, pots, bd.disk_radius(probe), vis_box)
    F, P = np.load("/mnt/data/fcunet/fcunet_peaks.npz"), np.load(E / "predictions.npz")
    rows = []
    for (m, k, c, t), pts in methods(i, F, P, best_set).items():
        if final is not None and _key(m, k, c, t) not in final:
            continue
        s = bd.score(g, pts)
        row = dict(pattern=i, method=m, setting=k, cut=c, t=t, **{q: s[q] for q in COLS})
        if final is not None:
            for name, im in (("acc_pots", pots), ("acc_img", img)):
                row[name] = weighted_accuracy(im, g["points"], pts, g["beam"], gt_required=g["visible"])["accuracy"]
        rows.append(row)
    return rows


def run(meta, od, vis, best_set=None, final=None):
    jobs = [(i, r, od, vis, best_set, final) for i, r in meta.iterrows()]
    with Pool(4) as pool:
        rows = [x for part in pool.imap_unordered(work, jobs, chunksize=4) for x in part]
    return pd.DataFrame(rows).join(meta[["part", "angle", "thickness"]], on="pattern")


def choose(sw):
    keys = ["method", "setting", "cut", "t"]
    v = sw[sw.part == VAL].fillna({"cut": -1, "t": -1}).groupby(keys).apply(pooled)
    best = v.reset_index().loc[lambda d: d.groupby("method").net_found.idxmax()]
    return {_key(r.method, r.setting, r.cut if r.cut >= 0 else np.nan, r.t if r.t >= 0 else np.nan) for r in best.itertuples()}


def main():
    S = json.loads((E / "settings.json").read_text())
    split = json.loads((E / "threshold_split.json").read_text())
    meta = pd.read_csv(E / "patterns.csv").set_index("pattern")
    meta["part"] = np.where(meta.material.isin(split["val"]), VAL, np.where(meta.material.isin(split["test"]), TEST, OLD))
    od = {f.name: f for f in bd.find_material_folders(json.loads((R / "ft_V/split.json").read_text())["args"]["root"])}

    if (E / "fcunet_sweep.csv.gz").exists():
        sw = pd.read_csv(E / "fcunet_sweep.csv.gz", keep_default_na=False, na_values=[""],
                         dtype={"setting": str}).fillna({"setting": ""})
    else:
        sw = run(meta, od, S["vis_box"])
        best_set = next(k for m, k, c, t in choose(sw) if m == "fcunet")
        sw = pd.concat([sw, run(meta, od, S["vis_box"], best_set=best_set)])
        sw.to_csv(E / "fcunet_sweep.csv.gz", index=False)
    chosen = choose(sw)
    best_set = next(k for m, k, c, t in chosen if m == "fcunet")
    print("chosen:", sorted(chosen), flush=True)

    # final pass at the chosen settings, with intensity-weighted accuracy (mean over patterns)
    fin = pd.concat([run(meta, od, S["vis_box"], best_set=None, final=chosen),
                     run(meta, od, S["vis_box"], best_set=best_set, final=chosen)])
    fin.to_csv(E / "fcunet_final_patterns.csv", index=False)
    out = []
    for (m, k, c, t), d in fin.groupby(["method", "setting", "cut", "t"], dropna=False):
        for part in (VAL, TEST, OLD):
            dp = d[d.part == part]
            out.append(dict(method=m, setting=k, cut=c, t=t, part=part, **pooled(dp).to_dict(),
                            acc_pots=dp.acc_pots.mean(), acc_img=dp.acc_img.mean()))
    b = pd.DataFrame(out)
    b.to_csv(E / "fcunet_best.csv", index=False)
    pd.set_option("display.width", 250)
    for part in (TEST, OLD):
        print(f"\n== {part}")
        print(b[b.part == part].set_index("method")[["setting", "cut", "t", "recall", "truly_false_share", "net_found",
                                                    "f1", "acc_pots", "acc_img", "found", "truly_false", "visible"]]
              .round(3).to_string())


if __name__ == "__main__":
    main()
