#!/usr/bin/env python
"""Intensity-weighted accuracy of disk positions (Munshi et al., npj Comput. Mater. 8, 254 (2022), eqs. 1-4):

    TP_int = sum of true-positive peak intensities  / sum of predicted peak intensities
    FP_int = sum of false-positive peak intensities / sum of predicted peak intensities
    FN_int = sum of false-negative peak intensities / sum of ground-truth peak intensities
    accuracy = TP_int / (TP_int + FP_int + FN_int)            (= TP_int / (1 + FN_int), since TP_int + FP_int = 1)

Adaptation: the paper reads intensities from its predicted structure-factor images. Here a peak's intensity is the
3 x 3 maximum of the LINEAR diffraction image at its position (dataMeas for simulations, the rescaled raw image for
experimental patterns), for predictions and labels alike. Matching uses our 3 px cutoff (the paper: 0.05 1/A,
about 2.3 px of its data). The direct beam (3 px) is excluded, as in the F1 scoring. Simulation: a prediction matched
to any ground-truth candidate is a true positive (as bd.score: spots below the visibility floor are not false);
false negatives are missed VISIBLE spots. Experimental: ground truth = r4 labels (conf >= 0.8).

Per pattern; the test-set value is the mean over patterns.

    python intensity_accuracy.py --models V_exp V_mix V_mix2 V_mix3b V_mix4b V_mix5b     # -> eval_exp2/intensity_accuracy*.csv
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

import benchmark_diffraction as bd

R = Path(__file__).resolve().parent / "diffraction_runs"


def peak_intensity(img_max3, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return np.zeros(0)
    rc = np.clip(np.round(pts).astype(int), 0, np.array(img_max3.shape) - 1)
    return img_max3[rc[:, 0], rc[:, 1]].astype(float)


def weighted_accuracy(img, gt, pred, beam, gt_required=None, cutoff=bd.CUTOFF, beam_r=bd.BEAM_R):
    """img: linear image; gt: (n, 2) ground-truth points; pred: (m, 2); gt_required: bool (n,) of the points that
    count as false negatives when missed (default all). -> dict(tp_int, fp_int, fn_int, accuracy, n_tp, n_fp, n_fn)."""
    from spotiflow.utils.matching import points_matching
    gt = np.asarray(gt, float).reshape(-1, 2)
    pred = np.asarray(pred, float).reshape(-1, 2)
    req = np.ones(len(gt), bool) if gt_required is None else np.asarray(gt_required, bool)
    keep_g = bd.far_from(gt, beam, beam_r) if len(gt) else np.zeros(0, bool)
    gt, req = gt[keep_g], req[keep_g]
    pred = pred[bd.far_from(pred, beam, beam_r)] if len(pred) else pred
    m3 = maximum_filter(np.asarray(img, np.float32), size=3)
    ip, ig = peak_intensity(m3, pred), peak_intensity(m3, gt)
    tp_p, tp_g = np.zeros(len(pred), bool), np.zeros(len(gt), bool)
    if len(pred) and len(gt):
        for a, b in points_matching(gt, pred, cutoff_distance=cutoff).matched_pairs:
            tp_g[a], tp_p[b] = True, True
    fn = req & ~tp_g
    sp, sg = ip.sum(), ig[req].sum()
    tp_int = ip[tp_p].sum() / sp if sp > 0 else 0.0
    fp_int = ip[~tp_p].sum() / sp if sp > 0 else 0.0
    fn_int = ig[fn].sum() / sg if sg > 0 else 0.0
    den = tp_int + fp_int + fn_int
    return dict(tp_int=tp_int, fp_int=fp_int, fn_int=fn_int, accuracy=tp_int / den if den > 0 else np.nan,
                n_tp=int(tp_p.sum()), n_fp=int((~tp_p).sum()), n_fn=int(fn.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--eval", default=str(R / "eval_exp2"))
    p.add_argument("--data", default="/mnt/data/4D-ImageNet/converted")
    p.add_argument("--eval-per-acq", type=int, default=500)
    a = p.parse_args()
    logging.getLogger("spotiflow").setLevel(logging.WARNING)
    import finetune_experimental as fe
    from eval_experimental import RUNS, at, predict_low
    from experimental_data import load_experimental
    from spotiflow.model import Spotiflow

    ev = Path(a.eval)
    one = pd.read_csv(ev / "one_threshold_summary.csv").set_index("model")
    split = json.loads((R / "exp_split.json").read_text())
    imgs, labels, meta = load_experimental(a.data, split["test"], min_conf=0.8, per_acquisition=a.eval_per_acq)
    xs = [fe.log_input(im) for im in imgs]
    sims = fe.load_simulation_test(R / "ft_V")
    rows = []
    for k in a.models:
        t = float(one.loc[k, "thr"])
        model = Spotiflow.from_folder(str(R / RUNS.get(k, f"ft_{k}") / "model"), map_location="cuda")
        for lw, im, g, (_, m) in zip(predict_low(model, xs, "cuda"), imgs, labels, meta.iterrows()):
            rows.append(dict(model=k, data="experimental", threshold=t, material=m.material, angle=np.nan,
                             **weighted_accuracy(im, g[:, :2], at(lw, t), np.array([m.beam_r, m.beam_c]))))
        for lw, it in zip(predict_low(model, [it["x"] for it in sims], "cuda"), sims):
            g = it["g"]
            rows.append(dict(model=k, data="simulation", threshold=t, material=it["material"], angle=it["angle"],
                             **weighted_accuracy(it["img"], g["points"], at(lw, t), g["beam"], gt_required=g["visible"])))
        print(k, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ev / "intensity_accuracy_patterns.csv", index=False)
    cols = ["accuracy", "tp_int", "fp_int", "fn_int"]
    s = df.groupby(["model", "data"])[cols].mean().unstack("data")
    s.columns = [f"{d}_{c}" for c, d in s.columns]
    s.insert(0, "threshold", df.groupby("model").threshold.first())
    s["mean_accuracy"] = (s.experimental_accuracy + s.simulation_accuracy) / 2
    s = s.loc[a.models]
    s.round(4).to_csv(ev / "intensity_accuracy.csv")
    per = df.groupby(["model", "data", "material"]).accuracy.mean().unstack(["data", "material"])
    per.round(4).to_csv(ev / "intensity_accuracy_by_material.csv")
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
