#!/usr/bin/env python
"""How much of the experimental "error" is due to the r4 labels rather than the models? (step 1 of the improvement plan)

The experimental test scores (eval_experimental.py) compare predictions with r4 disks of confidence >= 0.8. Three
things in that reference can make the models look worse than they are:

1. the direct beam: r4 lists it as a disk in almost every pattern, but predictions within BEAM_R of the beam are
   ignored, so this label is always a miss;
2. the 0.8 cut: r4 disks with score 0.5-0.8 are dropped, so a model that finds them gets a false positive;
3. disks r4 did not report at all (score < 0.5 or missed).

Scorings (each model at its own experimental threshold, same 6 test acquisitions x 500 patterns):
    current      labels r4 >= 0.8, beam label kept                     (= eval_experimental.py)
    no_beam      labels r4 >= 0.8, beam label removed
    all_r4       labels r4 >= 0.5, beam label removed
    lenient      r4 >= 0.8 must be found; r4 0.5-0.8 are optional (a hit on one is neither TP nor FP, a miss is not FN)
Also the F1-optimal threshold of each scoring on the validation acquisitions.

For 3 (no r4 label at all) the local contrast of every off-beam V_mix2 detection is measured on the raw counts:
    snr = (mean in r <= 2 px - median of annulus 5-8 px) / (1.4826 * MAD of annulus)
and compared with r4-labelled hits, missed labels and random background positions; crops are saved for an eye check.

Outputs in diffraction_runs/eval_exp/labels_check/: scores.csv, sweep.csv, detections.csv, snr_summary.csv, *.png

    python check_exp_labels.py
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_diffraction as bd
import finetune_experimental as fe
from experimental_data import load_experimental_conf
from eval_experimental import RUNS, predict_low

HERE = Path(__file__).parent
SWEEP = np.round(np.arange(0.10, 0.901, 0.02), 2)
MODELS = ("V_exp", "V_mix", "V_mix2")   # V (simulations only) has F1 0.04 on these data and is left out
BLUE, SKY, ORANGE, YELLOW, PINK, GREY = "#0072B2", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7", "#999999"


def load_with_conf(root, acquisitions, per_acquisition):
    return load_experimental_conf(root, acquisitions, per_acquisition)


def match(a, b, cutoff=bd.CUTOFF):
    """One-to-one matching (spotiflow's points_matching) -> (index in a, index in b) arrays."""
    from spotiflow.utils.matching import points_matching
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    pairs = np.asarray(points_matching(a, b, cutoff_distance=cutoff).matched_pairs, int).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def counts(pred, pts, conf, beam, scoring):
    """tp / n_pred / n_label of one pattern under one scoring."""
    pred = pred[bd.far_from(pred, beam, bd.BEAM_R)]
    if scoring == "current":
        g = pts[conf >= 0.8]
    else:
        keep = bd.far_from(pts, beam, bd.BEAM_R)
        pts, conf = pts[keep], conf[keep]
        g = pts[conf >= 0.8] if scoring in ("no_beam", "lenient") else pts
    ig, ip = match(g, pred)
    tp, n_pred, n_label = len(ig), len(pred), len(g)
    if scoring == "lenient":  # unmatched predictions on a 0.5-0.8 disk are not false positives
        rest = np.setdiff1d(np.arange(len(pred)), ip)
        _, iw = match(pts[conf < 0.8], pred[rest])
        n_pred -= len(iw)
    return tp, n_pred, n_label


def pooled(rows):
    df = pd.DataFrame(rows, columns=["material", "tp", "n_pred", "n_label"])
    out = df.groupby("material")[["tp", "n_pred", "n_label"]].sum()
    out.loc["all"] = out.sum()
    out["precision"] = out.tp / out.n_pred.clip(lower=1)
    out["recall"] = out.tp / out.n_label
    out["f1"] = 2 * out.tp / (out.n_pred + out.n_label)
    return out


def score(low, pts, conf, meta, thr, scoring):
    rows = []
    for lw, p, c, (_, m) in zip(low, pts, conf, meta.iterrows()):
        pred = lw[0][lw[1] >= thr]
        rows.append((m.material, *counts(pred, p, c, (m.beam_r, m.beam_c), scoring)))
    return pooled(rows)


def local_snr(img, pts):
    """Contrast of each point: (mean r<=2 - median annulus 5-8) / robust std of the annulus. NaN near the border."""
    yy, xx = np.mgrid[-8:9, -8:9]
    rr = np.hypot(yy, xx)
    core, ring = rr <= 2, (rr >= 5) & (rr <= 8)
    pad = np.pad(img, 8, mode="constant", constant_values=np.nan)
    out = np.full(len(pts), np.nan)
    for k, (r, c) in enumerate(np.round(np.asarray(pts).reshape(-1, 2)).astype(int)):
        if not (0 <= r < img.shape[0] and 0 <= c < img.shape[1]):
            continue
        w = pad[r:r + 17, c:c + 17]
        a = w[ring]
        if np.isnan(a).any():
            continue
        mad = 1.4826 * np.median(np.abs(a - np.median(a)))
        out[k] = (w[core].mean() - np.median(a)) / max(mad, 1e-6)
    return out


def classify(img, pred, pts, conf, beam, rng):
    """Off-beam predictions -> category by the r4 label they hit; plus missed >=0.8 labels and background points."""
    beam = np.asarray(beam)
    pred = pred[bd.far_from(pred, beam, bd.BEAM_R)]
    keep = bd.far_from(pts, beam, bd.BEAM_R)
    pts, conf = pts[keep], conf[keep]
    strong, weak = pts[conf >= 0.8], pts[conf < 0.8]
    cat = np.full(len(pred), "no r4 label", dtype=object)
    ig, ip = match(strong, pred)
    cat[ip] = "r4 >= 0.8"
    rest = np.setdiff1d(np.arange(len(pred)), ip)
    _, iw = match(weak, pred[rest])
    cat[rest[iw]] = "r4 0.5-0.8"
    d = (np.min(np.hypot(*(pred[:, None] - pts[None]).transpose(2, 0, 1)), axis=1)
         if len(pts) and len(pred) else np.full(len(pred), np.inf))
    missed = strong[np.setdiff1d(np.arange(len(strong)), ig)]
    # background: random positions >= 8 px from every r4 disk, prediction and the beam
    taken = np.concatenate([pts, pred, beam[None]]) if len(pred) else np.concatenate([pts, beam[None]])
    cand = rng.uniform([8, 8], np.array(img.shape) - 8, size=(60, 2))
    cand = cand[np.min(np.hypot(*(cand[:, None] - taken[None]).transpose(2, 0, 1)), axis=1) >= 8][:10]
    return pred, cat, d, missed, cand


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/mnt/data/4D-ImageNet/converted")
    ap.add_argument("--split", default=str(HERE / "diffraction_runs/exp_split.json"))
    ap.add_argument("--runs", default=str(HERE / "diffraction_runs"))
    ap.add_argument("--out", default=str(HERE / "diffraction_runs/eval_exp/labels_check"))
    ap.add_argument("--eval-per-acq", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--thresholds", default=str(HERE / "diffraction_runs/eval_exp/thresholds.json"),
                    help="thresholds.json of eval_experimental.py (each model's experimental threshold)")
    ap.add_argument("--teacher", default="V_mix2", help="model whose detections are analysed (contrast, galleries)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the threshold sweep (slow, ~50 min)")
    args = ap.parse_args()
    logging.getLogger("spotiflow").setLevel(logging.WARNING)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spotiflow.model import Spotiflow

    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    thr = {k: v["exp"] for k, v in json.loads(Path(args.thresholds).read_text()).items()}
    data = {role: load_with_conf(args.data, split[role], args.eval_per_acq) for role in ("val", "test")}

    cache = out / "pred_low.pkl"
    low = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    if any(k not in low for k in args.models):  # predict the models missing from the cache
        for k in [k for k in args.models if k not in low]:
            m = Spotiflow.from_folder(str(runs / RUNS.get(k, f"ft_{k}") / "model"), map_location=args.device)
            low[k] = {role: predict_low(m, [fe.log_input(im) for im in data[role][0]], args.device)
                      for role in data}
            print(f"{k} predicted", flush=True)
        cache.write_bytes(pickle.dumps(low))

    # --- 1+2: scorings -------------------------------------------------------------------------------------------
    scorings = ("current", "no_beam", "all_r4", "lenient")
    _, pts, conf, meta = data["test"]
    tab, sweep = [], []
    for k in args.models:
        for s in scorings:
            tab.append(score(low[k]["test"], pts, conf, meta, thr[k], s).assign(model=k, scoring=s, threshold=thr[k]))
            for role in (() if args.no_sweep else ("val", "test")):
                _, p_, c_, m_ = data[role]
                for t in SWEEP:
                    a = score(low[k][role], p_, c_, m_, float(t), s).loc["all"]
                    sweep.append(dict(model=k, scoring=s, split=role, threshold=float(t), **a.to_dict()))
    tab = pd.concat(tab).reset_index().set_index(["model", "scoring", "material"]).round(4)
    tab.to_csv(out / "scores.csv")
    sweep = pd.DataFrame(sweep, columns=["model", "scoring", "split", "threshold", "tp", "n_pred", "n_label",
                                         "precision", "recall", "f1"]).round(4)
    if not args.no_sweep:
        sweep.to_csv(out / "sweep.csv", index=False)
    print(tab.xs("all", level="material")[["threshold", "precision", "recall", "f1"]].to_string(), flush=True)

    best = []
    for (k, s), g in sweep[sweep.split == "val"].groupby(["model", "scoring"]):
        t = g.loc[g.f1.idxmax(), "threshold"]
        r = sweep[(sweep.split == "test") & (sweep.model == k) & (sweep.scoring == s) & (sweep.threshold == t)].iloc[0]
        best.append(dict(model=k, scoring=s, val_best_threshold=t, test_precision=r.precision,
                         test_recall=r.recall, test_f1=r.f1))
    best = pd.DataFrame(best)
    if not args.no_sweep:
        best.round(4).to_csv(out / "best_threshold.csv", index=False)
    print(best.round(3).to_string(), flush=True)

    # --- 3: detections without an r4 label (V_mix2) -------------------------------------------------------------
    imgs = data["test"][0]
    rng = np.random.default_rng(0)
    det, pts_rows = [], []
    for i, (lw, img, p, c, (_, m)) in enumerate(zip(low[args.teacher]["test"], imgs, pts, conf, meta.iterrows())):
        pred, cat, d, missed, bg = classify(img, lw[0][lw[1] >= thr[args.teacher]], p, c, (m.beam_r, m.beam_c), rng)
        prob = lw[1][lw[1] >= thr[args.teacher]][bd.far_from(lw[0][lw[1] >= thr[args.teacher]], (m.beam_r, m.beam_c), bd.BEAM_R)]
        for kind, P, extra in (("pred", pred, dict(cat=cat, dist_r4=d, prob=prob)),
                               ("missed r4 >= 0.8", missed, {}), ("background", bg, {})):
            snr = local_snr(img, P)
            for j in range(len(P)):
                row = dict(pattern=i, material=m.material, kind=kind, r=P[j, 0], c=P[j, 1], snr=snr[j])
                row.update({key: v[j] for key, v in extra.items()})
                det.append(row)
    det = pd.DataFrame(det)
    det["group"] = np.where(det.kind == "pred", "detected, " + det.cat.astype(str), det.kind)
    det.round(3).to_csv(out / "detections.csv", index=False)

    nolab = det.group == "detected, no r4 label"
    det.loc[nolab, "near"] = pd.cut(det.loc[nolab, "dist_r4"], [0, 6, 10, np.inf], right=False,
                                     labels=["3-6 px from an r4 disk", "6-10 px", "> 10 px"])
    bg95 = det.loc[det.group == "background", "snr"].quantile(0.95)
    tp05 = det.loc[det.group == "detected, r4 >= 0.8", "snr"].quantile(0.05)
    summ = det.groupby("group").snr.describe(percentiles=[0.05, 0.25, 0.5, 0.75])
    summ["frac_above_bg95"] = det.groupby("group").snr.apply(lambda s: (s > bg95).mean())
    summ["frac_above_tp05"] = det.groupby("group").snr.apply(lambda s: (s > tp05).mean())
    summ.round(3).to_csv(out / "snr_summary.csv")
    print(f"\nbackground snr 95th pct {bg95:.2f}, r4>=0.8 hits 5th pct {tp05:.2f}")
    print(summ.round(2).to_string())
    print("\nno-label detections by distance to nearest r4 disk:")
    print(det[nolab].groupby("near", observed=True).agg(n=("snr", "size"), median_snr=("snr", "median"),
                                                        above_bg95=("snr", lambda s: (s > bg95).mean())).round(2))
    print("\nno-label detections per material (share of all detections):")
    print((det[det.kind == "pred"].groupby("material").cat.value_counts(normalize=True).unstack().round(3)))

    # --- plots -------------------------------------------------------------------------------------------------
    order = ["background", "detected, no r4 label", "detected, r4 0.5-0.8", "missed r4 >= 0.8", "detected, r4 >= 0.8"]
    col = dict(zip(order, [GREY, YELLOW, SKY, PINK, BLUE]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-5, 60, 66)
    for g in order:
        s = det.loc[det.group == g, "snr"].dropna().clip(-5, 60)
        ax.hist(s, bins, histtype="step", lw=2, color=col[g], density=True,
                ls="--" if g in ("background", "missed r4 >= 0.8") else "-", label=f"{g} (n={len(s)})")
    ax.axvline(bg95, color=GREY, ls=":", label=f"95 % of background below {bg95:.1f}")
    ax.set(xlabel="local contrast snr (raw counts, clipped to 60)", ylabel="density",
           title=f"{args.teacher} detections on the test acquisitions: contrast by r4 label")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "snr_hist.png", dpi=120)

    def gallery(sel, name, title, n=48):
        sel = sel.sample(min(n, len(sel)), random_state=0)
        fig, axs = plt.subplots(6, 8, figsize=(13, 10.5))
        for a in axs.ravel():
            a.axis("off")
        for a, (_, d) in zip(axs.ravel(), sel.iterrows()):
            img = np.log1p(imgs[int(d.pattern)] / imgs[int(d.pattern)].max() * 1000)
            r, c = int(round(d.r)), int(round(d.c))
            w = np.pad(img, 16)[r:r + 33, c:c + 33]
            a.imshow(w, cmap="gray", vmin=0, vmax=np.percentile(img, 99.9))
            pp, cc, dd = pts[int(d.pattern)], conf[int(d.pattern)], (r - 16, c - 16)
            for q, cf in zip(pp, cc):
                y, x = q[0] - dd[0], q[1] - dd[1]
                if 0 <= y < 33 and 0 <= x < 33:
                    a.plot(x, y, "o" if cf >= 0.8 else "s", mfc="none", mec=SKY if cf >= 0.8 else ORANGE, ms=9)
            a.plot(d.c - dd[1], d.r - dd[0], "+", color=YELLOW, ms=12, mew=2)
            a.set_title(f"{d.material} snr {d.snr:.1f}", fontsize=7)
        fig.suptitle(f"{title}\nyellow + : the point, sky circle: r4 >= 0.8, orange square: r4 0.5-0.8 "
                     "(33x33 px, log scale)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / name, dpi=110)

    gallery(det[nolab & (det.dist_r4 >= 6)], "gallery_no_label.png", f"{args.teacher} detections with no r4 disk within 6 px")
    gallery(det[nolab & (det.dist_r4 < 6)], "gallery_no_label_near.png", f"{args.teacher} detections 3-6 px from an r4 disk")
    gallery(det[det.group == "missed r4 >= 0.8"], "gallery_missed.png", f"r4 >= 0.8 disks {args.teacher} missed")
    gallery(det[det.group == "detected, r4 >= 0.8"], "gallery_hits.png", f"{args.teacher} hits on r4 >= 0.8 (reference)")
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
