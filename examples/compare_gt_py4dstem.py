#!/usr/bin/env python
"""Compare our ground-truth spot positions (benchmark_diffraction.ground_truth: skimage peak_local_max on dataPots,
min_distance 3, 2 % of the strongest peak, whole pixels) with py4DSTEM's get_maxima_2D on the same dataPots slices.

Runs in the py4DSTEM environment (it does not need spotiflow):
    /mnt/data/envs/py4dstem/bin/python compare_gt_py4dstem.py --out diffraction_runs/gt_py4dstem

Variants of get_maxima_2D (all: subpixel 'poly', sigma 0.5, minRelativeIntensity 0.02, edgeBoundary 2):
    given     minSpacing 4, maxNumPeaks 100     (the settings as proposed)
    nocap     minSpacing 4, maxNumPeaks 100000
    nocap_s3  minSpacing 3, maxNumPeaks 100000  (spacing as ours)
The direct beam (within 3 px of the beam centre) is removed from every set, as in ours.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py4DSTEM.preprocess.utils import get_maxima_2D
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import benchmark_diffraction as bd

ROOTS = ["/mnt/data/DINOV3/dinov3/dummy_data/selected_folders", "/mnt/data/DINOV3/dinov3/dummy_data"]
COMMON = dict(subpixel="poly", sigma=0.5, minRelativeIntensity=0.02, edgeBoundary=2)
VARIANTS = {"given": dict(minSpacing=4, maxNumPeaks=100),
            "nocap": dict(minSpacing=4, maxNumPeaks=100000),
            "nocap_s3": dict(minSpacing=3, maxNumPeaks=100000)}
BLUE, ORANGE, SKY, PINK, YELLOW = "#0072B2", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"


def py4d(pots, beam, **kw):
    pk = get_maxima_2D(np.asarray(pots, np.float64), **COMMON, **kw)
    pts = np.stack([pk["x"], pk["y"]], axis=1).astype(float).reshape(-1, 2)   # x = first axis (row), y = second (col)
    keep = bd.far_from(pts, beam, bd.BEAM_R)
    return pts[keep], np.asarray(pk["intensity"], float)[keep]


def match(a, b, cutoff):
    """One-to-one matching within cutoff (Hungarian on the distance matrix). -> index pairs (ia, ib)."""
    if not len(a) or not len(b):
        return np.zeros(0, int), np.zeros(0, int)
    d = np.hypot(*(a[:, None] - b[None]).transpose(2, 0, 1))
    big = d > cutoff
    d[big] = 1e6
    ia, ib = linear_sum_assignment(d)
    ok = ~big[ia, ib]
    return ia[ok], ib[ok]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--thicknesses", nargs="*", type=float, default=None, help="default: all 25")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, offsets, examples = [], [], []
    for folder, material, uid, angle, h5 in bd.iter_simulations(ROOTS):
        with h5py.File(h5, "r") as h:
            probe = np.asarray(h["dataProbe"], np.float32)
            r = bd.disk_radius(probe)
            thick = h["thicknesses"][:]
            for t_idx, t in enumerate(thick):
                if a.thicknesses and float(t) not in a.thicknesses:
                    continue
                img = np.asarray(h["dataMeas"][..., t_idx], np.float32)
                pots = np.asarray(h["dataPots"][..., t_idx], np.float32)
                g = bd.ground_truth(img, pots, r, 2 * int(np.ceil(r)) + 1)
                ours, vis = g["points"].astype(float), g["visible"]
                pmax = pots.max()
                row = dict(material=material, angle=angle, thickness=float(t), n_ours=len(ours), n_vis=int(vis.sum()))
                for name, kw in VARIANTS.items():
                    pts, inten = py4d(pots, g["beam"], **kw)
                    ia, ib = match(ours, pts, bd.CUTOFF)
                    i1, _ = match(ours, pts, 1.0)
                    only_ours = np.setdiff1d(np.arange(len(ours)), ia)
                    only_p = np.setdiff1d(np.arange(len(pts)), ib)
                    row.update({f"{name}_n": len(pts), f"{name}_matched": len(ia), f"{name}_within1px": len(i1),
                                f"{name}_only_ours": len(only_ours), f"{name}_only_ours_vis": int(vis[only_ours].sum()),
                                f"{name}_only_py4d": len(only_p)})
                    if name == "nocap":
                        d = pts[ib] - ours[ia]
                        offsets.append(pd.DataFrame(dict(angle=angle, dy=d[:, 0], dx=d[:, 1],
                                                         pots_rel=pots[ours[ia, 0].astype(int), ours[ia, 1].astype(int)] / pmax)))
                        row["nocap_only_ours_pots_med"] = float(np.median(pots[tuple(ours[only_ours].astype(int).T)] / pmax)) if len(only_ours) else np.nan
                        row["nocap_only_py4d_int_med"] = float(np.median(inten[only_p] / pmax)) if len(only_p) else np.nan
                    if name == "given" and material == "BW" and t == 100:
                        examples.append((angle, img, pots, ours, vis, pts, g["beam"]))
                rows.append(row)
        print(material, angle, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_pattern.csv", index=False)
    off = pd.concat(offsets)
    off.to_csv(out / "offsets_nocap.csv.gz", index=False)

    lines = [f"# Ground truth: ours (skimage peak_local_max) vs py4DSTEM get_maxima_2D on dataPots",
             f"{len(df)} patterns ({df.material.nunique()} materials x {df.angle.nunique()} angles x thicknesses); "
             f"matching one-to-one within {bd.CUTOFF} px; direct beam removed from all sets", ""]
    for name, kw in VARIANTS.items():
        s = df[["n_ours", "n_vis", f"{name}_n", f"{name}_matched", f"{name}_within1px", f"{name}_only_ours",
                f"{name}_only_ours_vis", f"{name}_only_py4d"]].sum()
        lines.append(f"## {name} {kw}")
        lines.append(f"ours {int(s.n_ours)} spots ({int(s.n_vis)} visible); py4DSTEM {int(s[f'{name}_n'])}; matched "
                     f"{int(s[f'{name}_matched'])} = {s[f'{name}_matched'] / s.n_ours:.1%} of ours "
                     f"({s[f'{name}_within1px'] / s.n_ours:.1%} within 1 px); only ours {int(s[f'{name}_only_ours'])} "
                     f"(visible {int(s[f'{name}_only_ours_vis'])} = {s[f'{name}_only_ours_vis'] / s.n_vis:.1%} of visible labels); "
                     f"only py4DSTEM {int(s[f'{name}_only_py4d'])}")
        by = df.groupby("angle")[["n_ours", "n_vis", f"{name}_n", f"{name}_matched", f"{name}_only_ours_vis", f"{name}_only_py4d"]].sum()
        by["matched_frac"] = (by[f"{name}_matched"] / by.n_ours).round(3)
        by["vis_missing_frac"] = (by[f"{name}_only_ours_vis"] / by.n_vis).round(3)
        lines.append(by.to_string())
        if name == "given":
            capped = (df.given_n >= 100 - 1)
            lines.append(f"patterns where py4DSTEM hit maxNumPeaks=100 (after beam removal >= 99): {capped.mean():.1%}")
        lines.append("")
    lines.append("## position offsets py4DSTEM (nocap, sub-pixel) - ours (whole pixels), matched pairs")
    lines.append(off.groupby("angle")[["dy", "dx"]].agg(["mean", "median", "std"]).round(3).to_string())
    rr = np.hypot(off.dy, off.dx)
    lines.append(f"|offset|: median {np.median(rr):.3f} px, 95 % {np.percentile(rr, 95):.3f} px, max {rr.max():.3f} px")
    lines.append(f"nocap: unmatched ours, median dataPots / max {df.nocap_only_ours_pots_med.median():.3f}; "
                 f"unmatched py4DSTEM, median intensity / max {df.nocap_only_py4d_int_med.median():.3f}")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # figures
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    for ax, ang in zip(axs, sorted(off.angle.unique())):
        o = off[off.angle == ang]
        ax.hist2d(o.dx, o.dy, bins=60, range=[[-1, 1], [-1, 1]], cmap="cividis")
        ax.set(title=f"{ang:g} mrad: py4DSTEM - ours (px), n={len(o)}", xlabel="dx (col)", ylabel="dy (row)", aspect="equal")
    fig.savefig(out / "offsets.png", dpi=110)
    plt.close(fig)
    fig, axs = plt.subplots(1, len(examples), figsize=(6 * len(examples), 6.4), constrained_layout=True)
    for ax, (ang, img, pots, ours, vis, pts, beam) in zip(np.atleast_1d(axs), examples):
        ax.imshow(np.log10(np.maximum(img, img.max() * 1e-5)), cmap="gray")
        ax.scatter(ours[:, 1], ours[:, 0], s=70, facecolors="none", edgecolors=SKY, lw=1.3, label=f"ours ({len(ours)})")
        ax.scatter(pts[:, 1], pts[:, 0], s=25, c=ORANGE, marker="x", lw=1.3, label=f"py4DSTEM given ({len(pts)})")
        ax.set_title(f"BW {ang:g} mrad 100 Å (dataMeas, log)")
        ax.legend(loc="lower right", fontsize=9)
        ax.axis("off")
    fig.savefig(out / "example_BW.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
