#!/usr/bin/env python
"""Which ground-truth positions are more accurate: ours (skimage peak_local_max on dataPots, whole pixels) or
py4DSTEM get_maxima_2D (sub-pixel 'poly')?  Two independent references:

1. Lattice fit: Bragg spots lie on a 2D lattice p = p0 + h a + k b. For each method, index its spots (h, k) and fit
   p0, a, b by least squares to that method's own positions; the residual RMS is the position error (plus any
   true deviation from a lattice, which is the same for both methods).
2. Disk centres at 1 mrad: isolated, bright disks in dataMeas (no other spot within 2r + 2 px, centre >= 10x the
   visibility floor); centre = intensity-weighted centroid within r + 1 px. Distance of each method's position.

    /mnt/data/envs/py4dstem/bin/python gt_accuracy_check.py --out diffraction_runs/gt_py4dstem
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py
import numpy as np
import pandas as pd
from py4DSTEM.preprocess.utils import get_maxima_2D
from scipy.spatial import cKDTree

import benchmark_diffraction as bd
from compare_gt_py4dstem import ROOTS, match

METHODS = {"py4d_s0.5": dict(sigma=0.5), "py4d_s0": dict(sigma=0)}


def py4d(pots, beam, sigma):
    pk = get_maxima_2D(np.asarray(pots, np.float64), subpixel="poly", sigma=sigma, minRelativeIntensity=0.02,
                       edgeBoundary=2, minSpacing=3, maxNumPeaks=100000)
    pts = np.stack([pk["x"], pk["y"]], axis=1).astype(float).reshape(-1, 2)
    return pts[bd.far_from(pts, beam, bd.BEAM_R)]


def basis(pts):
    """Two shortest non-collinear lattice vectors from nearest-neighbour differences (whole-pixel positions)."""
    d, i = cKDTree(pts).query(pts, k=7)
    v = (pts[i[:, 1:]] - pts[:, None]).reshape(-1, 2)
    v = v[np.hypot(*v.T) > 0.5]
    v = v * np.where((v[:, 0] < 0) | ((v[:, 0] == 0) & (v[:, 1] < 0)), -1, 1)[:, None]   # one half-plane
    vals, counts = np.unique(np.round(v).astype(int), axis=0, return_counts=True)
    order = np.argsort(-counts)
    cand = vals[order[:12]].astype(float)
    a = min(cand[:4], key=lambda x: np.hypot(*x))
    for b in sorted(cand, key=lambda x: np.hypot(*x)):
        if abs(np.cross(a, b)) > 0.3 * np.hypot(*a) * np.hypot(*b):
            return a, b
    return None


def lattice_residuals(ref_int, sets):
    """Index the matched spots with the basis from ref_int, then fit a lattice to each set separately."""
    ab = basis(ref_int)
    if ab is None:
        return None
    A = np.stack(ab, axis=1)                                  # 2 x 2, columns a, b
    p0 = ref_int[np.argmin(np.hypot(*(ref_int - ref_int.mean(0)).T))]
    hk = np.linalg.solve(A, (ref_int - p0).T).T
    hk_r = np.round(hk)
    ok = np.all(np.abs(hk - hk_r) < 0.25, axis=1)             # spots that index cleanly (both methods share them)
    if ok.sum() < 12:
        return None
    X = np.column_stack([np.ones(ok.sum()), hk_r[ok]])        # [1, h, k]
    out = {}
    for name, p in sets.items():
        coef, *_ = np.linalg.lstsq(X, p[ok], rcond=None)
        res = p[ok] - X @ coef
        out[name] = np.hypot(*res.T)
    out["n"] = int(ok.sum())
    return out


def disk_centre(img, c, r):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    m = (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= (r + 1) ** 2
    w = img * m
    return np.array([(w * yy).sum(), (w * xx).sum()]) / w.sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--thicknesses", nargs="*", type=float, default=[20, 100, 300, 500])
    a = p.parse_args()
    out = Path(a.out)
    lat_rows, disk_rows = [], []
    for folder, material, uid, angle, h5 in bd.iter_simulations(ROOTS):
        with h5py.File(h5, "r") as h:
            r = bd.disk_radius(np.asarray(h["dataProbe"], np.float32))
            for t_idx, t in enumerate(h["thicknesses"][:]):
                if float(t) not in a.thicknesses:
                    continue
                img = np.asarray(h["dataMeas"][..., t_idx], np.float64)
                pots = np.asarray(h["dataPots"][..., t_idx], np.float64)
                g = bd.ground_truth(img, pots, r, 2 * int(np.ceil(r)) + 1)
                ours = g["points"].astype(float)
                sets = {"ours": ours}
                for name, kw in METHODS.items():
                    q = py4d(pots, g["beam"], **kw)
                    ia, ib = match(ours, q, bd.CUTOFF)
                    full = np.full_like(ours, np.nan)
                    full[ia] = q[ib]
                    sets[name] = full
                both = np.all([np.isfinite(s).all(1) for s in sets.values()], axis=0)
                lr = lattice_residuals(ours[both], {k: v[both] for k, v in sets.items()})
                if lr is not None:
                    lat_rows.append(dict(material=material, angle=angle, thickness=t, n=lr["n"],
                                         **{f"rms_{k}": float(np.sqrt(np.mean(lr[k] ** 2))) for k in sets}))
                if angle == 1.0:
                    tree = cKDTree(ours)
                    nn = tree.query(ours, k=2)[0][:, 1]
                    bright = img[tuple(ours.astype(int).T)] >= 10 * g["floor"]
                    for i in np.where((nn > 2 * r + 2) & bright & both)[0]:
                        c = disk_centre(img, ours[i], r)
                        disk_rows.append(dict(material=material, thickness=t,
                                              **{f"d_{k}": float(np.hypot(*(sets[k][i] - c))) for k in sets}))
        print(material, angle, flush=True)
    lat, disk = pd.DataFrame(lat_rows), pd.DataFrame(disk_rows)
    lat.to_csv(out / "accuracy_lattice.csv", index=False)
    disk.to_csv(out / "accuracy_disk_centres.csv", index=False)
    lines = ["# Accuracy of the ground-truth positions", "",
             f"## 1. Lattice fit residual RMS (px), {len(lat)} patterns, median over patterns",
             lat.groupby("angle")[[c for c in lat if c.startswith("rms_")]].median().round(4).to_string(),
             "all: " + lat[[c for c in lat if c.startswith("rms_")]].median().round(4).to_string().replace("\n", ", "),
             f"py4d_s0 better than ours in {(lat.rms_py4d_s0 < lat.rms_ours).mean():.1%} of patterns; "
             f"py4d_s0.5 better in {(lat['rms_py4d_s0.5'] < lat.rms_ours).mean():.1%}", "",
             f"## 2. Distance to the measured disk centre at 1 mrad (px), {len(disk)} isolated disks",
             disk[[c for c in disk if c.startswith("d_")]].describe(percentiles=[.5, .9]).round(4).to_string()]
    (out / "accuracy.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
