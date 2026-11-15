#!/usr/bin/env python
"""Training-free fixes for the new simulated materials: post-processing of the stored detections (eval_sim93.py,
threshold 0.05) of V, V_mix2 and V_mix4b. Every variant and its threshold are chosen on the 31 new validation
materials by the goal score (net found = (visible spots found - truly false detections) / visible spots) and scored
on the 62 new test materials and the 3 old test materials.

Variants
    base           probability threshold only
    angle          a separate threshold per probe angle (chosen per angle on the validation materials)
    nms<k>         minimum distance: of detections closer than d = max(3, min(k * r, 0.6 * s)) keep the most probable
                   (r = probe disk radius, s = median nearest-neighbour distance of the confident detections)
    lattice<t>     fit the 2D lattice p = p0 + h a + k b to the confident detections (prob >= 0.5; the direct beam is a
                   lattice point) and keep detections within t px of a lattice point; skipped (all kept) when the
                   lattice explains < 60 % of the confident detections
    refine         disk-centre refinement: move each detection to the maximum of the image's disk-template mean
                   (disk of radius r on the log input) within min(r, 0.4 s) px; duplicates (< 3 px) merged
    refine+lattice<t>, refine+nms<k>

    python postproc_sim93.py      # -> diffraction_runs/sim93_eval/postproc_sweep.csv.gz, postproc_best.csv
"""
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

import benchmark_diffraction as bd

R = Path(__file__).resolve().parent / "diffraction_runs"
E = R / "sim93_eval"
NEW = Path("/mnt/data/sim_diffraction_gdrive")
THRESHOLDS = np.round(np.arange(0.10, 0.651, 0.05), 2)
VARIANTS = ["base", "nms0.5", "nms0.75", "nms1.0", "lattice1.5", "lattice2.5", "refine", "refine+lattice2.5",
            "refine+nms0.75"]
COLS = ("n_vis", "n_pred", "found_vis", "false", "false_near_spot", "false_on_weak_peak", "truly_false")


# ---------------------------------------------------------------------------------------------------- post-processing
def spacing_of(p):
    return float(np.median(cKDTree(p).query(p, k=2)[0][:, 1])) if len(p) > 2 else np.inf


def nms(p, prob, d):
    keep, taken = [], np.zeros(len(p), bool)
    tree = cKDTree(p) if len(p) else None
    for i in np.argsort(-prob):
        if taken[i]:
            continue
        keep.append(i)
        taken[tree.query_ball_point(p[i], d)] = True
    return np.array(sorted(keep), int)


def lattice_basis(p, beam):
    """Two shortest common non-collinear neighbour vectors of the points (and the beam), refined by least squares."""
    q = np.vstack([p, beam[None]])
    if len(q) < 6:
        return None
    _, idx = cKDTree(q).query(q, k=min(7, len(q)))
    v = (q[idx[:, 1:]] - q[:, None]).reshape(-1, 2)
    v = v[np.hypot(*v.T) > 2]
    v = v * np.where((v[:, 0] < 0) | ((v[:, 0] == 0) & (v[:, 1] < 0)), -1, 1)[:, None]
    vals, counts = np.unique(np.round(v).astype(int), axis=0, return_counts=True)
    cand = vals[np.argsort(-counts)[:12]].astype(float)
    a = min(cand[:4], key=lambda x: np.hypot(*x))
    b = next((c for c in sorted(cand, key=lambda x: np.hypot(*x))
              if abs(a[0] * c[1] - a[1] * c[0]) > 0.3 * np.hypot(*a) * np.hypot(*c)), None)
    if b is None:
        return None
    A = np.stack([a, b], axis=1)
    hk = np.linalg.solve(A, (q - beam).T).T
    ok = np.all(np.abs(hk - np.round(hk)) < 0.25, axis=1)
    if ok.sum() < 6:
        return None
    X = np.column_stack([np.ones(ok.sum()), np.round(hk[ok])])
    coef, *_ = np.linalg.lstsq(X, q[ok], rcond=None)                   # rows: p0, a, b
    a2, b2 = coef[1], coef[2]
    if abs(a2[0] * b2[1] - a2[1] * b2[0]) < 0.3 * np.hypot(*a2) * np.hypot(*b2):
        return None                                                    # degenerate fit
    return coef


def lattice_dist(p, coef):
    p0, A = coef[0], coef[1:].T
    hk = np.round(np.linalg.solve(A, (p - p0).T).T)
    return np.hypot(*(p - (p0 + hk @ coef[1:])).T)


def lattice_filter(p, prob, beam, tol, conf=0.5):
    """Indices of the detections kept; all of them when no reliable lattice can be fitted."""
    c = p[prob >= conf]
    if len(c) < 8:
        c = p[np.argsort(-prob)[:15]]
    try:
        coef = lattice_basis(c, beam)
        if coef is None or np.mean(lattice_dist(c, coef) < 1.5) < 0.6:
            return np.arange(len(p))
        return np.where(lattice_dist(p, coef) <= tol)[0]
    except np.linalg.LinAlgError:
        return np.arange(len(p))


def refine(p, prob, x, r, s):
    """Move each detection to the maximum of the disk-mean image within min(r, 0.4 s) px; merge duplicates."""
    if r < 3.5 or not len(p):
        return p, prob
    rr = int(np.ceil(r))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disk = (yy ** 2 + xx ** 2 <= r ** 2).astype(float)
    m = fftconvolve(x, disk / disk.sum(), mode="same")
    w = max(1, int(round(min(r, 0.4 * s))))
    out = p.copy()
    for i, (y, x0) in enumerate(np.round(p).astype(int)):
        y0, y1, x1, x2 = max(0, y - w), min(m.shape[0], y + w + 1), max(0, x0 - w), min(m.shape[1], x0 + w + 1)
        win = m[y0:y1, x1:x2]
        dy, dx = np.unravel_index(np.argmax(win), win.shape)
        out[i] = (y0 + dy, x1 + dx)
    keep = nms(out, prob, 3.0)
    return out[keep], prob[keep]


def apply(variant, p, prob, beam, x, r):
    s = spacing_of(p[prob >= 0.5]) if (prob >= 0.5).sum() > 2 else spacing_of(p)
    if variant.startswith("refine"):
        p, prob = refine(p, prob, x, r, s)
        variant = variant[len("refine+"):] if "+" in variant else "base"
    if variant.startswith("nms"):
        k = nms(p, prob, max(3.0, min(float(variant[3:]) * r, 0.6 * s)))
        p, prob = p[k], prob[k]
    elif variant.startswith("lattice"):
        k = lattice_filter(p, prob, beam, float(variant[7:]))
        p, prob = p[k], prob[k]
    return p, prob


# ---------------------------------------------------------------------------------------------------- evaluation
def load(r, old_dirs):
    folder = NEW / r.folder if r.dataset == "new" else old_dirs[r.folder]
    with h5py.File(folder / f"{r.uid}_training.h5", "r") as h:
        ti = int(np.where(h["thicknesses"][:] == r.thickness)[0][0])
        return h["dataMeas"][..., ti], h["dataPots"][..., ti], h["dataProbe"][:]


def work(args):
    i, r, models, old_dirs, vis_box, log_scale = args
    img, pots, probe = load(r, old_dirs)
    rad = bd.disk_radius(probe)
    g = bd.ground_truth(img, pots, rad, vis_box)
    x = bd.spotiflow_input(img, probe, "log", log_scale=log_scale)
    P = np.load(E / "predictions.npz")
    rows = []
    for k in models:
        pts, prob = P[f"{k}_{i}_pts"], P[f"{k}_{i}_prob"]
        pts, prob = pts[prob >= THRESHOLDS[0]], prob[prob >= THRESHOLDS[0]]
        for v in VARIANTS:
            for t in THRESHOLDS:
                pv, bv = apply(v, pts[prob >= t], prob[prob >= t], g["beam"], x, rad)
                s = bd.score(g, pv)
                rows.append(dict(model=k, pattern=i, variant=v, threshold=t, **{c: s[c] for c in COLS}))
    return rows


def pooled(d):
    t = d[list(COLS)].sum()
    P_, R_ = 1 - t["false"] / max(t.n_pred, 1), t.found_vis / max(t.n_vis, 1)
    return pd.Series(dict(recall=R_, truly_false_share=t.truly_false / max(t.n_pred, 1), precision=P_,
                          f1=2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0, net_found=(t.found_vis - t.truly_false) / max(t.n_vis, 1),
                          found=int(t.found_vis), truly_false=int(t.truly_false), visible=int(t.n_vis)))


def main():
    S = json.loads((E / "settings.json").read_text())
    split = json.loads((E / "threshold_split.json").read_text())
    meta = pd.read_csv(E / "patterns.csv").set_index("pattern")
    meta["part"] = np.where(meta.material.isin(split["val"]), "new validation (31)",
                            np.where(meta.material.isin(split["test"]), "new test (62)", "old test"))
    old_dirs = {f.name: f for f in bd.find_material_folders(json.loads((R / "ft_V/split.json").read_text())["args"]["root"])}
    jobs = [(i, r, S["models"], old_dirs, S["vis_box"], S["log_scale"]) for i, r in meta.iterrows()]
    with Pool(4) as pool:
        rows = [row for part in pool.imap_unordered(work, jobs, chunksize=4) for row in part]
    sw = pd.DataFrame(rows).join(meta[["part", "angle"]], on="pattern")
    sw.to_csv(E / "postproc_sweep.csv.gz", index=False)

    best = []
    for k in S["models"]:
        val = sw[(sw.model == k) & (sw.part == "new validation (31)")]
        for v in VARIANTS:
            vv = val[val.variant == v].groupby("threshold").apply(pooled)
            t = float(vv.net_found.idxmax())
            for part in ("new test (62)", "old test"):
                d = sw[(sw.model == k) & (sw.part == part) & (sw.variant == v) & (sw.threshold == t)]
                best.append(dict(model=k, variant=v, threshold=str(t), part=part, **pooled(d).to_dict()))
        # threshold per angle (base variant)
        va = val[val.variant == "base"].groupby(["angle", "threshold"]).apply(pooled).net_found
        t_ang = {a: float(va.loc[a].idxmax()) for a in sorted(val.angle.unique())}
        for part in ("new test (62)", "old test"):
            d = sw[(sw.model == k) & (sw.part == part) & (sw.variant == "base")]
            d = pd.concat([d[(d.angle == a) & (d.threshold == t)] for a, t in t_ang.items()])
            best.append(dict(model=k, variant="angle", threshold=json.dumps({f"{a:g}": t for a, t in t_ang.items()}),
                             part=part, **pooled(d).to_dict()))
    b = pd.DataFrame(best)
    b.to_csv(E / "postproc_best.csv", index=False)
    pd.set_option("display.width", 250)
    print(b[b.part == "new test (62)"].set_index(["model", "variant"])[["threshold", "recall", "truly_false_share",
                                                                        "net_found", "f1", "found", "truly_false"]].round(3).to_string())
    print(b[b.part == "old test"].set_index(["model", "variant"])[["threshold", "recall", "truly_false_share", "net_found", "f1"]].round(3).to_string())


if __name__ == "__main__":
    main()
