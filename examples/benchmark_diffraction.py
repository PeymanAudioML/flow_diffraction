#!/usr/bin/env python
"""Spotiflow Bragg-disk benchmark of examples/load_diffraction.ipynb, for EVERY material, probe angle and thickness.

The notebook runs one material (MATERIAL_INDEX), one simulation per angle and two thicknesses (20, 40 Å).
This script loops over

    every material folder in --root
      every simulation (row) of its simulation_dataframe.pkl   -> probe semi-angle
        every thickness stored in <uid>_training.h5            -> h["thicknesses"]

and writes one result row per pattern to a CSV (appended as it goes, so a crashed or
interrupted run resumes where it stopped), plus summary tables and plots.

Choosing the pipeline
---------------------
The notebook picks settings by angle (1 / 2 / 4 mrad), which only holds for BW: what actually
matters is the overlap ratio eta = 2 r / s (disk diameter / spot spacing). With --select eta
(default) the pipeline is picked per pattern from eta, so other materials and angles get the
right regime. --select angle reproduces the notebook (nearest of 1 / 2 / 4 mrad).
The spacing s is the median nearest-neighbour distance of the dataPots peaks; this uses
the ground truth only for the lattice spacing, which is known for a simulated crystal anyway.

Usage
-----
    python benchmark_diffraction.py --root /mnt/data/DINOV3/dinov3/dummy_data/selected_folders --out results/
    python benchmark_diffraction.py --root ... --out results/ --materials BW --angles 1 2 --thicknesses 20 40   # subset
    python benchmark_diffraction.py --out results/ --summarize-only                                              # tables + plots again
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import sobel

# ---------------------------------------------------------------------------------------------
# Settings (the notebook's SHARED settings: the same for every material)
# ---------------------------------------------------------------------------------------------

# --select eta: (upper bound of eta, method, settings). BW has eta 0.6 / 1.1 / 2.3 at 1 / 2 / 4 mrad.
REGIMES = [
    (0.85, "log",  dict(log_scale=1000, prob_thresh=0.5)),                   # separate disks
    (1.7,  "ring", dict(log_scale=10000, edge_sigma=0.5, prob_thresh=0.5)),  # touching disks
    (np.inf, "ring", dict(log_scale=10000, edge_sigma=1, prob_thresh=0.3)),  # overlapping disks
]
# --select angle: the notebook's SHARED dict; an angle not listed uses the nearest one.
BY_ANGLE = {
    1: ("log",  dict(log_scale=1000, prob_thresh=0.5)),
    2: ("ring", dict(log_scale=10000, edge_sigma=0.5, prob_thresh=0.5)),
    4: ("ring", dict(log_scale=10000, edge_sigma=1, prob_thresh=0.3)),
}

CUTOFF = 3     # px: a prediction within this distance of a ground-truth spot is correct
NEAR_R = 5     # px: a false detection 3-5 px from a missed spot is "near a real spot"
BEAM_R = 3     # px: predictions / ground truth this close to the central beam are ignored
GT_REL = 0.02  # ground truth = dataPots local maxima above 2 % of the maximum
FLOOR_REL = 1e-4

KEY = ["folder", "uid", "thickness"]   # identifies one pattern (used for resuming)
# fixed column order: failed and successful rows are appended to the same CSV
COLUMNS = KEY + [
    "material", "angle", "radius", "spacing", "eta", "method", "settings",
    "n_gt", "n_vis", "n_vis_reachable", "n_pred", "found_vis", "found_not_vis", "missed_vis",
    "false", "false_near_spot", "false_on_weak_peak", "truly_false",
    "precision", "recall_vis", "recall_vis_reachable", "recall_all", "loc_dy", "loc_dx", "loc_rms",
    "seconds", "error",
]


# ---------------------------------------------------------------------------------------------
# Geometry, ground truth, Spotiflow input (same logic as the notebook, factored out)
# ---------------------------------------------------------------------------------------------

def beam_centre(img):
    return np.argwhere(img > img.max() / 2).mean(axis=0)


def disk_radius(probe):
    return float(np.sqrt((probe > probe.max() / 2).sum() / np.pi))


def far_from(pts, centre, r):
    pts = np.asarray(pts).reshape(-1, 2)
    return np.hypot(*(pts - centre).T) > r


def ground_truth(img, gt, r, vis_box):
    beam = beam_centre(img)
    points = peak_local_max(gt, min_distance=3, threshold_rel=GT_REL, exclude_border=False)
    points = points[far_from(points, beam, BEAM_R)]
    floor = img.max() * FLOOR_REL
    visible = maximum_filter(img, size=vis_box)[points[:, 0], points[:, 1]] >= floor
    reachable = far_from(points, beam, r + 3)       # outside the beam mask used by ring correlation
    peaks_1pct = peak_local_max(gt, min_distance=3, threshold_rel=GT_REL / 2, exclude_border=False)
    weak = peaks_1pct[gt[tuple(peaks_1pct.T)] < gt[tuple(points.T)].min()] if len(points) else peaks_1pct
    spacing = float(np.median(cKDTree(points).query(points, k=2)[0][:, 1])) if len(points) > 1 else np.nan
    return dict(beam=beam, points=points, visible=visible, reachable=reachable, floor=floor, weak=weak, spacing=spacing)


def local_contrast(img, points, spacing, window=2.0, pct=20, floor=None):
    """3 x 3 maximum of dataMeas at each point divided by the local background: the pct-th percentile of dataMeas
    in a square window of half-width window * spacing around the point. Zero backgrounds are replaced by
    floor * 1e-3 so that points on a dark background get a large (finite) contrast."""
    points = np.asarray(points, int).reshape(-1, 2)
    hw = max(1, int(round(window * spacing)))
    peak = maximum_filter(img, size=3)[points[:, 0], points[:, 1]]
    bg = np.array([np.percentile(img[max(0, y - hw):y + hw + 1, max(0, x - hw):x + hw + 1], pct) for y, x in points])
    tiny = (img.max() * FLOOR_REL if floor is None else floor) * 1e-3
    return peak / np.maximum(bg, tiny)


def visible_local(img, g, k=10, window=2.0, pct=20):
    """Local-contrast visibility (section 6 of 5_diffraction_finetune.ipynb): the global floor rule AND
    3 x 3 maximum >= k x local background. Not used by the default pipeline."""
    return g["visible"] & (local_contrast(img, g["points"], g["spacing"], window, pct, g["floor"]) >= k)


def choose_pipeline(select, angle, eta):
    if select == "angle":
        return BY_ANGLE[min(BY_ANGLE, key=lambda a: abs(a - angle))]
    for eta_max, method, kw in REGIMES:
        if eta < eta_max:
            return method, kw
    raise ValueError(eta)


def spotiflow_input(img, probe, method, log_scale, edge_sigma=None, pad=True, **_):
    """Spotiflow input: "log" image or "ring" correlation. pad=False reproduces load_diffraction.ipynb exactly
    (circular FFT correlation); pad=True zero-pads so spots near one edge do not wrap onto the other."""
    x = np.log1p(img / img.max() * log_scale)
    if method == "log":
        return x
    r = disk_radius(probe)
    cy, cx = beam_centre(img)
    yy, xx = np.indices(img.shape)
    edges = sobel(gaussian_filter(x, edge_sigma)) * (np.hypot(yy - cy, xx - cx) > r + 3)
    ring = sobel(gaussian_filter(probe.astype(float), edge_sigma))
    if not pad:  # exactly as in load_diffraction.ipynb
        F = np.fft.fft2(edges) * np.conj(np.fft.fft2(np.fft.ifftshift(ring)))
        return np.clip(np.real(np.fft.ifft2(F)), 0, None)
    # zero-pad so the FFT correlation does not wrap spots from one edge onto the other
    # (the notebook correlates without padding)
    pad = int(np.ceil(2 * r + 4))
    H, W = edges.shape
    E = np.pad(edges, pad)
    R = np.zeros_like(E)
    ph, pw = ring.shape
    R[:ph, :pw] = ring
    R = np.roll(R, (-(ph // 2), -(pw // 2)), axis=(0, 1))      # ring centre -> origin, as ifftshift in the notebook
    corr = np.real(np.fft.ifft2(np.fft.fft2(E) * np.conj(np.fft.fft2(R))))
    return np.clip(corr[pad:pad + H, pad:pad + W], 0, None)


# ---------------------------------------------------------------------------------------------
# Scoring (same categories as score_spots in the notebook)
# ---------------------------------------------------------------------------------------------

def score(g, pred):
    from spotiflow.utils.matching import points_matching

    pts, vis, reach = g["points"], g["visible"], g["reachable"]
    pred = np.asarray(pred, float).reshape(-1, 2)
    pred = pred[far_from(pred, g["beam"], BEAM_R)]
    m = points_matching(pts, pred, cutoff_distance=CUTOFF)
    gi = np.array([a for a, _ in m.matched_pairs], dtype=int)
    pj = np.array([b for _, b in m.matched_pairs], dtype=int)
    fp = np.array(sorted(m.false_positives), dtype=int)
    fn = np.array(sorted(m.false_negatives), dtype=int)
    m_off = points_matching(pts[fn].reshape(-1, 2), pred[fp].reshape(-1, 2), cutoff_distance=NEAR_R)
    rest = fp[sorted(m_off.false_positives)] if len(fp) else fp
    m_weak = points_matching(g["weak"].reshape(-1, 2).astype(float), pred[rest].reshape(-1, 2), cutoff_distance=CUTOFF)
    delta = pred[pj] - pts[gi] if len(gi) else np.zeros((0, 2))
    n_vis, n_vis_reach = int(vis.sum()), int((vis & reach).sum())
    found_vis = int(vis[gi].sum()) if len(gi) else 0
    found_vis_reach = int((vis & reach)[gi].sum()) if len(gi) else 0
    return dict(
        n_gt=len(pts), n_vis=n_vis, n_vis_reachable=n_vis_reach, n_pred=len(pred),
        found_vis=found_vis, found_not_vis=len(gi) - found_vis, missed_vis=n_vis - found_vis,
        false=len(fp), false_near_spot=len(m_off.matched_pairs), false_on_weak_peak=len(m_weak.matched_pairs),
        truly_false=len(m_weak.false_positives),
        precision=len(gi) / len(pred) if len(pred) else np.nan,
        recall_vis=found_vis / n_vis if n_vis else np.nan,
        recall_vis_reachable=found_vis_reach / n_vis_reach if n_vis_reach else np.nan,
        recall_all=len(gi) / len(pts) if len(pts) else np.nan,
        loc_dy=delta[:, 0].mean() if len(delta) else np.nan,   # systematic offset of predictions
        loc_dx=delta[:, 1].mean() if len(delta) else np.nan,
        loc_rms=float(np.sqrt((delta ** 2).sum(1).mean())) if len(delta) else np.nan,
    )


# ---------------------------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------------------------

def find_material_folders(roots):
    """Material folders = folders containing simulation_dataframe.pkl. Each root may be such a folder
    itself (e.g. the SiO2 folder) or contain them (e.g. selected_folders/). Duplicates are removed."""
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    found = {}
    for root in map(Path, roots):
        cands = [root] if (root / "simulation_dataframe.pkl").exists() else sorted(root.iterdir())
        for p in cands:
            if p.is_dir() and (p / "simulation_dataframe.pkl").exists():
                found[p.resolve()] = p
    if not found:
        raise SystemExit(f"no folder with simulation_dataframe.pkl in {roots}")
    return sorted(found.values(), key=lambda p: p.name)


def iter_simulations(root, materials=None, angles=None):
    """Yield (folder, material, uid, angle, h5 path) for every simulation of every material folder.
    `root` is one path or a list of paths (see find_material_folders)."""
    folders = find_material_folders(root)
    for folder in folders:
        meta = pd.read_pickle(folder / "simulation_dataframe.pkl")
        material = str(meta["pretty_formula"].iloc[0]) if "pretty_formula" in meta else folder.name
        if materials and material not in materials and folder.name not in materials:
            continue
        for uid, row in meta.iterrows():
            angle = float(row["probeSemiangle"])
            if angles and not any(np.isclose(angle, a) for a in angles):
                continue
            h5 = folder / f"{uid}_training.h5"
            if not h5.exists():
                logging.warning("missing %s", h5)
                continue
            yield folder, material, uid, angle, h5


# ---------------------------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------------------------

def run(args):
    import h5py
    from spotiflow.model import Spotiflow
    from spotiflow.utils import normalize

    logging.getLogger("spotiflow").setLevel(logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "results.csv"
    done = set()
    if csv.exists() and not args.overwrite:
        prev = pd.read_csv(csv, dtype={"uid": str})
        prev = prev[prev["error"].isna()]          # failed patterns are retried
        prev.reindex(columns=COLUMNS).to_csv(csv, index=False)
        done = {(f, u, str(float(t))) for f, u, t in prev[KEY].values}
        print(f"resuming: {len(done)} patterns already in {csv}")
    elif csv.exists():
        csv.unlink()

    model = Spotiflow.from_pretrained(args.model, map_location=args.device, verbose=False)
    pred_dir = out / "predictions"
    if args.save_preds:
        pred_dir.mkdir(exist_ok=True)

    n_new = 0
    for folder, material, uid, angle, h5 in iter_simulations(args.root, args.materials, args.angles):
        with h5py.File(h5, "r") as h:
            thicknesses = np.asarray(h["thicknesses"][:])
            probe = h["dataProbe"][:]
            r = disk_radius(probe)
            for t_idx, thickness in enumerate(thicknesses):
                if args.thicknesses and not any(np.isclose(thickness, t) for t in args.thicknesses):
                    continue
                key = (folder.name, str(uid), str(float(thickness)))
                if key in done:
                    continue
                row = dict(folder=folder.name, material=material, uid=str(uid), angle=angle,
                           thickness=float(thickness), radius=r)
                t0 = time.time()
                try:
                    img = np.asarray(h["dataMeas"][..., t_idx], dtype=np.float32)
                    gt = np.asarray(h["dataPots"][..., t_idx])
                    g = ground_truth(img, gt, r, args.vis_box)
                    eta = 2 * r / g["spacing"]
                    method, kw = choose_pipeline(args.select, angle, eta)
                    x = spotiflow_input(img, probe, method, **kw)
                    pred, _ = model.predict(
                        x, min_distance=3, prob_thresh=kw["prob_thresh"], subpix=True, verbose=False,
                        normalizer=lambda im: normalize(im, 1, 99.8), device=args.device)
                    row.update(spacing=g["spacing"], eta=eta, method=method, settings=json.dumps(kw), **score(g, pred))
                    if args.save_preds:
                        np.savez_compressed(pred_dir / f"{folder.name}_{uid}_{float(thickness):g}.npz",
                                            pred=pred, gt=g["points"], visible=g["visible"], reachable=g["reachable"])
                except Exception as e:  # keep going; the error is recorded in the CSV
                    logging.exception("failed: %s", key)
                    row["error"] = f"{type(e).__name__}: {e}"
                row["seconds"] = round(time.time() - t0, 2)
                pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(csv, mode="a", header=not csv.exists(), index=False)
                n_new += 1
                print(f"{material:>12} {angle:g} mrad {float(thickness):6.1f} Å  "
                      f"eta {row.get('eta', np.nan):.2f} {row.get('method', '-'):>4}  "
                      f"recall_vis {row.get('recall_vis', np.nan):.2f}  precision {row.get('precision', np.nan):.2f}"
                      + (f"  ERROR {row['error']}" if "error" in row else ""))
    print(f"\n{n_new} new patterns written to {csv}")
    summarize(out)


# ---------------------------------------------------------------------------------------------
# Summary tables and plots
# ---------------------------------------------------------------------------------------------

def pooled(df, by):
    """Pooled (micro-averaged) metrics: sum the counts first, then divide.
    The mean of per-pattern ratios would let patterns with few visible spots dominate."""
    s = df.groupby(by)[["n_vis", "n_vis_reachable", "found_vis", "found_not_vis", "n_pred", "false", "truly_false"]].sum()
    s["n_patterns"] = df.groupby(by).size()
    s["recall_vis"] = s.found_vis / s.n_vis.replace(0, np.nan)
    s["precision"] = 1 - s["false"] / s.n_pred.replace(0, np.nan)
    return s.round(3)


def summarize(out):
    out = Path(out)
    df = pd.read_csv(out / "results.csv", dtype={"uid": str})
    if "error" in df:
        bad = df[df.error.notna()]
        if len(bad):
            print(f"\n{len(bad)} patterns failed (see 'error' column), e.g.\n{bad[KEY + ['error']].head().to_string()}")
        df = df[df.error.isna()]
    if df.empty:
        print("no successful patterns yet")
        return
    df["eta_bin"] = pd.cut(df.eta, [0, 0.85, 1.2, 1.7, 2.5, np.inf])
    df["thickness_bin"] = pd.cut(df.thickness, [0, 50, 100, 200, 300, 400, np.inf])
    tables = {
        "by_material_angle": pooled(df, ["material", "angle"]),
        "by_angle_thickness": pooled(df, ["angle", "thickness_bin"]),
        "by_angle_thickness_all": pooled(df, ["angle", "thickness"]),
        "by_eta": pooled(df, ["eta_bin"]),
        "by_material": pooled(df, ["material"]),
    }
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for name, t in tables.items():
            t.to_csv(out / f"summary_{name}.csv")
            print(f"\n=== {name} (pooled over patterns) ===\n{t.to_string()}")
    off = df[["loc_dy", "loc_dx"]].abs().max(axis=1)
    if (off > 0.5).any():
        print(f"\nWARNING: {int((off > 0.5).sum())} patterns have a mean prediction offset > 0.5 px "
              f"(check probe centring / template shift), see loc_dy, loc_dx")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
    markers = "osD^vP*X"
    angles = sorted(df.angle.unique())
    fig, axs = plt.subplots(1, len(angles), figsize=(5 * len(angles), 4), sharey=True, squeeze=False)
    for ax, angle in zip(axs[0], angles):
        for i, (mat, d) in enumerate(df[df.angle == angle].groupby("material")):
            d = d.sort_values("thickness")
            ax.plot(d.thickness, d.recall_vis, marker=markers[i % 8], color=colors[i % 8], label=mat)
        ax.set_title(f"{angle:g} mrad")
        ax.set_xlabel("thickness (Å)")
        ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("recall of visible spots")
    axs[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "recall_vs_thickness.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (mat, d) in enumerate(df.groupby("material")):
        ax.scatter(d.eta, d.recall_vis, s=14, marker=markers[i % 8], color=colors[i % 8], label=mat, alpha=0.7)
    ax.set_xlabel("overlap ratio η = 2r / s")
    ax.set_ylabel("recall of visible spots")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "recall_vs_eta.png", dpi=150)
    print(f"\nplots: {out / 'recall_vs_thickness.png'}, {out / 'recall_vs_eta.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", nargs="+", help="folder(s) with one sub-folder per material, or material folders themselves")
    p.add_argument("--out", required=True, help="output folder (results.csv, summaries, plots)")
    p.add_argument("--materials", nargs="*", help="pretty_formula or folder names (default: all)")
    p.add_argument("--angles", nargs="*", type=float, help="probe semi-angles in mrad (default: all)")
    p.add_argument("--thicknesses", nargs="*", type=float, help="thicknesses in Å (default: all in each file)")
    p.add_argument("--select", choices=["eta", "angle"], default="eta", help="pick the pipeline by overlap ratio or by angle")
    p.add_argument("--vis-box", type=int, default=3, help="box size for spot visibility (notebook: 3)")
    p.add_argument("--model", default="general", help="pretrained Spotiflow model name")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--save-preds", action="store_true", help="also save predictions / ground truth per pattern (.npz)")
    p.add_argument("--overwrite", action="store_true", help="start from scratch instead of resuming")
    p.add_argument("--summarize-only", action="store_true", help="only rebuild tables and plots from results.csv")
    args = p.parse_args()
    if args.summarize_only:
        summarize(args.out)
    else:
        if not args.root:
            p.error("--root is required unless --summarize-only")
        run(args)


if __name__ == "__main__":
    main()
