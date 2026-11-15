#!/usr/bin/env python
"""Describe a folder of simulated diffraction data (<folder>/<uid>_training.h5) from the h5 files only and compare
it with the 24 training materials.

Per pattern (folder, uid, thickness): probe disk radius, spot count (ground truth = dataPots local maxima above
2 % of the maximum, as benchmark_diffraction.ground_truth), visible spots, spacing, overlap ratio eta = 2 r / s,
beam share of the intensity, dynamic range. The probe angle of the new data is estimated from the probe radius
with the radius -> angle relation of the reference data (whose simulation_dataframe.pkl files we already use).
The new data's .pkl files are NOT opened.

    python -I analyze_sim_data.py --new /mnt/data/sim_diffraction_gdrive --out diffraction_runs/sim93_analysis
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchmark_diffraction as bd

REF = "/mnt/data/DINOV3/dinov3/dummy_data/selected_folders"
BLUE, ORANGE, SKY, PINK, YELLOW = "#0072B2", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"


def describe(h5, dataset):
    rows = []
    with h5py.File(h5, "r") as h:
        probe = np.asarray(h["dataProbe"], np.float32)
        thick = h["thicknesses"][:]
        meas, pots = h["dataMeas"], h["dataPots"]
        r = bd.disk_radius(probe)
        vis_box = 2 * int(np.ceil(r)) + 1
        phash = hashlib.md5(np.asarray(pots[..., 0]).tobytes()).hexdigest()
        for t_idx, t in enumerate(thick):
            img = np.asarray(meas[..., t_idx], np.float32)
            gt = np.asarray(pots[..., t_idx], np.float32)
            row = dict(dataset=dataset, folder=h5.parent.name, uid=h5.stem.replace("_training", ""),
                       thickness=int(t), radius=r, pots_hash=phash, finite=bool(np.isfinite(img).all() and np.isfinite(gt).all()))
            if not row["finite"] or img.max() <= 0:
                rows.append(row)
                continue
            g = bd.ground_truth(img, gt, r, vis_box)
            beam = g["beam"]
            yy, xx = np.indices(img.shape)
            in_beam = np.hypot(yy - beam[0], xx - beam[1]) <= r + 1
            pos = img[img > 0]
            row.update(n_gt=len(g["points"]), n_vis=int(g["visible"].sum()), spacing=g["spacing"],
                       eta=2 * r / g["spacing"] if g["spacing"] > 0 else np.nan,
                       beam_share=float(img[in_beam].sum() / img.sum()),
                       dyn_range=float(np.log10(img.max() / np.percentile(pos, 1))) if len(pos) else np.nan,
                       beam_off=float(np.hypot(*(beam - np.array(img.shape) / 2))))
            rows.append(row)
    return rows


def ref_angles(ref):
    """probe radius -> angle from the reference data's pkl files (already used for training)."""
    out = []
    for folder, material, uid, angle, h5 in bd.iter_simulations(ref):
        with h5py.File(h5, "r") as h:
            out.append(dict(folder=folder.name, uid=uid, material=material, angle=angle,
                            radius=bd.disk_radius(np.asarray(h["dataProbe"], np.float32))))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True)
    p.add_argument("--ref", default=REF)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, root in (("new", a.new), ("ref", a.ref)):
        files = sorted(Path(root).glob("*/*_training.h5"))
        for i, f in enumerate(files):
            rows += describe(f, name)
            if i % 50 == 0:
                print(name, i, len(files), flush=True)
    df = pd.DataFrame(rows)

    ra = ref_angles(a.ref)
    ra.to_csv(out / "ref_radius_angle.csv", index=False)
    cal = ra.groupby("angle").radius.agg(["mean", "min", "max"])
    df["angle_est"] = [cal.index[np.argmin(abs(cal["mean"].values - r))] for r in df.radius]
    df = df.merge(ra[["uid", "material", "angle"]], on="uid", how="left")
    df.to_csv(out / "patterns.csv", index=False)

    # duplicates: same dataPots between the new and the reference data
    hn, hr = set(df[df.dataset == "new"].pots_hash), set(df[df.dataset == "ref"].pots_hash)
    dup_new = df[(df.dataset == "new")].groupby("pots_hash").folder.nunique()

    ok = df[df.get("n_gt").notna()]
    lines = ["# Simulated data: new 93 folders vs the 24 reference folders", ""]
    lines.append(f"probe radius -> angle (reference): {cal.round(2).to_dict('index')}")
    for name, d in ok.groupby("dataset"):
        lines.append(f"\n## {name}: {d.folder.nunique()} folders, {d.uid.nunique()} simulations, {len(d)} patterns, "
                     f"non-finite patterns {int((~df[df.dataset == name].finite).sum())}")
        lines.append("simulations per estimated angle: " + str(d.drop_duplicates('uid').angle_est.value_counts().sort_index().to_dict()))
        lines.append(d[["radius", "n_gt", "n_vis", "spacing", "eta", "beam_share", "dyn_range", "beam_off"]]
                     .describe(percentiles=[.1, .5, .9]).round(3).to_string())
    lines.append(f"\nidentical dataPots shared with reference: {len(hn & hr)}; new hashes used by >1 folder: {int((dup_new > 1).sum())}")
    eta = ok.groupby("dataset").eta.apply(lambda e: pd.cut(e, [0, .85, 1.7, np.inf], labels=["separate", "touching", "overlap"]).value_counts(normalize=True).round(3))
    lines.append("\neta regimes (share of patterns):\n" + eta.unstack().to_string())
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    plots(ok, a.new, out)


def plots(ok, new_root, out):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for k, (col, lab, log) in enumerate([("radius", "probe disk radius (px)", False), ("n_vis", "visible spots per pattern", True),
                                         ("spacing", "spot spacing (px)", False), ("eta", "overlap ratio eta = 2r/s", True),
                                         ("beam_share", "beam share of intensity", False), ("dyn_range", "dynamic range (log10)", False)]):
        a = ax.flat[k]
        v = ok[col].replace([np.inf, -np.inf], np.nan).dropna()
        bins = np.geomspace(max(v[v > 0].min(), 1e-3), v.max(), 40) if log else np.linspace(v.min(), v.max(), 40)
        for name, c, hatch in (("ref", BLUE, None), ("new", ORANGE, "//")):
            a.hist(ok.loc[ok.dataset == name, col].dropna(), bins=bins, density=True, histtype="step", lw=2, color=c,
                   ls="-" if name == "new" else "--", label=f"{name} ({ok[ok.dataset == name].folder.nunique()} folders)")
        if log:
            a.set_xscale("log")
        if col == "eta":
            for x in (0.85, 1.7):
                a.axvline(x, color="k", lw=0.8, ls=":")
        a.set_xlabel(lab)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=110)
    plt.close(fig)

    # gallery: 12 new patterns spread over eta, thickness 100 A
    d = ok[(ok.dataset == "new") & (ok.thickness == 100)].dropna(subset=["eta"]).sort_values("eta")
    pick = d.iloc[np.linspace(0, len(d) - 1, 12).astype(int)]
    fig, ax = plt.subplots(3, 4, figsize=(16, 12.5))
    for a, (_, r) in zip(ax.flat, pick.iterrows()):
        h5 = Path(new_root) / r.folder / f"{r.uid}_training.h5"
        with h5py.File(h5, "r") as h:
            t_idx = list(h["thicknesses"][:]).index(r.thickness)
            img = np.asarray(h["dataMeas"][..., t_idx], np.float32)
            gt = np.asarray(h["dataPots"][..., t_idx], np.float32)
        g = bd.ground_truth(img, gt, r.radius, 2 * int(np.ceil(r.radius)) + 1)
        a.imshow(np.log1p(1000 * img / img.max()), cmap="gray")
        pv, pn = g["points"][g["visible"]], g["points"][~g["visible"]]
        a.scatter(pv[:, 1], pv[:, 0], s=40, facecolors="none", edgecolors=SKY, lw=1, marker="o")
        a.scatter(pn[:, 1], pn[:, 0], s=25, color=ORANGE, lw=1, marker="x")
        a.set_title(f"{r.folder[:8]}  ~{r.angle_est:g} mrad  eta {r.eta:.2f}\n{r.n_vis} visible / {r.n_gt} spots", fontsize=9)
        a.axis("off")
    fig.suptitle("New simulations, thickness 100 Å (log image; sky circles = visible spots, orange x = not visible)")
    fig.tight_layout()
    fig.savefig(out / "gallery.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
