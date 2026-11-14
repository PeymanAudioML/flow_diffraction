#!/usr/bin/env python
"""4D-ImageNet experimental data (29 acquisitions, 6 materials, 2.1 mrad) for Spotiflow fine-tuning.

Dataset  https://doi.org/10.57760/sciencedb.nbsdc.00281 (CC BY 4.0)
Code     https://github.com/Gaiya69-rgb/4D-ImageNet      Paper  https://arxiv.org/abs/2609.21593

Convert (once):  raw/exp/<material>/<acq>.pkl + raw/label/<material>/<acq>.json  ->  converted/<acq>.h5
    python experimental_data.py --raw /mnt/data/4D-ImageNet/raw --out /mnt/data/4D-ImageNet/converted

  images        float32 (N, H, W)  raw detector counts (not log), rescaled so the disk radius is r_target_px
  points        float32 (M, 2)     disk centres (row, col) of all patterns, rescaled; the JSON stores (x, y)
  point_offsets int64 (N + 1)      disks of pattern i = points[point_offsets[i]:point_offsets[i + 1]]
  confidence    float32 (M)        r4 score per disk (all disks the authors kept, score > 0.5)
  beam          float32 (N, 2)     r4 direct beam (row, col), rescaled; not in points
  scan_pos      int32 (N, 2)       scan position (x, y) as stored in the PKL
  attrs         material, acquisition, source_file, camera_length, fov, r_exp_px, r_target_px, scale, ...

The disk labels are predictions of the authors' detector r4, not ground truth.
PKL files can run code when unpickled: they are read with an unpickler that only allows numpy arrays.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

R_TARGET_PX = 4.9  # our simulations at 2 mrad have r ~ 4.7 px; these data are at 2.1 mrad


# ---------------------------------------------------------------------------------------------
# Safe reading of the authors' PKL files
# ---------------------------------------------------------------------------------------------

class NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that refuses everything except the numpy types needed to rebuild arrays."""
    ALLOWED = {
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"), ("numpy.core.numeric", "_frombuffer"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module.replace("numpy.core.", "numpy._core."), name)


def read_pkl(path):
    with open(path, "rb") as f:
        return NumpyOnlyUnpickler(f).load()


# ---------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------

def beam_radius(img, beam_rc):
    """Radius of the direct-beam disk: the connected area above half its maximum, as disk_radius() in
    benchmark_diffraction.py, but restricted to the blob at the beam position."""
    from scipy.ndimage import label
    r, c = np.clip(np.round(beam_rc).astype(int), 0, np.array(img.shape) - 1)
    y0, x0 = max(r - 20, 0), max(c - 20, 0)
    win = img[y0:r + 21, x0:c + 21]
    level = 0.5 * np.percentile(img[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4], 95)  # robust to hot pixels
    lab, _ = label(win > level)
    return float(np.sqrt((lab == lab[r - y0, c - x0]).sum() / np.pi)) if lab[r - y0, c - x0] else np.nan


def rescale(images, scale):
    """Bilinear zoom of (N, H, W) images; returns the images and the coordinate map old -> new."""
    from scipy.ndimage import zoom
    out = np.stack([zoom(im, scale, order=1, grid_mode=True, mode="grid-constant") for im in images])
    s = np.array(out.shape[1:]) / np.array(images.shape[1:])     # exact per-axis factor after rounding
    return out.astype(np.float32), lambda rc: (np.asarray(rc) + 0.5) * s - 0.5


def convert(pkl_path, json_path, out_path, r_target=R_TARGET_PX, n_radius=100):
    import h5py
    d = read_pkl(pkl_path)
    lab = json.loads(Path(json_path).read_text())
    images = np.asarray(d["diffraction_patterns"], np.float32)
    preds = sorted(lab["predictions"], key=lambda p: p["index"])
    assert len(preds) == len(images), (len(preds), len(images))

    beam = np.array([[p["center_beam"]["y"], p["center_beam"]["x"]] for p in preds], np.float32)
    pts = [np.array([[k["y"], k["x"]] for k in p["disks"]], np.float32).reshape(-1, 2) for p in preds]
    conf = [np.array([k["score"] for k in p["disks"]], np.float32) for p in preds]

    pick = np.linspace(0, len(images) - 1, min(n_radius, len(images))).astype(int)
    r_exp = float(np.nanmedian([beam_radius(images[i], beam[i]) for i in pick]))
    scale = r_target / r_exp
    images, to_new = rescale(images, scale)

    offsets = np.concatenate([[0], np.cumsum([len(p) for p in pts])]).astype(np.int64)
    with h5py.File(out_path, "w") as h:
        h["images"] = images
        h["points"] = to_new(np.concatenate(pts)).astype(np.float32)
        h["point_offsets"] = offsets
        h["confidence"] = np.concatenate(conf)
        h["beam"] = to_new(beam).astype(np.float32)
        h["scan_pos"] = np.asarray(d["scan_positions"], np.int32)
        h.attrs.update(
            material=str(lab.get("material") or d.get("material")), acquisition=Path(pkl_path).stem,
            source_file=str(lab.get("source_file")), camera_length=str(d.get("camera_length")),
            fov=str(d.get("fov")), r4_version=str(lab["model"]["version"]),
            r4_confidence_threshold=float(lab["model"]["confidence_threshold"]),
            r_exp_px=r_exp, r_target_px=r_target, scale=scale, original_shape=np.asarray(d["diffraction_patterns"].shape[1:]),
        )
    return dict(acquisition=Path(pkl_path).stem, n=len(images), shape=images.shape[1:], r_exp_px=r_exp, scale=scale)


# ---------------------------------------------------------------------------------------------
# Loading for training (the form Spotiflow.fit() takes: lists of (H, W) images and (n, 2) points)
# ---------------------------------------------------------------------------------------------

def even_subset(scan_pos, n):
    """n patterns spread evenly over the scan: order by scan row then column, take evenly spaced ones."""
    order = np.lexsort((scan_pos[:, 0], scan_pos[:, 1]))
    if n is None or n <= 0 or n >= len(order):
        return np.sort(order)
    return np.sort(order[np.linspace(0, len(order) - 1, n).astype(int)])


def load_experimental(root, acquisitions, min_conf=0.8, per_acquisition=1000, seed=0):
    """-> images (list of float32 (H, W)), points (list of float32 (n, 2), row/col), meta (DataFrame).
    `seed` is unused (the subset is deterministic, spread over the scan) and kept for a stable signature."""
    import h5py
    import pandas as pd
    images, points, rows = [], [], []
    for acq in acquisitions:
        with h5py.File(Path(root) / f"{acq}.h5", "r") as h:
            idx = even_subset(h["scan_pos"][:], per_acquisition)
            off, P, C = h["point_offsets"][:], h["points"][:], h["confidence"][:]
            imgs = h["images"][np.sort(idx)]
            beam = h["beam"][:]
            mat = h.attrs["material"]
            for k, i in enumerate(idx):
                p, c = P[off[i]:off[i + 1]], C[off[i]:off[i + 1]]
                images.append(imgs[k])
                points.append(p[c >= min_conf].astype(np.float32))
                rows.append(dict(acquisition=acq, material=mat, index=int(i), beam_r=beam[i, 0], beam_c=beam[i, 1]))
    return images, points, pd.DataFrame(rows)



def load_experimental_conf(root, acquisitions, per_acquisition=1000):
    """As load_experimental, but every r4 disk (score > 0.5) with its score.
    -> images (list of float32 (H, W)), points (list of (n, 2)), scores (list of (n,)), meta (DataFrame)."""
    import h5py
    import pandas as pd
    images, points, conf, rows = [], [], [], []
    for acq in acquisitions:
        with h5py.File(Path(root) / f"{acq}.h5", "r") as h:
            idx = even_subset(h["scan_pos"][:], per_acquisition)
            off, P, C = h["point_offsets"][:], h["points"][:], h["confidence"][:]
            imgs, beam, mat = h["images"][np.sort(idx)], h["beam"][:], h.attrs["material"]
            for k, i in enumerate(idx):
                images.append(imgs[k].astype(np.float32))
                points.append(P[off[i]:off[i + 1]].astype(np.float32))
                conf.append(C[off[i]:off[i + 1]].astype(np.float32))
                rows.append(dict(acquisition=acq, material=mat, index=int(i), beam_r=beam[i, 0], beam_c=beam[i, 1]))
    return images, points, conf, pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="/mnt/data/4D-ImageNet/raw")
    p.add_argument("--out", default="/mnt/data/4D-ImageNet/converted")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pkl in sorted((raw / "exp").rglob("*.pkl")):
        js = raw / "label" / pkl.parent.name.replace("Ag+Au", "Au+Ag") / f"{pkl.stem}.json"
        dst = out / f"{pkl.stem}.h5"
        if dst.exists() and not args.overwrite:
            print(f"skip (exists) {dst.name}")
            continue
        info = convert(pkl, js, dst)
        print(f"{info['acquisition']}: {info['n']} patterns, {info['shape']} px, r_exp {info['r_exp_px']:.2f} px, "
              f"scale {info['scale']:.3f}", flush=True)


if __name__ == "__main__":
    main()
