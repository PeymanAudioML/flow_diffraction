#!/usr/bin/env python
"""py4DSTEM's FCU-Net (Munshi et al. 2022, npj Comput Mater 8:254) on the patterns of eval_sim93.py.

Runs in the FCU-Net environment (py4DSTEM + TensorFlow 2.15 + crystal4D; weights downloaded by py4DSTEM to
/mnt/data/fcunet/tmp):
    /mnt/data/envs/fcunet/bin/python fcunet_sim93.py

Step 1: the network predicts the potential image from dataMeas and dataProbe, exactly as
py4DSTEM.find_Bragg_disks_aiml does: the averaged output of NUM_ATTEMPTS passes (Monte-Carlo dropout).
Saved as float16 in /mnt/data/fcunet/pred_pots.npy (pattern order of sim93_eval/patterns.csv).
Step 2: py4DSTEM's get_maxima_2D on the prediction (the find_Bragg_disks_aiml peak step) for each setting in PEAKS,
at a low cutoff (minRelativeIntensity 0.002); every peak keeps its relative intensity so that the cutoff can be
tuned afterwards (score_fcunet_sim93.py). -> /mnt/data/fcunet/fcunet_peaks.npz (188 MB, not in git)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "diffraction_runs"
E = R / "sim93_eval"
NEW = Path("/mnt/data/sim_diffraction_gdrive")
OUT = Path("/mnt/data/fcunet")
NUM_ATTEMPTS = 5
BATCH = 16
LOW = 0.002
PEAKS = {f"s{s}_m{m}": dict(sigma=s, minSpacing=m) for s in (0, 0.5) for m in (2, 4, 6)}


def old_dirs():
    import json
    import benchmark_diffraction as bd
    root = json.loads((R / "ft_V/split.json").read_text())["args"]["root"]
    return {f.name: f for f in bd.find_material_folders(root)}


def load(r, od):
    folder = NEW / r.folder if r.dataset == "new" else od[r.folder]
    with h5py.File(folder / f"{r.uid}_training.h5", "r") as h:
        ti = int(np.where(h["thicknesses"][:] == r.thickness)[0][0])
        return h["dataMeas"][..., ti].astype(np.float32), h["dataProbe"][:].astype(np.float32)


def predict(meta):
    import os
    import tensorflow as tf
    from py4DSTEM.braggvectors.diskdetection_aiml import _get_latest_model
    cwd = os.getcwd()
    os.chdir(OUT)                                    # the model is cached in ./tmp
    model = _get_latest_model()
    os.chdir(cwd)
    od = old_dirs()
    out = np.lib.format.open_memmap(OUT / "pred_pots.npy", "w+", np.float16, (len(meta), 256, 256))
    rows = list(meta.itertuples())
    for b in range(0, len(rows), BATCH):
        dp, pr = zip(*(load(r, od) for r in rows[b:b + BATCH]))
        dp, pr = np.stack(dp)[..., None], np.stack(pr)[..., None]
        acc = np.zeros(dp.shape[:3], np.float64)
        for _ in range(NUM_ATTEMPTS):
            acc += model([dp, pr], training=False).numpy()[..., 0]   # MC-dropout layers stay active
        out[b:b + len(rows[b:b + BATCH])] = acc / NUM_ATTEMPTS
        if b % (BATCH * 10) == 0:
            print(f"predicted {b + len(dp)}/{len(rows)}", flush=True)
    out.flush()


def peaks():
    from py4DSTEM.process.utils import get_maxima_2D
    P = np.load(OUT / "pred_pots.npy", mmap_mode="r")
    res = {}
    for i in range(len(P)):
        img = np.asarray(P[i], np.float64)
        for k, s in PEAKS.items():
            pk = get_maxima_2D(img, subpixel="poly", minRelativeIntensity=LOW, edgeBoundary=2, maxNumPeaks=100000, **s)
            pts = np.stack([pk["x"], pk["y"]], 1).astype(np.float32).reshape(-1, 2)
            rel = (np.asarray(pk["intensity"]) / max(pk["intensity"].max(), 1e-12)) if len(pk) else np.zeros(0)
            res[f"{k}_{i}_pts"], res[f"{k}_{i}_rel"] = pts, rel.astype(np.float32)
        if i % 200 == 0:
            print(f"peaks {i}/{len(P)}", flush=True)
    np.savez_compressed(OUT / "fcunet_peaks.npz", **res)


if __name__ == "__main__":
    meta = pd.read_csv(E / "patterns.csv")
    if "--peaks-only" not in sys.argv:
        predict(meta)
    peaks()
