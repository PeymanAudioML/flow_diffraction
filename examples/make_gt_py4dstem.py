#!/usr/bin/env python
"""Ground-truth spot positions from py4DSTEM for every simulated pattern, stored for benchmark_diffraction.ground_truth.

Runs in the py4DSTEM environment (py4DSTEM is not installed in the spotiflow environment):
    /mnt/data/envs/py4dstem/bin/python make_gt_py4dstem.py            # all reference + new simulations

For every <uid>_training.h5 and thickness, py4DSTEM's get_maxima_2D on dataPots with

    subpixel="poly", sigma=0.5, minRelativeIntensity=0.02, minSpacing=4, edgeBoundary=2, maxNumPeaks=100000

(the requested settings; maxNumPeaks raised from 100 so that no spot is cut: most patterns have more than 100) gives
the spots, and the same call with minRelativeIntensity=0.01 the weaker 1-2 % peaks. Positions are (row, col):
py4DSTEM's x is the first array axis. Stored per simulation in <cache>/<folder>/<uid>.npz (keys t<i>_points,
t<i>_intensity, t<i>_weak for thickness index i) plus <cache>/index.json: md5 of the float32 dataPots slice ->
[npz path, thickness index], which ground_truth uses to find a pattern's spots from the array alone.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py
import numpy as np
from py4DSTEM.preprocess.utils import get_maxima_2D

import benchmark_diffraction as bd

ROOTS = ["/mnt/data/DINOV3/dinov3/dummy_data/selected_folders", "/mnt/data/DINOV3/dinov3/dummy_data"]
NEW = "/mnt/data/sim_diffraction_gdrive"
SETTINGS = dict(subpixel="poly", sigma=0.5, minRelativeIntensity=0.02, minSpacing=4, edgeBoundary=2, maxNumPeaks=100000)


def pots_key(pots):
    return hashlib.md5(np.ascontiguousarray(pots, dtype=np.float32).tobytes()).hexdigest()


def peaks(pots, **over):
    pk = get_maxima_2D(np.asarray(pots, np.float64), **{**SETTINGS, **over})
    pts = np.stack([pk["x"], pk["y"]], axis=1).astype(np.float32).reshape(-1, 2)
    return pts, np.asarray(pk["intensity"], np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=bd.GT_CACHE)
    a = p.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    files = [f for d in bd.find_material_folders(ROOTS) for f in sorted(d.glob("*_training.h5"))]
    files += sorted(Path(NEW).glob("*/*_training.h5"))
    index = {}
    for n, h5 in enumerate(files):
        out = {}
        with h5py.File(h5, "r") as h:
            pots_all = h["dataPots"]
            for ti in range(pots_all.shape[-1]):
                pots = np.asarray(pots_all[..., ti], np.float32)
                pts, inten = peaks(pots)
                weak, _ = peaks(pots, minRelativeIntensity=0.01)
                if len(weak) and len(pts):
                    d = np.min(np.hypot(*(weak[:, None] - pts[None]).transpose(2, 0, 1)), axis=1)
                    weak = weak[d > 1.0]                      # 1-2 % peaks: not already a 2 % spot
                out[f"t{ti}_points"], out[f"t{ti}_intensity"], out[f"t{ti}_weak"] = pts, inten, weak
                npz = cache / h5.parent.name / f"{h5.stem.replace('_training', '')}.npz"
                index[pots_key(pots)] = [str(npz.relative_to(cache)), ti]
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **out)
        if n % 25 == 0:
            print(f"{n + 1}/{len(files)} {h5.parent.name}", flush=True)
    (cache / "index.json").write_text(json.dumps(dict(settings=SETTINGS, weak_minRelativeIntensity=0.01,
                                                      patterns=index)))
    print(f"done: {len(files)} simulations, {len(index)} patterns -> {cache}")


if __name__ == "__main__":
    main()
