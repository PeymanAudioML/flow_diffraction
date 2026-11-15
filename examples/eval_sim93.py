#!/usr/bin/env python
"""Zero-shot test of trained models on the 93 new simulated materials (/mnt/data/sim_diffraction_gdrive), with
the 3 simulation test materials of ft_V (BW, Co3(O2F)2, SiO2) as the reference ("old test").

Same input and scoring as V's simulation test: log x1000 input, ground truth = dataPots peaks > 2 % (beam removed),
targets = visible spots (3 x 3 box >= 1e-4 of the max), bd.score (3 px matching). Thicknesses 20, 40, 100, 200,
500 A (V's evaluation thicknesses). The probe angle of a new simulation is read from its probe disk radius
(2.59 / 4.69 / 9.85 px = 1 / 2 / 4 mrad, exactly as in the 24 reference folders); the new .pkl files are not opened.

Writes <out>/patterns.csv (one row per pattern), sweep.csv (bd.score per model, pattern, threshold), and
predictions.npz (points + probabilities at threshold 0.05, for plots).

    python eval_sim93.py --models V V_mix2 V_mix4b --out diffraction_runs/sim93_eval
"""
import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import benchmark_diffraction as bd
from intensity_accuracy import weighted_accuracy

HERE = Path(__file__).resolve().parent
R = HERE / "diffraction_runs"
NEW = "/mnt/data/sim_diffraction_gdrive"
RADIUS_TO_ANGLE = {2.59: 1.0, 4.69: 2.0, 9.85: 4.0}
SWEEP = np.round(np.arange(0.30, 0.801, 0.02), 2)


def angle_from_radius(r):
    k = min(RADIUS_TO_ANGLE, key=lambda x: abs(x - r))
    assert abs(k - r) < 0.05, f"unexpected probe radius {r:.3f}"
    return RADIUS_TO_ANGLE[k]


def model_threshold(k):
    """One threshold per model: the 8.4 value of notebook 6 for the mixed models; V's own (simulation) threshold."""
    one = pd.read_csv(R / "eval_exp2/one_threshold_summary.csv").set_index("model")
    if k == "V":
        import yaml
        return float(yaml.safe_load((R / "ft_V/model/thresholds.yaml").read_text())["prob_thresh_best"][0])
    return float(one.loc[k, "thr"])


def new_patterns(root, thicknesses, vis_box, log_scale):
    for h5 in sorted(Path(root).glob("*/*_training.h5")):
        with h5py.File(h5, "r") as h:
            probe = np.asarray(h["dataProbe"], np.float32)
            r = bd.disk_radius(probe)
            angle = angle_from_radius(r)
            for t_idx, t in enumerate(h["thicknesses"][:]):
                if float(t) not in thicknesses:
                    continue
                img = np.asarray(h["dataMeas"][..., t_idx], np.float32)
                pots = np.asarray(h["dataPots"][..., t_idx], np.float32)
                g = bd.ground_truth(img, pots, r, vis_box)
                yield dict(dataset="new", folder=h5.parent.name, uid=h5.stem.replace("_training", ""), material=h5.parent.name[:8],
                           angle=angle, thickness=float(t), radius=r, img=img, pots=pots, g=g,
                           x=bd.spotiflow_input(img, probe, "log", log_scale=log_scale).astype(np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["V", "V_mix2", "V_mix4b"])
    p.add_argument("--new", default=NEW)
    p.add_argument("--out", default=str(R / "sim93_eval"))
    a = p.parse_args()
    logging.getLogger("spotiflow").setLevel(logging.WARNING)
    import finetune_experimental as fe
    from eval_experimental import RUNS, predict_low
    from spotiflow.model import Spotiflow

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    va = json.loads((R / "ft_V/split.json").read_text())["args"]
    thick, vis_box, log_scale = [float(t) for t in va["eval_thicknesses"]], int(va["vis_box"]), float(va["log_scale"])

    items = list(new_patterns(a.new, thick, vis_box, log_scale))
    for it in fe.load_simulation_test(R / "ft_V"):           # reference: V's 3 simulation test materials
        pots = None
        items.append(dict(dataset="old test", folder=it["folder"], uid=it["uid"], material=it["material"], angle=it["angle"],
                          thickness=it["thickness"], radius=bd.disk_radius(it["probe"]), img=it["img"], pots=pots,
                          g=it["g"], x=it["x"]))
    # old test dataPots (for the dataPots-weighted accuracy)
    roots = va["root"]
    for it in items:
        if it["pots"] is None:
            folder = next(f for f in bd.find_material_folders(roots) if f.name == it["folder"])
            with h5py.File(folder / f"{it['uid']}_training.h5", "r") as h:
                ti = int(np.where(h["thicknesses"][:] == it["thickness"])[0][0])
                it["pots"] = np.asarray(h["dataPots"][..., ti], np.float32)
    print(f"{sum(i['dataset'] == 'new' for i in items)} new patterns, {sum(i['dataset'] != 'new' for i in items)} old test patterns", flush=True)

    meta = []
    for it in items:
        g, img = it["g"], it["img"]
        yy, xx = np.indices(img.shape)
        in_beam = np.hypot(yy - g["beam"][0], xx - g["beam"][1]) <= it["radius"] + 1
        meta.append(dict(dataset=it["dataset"], folder=it["folder"], uid=it["uid"], material=it["material"],
                         angle=it["angle"], thickness=it["thickness"], n_gt=len(g["points"]), n_vis=int(g["visible"].sum()),
                         spacing=g["spacing"], eta=2 * it["radius"] / g["spacing"] if g["spacing"] > 0 else np.nan,
                         beam_share=float(img[in_beam].sum() / img.sum())))
    pd.DataFrame(meta).to_csv(out / "patterns.csv", index_label="pattern")

    rows, preds = [], {}
    for k in a.models:
        t_one = model_threshold(k)
        model = Spotiflow.from_folder(str(R / RUNS.get(k, f"ft_{k}") / "model"), map_location="cuda")
        low = predict_low(model, [it["x"] for it in items], "cuda")
        preds[k] = low
        for i, (it, (pts, prob)) in enumerate(zip(items, low)):
            for t in sorted(set(SWEEP) | {round(t_one, 3)}):
                pr = pts[prob >= t]
                s = bd.score(it["g"], pr)
                row = dict(model=k, pattern=i, threshold=t, one=bool(np.isclose(t, t_one)),
                           **{c: s[c] for c in ("n_vis", "n_pred", "found_vis", "found_not_vis", "missed_vis", "false",
                                                 "false_near_spot", "false_on_weak_peak", "truly_false")})
                if row["one"]:
                    g = it["g"]
                    row["acc_img"] = weighted_accuracy(it["img"], g["points"], pr, g["beam"], gt_required=g["visible"])["accuracy"]
                    row["acc_pots"] = weighted_accuracy(it["pots"], g["points"], pr, g["beam"], gt_required=g["visible"])["accuracy"]
                rows.append(row)
        print(k, "done", flush=True)
    pd.DataFrame(rows).to_csv(out / "sweep.csv.gz", index=False)
    np.savez_compressed(out / "predictions.npz", **{f"{k}_{i}_pts": pp for k, low in preds.items() for i, (pp, _) in enumerate(low)},
                        **{f"{k}_{i}_prob": pb for k, low in preds.items() for i, (_, pb) in enumerate(low)})
    json.dump(dict(models=a.models, thresholds={k: model_threshold(k) for k in a.models}, thicknesses=thick,
                   vis_box=vis_box, log_scale=log_scale), open(out / "settings.json", "w"), indent=1)


if __name__ == "__main__":
    main()
