#!/usr/bin/env python
"""Fine-tune a pretrained Spotiflow model on simulated CBED patterns, then compare it with the
zero-shot pipelines of load_diffraction.ipynb on materials it has never seen.

Data      every material / simulation / thickness under --root (same layout as the notebook)
Input     --input log  : log1p(dataMeas / max * log_scale)    (default: the network learns the disks itself)
          --input ring : ring-correlation map (settings chosen by overlap ratio, as in benchmark_diffraction.py)
Targets   dataPots peaks > 2 % (central beam removed); --targets visible (default) keeps only spots
          visible in dataMeas, --targets all also asks the model to place spots it cannot see.
          Per angle: --targets 1:visible,2:visible,4:all
Zero-shot the baseline reproduces load_diffraction.ipynb: shared settings chosen by angle, circular FFT
          correlation (--zero-shot-select eta and padding are options of benchmark_diffraction.py only)
Split     by MATERIAL (train / val / test), so test numbers measure generalisation to new crystals.
          With fewer than 3 materials it falls back to splitting by simulation and warns.
Eval      on the test materials: pretrained zero-shot pipeline vs the fine-tuned model,
          same scoring as benchmark_diffraction.py (pooled recall of visible spots, precision).

Needs benchmark_diffraction.py in the same folder.

    python finetune_diffraction.py --root /mnt/data/DINOV3/dinov3/dummy_data/selected_folders --out ft_run/ --epochs 30
    python finetune_diffraction.py --root ... --out ft_run/ --test-materials BW --val-materials Co3(O2F)2
    python finetune_diffraction.py --root ... --out ft_run/ --eval-only          # re-evaluate a trained model
    python finetune_diffraction.py --root ... --out ft_A/ --targets 1:visible,2:visible,4:all --eval-thicknesses 20 40 100 200 500
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_diffraction as bd


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------

def parse_targets(spec):
    """'visible' / 'all' for every angle, or per angle: '1:visible,2:visible,4:all'."""
    if spec in ("visible", "all"):
        return lambda angle: spec
    table = {}
    for part in spec.split(","):
        a, kind = part.split(":")
        if kind not in ("visible", "all"):
            raise SystemExit(f"--targets: unknown target {kind!r} (use visible or all)")
        table[float(a)] = kind

    def target(angle):
        for a, kind in table.items():
            if np.isclose(a, angle):
                return kind
        raise SystemExit(f"--targets {spec!r} gives no target for {angle:g} mrad")
    return target


def make_input(img, probe, kind, log_scale, eta, angle):
    if kind == "log":
        return bd.spotiflow_input(img, probe, "log", log_scale=log_scale).astype(np.float32)
    method, kw = bd.choose_pipeline("angle", angle, eta)  # the reference's ring settings for this angle
    return bd.spotiflow_input(img, probe, method, pad=False, **kw).astype(np.float32)


def load_patterns(args):
    """All patterns as a list of dicts (image, targets, ground truth for scoring, metadata)."""
    import h5py

    target = parse_targets(args.targets)
    items = []
    for folder, material, uid, angle, h5 in bd.iter_simulations(args.root, args.materials, args.angles):
        with h5py.File(h5, "r") as h:
            probe = h["dataProbe"][:]
            r = bd.disk_radius(probe)
            for t_idx, thickness in enumerate(np.asarray(h["thicknesses"][:])):
                if args.thicknesses and not any(np.isclose(thickness, t) for t in args.thicknesses):
                    continue
                try:
                    img = np.asarray(h["dataMeas"][..., t_idx], dtype=np.float32)
                    gt = np.asarray(h["dataPots"][..., t_idx])
                except Exception as e:
                    logging.warning("skipping %s %s %.0f Å: %s", folder.name, uid, thickness, e)
                    continue
                g = bd.ground_truth(img, gt, r, args.vis_box)
                eta = 2 * r / g["spacing"]
                kind = target(angle)
                keep = g["visible"] if kind == "visible" else np.ones(len(g["points"]), bool)
                items.append(dict(
                    folder=folder.name, material=material, uid=str(uid), angle=angle, thickness=float(thickness),
                    eta=eta, img=img, probe=probe, g=g, target=kind,
                    x=make_input(img, probe, args.input, args.log_scale, eta, angle),
                    y=g["points"][keep].astype(np.float32),
                ))
    if not items:
        raise SystemExit("no patterns found")
    return items


def split_materials(items, args):
    """Material-level split. Returns {'train': [...], 'val': [...], 'test': [...]} of material names."""
    mats = sorted({it["material"] for it in items})
    test = set(args.test_materials or [])
    val = set(args.val_materials or [])
    rest = [m for m in mats if m not in test | val]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rest)
    n = len(mats)
    if not test:
        k = max(1, round(0.15 * n)) if n >= 3 else 0
        test, rest = set(rest[:k]), rest[k:]
    if not val:
        k = max(1, round(0.15 * n)) if n >= 3 else 0
        val, rest = set(rest[:k]), rest[k:]
    return dict(train=sorted(rest), val=sorted(val), test=sorted(test))


def assign_split(items, args):
    split = split_materials(items, args)
    if split["train"] and split["val"] and split["test"]:
        by_mat = {m: s for s, ms in split.items() for m in ms}
        for it in items:
            it["split"] = by_mat[it["material"]]
        split["level"] = "material"
    else:
        # too few materials: split simulations instead (test numbers are then NOT on unseen crystals)
        logging.warning("fewer than 3 materials: splitting by simulation; test materials were also seen in training")
        sims = sorted({(it["folder"], it["uid"]) for it in items})
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(sims))
        n_test = n_val = max(1, round(0.15 * len(sims)))
        label = {sims[i]: ("test" if k < n_test else "val" if k < n_test + n_val else "train") for k, i in enumerate(order)}
        for it in items:
            it["split"] = label[(it["folder"], it["uid"])]
        split = {s: sorted({f"{f}/{u}" for (f, u), l in label.items() if l == s}) for s in ("train", "val", "test")}
        split["level"] = "simulation"
    return split


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

def train(items, args, out):
    from lightning.pytorch.loggers import CSVLogger
    from spotiflow.model import Spotiflow, SpotiflowModelConfig, SpotiflowTrainingConfig
    import lightning.pytorch as pl

    pl.seed_everything(args.seed)  # initialisation, augmentation and data order, so runs differ only in their settings
    tr = [it for it in items if it["split"] == "train"]
    va = [it for it in items if it["split"] == "val"]
    print(f"training on {len(tr)} patterns, validating on {len(va)}")
    if args.from_scratch:
        model = Spotiflow(SpotiflowModelConfig(sigma=args.sigma))
        base = "scratch"
    else:
        model = Spotiflow.from_pretrained(args.base, map_location=args.device, verbose=False)
        base = args.base
    cfg = SpotiflowTrainingConfig(
        num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, crop_size=args.crop_size,
        finetuned_from=base, optimize_threshold=True, early_stopping_patience=args.patience,
    )
    model.fit(
        [it["x"] for it in tr], [it["y"] for it in tr],
        [it["x"] for it in va], [it["y"] for it in va],
        save_dir=str(out / "model"), train_config=cfg, device=args.device,
        # Spotiflow's training loop needs a logger object; CSV needs no extra packages (losses in OUT/logs/)
        logger=CSVLogger(save_dir=str(out / "logs"), name="") if args.logger == "csv" else args.logger,
        num_workers=args.num_workers, deterministic=False,
    )
    return model


# ---------------------------------------------------------------------------------------------
# Evaluation: zero-shot pipeline (benchmark) vs fine-tuned model, on the test split
# ---------------------------------------------------------------------------------------------

def evaluate(items, args, out, finetuned):
    from spotiflow.model import Spotiflow
    from spotiflow.utils import normalize

    pretrained = Spotiflow.from_pretrained(args.base, map_location=args.device, verbose=False)
    rows = []
    test = [it for it in items if it["split"] == "test"
            and (not args.eval_thicknesses or any(np.isclose(it["thickness"], t) for t in args.eval_thicknesses))]
    print(f"evaluating on {len(test)} test patterns")
    for it in test:
        meta = {k: it[k] for k in ("folder", "material", "uid", "angle", "thickness", "eta", "target")}
        # zero-shot: the shared pipeline of load_diffraction.ipynb for this angle (no FFT padding)
        method, kw = bd.choose_pipeline("angle", it["angle"], it["eta"])
        x0 = bd.spotiflow_input(it["img"], it["probe"], method, pad=False, **kw)
        p0, _ = pretrained.predict(x0, min_distance=3, prob_thresh=kw["prob_thresh"], subpix=True, verbose=False,
                                   normalizer=lambda im: normalize(im, 1, 99.8), device=args.device)
        rows.append(dict(meta, model=f"zero-shot {args.base}", input=method, **bd.score(it["g"], p0)))
        # fine-tuned: its own input, threshold optimised on the validation materials
        p1, _ = finetuned.predict(it["x"], min_distance=args.min_distance, subpix=True, verbose=False, device=args.device)
        rows.append(dict(meta, model="fine-tuned", input=args.input, **bd.score(it["g"], p1)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_results.csv", index=False)
    if df.empty:
        print("no test patterns")
        return df

    df["eta_bin"] = pd.cut(df.eta, [0, 0.85, 1.2, 1.7, 2.5, np.inf])
    df["thickness_bin"] = pd.cut(df.thickness, [0, 50, 100, 200, 300, 400, np.inf])
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        for by in (["model"], ["angle", "model"], ["angle", "thickness", "model"], ["eta_bin", "model"],
                   ["material", "model"]):
            t = bd.pooled(df, by)[["n_patterns", "n_vis", "found_vis", "found_not_vis", "n_pred", "false",
                                   "truly_false", "recall_vis", "precision"]]
            name = "_".join(b for b in by if b != "model") or "overall"
            t.to_csv(out / f"test_summary_{name}.csv")
            print(f"\n=== test materials, {name} (pooled) ===\n{t.to_string()}")
    return df


# ---------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, nargs="+",
                   help="folder(s) with one sub-folder per material, or material folders themselves")
    p.add_argument("--out", required=True)
    p.add_argument("--materials", nargs="*", help="restrict to these materials (default: all)")
    p.add_argument("--angles", nargs="*", type=float)
    p.add_argument("--thicknesses", nargs="*", type=float)
    p.add_argument("--test-materials", nargs="*", help="held-out materials for the final numbers (default: ~15%%, random)")
    p.add_argument("--val-materials", nargs="*", help="materials for early stopping / threshold (default: ~15%%, random)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input", choices=["log", "ring"], default="log")
    p.add_argument("--log-scale", type=float, default=1000)
    p.add_argument("--targets", default="visible",
                   help="visible | all | per angle, e.g. 1:visible,2:visible,4:all")
    p.add_argument("--eval-thicknesses", nargs="*", type=float,
                   help="evaluate only these thicknesses (default: all test patterns)")
    p.add_argument("--vis-box", type=int, default=3)
    p.add_argument("--base", default="general", help="pretrained model to start from (and zero-shot baseline)")
    p.add_argument("--from-scratch", action="store_true", help="train a new model instead of fine-tuning")
    p.add_argument("--sigma", type=float, default=1.0, help="heatmap sigma, only with --from-scratch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=256, help="training crop (clipped to the image size)")
    p.add_argument("--patience", type=int, default=0, help="early stopping patience in epochs (0 = off)")
    p.add_argument("--min-distance", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--logger", default="csv", choices=["csv", "tensorboard", "wandb"])
    p.add_argument("--eval-only", action="store_true", help="skip training, load OUT/model")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("spotiflow").setLevel(logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = load_patterns(args)
    split = assign_split(items, args)
    counts = pd.Series([it["split"] for it in items]).value_counts().to_dict()
    print(f"{len(items)} patterns; split by {split['level']}: "
          + ", ".join(f"{s} {split[s]} ({counts.get(s, 0)} patterns)" for s in ("train", "val", "test")))
    (out / "split.json").write_text(json.dumps(dict(split, args=vars(args)), indent=2, default=str))
    for it in items:  # raw images / probes are only needed to score the test split
        if it["split"] != "test":
            it.pop("img"), it.pop("probe")

    if args.eval_only:
        from spotiflow.model import Spotiflow
        model = Spotiflow.from_folder(str(out / "model"), map_location=args.device)
    else:
        train(items, args, out)
        from spotiflow.model import Spotiflow
        model = Spotiflow.from_folder(str(out / "model"), map_location=args.device)  # best checkpoint + tuned threshold
    evaluate(items, args, out, model)


if __name__ == "__main__":
    main()
