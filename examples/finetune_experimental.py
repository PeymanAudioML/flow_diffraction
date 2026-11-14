#!/usr/bin/env python
"""Continue fine-tuning our simulation-trained model (ft_V) on the 4D-ImageNet experimental patterns,
then show how the loss dropped and how the model performs before and after.

Data     converted HDF5 files of experimental_data.py (29 acquisitions, 6 materials, 2.1 mrad)
Targets  r4 disk centres with confidence >= --min-conf (the authors' detector, not ground truth)
Split    by acquisition (exp_split.json): one test acquisition per material, 3 for validation, the rest train
Input    log1p(img / max * 1000), the same as ft_V
Results  OUT/loss.png               train / validation loss and validation F1 per epoch
         OUT/performance.png        before vs after: experimental test acquisitions (vs r4 labels)
                                    and the simulation test materials of ft_V (recall of visible spots)
         OUT/results_experimental.csv, OUT/results_simulation.csv

    python finetune_experimental.py --out diffraction_runs/ft_V_exp --epochs 10
    python finetune_experimental.py --out diffraction_runs/ft_V_mix --epochs 10 --mix-sim   # one model for both
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_diffraction as bd
from experimental_data import load_experimental, load_experimental_conf
from train_extras import F1Checkpoint, F1EarlyStopping, pseudo_labels, use_weighted_dataset

HERE = Path(__file__).parent
OKABE = dict(before="#0072B2", after="#E69F00", train="#0072B2", val="#D55E00", f1="#009E73")


def make_split(root, path, seed=0):
    """One test acquisition per material, 3 validation acquisitions from the materials with most data, rest train."""
    import h5py
    acqs = {}
    for f in sorted(Path(root).glob("*.h5")):
        with h5py.File(f, "r") as h:
            acqs[f.stem] = str(h.attrs["material"])
    rng = np.random.default_rng(seed)
    by_mat = {m: sorted(a for a, mm in acqs.items() if mm == m) for m in sorted(set(acqs.values()))}
    test = [by_mat[m][rng.integers(len(by_mat[m]))] for m in by_mat]
    left = {m: [a for a in v if a not in test] for m, v in by_mat.items()}
    val = [left[m][rng.integers(len(left[m]))] for m in sorted(left, key=lambda m: -len(left[m]))[:3]]
    train = sorted(a for a in acqs if a not in test + val)
    split = dict(train=train, val=sorted(val), test=sorted(test), materials=acqs)
    Path(path).write_text(json.dumps(split, indent=2))
    return split


def write_manifest(root, split, path, min_conf):
    import h5py
    role = {a: r for r in ("train", "val", "test") for a in split[r]}
    rows = []
    for a in split["materials"]:
        with h5py.File(Path(root) / f"{a}.h5", "r") as h:
            off, conf = h["point_offsets"][:], h["confidence"][:]
            kept = np.add.reduceat(conf >= min_conf, off[:-1]) if len(conf) else np.zeros(len(off) - 1)
            rows.append(dict(acquisition=a, material=h.attrs["material"], role=role[a], n_patterns=len(off) - 1,
                             shape=h["images"].shape[1:], median_disks=float(np.median(np.diff(off))),
                             median_disks_kept=float(np.median(kept)), median_confidence=round(float(np.median(conf)), 3),
                             r_exp_px=round(float(h.attrs["r_exp_px"]), 2), scale=round(float(h.attrs["scale"]), 3)))
    pd.DataFrame(rows).sort_values(["role", "material"]).to_csv(path, index=False)


def log_input(img):
    return np.log1p(img / img.max() * 1000).astype(np.float32)


def score_experimental(model, xs, pts, meta, device):
    """Precision / recall / F1 against the r4 labels, pooled per material (predictions on the beam ignored)."""
    from spotiflow.utils.matching import points_matching
    rows = []
    for x, g, (_, m) in zip(xs, pts, meta.iterrows()):
        p, _ = model.predict(x, min_distance=3, subpix=True, verbose=False, device=device)
        p = p[bd.far_from(p, (m.beam_r, m.beam_c), bd.BEAM_R)]
        mt = points_matching(g, p, cutoff_distance=bd.CUTOFF)
        rows.append(dict(material=m.material, acquisition=m.acquisition, tp=len(mt.matched_pairs),
                         n_pred=len(p), n_label=len(g)))
    df = pd.DataFrame(rows)
    out = df.groupby("material")[["tp", "n_pred", "n_label"]].sum()
    out.loc["all"] = out.sum()
    out["precision"] = out.tp / out.n_pred
    out["recall"] = out.tp / out.n_label
    out["f1"] = 2 * out.tp / (out.n_pred + out.n_label)
    return out.round(3)


def score_simulation(model, sims, device, prob_thresh=None):
    """ft_V's test materials (BW, Co3(O2F)2, SiO2): recall of visible spots and precision, pooled per angle.
    prob_thresh None = the model's own tuned threshold."""
    rows = [dict(angle=it["angle"], **bd.score(it["g"], model.predict(it["x"], prob_thresh=prob_thresh, min_distance=3,
                                                                       subpix=True, verbose=False, device=device)[0]))
            for it in sims]
    df = pd.DataFrame(rows).groupby("angle")[["n_vis", "found_vis", "found_not_vis", "n_pred", "false"]].sum()
    df["recall_vis"] = df.found_vis / df.n_vis
    df["precision"] = 1 - df["false"] / df.n_pred
    return df.round(3)


def tune_sim_threshold(model, base_model, sims_val, device, grid=np.round(np.arange(0.30, 0.81, 0.02), 2)):
    """Detection threshold for SIMULATED patterns: the lowest threshold at which `model` is as precise as
    `base_model` (ft_V, at its own threshold) on the simulation validation materials, pooled over angles.
    The model's own threshold is tuned on the mixed validation set and is too low for simulations
    (many false detections in the overlapping disks at 4 mrad)."""
    def pooled(m, thr):
        s = score_simulation(m, sims_val, device, thr)
        return 1 - s["false"].sum() / s.n_pred.sum()
    target = pooled(base_model, None)
    for thr in grid:
        p = pooled(model, float(thr))
        if p >= target:
            return float(thr), float(target), float(p)
    return float(grid[-1]), float(target), float(p)


def load_simulation(base_run, role):
    """Simulation patterns of ft_V's split ('train', 'val' or 'test'), with ft_V's input and targets.
    The test set uses ft_V's evaluation thicknesses, train / val all thicknesses (as ft_V was trained)."""
    import finetune_diffraction as fd
    split = json.loads((base_run / "split.json").read_text())
    a = split["args"]
    ns = argparse.Namespace(root=a["root"], materials=split[role], angles=None,
                            thicknesses=a["eval_thicknesses"] if role == "test" else a["thicknesses"],
                            targets=a["targets"], vis_box=a["vis_box"], input=a["input"], log_scale=a["log_scale"])
    return fd.load_patterns(ns)


def load_simulation_test(base_run):
    return load_simulation(base_run, "test")


def plot_loss(out):
    import matplotlib.pyplot as plt
    m = pd.read_csv(next((out / "logs").rglob("metrics.csv")))
    tr = m.dropna(subset=["train_loss"]).groupby("epoch").train_loss.last()
    va = m.dropna(subset=["val_loss"]).groupby("epoch")[["val_loss", "val_f1"]].last()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    a1.plot(tr.index + 1, tr.values, "-o", c=OKABE["train"], label="train loss")
    a1.plot(va.index + 1, va.val_loss.values, "--s", c=OKABE["val"], label="validation loss")
    a1.set(xlabel="epoch", ylabel="loss", title="Loss (heatmap + flow)")
    a1.legend()
    a2.plot(va.index + 1, va.val_f1.values, "-^", c=OKABE["f1"])
    a2.set(xlabel="epoch", ylabel="F1", title="Validation F1 (validation set)", ylim=(0, 1))
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.savefig(out / "loss.png", dpi=120)
    return tr, va


def plot_performance(exp, sim, out):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, metric in zip(axs[:2], ("f1", "recall")):
        e = exp[metric].unstack("model")[["before", "after"]]
        x = np.arange(len(e))
        for k, (name, hatch) in enumerate((("before", ""), ("after", "//"))):
            ax.bar(x + (k - 0.5) * 0.4, e[name], 0.4, color=OKABE[name], hatch=hatch, edgecolor="black", label=name)
        ax.set_xticks(x, e.index)
        ax.set(ylim=(0, 1), title=f"Experimental test acquisitions: {metric} (vs r4 labels)")
        ax.legend()
    s = sim.reset_index()
    for name, mk, ls in (("before", "o", "-"), ("after", "s", "--")):
        d = s[s.model == name]
        axs[2].plot(d.angle, d.recall_vis, ls, marker=mk, c=OKABE[name], label=f"{name}: recall (visible)")
        axs[2].plot(d.angle, d.precision, ls, marker=mk, mfc="white", c=OKABE[name], label=f"{name}: precision")
    axs[2].set(xlabel="probe angle (mrad)", ylim=(0, 1.02), xticks=[1, 2, 4],
               title="Simulation test materials (BW, Co3(O2F)2, SiO2)")
    axs[2].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3, axis="y")
    fig.savefig(out / "performance.png", dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="/mnt/data/4D-ImageNet/converted")
    p.add_argument("--base-run", default=str(HERE / "diffraction_runs/ft_V"), help="run whose model/ we continue from")
    p.add_argument("--out", default=str(HERE / "diffraction_runs/ft_V_exp"))
    p.add_argument("--split", default=str(HERE / "diffraction_runs/exp_split.json"))
    p.add_argument("--min-conf", type=float, default=0.8)
    p.add_argument("--per-acq", type=int, default=500, help="training patterns per acquisition, spread over the scan")
    p.add_argument("--eval-per-acq", type=int, default=200, help="patterns per val/test acquisition")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mix-sim", action="store_true",
                   help="also train / validate on ft_V's simulation train / val materials (all angles), so the model "
                        "keeps working on simulations")
    p.add_argument("--sim-repeat", type=int, default=7,
                   help="with --mix-sim: repeat the simulation training patterns this many times (1350 x 7 ~ the "
                        "10000 experimental ones) so both get similar weight")
    p.add_argument("--sim-val-to-train", action="store_true",
                   help="with --mix-sim: also train on ft_V's simulation validation materials (no simulation "
                        "validation; validation is experimental only). The 3 simulation test materials stay held out")
    p.add_argument("--patience", type=int, default=0, help="early stopping on val loss (epochs, 0 = off)")
    p.add_argument("--lr-patience", type=int, default=10,
                   help="halve the learning rate after this many epochs without a better val loss")
    p.add_argument("--select", choices=("loss", "f1"), default="loss",
                   help="best.pt = lowest val_loss (spotiflow default) or highest val_f1")
    p.add_argument("--stop-patience", type=int, default=0,
                   help="early stopping: stop after this many epochs without a better val_f1 (0 = off)")
    p.add_argument("--min-epochs", type=int, default=10, help="no early stopping before this many epochs")
    p.add_argument("--weighted-labels", action="store_true",
                   help="train on every r4 disk (score > 0.5) with target peak height min(1, score / min_conf) "
                        "instead of only disks with score >= min_conf (validation labels unchanged)")
    p.add_argument("--pseudo-from", default=None,
                   help="self-training: add confident detections of this run's model (teacher) that have no r4 "
                        "label to the experimental training labels")
    p.add_argument("--pseudo-thresh", type=float, default=0.7, help="teacher probability for a pseudo label")
    args = p.parse_args()

    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger
    from spotiflow.model import Spotiflow, SpotiflowTrainingConfig

    out, base = Path(args.out), Path(args.base_run)
    out.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed)
    split = json.loads(Path(args.split).read_text()) if Path(args.split).exists() else make_split(args.data, args.split, args.seed)
    write_manifest(args.data, split, Path(args.split).with_name("exp_manifest.csv"), args.min_conf)
    print({k: len(split[k]) for k in ("train", "val", "test")}, "acquisitions; test:", split["test"])

    def load(role, n):
        imgs, pts, meta = load_experimental(args.data, split[role], args.min_conf, n)
        return [log_input(im) for im in imgs], pts, meta

    if args.weighted_labels or args.pseudo_from:
        imgs, P, C, mtr = load_experimental_conf(args.data, split["train"], args.per_acq)
        xtr = [log_input(im) for im in imgs]
        del imgs
        if args.weighted_labels:
            ytr = [np.c_[p, np.minimum(1.0, c / args.min_conf)].astype(np.float32) for p, c in zip(P, C)]
            print(f"weighted labels: {sum(map(len, P))} r4 disks (score > 0.5) instead of "
                  f"{sum(int((c >= args.min_conf).sum()) for c in C)} with score >= {args.min_conf}")
        else:
            ytr = [p[c >= args.min_conf] for p, c in zip(P, C)]
        if args.pseudo_from:
            from spotiflow.model import Spotiflow as _S
            teacher = _S.from_folder(str(Path(args.pseudo_from) / "model"), map_location=args.device)
            extra = pseudo_labels(teacher, xtr, P, mtr, args.pseudo_thresh, args.device)  # away from ALL r4 disks
            n_add = sum(map(len, extra))
            n_lab = sum(map(len, ytr))
            print(f"pseudo labels from {args.pseudo_from} (prob >= {args.pseudo_thresh}): +{n_add} points "
                  f"({n_add / n_lab:.1%} of {n_lab} labels)", flush=True)
            np.savez_compressed(out / "pseudo_labels.npz", *extra)
            (out / "pseudo_labels.json").write_text(json.dumps(dict(
                teacher=str(args.pseudo_from), prob_thresh=args.pseudo_thresh, n_added=int(n_add), n_labels=int(n_lab),
                per_material=pd.Series([len(e) for e in extra]).groupby(mtr.material.values).sum().to_dict()),
                indent=1))
            ytr = [np.concatenate([y, e if y.shape[1] == 2 else np.c_[e, np.ones(len(e))]]).astype(np.float32)
                   for y, e in zip(ytr, extra)]
            del teacher
        del P, C
    else:
        xtr, ytr, _ = load("train", args.per_acq)
    xva, yva, _ = load("val", args.eval_per_acq)
    xte, yte, mte = load("test", args.eval_per_acq)
    sims = load_simulation_test(base)
    print(f"train {len(xtr)}, val {len(xva)}, test {len(xte)} experimental patterns; {len(sims)} simulation test patterns")
    if args.mix_sim:  # keep only input and targets (the scoring extras are not needed for training)
        sim_tr = load_simulation(base, "train") + (load_simulation(base, "val") if args.sim_val_to_train else [])
        s_tr = [(it["x"], it["y"]) for it in sim_tr]
        s_va = [] if args.sim_val_to_train else [(it["x"], it["y"]) for it in load_simulation(base, "val")]
        s_y = [y if ytr[0].shape[1] == 2 else np.c_[y, np.ones(len(y))].astype(np.float32) for _, y in s_tr]
        xtr += [x for x, _ in s_tr] * args.sim_repeat
        ytr += s_y * args.sim_repeat
        xva += [x for x, _ in s_va]
        yva += [y for _, y in s_va]
        print(f"+ simulation: {len(s_tr)} train patterns x {args.sim_repeat}, {len(s_va)} val patterns "
              f"-> train {len(xtr)}, val {len(xva)}")

    before = Spotiflow.from_folder(str(base / "model"), map_location=args.device)
    exp_before, sim_before = score_experimental(before, xte, yte, mte, args.device), score_simulation(before, sims, args.device)

    cfg = SpotiflowTrainingConfig(num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, crop_size=256,
                                  finetuned_from=str(base), optimize_threshold=True,
                                  early_stopping_patience=args.patience, lr_reduce_patience=args.lr_patience)
    model = Spotiflow.from_folder(str(base / "model"), map_location=args.device)  # a fresh copy of ft_V to train
    callbacks = [F1Checkpoint(out / "model", cfg)] if args.select == "f1" else []
    if args.stop_patience > 0:
        callbacks.append(F1EarlyStopping(args.stop_patience, args.min_epochs))
    if ytr[0].shape[1] == 3:
        use_weighted_dataset()
    model.fit(xtr, ytr, xva, yva, save_dir=str(out / "model"), train_config=cfg, device=args.device,
              logger=CSVLogger(save_dir=str(out / "logs"), name=""), num_workers=0, deterministic=False,
              callbacks=callbacks,
              defer_normalization=True)  # normalise per sample: no copy of every (repeated) image in RAM
    after = Spotiflow.from_folder(str(out / "model"), map_location=args.device)  # best checkpoint + tuned threshold
    sim_thr = None
    if args.mix_sim:  # a separate, higher threshold for simulated patterns (see tune_sim_threshold)
        # without simulation validation materials the rule runs on the training simulations (optimistic for both
        # models, but the same for both: ft_V was trained on most of them too)
        tune_on = "train" if args.sim_val_to_train else "val"
        sim_thr, target, prec = tune_sim_threshold(after, before, sim_tr if args.sim_val_to_train
                                                   else load_simulation(base, "val"), args.device)
        (out / "model" / "threshold_sim.json").write_text(json.dumps(
            dict(prob_thresh_sim=sim_thr, rule=f"lowest threshold with ft_V's pooled precision on the simulation "
                                               f"{tune_on} materials", target_precision=round(target, 3),
                 val_precision=round(prec, 3)), indent=2))
        print(f"simulation threshold {sim_thr} (val precision {prec:.3f} >= ft_V's {target:.3f})")
    exp_after = score_experimental(after, xte, yte, mte, args.device)
    sim_after = score_simulation(after, sims, args.device, sim_thr)

    exp = pd.concat({"before": exp_before, "after": exp_after}, names=["model"]).swaplevel().sort_index()
    sim = pd.concat({"before": sim_before, "after": sim_after}, names=["model"]).swaplevel().sort_index()
    exp.to_csv(out / "results_experimental.csv")
    sim.to_csv(out / "results_simulation.csv")
    tr, va = plot_loss(out)
    plot_performance(exp, sim, out)

    print("\n=== loss per epoch ===")
    print(pd.concat([tr.rename("train_loss"), va], axis=1).round(3).to_string())
    print(f"\n=== experimental test acquisitions vs r4 labels (confidence >= {args.min_conf}) ===")
    print(exp.to_string())
    print("\n=== simulation test materials of ft_V (visible spots) ===")
    print(sim.to_string())
    print(f"\nplots: {out / 'loss.png'}, {out / 'performance.png'}")


if __name__ == "__main__":
    main()
