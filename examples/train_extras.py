"""Training extras for finetune_experimental.py (spotiflow itself is not changed).

- F1Checkpoint      keep best.pt at the highest val_f1 (spotiflow's default keeps the lowest val_loss)
- F1EarlyStopping   stop when val_f1 has not improved for `patience` epochs (spotiflow stores
                    early_stopping_patience in the config but never uses it)
- WeightedSpotsDataset  heatmap targets with a peak height per point (column 3 of the point array) instead of 1,
                    so uncertain labels count less. Point arrays with 2 columns behave exactly as SpotsDataset.
- pseudo_labels     confident detections of a teacher model that have no r4 label (self-training)
"""
import json
import logging

import lightning.pytorch as pl
import numpy as np
import torch
from spotiflow.data import SpotsDataset
from spotiflow.model import SpotiflowModelCheckpoint
from spotiflow import utils as su

log = logging.getLogger(__name__)


class F1Checkpoint(SpotiflowModelCheckpoint):
    """SpotiflowModelCheckpoint that keeps the model with the highest val_f1."""

    def __init__(self, logdir, train_config):
        super().__init__(logdir=logdir, train_config=train_config, monitor="val_f1")
        self._best = -np.inf
        self.history = []

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero and not trainer.sanity_checking:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            trainer.logger.log_metrics({"learning_rate": lr}, step=trainer.global_step)
            value = float(trainer.logged_metrics["val_f1"])
            self.history.append(dict(epoch=trainer.current_epoch, val_f1=value,
                                     val_loss=float(trainer.logged_metrics["val_loss"])))
            if value > self._best:
                self._best = value
                print(f"epoch {trainer.current_epoch}: best val_f1 {value:.4f}, saved best.pt", flush=True)
                pl_module.model.save(self._logdir, which="best")
            (self._logdir / "selection.json").write_text(json.dumps(
                dict(monitor="val_f1 (max)", best_val_f1=self._best,
                     best_epoch=max(self.history, key=lambda h: h["val_f1"])["epoch"], history=self.history), indent=1))


class F1EarlyStopping(pl.Callback):
    """Stop when val_f1 has not improved by more than min_delta for `patience` validations (after min_epochs)."""

    def __init__(self, patience, min_epochs=10, min_delta=1e-3):
        self.patience, self.min_epochs, self.min_delta = patience, min_epochs, min_delta
        self.best, self.wait = -np.inf, 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = float(trainer.logged_metrics["val_f1"])
        if value > self.best + self.min_delta:
            self.best, self.wait = value, 0
        else:
            self.wait += 1
        if self.wait >= self.patience and trainer.current_epoch + 1 >= self.min_epochs:
            print(f"early stopping after epoch {trainer.current_epoch}: val_f1 not better than {self.best:.4f} "
                  f"for {self.wait} epochs", flush=True)
            trainer.should_stop = True


class WeightedSpotsDataset(SpotsDataset):
    """SpotsDataset whose point arrays may have a 3rd column: the target peak height (0-1] of that point."""

    def __init__(self, images, centers, *args, **kwargs):
        weights = None
        if len(centers) and centers[0].ndim == 2 and centers[0].shape[1] == 3:
            weights = [np.asarray(c[:, 2], np.float32) for c in centers]
            centers = [np.ascontiguousarray(c[:, :2], dtype=np.float32) for c in centers]
        super().__init__(images, centers, *args, **kwargs)
        self._weighted = weights is not None
        if self._weighted:  # the class column added by SpotsDataset (all 0) now carries the weight
            self._centers = [np.concatenate([c[:, :2], w[:, None]], axis=1).astype(np.float32)
                             for c, w in zip(self._centers, weights)]

    def __getitem__(self, idx):
        if not self._weighted:
            return super().__getitem__(idx)
        img, centers = self.images[idx], self._centers[idx]
        if self._defer_normalization:
            img = self._normalizer(img)
        img = torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(1)          # (1, 1, H, W)
        centers = torch.from_numpy(centers.copy()).unsqueeze(0)
        img, centers = self.augmenter(img, centers)                           # the last column is carried along
        img, centers = img.squeeze(0), centers.squeeze(0)
        pts, w = centers[:, :2].numpy(), centers[:, 2].numpy()
        shape = img.shape[-2:]
        ret = {"img": img.float(),
               "pts": torch.cat([centers[:, :2], torch.zeros(len(centers), 1)], dim=1).float()}
        if self._compute_flow:
            ret["flow"] = torch.from_numpy(su.points_to_flow(pts, shape, sigma=self._sigma).transpose((2, 0, 1))).float()
        hm = np.zeros(shape, np.float32)
        wq = np.round(w * 20) / 20                                             # <= 20 levels: few Gaussian renders
        for level in np.unique(wq):
            hm = np.maximum(hm, level * su.points_to_prob(pts[wq == level], shape, mode=self._mode, sigma=self._sigma))
        hm = hm[None]
        for lv, ds in enumerate(self._downsample_factors):
            ret[f"heatmap_lv{lv}"] = torch.from_numpy(su.multiscale_decimate(hm, ds).copy())
        return ret


def use_weighted_dataset():
    """Make Spotiflow.fit build WeightedSpotsDataset (identical to SpotsDataset for 2-column point arrays)."""
    import spotiflow.model.spotiflow as sm
    sm.SpotsDataset = WeightedSpotsDataset


def pseudo_labels(teacher, xs, labels, meta, prob_thresh, device, cutoff=3.0, beam_r=6.0, border=8):
    """Teacher detections with probability >= prob_thresh that are > cutoff px from every label, > beam_r px from
    the direct beam and >= border px from the image edge. -> list of (k, 2) arrays (one per image)."""
    out = []
    for x, g, (_, m) in zip(xs, labels, meta.iterrows()):
        p, _ = teacher.predict(x, prob_thresh=prob_thresh, min_distance=3, subpix=True, verbose=False, device=device)
        p = np.asarray(p, np.float32).reshape(-1, 2)
        keep = np.hypot(p[:, 0] - m.beam_r, p[:, 1] - m.beam_c) > beam_r
        keep &= np.all((p >= border) & (p < np.array(x.shape) - border), axis=1)
        if len(g):
            keep &= np.min(np.hypot(*(p[:, None] - np.asarray(g)[None, :, :2]).transpose(2, 0, 1)), axis=1) > cutoff
        out.append(p[keep])
    return out
