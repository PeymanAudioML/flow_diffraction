"""Append section 8 (V_mix3-5: F1 checkpoint, early stopping, weighted r4 labels, self-training) to
6_experimental_test_results.ipynb and execute only the new cells (plus the setup cell). Run from examples/:
    python3 diffraction_runs/queue_step23/add_sec8.py
Re-running replaces an existing section 8."""
import copy

import nbformat as nbf
from nbclient import NotebookClient

P = "6_experimental_test_results.ipynb"
nb = nbf.read(P, 4)
cut = next((i for i, c in enumerate(nb.cells) if c.source.startswith("## 8.")), len(nb.cells))
nb.cells = nb.cells[:cut]
new = []
md = lambda s: new.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: new.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
## 8. Better checkpoint choice and better labels (V_mix3, V_mix4, V_mix5)

Three runs with V_mix2's settings (ft_V start, 20k experimental + 1650 × 12 simulated training patterns, lr 2e-4,
batch 16, at most 30 epochs). Each adds one change:

| run | change compared with V_mix2 |
|---|---|
| **V_mix3b** | best.pt = epoch with the **highest validation F1** (V_mix2: lowest validation loss); all 30 epochs |
| **V_mix4b** | V_mix3b + **score-weighted r4 labels**: every r4 disk (score > 0.5) is a training target, with peak height min(1, score / 0.8). Disks ≥ 0.8 count fully, a disk of score 0.5 counts 0.63 |
| **V_mix5b** | V_mix3b + **self-training**: V_mix2's detections with probability ≥ 0.55 that are > 3 px from every r4 disk (and away from the beam and the image edge) are added as labels. At 0.55 these have the same contrast as r4 hits (section 7: median snr 4.0) |

- **A first try with early stopping** (V_mix3, V_mix4, V_mix5: stop after 6 epochs without a better validation F1,
  at least 10 epochs) stopped too soon: at lr 2e-4 validation F1 swings by ±0.05 between epochs and rises only after
  the learning rate is halved (V_mix2: epoch 15). V_mix3 stopped after 10 epochs with its best at epoch 0. The "b" runs
  train all 30 epochs; the early-stopped runs are listed at the end of 8.2 for comparison.
- **Validation is the same for all runs:** the 3 experimental validation acquisitions with r4 ≥ 0.8 labels. Validation
  F1 is therefore experimental only; V_mix2's setup has no simulation validation set (`--sim-val-to-train`).
- **Testing is the same as in sections 3–4:** `eval_experimental.py`, results in `diffraction_runs/eval_exp2/`. The
  label scorings of section 7 are in `eval_exp2/labels_check/`.
- **Code:** `train_extras.py` (callbacks, weighted dataset, pseudo labels) and new options of `finetune_experimental.py`.
  The queue is `diffraction_runs/queue_step23/run.sh`.
''')
code(r'''
EV2 = R / "eval_exp2"
NEW = ["V_exp", "V_mix2", "V_mix3b", "V_mix4b", "V_mix5b"]
EARLY = ["V_mix3", "V_mix4", "V_mix5"]           # first try, early stopped
for k in NEW[2:] + EARLY:
    RUNS[k] = R / f"ft_{k}"
STYLE.update({"V_mix3b": (SKY, "v"), "V_mix4b": (ORANGE, "s"), "V_mix5b": ("#000000", "X")})
DESC.update({"V_mix3b": "V_mix3b: V_mix2 + F1 checkpoint",
             "V_mix4b": "V_mix4b: V_mix3b + score-weighted r4 labels (> 0.5)",
             "V_mix5b": "V_mix5b: V_mix3b + self-training (V_mix2 pseudo labels)"})
NEW = [k for k in NEW if (RUNS[k] / "model/best.pt").exists()]
thr2 = json.loads((EV2 / "thresholds.json").read_text())
print("runs:", NEW)
print("thresholds (experimental / simulation):", {k: (thr2[k]["exp"], thr2[k]["sim"]) for k in NEW})
for k in NEW:
    pj = RUNS[k] / "pseudo_labels.json"
    if pj.exists():
        print(f"{k} pseudo labels:", json.loads(pj.read_text()))
''')
md(r'''
### 8.1 Training curves

Large marker: the kept checkpoint (V_mix2: lowest validation loss; V_mix3–5: highest validation F1). A line that ends
before epoch 29 was stopped early. V_mix2's line ends in epoch 26 because of the machine shutdown.
''')
code(r'''
fig, axs = plt.subplots(1, 3, figsize=(17, 4), constrained_layout=True)
for k in [k for k in NEW if k != "V_exp"]:
    col, mk = STYLE[k]
    m = pd.read_csv(sorted(RUNS[k].glob("logs/*/metrics.csv"))[-1])
    tr = m.dropna(subset=["train_loss"]).groupby("epoch").train_loss.last()
    v = m.dropna(subset=["val_loss"]).groupby("epoch")[["val_loss", "val_f1"]].last()
    sel = RUNS[k] / "model/selection.json"
    best = json.loads(sel.read_text())["best_epoch"] if sel.exists() else int(v.val_loss.idxmin())
    how = "highest val F1" if sel.exists() else "lowest val loss"
    kw = dict(color=col, marker=mk, ms=4, mec="black" if col == "#000000" else col)
    axs[0].plot(tr.index, tr, label=DESC[k], **kw)
    axs[1].plot(v.index, v.val_loss, ls="--", label=f"{k}: kept epoch {best} ({how})", **kw)
    axs[2].plot(v.index, v.val_f1, label=f"{k}: kept epoch {best}, val F1 {v.val_f1[best]:.3f}", **kw)
    for ax, y in ((axs[1], v.val_loss[best]), (axs[2], v.val_f1[best])):
        ax.plot(best, y, marker=mk, ms=11, color=col, mec="black", mew=1.5, ls="none")
    print(f"{k}: {len(v)} epochs, kept epoch {best} ({how}): val F1 {v.val_f1[best]:.3f}, "
          f"val loss {v.val_loss[best]:.3f}; max val F1 {v.val_f1.max():.3f} (epoch {int(v.val_f1.idxmax())})")
axs[0].set_title("training loss (V_mix4 / V_mix5 train on different labels)")
axs[1].set_title("validation loss (lower = better)")
axs[2].set_title("validation F1 (higher = better)")
for ax in axs:
    ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
plt.show()
''')
md(r'''
### 8.2 Experimental test

Same 6 test acquisitions × 500 patterns as section 3; each model at its own experimental threshold. The first table
uses the r4 ≥ 0.8 labels (as in section 3); the second shows the fairer scorings of section 7 (beam label removed;
r4 0.5–0.8 optional).
''')
code(r'''
exp2 = pd.read_csv(EV2 / "exp_test.csv").set_index(["model", "material"])
e = pd.concat({k: exp2.loc[k] for k in NEW}, axis=1)
print("vs r4 >= 0.8 labels (as section 3):")
print(e.loc[:, (slice(None), ["precision", "recall", "f1"])].round(3).to_string())
lc2 = EV2 / "labels_check/scores.csv"
if lc2.exists():
    sc2 = pd.read_csv(lc2).set_index(["model", "scoring", "material"]).xs("all", level="material")
    print("\nall test materials, by scoring (section 7):")
    print(sc2.loc[[k for k in NEW if k in sc2.index.get_level_values(0)]][["threshold", "precision", "recall", "f1"]]
          .unstack("scoring").round(3).to_string())
early = [k for k in EARLY if k in exp2.index.get_level_values(0)]
if early:
    print("\nfirst try, early stopped (vs r4 >= 0.8, all test materials):")
    print(pd.concat({k: exp2.loc[(k, "all")] for k in early}, axis=1).T[["precision", "recall", "f1"]].round(3)
          .assign(threshold=[thr2[k]["exp"] for k in early]).to_string())

mats = [m for m in e.index if m != "all"] + ["all"]
fig, axs = plt.subplots(1, 3, figsize=(18, 4.4), constrained_layout=True)
w = 0.8 / len(NEW)
hatch = ["", "//", "..", "xx", "\\\\"]
for ax, metric in zip(axs, ("f1", "recall", "precision")):
    x = np.arange(len(mats))
    for i, k in enumerate(NEW):
        col, _ = STYLE[k]
        ax.bar(x + (i - (len(NEW) - 1) / 2) * w, e.loc[mats, (k, metric)].values, w, color=col,
               edgecolor="white" if col == "#000000" else "black", hatch=hatch[i],
               label=f"{DESC[k]} (threshold {thr2[k]['exp']:.2f})")
    ax.set_xticks(x, mats)
    ax.set(ylim=(0, 1), title=f"experimental test: {metric} (vs r4 >= 0.8)")
    ax.grid(alpha=0.3, axis="y")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17), fontsize=10)
plt.show()
''')
md(r'''
**Same threshold rule for every model.** The thresholds above come from different rules: spotiflow's
`optimize_threshold` (random 128-px crops of the validation patterns; V_exp, V_mix, V_mix3–5) or whole validation
patterns (V_mix2). The crop rule gives lower thresholds (0.31–0.40) than the whole-pattern rule (0.44–0.48). Here
every model uses the threshold with the best F1 on the **whole validation patterns** (grid step 0.02, from
`exp_sweep.csv`), and the test scores are read at that threshold.
''')
code(r'''
sw = pd.read_csv(EV2 / "exp_sweep.csv")
rows = []
for k, g in sw.groupby("model"):
    v = g[g.split == "val"]
    t = v.loc[v.f1.idxmax(), "threshold"]
    te = g[(g.split == "test") & (g.threshold == t)].iloc[0]
    own = g[(g.split == "test")].iloc[(g[g.split == "test"].threshold - thr2[k]["exp"]).abs().argmin()]
    rows.append(dict(model=k, own_threshold=thr2[k]["exp"], test_f1_own=own.f1, whole_val_threshold=t,
                     val_f1=v.f1.max(), test_precision=te.precision, test_recall=te.recall, test_f1=te.f1))
fair = pd.DataFrame(rows).set_index("model").drop(index="V", errors="ignore")
print("test_f1_own: at the model's own threshold (nearest grid point); test_*: at the whole-pattern val threshold")
display(fair.round(3))
fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
x = np.arange(len(fair))
ax.bar(x - 0.2, fair.test_f1_own, 0.4, color=SKY, edgecolor="black", label="own threshold (as in the tables above)")
ax.bar(x + 0.2, fair.test_f1, 0.4, color=BLUE, edgecolor="black", hatch="//",
       label="threshold best on whole validation patterns")
ax.set_xticks(x, fair.index)
ax.set(ylim=(0.5, 0.75), ylabel="test F1 (all materials)", title="experimental test F1: effect of the threshold rule")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)
plt.show()
''')
md(r'''
### 8.3 Simulation test

Same 45 simulation test patterns as section 4 (BW, Co₃(O₂F)₂, SiO₂ at 1, 2 and 4 mrad). `own` = the experimental
threshold, `sim` = the separate simulation threshold (tune_sim_threshold rule).
''')
code(r'''
sim2 = pd.read_csv(EV2 / "sim_test.csv").set_index(["model", "rule", "angle"])
cols = {}
for k in NEW:
    for rule in ("own", "sim"):
        if rule == "sim" and thr2[k]["sim"] == thr2[k]["exp"]:
            continue
        cols[f"{k} {rule} ({thr2[k]['exp' if rule == 'own' else 'sim']:.2f})"] = sim2.loc[(k, rule)]
s2 = pd.concat(cols, axis=1)
print(s2.loc[:, (slice(None), ["recall_vis", "precision"])].round(3).T.to_string())

fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
for k in NEW:
    col, mk = STYLE[k]
    s = sim2.loc[(k, "own")]
    kw = dict(color=col, marker=mk, ms=7, mec="black" if col == "#000000" else col)
    axs[0].plot(s.index, s.recall_vis, label=f"{DESC[k]}; threshold {thr2[k]['exp']:.2f}", **kw)
    axs[1].plot(s.index, s.precision, label=f"{DESC[k]}; threshold {thr2[k]['exp']:.2f}", **kw)
for ax, t in zip(axs, ("visible recall", "precision")):
    ax.set(xlabel="probe semi-angle (mrad)", xticks=[1, 2, 4], ylim=(0, 1.02),
           title=f"simulation test: {t} (each model's experimental threshold)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.show()
''')
md(r'''
### Results of section 8

*(written after the runs finished)*
''')

tmp = nbf.v4.new_notebook(cells=[copy.deepcopy(nb.cells[1])] + new, metadata=nb.metadata)
NotebookClient(tmp, timeout=1200, kernel_name="spotiflow", resources={"metadata": {"path": "."}}).execute()
nb.cells += tmp.cells[1:]
nbf.write(nb, P)
print("section 8 added:", len(new), "cells")
