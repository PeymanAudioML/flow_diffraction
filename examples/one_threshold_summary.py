#!/usr/bin/env python
"""Score every model on the experimental AND the simulation test at ONE threshold: the threshold with the highest
F1 on the whole experimental validation patterns (from eval_experimental.py's exp_sweep.csv).

    python one_threshold_summary.py --eval diffraction_runs/eval_exp2 --models V V_exp V_mix ...
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

import finetune_experimental as fe
from eval_experimental import RUNS, pooled_sim, predict_low, sim_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", default="diffraction_runs/eval_exp2")
    p.add_argument("--models", nargs="+", required=True)
    a = p.parse_args()
    logging.getLogger("spotiflow").setLevel(logging.WARNING)
    from spotiflow.model import Spotiflow

    R, ev = Path("diffraction_runs"), Path(a.eval)
    sw = pd.read_csv(ev / "exp_sweep.csv")
    sims = fe.load_simulation_test(R / "ft_V")
    rows = []
    for k in a.models:
        g = sw[sw.model == k]
        v = g[g.split == "val"]
        t = float(v.loc[v.f1.idxmax(), "threshold"])
        te = g[(g.split == "test") & (g.threshold == t)].iloc[0]
        m = Spotiflow.from_folder(str(R / RUNS.get(k, f"ft_{k}") / "model"), map_location="cuda")
        low = predict_low(m, [it["x"] for it in sims], "cuda")
        ps = pooled_sim(sim_rows(low, sims, t))
        s = ps[["n_vis", "found_vis", "n_pred", "false"]].sum()
        rec, prec = s.found_vis / s.n_vis, 1 - s["false"] / s.n_pred
        rows.append(dict(model=k, thr=t, exp_TP=int(te.tp), exp_FP=int(te.n_pred - te.tp), exp_FN=int(te.n_label - te.tp),
                         exp_P=te.precision, exp_R=te.recall, exp_F1=te.f1,
                         sim_TP=int(s.found_vis), sim_FP=int(s["false"]), sim_FN=int(s.n_vis - s.found_vis),
                         sim_P=prec, sim_R=rec, sim_F1=2 * prec * rec / (prec + rec),
                         **{f"simR_{x}": ps.loc[x, "recall_vis"] for x in ps.index},
                         **{f"simP_{x}": ps.loc[x, "precision"] for x in ps.index}))
        print(k, "done", flush=True)
    d = pd.DataFrame(rows).set_index("model")
    d["mean_F1"] = (d.exp_F1 + d.sim_F1) / 2
    pd.set_option("display.width", 250)
    print(d.round(3).to_string())
    d.round(4).to_csv(ev / "one_threshold_summary.csv")


if __name__ == "__main__":
    main()
