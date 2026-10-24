#!/usr/bin/env python
"""Create the README figures from the files in results/. Run after run_experiments.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alarmrca import viz  # noqa: E402
from alarmrca.data import DATASETS, load_dataset  # noqa: E402
from alarmrca.metrics import graph_metrics  # noqa: E402
from alarmrca.rca import explain_alarms, topo_rank  # noqa: E402

RESULTS, FIG = ROOT / "results", ROOT / "results" / "figures"
SHORT = {k: v["short"] for k, v in DATASETS.items()}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", name, flush=True)


def main() -> None:
    viz.style()
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    sens = pd.read_csv(RESULTS / "sensitivity.csv")

    # 1. F1 comparison
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    viz.plot_f1_bars(ax, metrics, SHORT)
    ax.set_title("How close the learned graph is to the expert graph")
    fig.text(0.5, -0.03, "Random: mean of 300 random DAGs with as many edges as the answer key.", ha="center",
             fontsize=8, color=viz.PALETTE["muted"])
    save(fig, "f1_comparison.png")

    # 2. edge-status matrices: 3 datasets x 3 methods
    names = list(DATASETS)
    fig, axes = plt.subplots(3, 3, figsize=(11, 11.4))
    for r, name in enumerate(names):
        g = np.load(RESULTS / "graphs" / f"{name}.npz")
        for c, (key, label) in enumerate((("lift", "Lift"), ("pc", "PC"), ("ttpm", "TTPM"))):
            m = graph_metrics(g[key], g["truth"])
            viz.plot_edge_matrix(axes[r, c], g[key], g["truth"], f"{SHORT[name]} | {label} | F1 {m['f1']:.2f}")
    handles = [matplotlib.patches.Patch(facecolor=viz.PALETTE[s] if s != "missed" else "#C9CED6",
                                        label=viz.STATUS_LABEL[s]) for s in ("correct", "wrong", "reversed", "missed")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "edge_matrices.png")

    # 3. node-link graph for the Wireless dataset: PC vs TTPM
    g = np.load(RESULTS / "graphs" / "18V_55N_Wireless.npz")
    pos = viz.circular_positions(g["truth"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax, key, label in ((axes[0], "pc", "PC"), (axes[1], "ttpm", "TTPM")):
        m = graph_metrics(g[key], g["truth"])
        viz.plot_graph(ax, g[key], g["truth"], pos, f"Wireless | {label} | F1 {m['f1']:.2f}, SHD {m['shd']}",
                       show_missed=False)
        ax.set_aspect("equal")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    viz.graph_legend(fig, loc="lower center", ncol=3, statuses=("correct", "wrong", "reversed"))
    save(fig, "graph_wireless.png")

    # 4. topology ablation
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    viz.plot_topology_ablation(ax, metrics, SHORT)
    ax.set_title("Effect of topology on TTPM")
    save(fig, "topology_ablation.png")

    # 5. delta sensitivity
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    viz.plot_delta_sensitivity(ax, sens, SHORT)
    ax.set_title("TTPM sensitivity to delta")
    save(fig, "delta_sensitivity.png")

    # 6. triage example (Microwave-24, TTPM graph): busiest burst vs the burst that is best explained
    name = "24V_439N_Microwave"
    ds = load_dataset(name)
    g = np.load(RESULTS / "graphs" / f"{name}.npz")
    width = 600
    base = int(ds.alarms["start"].min())
    counts = ((ds.alarms["start"] - base) // width).value_counts()

    def window_stats(b):
        t0 = int(b * width + base)
        sl = ds.alarms[(ds.alarms["start"] >= t0 - 60) & (ds.alarms["start"] < t0 + width)]
        ex = explain_alarms(sl, g["ttpm"], ds.topology, ds.n_devices, window=60, hops=2)
        return t0, ex[ex["start"] >= t0]

    cand = []
    for b in counts.index[:20]:
        t0, ex = window_stats(b)
        cand.append((len(ex) - int(ex["is_root"].sum()), int(b), t0, len(ex), int(ex["is_root"].sum())))
    busiest = window_stats(counts.index[0])[1]
    best = max(cand)
    t0, ex = window_stats(best[1])
    ex = ex[(ex["parent_row"] < 0) | ex["parent_row"].isin(ex.index)]
    info = {"busiest_alarms": len(busiest), "busiest_roots": int(busiest["is_root"].sum()),
            "example_alarms": len(ex), "example_roots": int(ex["is_root"].sum()), "dataset": name}
    (RESULTS / "triage_example.json").write_text(json.dumps(info))
    print(info, flush=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    viz.plot_swimlane(ax, ex, topo_rank(g["ttpm"]),
                      f"Example 10-minute burst ({SHORT[name]}): {len(ex)} alarms become {int(ex['is_root'].sum())} root alarms")
    save(fig, "triage_example.png")


if __name__ == "__main__":
    main()
