"""Static figures (matplotlib) for the notebook and README."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .metrics import edge_status
from .rca import topo_rank

PALETTE = {
    "ink": "#1F2933",
    "muted": "#8A94A6",
    "grid": "#E4E7EB",
    "correct": "#0E7C66",
    "wrong": "#D64933",
    "reversed": "#E09F3E",
    "missed": "#8A94A6",
    "Random": "#C5CCD6",
    "Lift": "#E09F3E",
    "PC": "#5B7DB1",
    "TTPM": "#0E7C66",
}
METHOD_ORDER = ["Random", "Lift", "PC", "TTPM"]
STATUS_LABEL = {
    "correct": "correct edge",
    "wrong": "wrong edge (not in answer key)",
    "reversed": "reversed direction",
    "missed": "missed edge",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"], "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.grid": True, "grid.color": PALETTE["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10,
    })


def labels(k: int) -> list[str]:
    return [f"A{i}" for i in range(k)]


def circular_positions(truth: np.ndarray) -> dict[int, tuple[float, float]]:
    """Nodes on a circle, ordered by the answer key's topological order (causes first)."""
    order = np.argsort(topo_rank(truth))
    k = len(order)
    ang = np.pi / 2 - 2 * np.pi * np.arange(k) / k
    return {int(n): (float(np.cos(a)), float(np.sin(a))) for n, a in zip(order, ang)}


def plot_graph(ax, est: np.ndarray, truth: np.ndarray, pos: dict, title: str = "",
               show_missed: bool = True) -> None:
    """Node-link drawing of `est`; edge color shows correct / wrong / reversed / missed."""
    k = truth.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(k))
    status = edge_status(est, truth)
    for s in ("missed", "wrong", "reversed", "correct"):
        if s == "missed" and not show_missed:
            continue
        edges = [e for e, st in status.items() if st == s]
        if not edges:
            continue
        nx.draw_networkx_edges(
            g, pos, edgelist=edges, ax=ax, edge_color=PALETTE[s], width=1.7 if s == "correct" else 1.1,
            style="dashed" if s == "missed" else "solid", alpha=0.55 if s == "missed" else 0.9,
            arrows=True, arrowsize=9, node_size=330, connectionstyle="arc3,rad=0.12",
        )
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=330, node_color="white", edgecolors=PALETTE["ink"], linewidths=1.0)
    nx.draw_networkx_labels(g, pos, labels=dict(enumerate(labels(k))), ax=ax, font_size=7, font_color=PALETTE["ink"])
    ax.set_title(title)
    ax.axis("off")


def graph_legend(fig_or_ax, loc="lower center", ncol=4, statuses=("correct", "wrong", "reversed", "missed")) -> None:
    handles = [Patch(facecolor=PALETTE[s], label=STATUS_LABEL[s]) for s in statuses]
    fig_or_ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, fontsize=9)


def plot_edge_matrix(ax, est: np.ndarray, truth: np.ndarray, title: str = "") -> None:
    """K x K matrix: row = cause, column = effect, color = edge status."""
    k = truth.shape[0]
    code = np.zeros((k, k), dtype=int)
    for (i, j), s in edge_status(est, truth).items():
        code[i, j] = {"correct": 1, "wrong": 2, "reversed": 3, "missed": 4}[s]
    cmap = ListedColormap(["#F5F7FA", PALETTE["correct"], PALETTE["wrong"], PALETTE["reversed"], "#C9CED6"])
    ax.imshow(code, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    step = 1 if k <= 18 else 2
    ax.set_xticks(range(0, k, step))
    ax.set_yticks(range(0, k, step))
    ax.set_xticklabels([f"{i}" for i in range(0, k, step)], fontsize=7)
    ax.set_yticklabels([f"{i}" for i in range(0, k, step)], fontsize=7)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("effect alarm type (column)")
    ax.set_ylabel("cause alarm type (row)")
    ax.set_title(title)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_f1_bars(ax, metrics: pd.DataFrame, short: dict[str, str] | None = None) -> None:
    """Directed-edge F1 per dataset for Random, Lift, PC and TTPM."""
    main = metrics[metrics["role"] == "main"]
    datasets = list(dict.fromkeys(main["dataset"]))
    width = 0.2
    x = np.arange(len(datasets))
    for n, method in enumerate(METHOD_ORDER):
        vals = [float(main[(main["dataset"] == d) & (main["method"] == method)]["f1"].iloc[0]) for d in datasets]
        bars = ax.bar(x + (n - 1.5) * width, vals, width, color=PALETTE[method], label=method)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if method == "TTPM" else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([(short or {}).get(d, d) for d in datasets])
    ax.set_ylabel("Directed-edge F1 (higher is better)")
    ax.set_ylim(0, max(0.75, main["f1"].max() + 0.1))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)


def plot_topology_ablation(ax, metrics: pd.DataFrame, short: dict[str, str] | None = None) -> None:
    """TTPM F1 by number of topology hops (0 = topology ignored)."""
    rows = metrics[(metrics["method"].str.startswith("TTPM")) & (metrics["dataset"].str.contains("Microwave"))]
    datasets = list(dict.fromkeys(rows["dataset"]))
    width = 0.24
    x = np.arange(len(datasets))
    names = {0: "no topology", 1: "up to 1 hop", 2: "up to 2 hops"}
    for n, h in enumerate((0, 1, 2)):
        vals = [float(rows[(rows["dataset"] == d) & (rows["hops"] == h)]["f1"].iloc[0]) for d in datasets]
        shade = ["#B7D9D1", "#6FB3A3", PALETTE["TTPM"]][n]
        bars = ax.bar(x + (n - 1) * width, vals, width, color=shade, label=names[h])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([(short or {}).get(d, d) for d in datasets])
    ax.set_ylabel("Directed-edge F1")
    ax.set_ylim(0, rows["f1"].max() * 1.3)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")


def plot_delta_sensitivity(ax, sens: pd.DataFrame, short: dict[str, str] | None = None) -> None:
    rows = sens[sens["method"] == "TTPM"]
    colors = [PALETTE["TTPM"], PALETTE["PC"], PALETTE["Lift"]]
    for color, (d, grp) in zip(colors, rows.groupby("dataset", sort=False)):
        grp = grp.sort_values("delta")
        ax.plot(grp["delta"], grp["f1"], marker="o", color=color, label=(short or {}).get(d, d))
    ax.axvline(0.1, color=PALETTE["muted"], linestyle=":", linewidth=1)
    ax.text(0.1, ax.get_ylim()[0], " gCastle default", color=PALETTE["muted"], fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("delta (decay per second; small = long memory)")
    ax.set_ylabel("Directed-edge F1")
    ax.legend(frameon=False, fontsize=9)


def plot_alarm_rate(ax, alarms: pd.DataFrame, unit: str = "day") -> None:
    sec = 86400 if unit == "day" else 3600
    t = (alarms["start"] - alarms["start"].min()) // sec
    counts = t.value_counts().sort_index()
    full = counts.reindex(range(int(counts.index.max()) + 1), fill_value=0)
    ax.fill_between(full.index, full.values, color=PALETTE["PC"], alpha=0.25, step="mid")
    ax.plot(full.index, full.values, color=PALETTE["PC"], linewidth=1, drawstyle="steps-mid")
    ax.set_xlabel("days since start of observation" if unit == "day" else "hours since start of observation")
    ax.set_ylabel("number of alarms")


def plot_swimlane(ax, explained: pd.DataFrame, rank: np.ndarray, title: str = "") -> None:
    """One lane per alarm type (topological order), time to the right. Root alarms are filled black
    dots; explained alarms are small hollow dots joined to their parent by a thin line."""
    ex = explained.copy()
    t0 = ex["start"].min()
    ex["x"] = ex["start"] - t0
    lane = {t: i for i, t in enumerate(np.argsort(rank))}
    ex["y"] = ex["alarm_id"].map(lane)
    kids = ex[~ex["is_root"]]
    for _, r in kids.iterrows():
        p = ex.loc[int(r["parent_row"])]
        ax.plot([p["x"], r["x"]], [p["y"], r["y"]], color=PALETTE["wrong"], alpha=0.25, linewidth=0.8, zorder=1)
    ax.scatter(kids["x"], kids["y"], s=16, facecolors="white", edgecolors=PALETTE["muted"], linewidths=0.9, zorder=2,
               label=f"explained ({len(kids)})")
    roots = ex[ex["is_root"]]
    ax.scatter(roots["x"], roots["y"], s=34, color=PALETTE["ink"], zorder=3, label=f"root alarm ({len(roots)})")
    ax.set_yticks(list(lane.values()))
    ax.set_yticklabels([f"A{t}" for t in lane], fontsize=7)
    ax.set_xlabel("seconds since the first alarm in the window")
    ax.set_ylabel("alarm type (causes on top)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title(title)
