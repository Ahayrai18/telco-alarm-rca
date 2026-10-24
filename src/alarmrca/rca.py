"""Alarm triage: turn an alarm-type graph into root alarms and the alarms they explain.

An alarm is "explained" when the graph gives its type a parent and an alarm of that parent type
occurred at most `window` seconds earlier on the same device or a neighbor (up to `hops` links).
Unexplained alarms are root alarms. Each alarm points to its nearest parent in time, so every root
alarm owns a tree of descendants.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .events import device_reach, preceded_by


def topo_rank(graph: np.ndarray) -> np.ndarray:
    """Topological rank of each alarm type (parents before children); cycles are broken as needed."""
    g = nx.DiGraph((np.asarray(graph) != 0).astype(int))
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    while not nx.is_directed_acyclic_graph(g):
        g.remove_edge(*nx.find_cycle(g)[0][:2])
    rank = np.zeros(g.number_of_nodes(), dtype=np.int64)
    for pos, node in enumerate(nx.topological_sort(g)):
        rank[node] = pos
    return rank


def explain_alarms(alarms: pd.DataFrame, graph: np.ndarray, topology: np.ndarray | None,
                   n_devices: int, window: int = 60, hops: int = 0) -> pd.DataFrame:
    """Add is_root, parent_row, parent_type, parent_device, lag and root_row columns to the alarm table."""
    df = alarms.reset_index(drop=True).copy()
    g = (np.asarray(graph) != 0).copy()
    np.fill_diagonal(g, False)
    rank = topo_rank(g)
    reach = device_reach(topology, n_devices, hops)
    n = len(df)
    a_id = df["alarm_id"].to_numpy()

    best_t = np.full(n, -np.inf)
    parent_row = np.full(n, -1, dtype=np.int64)
    parent_type = np.full(n, -1, dtype=np.int64)
    parent_dev = np.full(n, -1, dtype=np.int64)

    for child in range(g.shape[0]):
        rows = np.flatnonzero(a_id == child)
        if len(rows) == 0:
            continue
        for par in np.flatnonzero(g[:, child]):
            d = preceded_by(df, int(par), child, reach, window, rank=rank, detail=True)
            better = d["matched"] & (d["t_parent"] > best_t[rows])  # keep the most recent parent
            tgt = rows[better]
            best_t[tgt] = d["t_parent"][better]
            parent_row[tgt] = d["parent_row"][better]
            parent_type[tgt] = par
            parent_dev[tgt] = d["parent_device"][better]

    df["is_root"] = parent_row < 0
    df["parent_row"] = parent_row
    df["parent_type"] = parent_type
    df["parent_device"] = parent_dev
    df["lag"] = np.where(parent_row >= 0, df["start"].to_numpy() - np.where(np.isfinite(best_t), best_t, 0), np.nan)

    # follow parent pointers (always to earlier events) to find each alarm's root
    root_row = np.arange(n)
    order = np.lexsort((rank[a_id], df["start"].to_numpy()))
    for i in order:
        if parent_row[i] >= 0:
            root_row[i] = root_row[parent_row[i]]
    df["root_row"] = root_row
    return df


def triage_summary(explained: pd.DataFrame) -> dict:
    n = len(explained)
    n_root = int(explained["is_root"].sum())
    return {"n_alarms": n, "n_roots": n_root, "explained_frac": (1 - n_root / n) if n else 0.0}


def root_ranking(explained: pd.DataFrame) -> pd.DataFrame:
    """Root alarms sorted by the number of alarms they explain."""
    counts = explained.groupby("root_row").size().rename("n_total")
    roots = explained[explained["is_root"]].copy()
    roots["n_explained"] = counts.reindex(roots.index).fillna(1).astype(int) - 1
    cols = ["alarm_id", "device_id", "start", "n_explained"]
    return roots[cols].sort_values(["n_explained", "start"], ascending=[False, True])


def root_agreement(ref_is_root: np.ndarray, est_is_root: np.ndarray) -> dict:
    """How well the root alarms of an estimated graph match those of the reference (expert) graph."""
    ref, est = np.asarray(ref_is_root, bool), np.asarray(est_is_root, bool)
    both = int((ref & est).sum())
    precision = both / est.sum() if est.sum() else 0.0
    recall = both / ref.sum() if ref.sum() else 0.0
    return {"root_precision": float(precision), "root_recall": float(recall)}
