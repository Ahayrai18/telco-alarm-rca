"""Building blocks for alarm sequences: device neighborhoods and "preceding alarm" lookups."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def device_reach(topology: np.ndarray | None, n_devices: int, hops: int) -> np.ndarray:
    """Boolean matrix: reach[d, e] is True if device e is within `hops` links of d (d included)."""
    if topology is None or hops <= 0:
        return np.eye(n_devices, dtype=bool)
    dist = shortest_path(csr_matrix(np.asarray(topology) != 0), directed=False, unweighted=True)
    return np.isfinite(dist) & (dist <= hops)


def _csr_lists(reach: np.ndarray):
    deg = reach.sum(axis=1).astype(np.int64)
    ptr = np.concatenate([[0], np.cumsum(deg)])
    idx = np.nonzero(reach)[1]  # np.nonzero returns row-major order
    return ptr, idx, deg


def preceded_by(alarms: pd.DataFrame, parent_type: int, child_type: int, reach: np.ndarray,
                window: int, rank: np.ndarray | None = None, detail: bool = False):
    """For each alarm of `child_type`, is there an alarm of `parent_type` at most `window` seconds
    earlier on the same device or a neighbor (per `reach`)?

    rank=None : the parent must be in a strictly earlier second (used by the baseline).
    rank=array: a parent in the same second counts if rank[parent] < rank[child]. The total order
                (time, rank) keeps explanation chains acyclic.

    `alarms` must have index 0..n-1. Returns a boolean array (child alarms in table order); with
    detail=True returns a dict with matched, t_parent, parent_row, parent_device.
    """
    a_id = alarms["alarm_id"].to_numpy()
    start = alarms["start"].to_numpy().astype(np.int64)
    dev = alarms["device_id"].to_numpy().astype(np.int64)
    c_rows = np.flatnonzero(a_id == child_type)
    p_rows = np.flatnonzero(a_id == parent_type)
    n_child = len(c_rows)

    matched = np.zeros(n_child, dtype=bool)
    t_par = np.full(n_child, np.nan)
    p_row = np.full(n_child, -1, dtype=np.int64)
    p_dev = np.full(n_child, -1, dtype=np.int64)

    def pack():
        if detail:
            return {"matched": matched, "t_parent": t_par, "parent_row": p_row, "parent_device": p_dev}
        return matched

    if n_child == 0 or len(p_rows) == 0:
        return pack()

    # expand every child alarm to (child, neighbor device) lookup rows
    ptr, idx, deg = _csr_lists(reach)
    lens = deg[dev[c_rows]]
    row = np.repeat(np.arange(n_child), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    lookup = idx[np.repeat(ptr[dev[c_rows]], lens) + offs]

    tc, tp = start[c_rows], start[p_rows]
    if rank is None:
        k_child, k_par, tol = tc, tp, int(window)
    else:
        n_k = len(rank)
        k_child = tc * n_k + int(rank[child_type])
        k_par = tp * n_k + int(rank[parent_type])
        tol = int(window) * n_k + n_k - 1

    left = pd.DataFrame({"k": np.repeat(k_child, lens), "dev": lookup, "row": row,
                         "tc": np.repeat(tc, lens)}).sort_values("k", kind="stable")
    right = pd.DataFrame({"k": k_par, "dev": dev[p_rows], "tp": tp, "prow": p_rows}
                         ).sort_values("k", kind="stable")
    merged = pd.merge_asof(left, right, on="k", by="dev", direction="backward",
                           tolerance=tol, allow_exact_matches=False)
    ok = merged["tp"].notna() & ((merged["tc"] - merged["tp"]) <= window)
    best = merged[ok].sort_values("tp", kind="stable").drop_duplicates("row", keep="last")
    r = best["row"].to_numpy()
    matched[r] = True
    t_par[r] = best["tp"].to_numpy()
    p_row[r] = best["prow"].to_numpy().astype(np.int64)
    p_dev[r] = best["dev"].to_numpy().astype(np.int64)
    return pack()
