"""Graph-quality metrics for learned causal graphs."""
from __future__ import annotations

import numpy as np


def _binary(m: np.ndarray) -> np.ndarray:
    b = (np.asarray(m) != 0).copy()
    np.fill_diagonal(b, False)
    return b


def graph_metrics(est: np.ndarray, true: np.ndarray) -> dict:
    """Compare an estimated graph with the answer key (adjacency matrices, row = cause).

    Follows castle.metrics.MetricsDAG: precision, recall and F1 are strict on directed edges
    (an undirected edge counts as two entries); SHD is computed on the skeleton.
    """
    est, true = _binary(est), _binary(true)
    tp = int((est & true).sum())
    n_pred, n_true = int(est.sum()), int(true.sum())
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # skeleton (direction ignored)
    sk_e, sk_t = np.triu(est | est.T, 1), np.triu(true | true.T, 1)
    sk_tp = int((sk_e & sk_t).sum())
    sk_p = sk_tp / sk_e.sum() if sk_e.sum() else 0.0
    sk_r = sk_tp / sk_t.sum() if sk_t.sum() else 0.0
    sk_f1 = 2 * sk_p * sk_r / (sk_p + sk_r) if (sk_p + sk_r) else 0.0

    # directed edges whose direction is opposite to the answer key
    pure = est & ~est.T
    reversed_ = int((pure & true.T & ~true).sum())
    shd = int((sk_e & ~sk_t).sum() + (sk_t & ~sk_e).sum()) + reversed_

    return {
        "edges_pred": n_pred,
        "edges_true": n_true,
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_true - tp,
        "reversed": reversed_,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "skeleton_f1": round(float(sk_f1), 4),
        "shd": shd,
    }


def edge_status(est: np.ndarray, true: np.ndarray) -> dict[tuple[int, int], str]:
    """Label each edge: correct, wrong or reversed (estimated edges), missed (answer-key edges)."""
    est, true = _binary(est), _binary(true)
    out: dict[tuple[int, int], str] = {}
    k = est.shape[0]
    for i in range(k):
        for j in range(k):
            if est[i, j]:
                if true[i, j]:
                    out[(i, j)] = "correct"
                elif true[j, i]:
                    out[(i, j)] = "reversed"
                else:
                    out[(i, j)] = "wrong"
            elif true[i, j] and not est[j, i]:
                out[(i, j)] = "missed"
    return out


def random_dag(k: int, n_edges: int, seed: int = 0) -> np.ndarray:
    """Random DAG on k nodes with n_edges edges (random node order, edges drawn uniformly)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(k)
    slots = [(order[a], order[b]) for a in range(k) for b in range(a + 1, k)]
    g = np.zeros((k, k), dtype=int)
    for p in rng.choice(len(slots), size=min(n_edges, len(slots)), replace=False):
        g[slots[p]] = 1
    return g


def random_reference(true: np.ndarray, n_draws: int = 300, seed: int = 0, n_edges: int | None = None) -> dict:
    """Mean metrics of random DAGs. By default they have as many edges as the answer key; pass
    n_edges to get the chance level for a method that returns a different number of edges."""
    n_edges = int(_binary(true).sum()) if n_edges is None else int(n_edges)
    rows = [graph_metrics(random_dag(true.shape[0], n_edges, seed + i), true) for i in range(n_draws)]
    keys = ["precision", "recall", "f1", "skeleton_f1", "shd", "edges_pred"]
    return {key: float(np.mean([r[key] for r in rows])) for key in keys}
