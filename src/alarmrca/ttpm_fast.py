"""TTPM from gCastle with a per-child score cache.

gCastle's hill climbing re-scores every alarm type for each candidate edge, but adding j -> i only
changes the score of type i (the Hawkes likelihood factorizes per child). Caching the score of each
(child, parent set) gives the same graph much faster.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .compat import load_castle

_PC, TTPM, _MetricsDAG = load_castle()
from castle.common import Tensor  # noqa: E402,F401  (safe after load_castle)


class FastTTPM(TTPM):
    def __init__(self, *args, verbose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.trace_: list[dict] = []
        self.score_: float | None = None

    def _prepare_children(self) -> None:
        """Slice the per-event tensors once per alarm type."""
        times = self.tensor["times"].values
        self._total_times = float(times.sum())
        self._x, self._E = [], []
        for i in range(self._N):
            ind = np.flatnonzero(self._event_indexes == i)
            self._x.append(times[ind])
            self._E.append(np.ascontiguousarray(self._effect_tensor_decays[:, ind, :]))
        self._child_cache: dict = {}

    def _em_child(self, i: int, parent_col: np.ndarray):
        """EM for one alarm type; same loop as in TTPM._em, restricted to child i."""
        H, N = self._max_hop + 1, self._N
        pa = np.flatnonzero(parent_col == 1)
        alpha = np.ones((H, N)) * parent_col[None, :]
        mu = 1.0
        x, E = self._x[i], self._E[i]
        li = -100000000000.0
        while True:
            lam_sum = (self._decay_effects * alpha.T).sum() + mu * self._T
            lam = np.zeros(len(x)) + mu
            for k in range(H):
                lam += E[k] @ alpha[k]
            new_li = -lam_sum + (x * np.log(lam)).sum()
            if new_li - li < 0.1:
                return new_li, alpha, mu
            li = new_li
            mu = ((mu / lam) * x).sum() / self._T
            for j in pa:
                for k in range(H):
                    upper = ((alpha[k, j] * E[k][:, j] / lam) * x).sum()
                    lower = self._decay_effects[j, k]
                    alpha[k, j] = 0.0 if lower == 0 else upper / lower

    def _em(self, edge_mat):
        """Penalized log-likelihood of a graph, reusing cached per-child scores."""
        N, H = self._N, self._max_hop + 1
        if not nx.is_directed_acyclic_graph(nx.DiGraph(edge_mat - np.eye(N))):
            return -100000000000000, np.zeros([N, N]), np.zeros(N)
        alpha, mu, total = np.zeros((H, N, N)), np.ones(N), 0.0
        for i in range(N):
            col = edge_mat[:, i]
            key = (i, col.astype(np.int8).tobytes())
            hit = self._child_cache.get(key)
            if hit is None:
                hit = self._em_child(i, col)
                self._child_cache[key] = hit
            li, a_i, mu_i = hit
            total += li
            alpha[:, :, i] = a_i
            mu[i] = mu_i
        n_par = N + self._epsilon * edge_mat.sum() * H
        if self._penalty == "AIC":
            return total - n_par, alpha, mu
        if self._penalty == "BIC":
            return total - n_par * np.log(self._total_times) / 2, alpha, mu
        raise ValueError("penalty must be 'BIC' or 'AIC'.")

    def _hill_climb(self):
        t0 = time.time()
        self._get_effect_tensor_decays()
        self._prepare_children()
        edge_mat = np.eye(self._N, self._N)
        edge_mat[self.priori_knowledge.matrix == 1] = 1
        result = self._em(edge_mat)
        l_ret = result[0]
        for num_iter in range(self._max_iter):
            stop = True
            for cand in list(self._one_step_change_iterator(edge_mat)):
                res = self._em(cand)
                if res[0] > l_ret:
                    result, l_ret, stop, edge_mat = res, res[0], False, cand
            n_edges = int(edge_mat.sum() - self._N)
            self.trace_.append({"iter": num_iter + 1, "score": float(l_ret), "edges": n_edges})
            if self.verbose:
                print(f"  iter {num_iter + 1:3d}  score={l_ret:,.1f}  edges={n_edges}  "
                      f"{time.time() - t0:5.0f}s", flush=True)
            if stop:
                break
        self.score_ = float(l_ret)
        return result, edge_mat


@dataclass
class TTPMResult:
    graph: np.ndarray          # K x K, graph[i, j] = 1 means i -> j
    score: float               # penalized (BIC) score of the final graph
    runtime_s: float
    n_iter: int
    params: dict
    trace: list


def fit_ttpm(alarms: pd.DataFrame, topology: np.ndarray | None, n_types: int, *,
             delta: float = 0.1, max_hop: int = 0, epsilon: float = 1.0,
             max_iter: int = 600, penalty: str = "BIC", verbose: bool = False) -> TTPMResult:
    """Fit FastTTPM on an alarm table with columns alarm_id, device_id, start."""
    n_dev = int(alarms["device_id"].max()) + 1
    topo = np.asarray(topology, dtype=float) if topology is not None else np.zeros((n_dev, n_dev))
    X = pd.DataFrame({
        "event": alarms["alarm_id"].to_numpy(),
        "timestamp": alarms["start"].to_numpy(dtype=float),
        "node": alarms["device_id"].to_numpy(),
    })
    model = FastTTPM(topo, delta=delta, epsilon=epsilon, max_hop=max_hop,
                     penalty=penalty, max_iter=max_iter, verbose=verbose)
    t0 = time.time()
    model.learn(X)
    names = np.asarray(model._event_names, dtype=int)
    raw = np.asarray(model.causal_matrix, dtype=float)
    graph = np.zeros((n_types, n_types), dtype=int)
    graph[np.ix_(names, names)] = (raw != 0).astype(int)
    np.fill_diagonal(graph, 0)
    return TTPMResult(graph=graph, score=float(model.score_), runtime_s=time.time() - t0,
                      n_iter=len(model.trace_), trace=model.trace_,
                      params={"delta": delta, "max_hop": max_hop, "epsilon": epsilon,
                              "penalty": penalty, "max_iter": max_iter})
