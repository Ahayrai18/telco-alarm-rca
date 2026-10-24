"""Classic baseline: the PC algorithm (gCastle) on time-window tables."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.special import erfc

from .compat import load_castle


def cached_fisherz(data: np.ndarray):
    """Fisher-z test identical to gCastle's `CITest.fisherz_test`, but the correlation matrix is
    computed once, so each test costs O(|S|^3) instead of O(n * |S|^2)."""
    n = data.shape[0]
    corr = np.corrcoef(data.T)

    def test(_data, x, y, z):
        k = len(z)
        if k == 0:
            r = corr[x, y]
        else:
            idx = [x, y, *z]
            sub = corr[np.ix_(idx, idx)]
            try:
                pm = np.linalg.inv(sub)
            except np.linalg.LinAlgError:
                pm = np.linalg.pinv(sub)
            r = -1 * pm[0, 1] / math.sqrt(abs(pm[0, 0] * pm[1, 1]))
        r = min(0.99999, max(-0.99999, r))
        stat = math.sqrt(n - k - 3) * 0.5 * math.log1p((2 * r) / (1 - r))
        return None, None, float(erfc(abs(stat) / math.sqrt(2)))

    return test


def window_samples(alarms: pd.DataFrame, n_types: int, window: int = 600, per_device: bool = True,
                   active_only: bool = True) -> np.ndarray:
    """Turn the alarm log into a 0/1 table (samples x alarm types).

    per_device=True : one sample = one device in one time window.
    per_device=False: one sample = one time window for the whole network.
    active_only     : keep only samples that contain at least one alarm.
    """
    t = alarms["start"].to_numpy() - alarms["start"].min()
    b = (t // window).astype(np.int64)
    n_bins = int(b.max()) + 1
    if per_device:
        n_dev = int(alarms["device_id"].max()) + 1
        key = alarms["device_id"].to_numpy().astype(np.int64) * n_bins + b
        n_rows = n_dev * n_bins
    else:
        key, n_rows = b, n_bins
    m = coo_matrix((np.ones(len(key), dtype=np.int8), (key, alarms["alarm_id"].to_numpy())),
                   shape=(n_rows, n_types)).tocsr()
    m.data[:] = 1
    if active_only:
        m = m[np.asarray(m.sum(axis=1)).ravel() > 0]
    return m.toarray().astype(np.float64)


def run_pc(alarms: pd.DataFrame, n_types: int, window: int = 600, per_device: bool = True,
           active_only: bool = True, alpha: float = 0.05, ci_test: str = "fisherz",
           max_samples: int | None = None, seed: int = 0,
           cached: bool = True) -> tuple[np.ndarray, dict]:
    """Run PC. Returns a K x K matrix (an undirected edge appears as 1 in both directions)."""
    PC, _TTPM, _M = load_castle()
    X = window_samples(alarms, n_types, window, per_device, active_only)
    if max_samples and len(X) > max_samples:
        X = X[np.random.default_rng(seed).choice(len(X), max_samples, replace=False)]
    keep = np.flatnonzero(X.std(axis=0) > 0)  # PC cannot use constant columns
    Xk = X[:, keep]
    use_cached = cached and ci_test == "fisherz"
    model = PC(alpha=alpha, ci_test=cached_fisherz(Xk) if use_cached else ci_test)
    model.learn(Xk)
    g = np.zeros((n_types, n_types), dtype=int)
    g[np.ix_(keep, keep)] = (np.asarray(model.causal_matrix) != 0).astype(int)
    np.fill_diagonal(g, 0)
    return g, {"n_samples": int(len(X)), "n_vars": int(len(keep))}
