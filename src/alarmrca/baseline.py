"""Simple baseline: temporal association (lift) between alarm types."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import device_reach, preceded_by


def lagged_cooccurrence(alarms: pd.DataFrame, n_types: int, topology: np.ndarray | None, n_devices: int,
                        window: int = 60, hops: int = 0) -> dict[str, np.ndarray]:
    """Scores for every ordered pair (a -> b).

    confidence[a, b]: share of b alarms preceded by an a alarm within `window` seconds
                      (same device, or neighbors within `hops` links).
    lift[a, b]      : confidence divided by the chance level if a alarms arrived at random with
                      their historical rate (Poisson approximation per device).
    support[a, b]   : number of b alarms preceded by a.
    """
    reach = device_reach(topology, n_devices, hops)
    span = float(alarms["start"].max() - alarms["start"].min() + 1)
    a_id = alarms["alarm_id"].to_numpy()
    dev = alarms["device_id"].to_numpy()

    per_dev = np.zeros((n_devices, n_types))
    np.add.at(per_dev, (dev, a_id), 1.0)
    per_nbr = reach.astype(float) @ per_dev  # alarms of each type in every device's neighborhood

    conf = np.zeros((n_types, n_types))
    lift = np.zeros((n_types, n_types))
    support = np.zeros((n_types, n_types), dtype=int)
    for b in range(n_types):
        d_b = dev[a_id == b]
        n_b = len(d_b)
        if n_b == 0:
            continue
        expected = (1.0 - np.exp(-per_nbr[d_b] * window / span)).mean(axis=0)  # shape (n_types,)
        for a in range(n_types):
            if a == b:
                continue
            hit = preceded_by(alarms, a, b, reach, window)
            support[a, b] = int(hit.sum())
            conf[a, b] = hit.mean()
            lift[a, b] = conf[a, b] / max(expected[a], 1e-9)
    return {"confidence": conf, "lift": lift, "support": support}


def select_edges(scores: dict[str, np.ndarray], min_lift: float = 3.0, min_conf: float = 0.05,
                 min_support: int = 20) -> np.ndarray:
    """Keep pairs above all thresholds; if both a->b and b->a pass, keep the one with higher lift."""
    lift, conf, sup = scores["lift"], scores["confidence"], scores["support"]
    g = ((lift >= min_lift) & (conf >= min_conf) & (sup >= min_support)).astype(int)
    k = g.shape[0]
    for a in range(k):
        for b in range(a + 1, k):
            if g[a, b] and g[b, a]:
                keep_ab = (lift[a, b], conf[a, b]) >= (lift[b, a], conf[b, a])
                if keep_ab:
                    g[b, a] = 0
                else:
                    g[a, b] = 0
    return g
