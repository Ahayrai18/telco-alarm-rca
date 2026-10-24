import numpy as np
import pandas as pd

from alarmrca.events import device_reach, preceded_by
from alarmrca.rca import explain_alarms, root_ranking, topo_rank

# graph 0 -> 1 -> 2
G = np.zeros((3, 3), dtype=int)
G[0, 1] = G[1, 2] = 1
TOPO = np.array([[0, 1], [1, 0]])  # device 0 is a neighbor of device 1


def log(rows):
    return pd.DataFrame(rows, columns=["alarm_id", "device_id", "start"]).assign(end=lambda d: d["start"])


def test_chain_is_explained_and_root_collects_descendants():
    ex = explain_alarms(log([(0, 0, 100), (1, 0, 110), (2, 0, 120)]), G, None, 2, window=60)
    assert ex["is_root"].tolist() == [True, False, False]
    assert ex["parent_row"].tolist() == [-1, 0, 1]
    assert ex["root_row"].tolist() == [0, 0, 0]
    assert root_ranking(ex).iloc[0]["n_explained"] == 2


def test_window_limits_explanation():
    ex = explain_alarms(log([(0, 0, 100), (1, 0, 300)]), G, None, 2, window=60)
    assert ex["is_root"].tolist() == [True, True]


def test_topology_hops_matter():
    rows = log([(0, 1, 500), (1, 0, 520)])
    assert explain_alarms(rows, G, TOPO, 2, window=60, hops=0)["is_root"].tolist() == [True, True]
    assert explain_alarms(rows, G, TOPO, 2, window=60, hops=1)["is_root"].tolist() == [True, False]


def test_same_second_parent_counts_but_reverse_order_does_not():
    rows = log([(0, 0, 700), (1, 0, 700)])
    assert explain_alarms(rows, G, None, 2, window=60)["is_root"].tolist() == [True, False]
    rev = np.zeros((3, 3), dtype=int)
    rev[1, 0] = 1  # now 1 -> 0
    assert explain_alarms(rows, rev, None, 2, window=60)["is_root"].tolist() == [False, True]


def test_reciprocal_edges_never_create_cycles():
    both = np.zeros((3, 3), dtype=int)
    both[0, 1] = both[1, 0] = 1
    ex = explain_alarms(log([(0, 0, 10), (1, 0, 10)]), both, None, 1, window=60)
    assert ex["is_root"].sum() == 1


def test_topo_rank_breaks_cycles():
    cyc = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert sorted(topo_rank(cyc).tolist()) == [0, 1, 2]


def test_preceded_by_strict_by_default():
    df = log([(0, 0, 50), (1, 0, 50), (1, 0, 80)])
    reach = device_reach(None, 1, 0)
    assert preceded_by(df, 0, 1, reach, 60).tolist() == [False, True]
