import numpy as np
import pytest

from alarmrca.metrics import edge_status, graph_metrics, random_dag, random_reference


def test_perfect_graph():
    g = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    m = graph_metrics(g, g)
    assert (m["precision"], m["recall"], m["f1"], m["shd"]) == (1.0, 1.0, 1.0, 0)


def test_reversed_edge_counts_once_in_shd_and_hurts_f1():
    true = np.array([[0, 1], [0, 0]])
    est = true.T.copy()
    m = graph_metrics(est, true)
    assert m["reversed"] == 1 and m["shd"] == 1 and m["f1"] == 0.0 and m["skeleton_f1"] == 1.0
    assert edge_status(est, true) == {(1, 0): "reversed"}


def test_undirected_edge_is_two_directed_entries():
    true = np.array([[0, 1], [0, 0]])
    est = np.array([[0, 1], [1, 0]])
    m = graph_metrics(est, true)
    assert m["edges_pred"] == 2 and m["tp"] == 1 and m["precision"] == 0.5 and m["shd"] == 0


def test_random_reference_matches_edge_count():
    true = random_dag(10, 12, seed=1)
    assert random_reference(true, n_draws=20)["edges_pred"] == 12


def test_matches_gcastle_metricsdag():
    pytest.importorskip("castle")
    from alarmrca.compat import load_castle

    _, _, MetricsDAG = load_castle()
    rng = np.random.default_rng(0)
    for _ in range(60):
        k = int(rng.integers(5, 12))
        true = np.triu((rng.random((k, k)) < 0.3).astype(int), 1)
        est = (rng.random((k, k)) < 0.2).astype(int)
        np.fill_diagonal(est, 0)
        ref, mine = MetricsDAG(est.copy(), true.copy()).metrics, graph_metrics(est, true)
        # gCastle returns NaN when precision and recall are both 0; treated as 0 here
        assert abs(np.nan_to_num(ref["F1"]) - mine["f1"]) < 1e-3
        assert abs(np.nan_to_num(ref["precision"]) - mine["precision"]) < 1e-3
        assert ref["shd"] == mine["shd"]
