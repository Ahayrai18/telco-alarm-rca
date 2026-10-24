import numpy as np
import pytest

pytest.importorskip("castle")

from alarmrca.compat import load_castle  # noqa: E402
from alarmrca.data import load_dataset  # noqa: E402
from alarmrca.pc_method import cached_fisherz, run_pc  # noqa: E402


def test_cached_fisherz_equals_gcastle():
    from castle.common.independence_tests import CITest

    rng = np.random.default_rng(3)
    data = (rng.random((3000, 6)) < 0.3).astype(float)
    data[:, 1] = np.maximum(data[:, 1], data[:, 0] * (rng.random(3000) < 0.8))
    fast = cached_fisherz(data)
    for x, y, z in [(0, 1, []), (0, 2, [1]), (3, 4, [0, 1]), (1, 5, [2, 3, 4])]:
        assert fast(data, x, y, z)[2] == pytest.approx(CITest.fisherz_test(data, x, y, z)[2], rel=1e-6, abs=1e-12)


def test_pc_cached_graph_equals_gcastle_graph():
    ds = load_dataset("18V_55N_Wireless")
    a, _ = run_pc(ds.alarms, ds.n_types, max_samples=1500, cached=True)
    b, _ = run_pc(ds.alarms, ds.n_types, max_samples=1500, cached=False)
    assert np.array_equal(a, b)


def test_fast_ttpm_equals_stock_ttpm():
    import pandas as pd

    from alarmrca.ttpm_fast import FastTTPM

    _, TTPM, _ = load_castle()
    ds = load_dataset("18V_55N_Wireless")
    al = ds.alarms.iloc[:1200]
    X = pd.DataFrame({"event": al.alarm_id, "timestamp": al.start.astype(float), "node": al.device_id})
    topo = np.zeros((ds.n_devices, ds.n_devices))
    stock = TTPM(topo, max_hop=0, max_iter=5)
    stock.learn(X.copy())
    fast = FastTTPM(topo, max_hop=0, max_iter=5)
    fast.learn(X.copy())
    assert np.array_equal(np.asarray(stock.causal_matrix), np.asarray(fast.causal_matrix))
