import networkx as nx
import numpy as np
import pytest

from alarmrca.data import list_datasets, load_dataset


@pytest.mark.parametrize("name", list_datasets())
def test_dataset_is_consistent(name):
    ds = load_dataset(name)
    assert ds.alarms["alarm_id"].between(0, ds.n_types - 1).all()
    assert ds.alarms["start"].is_monotonic_increasing
    assert (ds.alarms["end"] >= ds.alarms["start"]).all()
    assert nx.is_directed_acyclic_graph(nx.DiGraph(ds.true_dag))
    assert np.trace(ds.true_dag) == 0
    if ds.topology is not None:
        assert ds.topology.shape == (ds.n_devices, ds.n_devices)
        assert (ds.topology == ds.topology.T).all()
