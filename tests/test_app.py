"""Smoke tests: the dashboard runs without errors for every dataset and every triage graph."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from alarmrca.data import list_datasets  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.mark.parametrize("dataset", list_datasets())
def test_app_runs_for_each_dataset(dataset):
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    at.selectbox(key="dataset").select(dataset).run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert len(at.metric) >= 8


def test_triage_works_for_every_graph_source():
    at = AppTest.from_file(APP, default_timeout=240).run()
    for source in ("truth", "ttpm", "pc", "lift"):
        at.selectbox(key="tri_graph").select(source).run()
        assert not at.exception, source
