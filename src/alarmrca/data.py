"""Loader for the three Huawei alarm datasets (released through gCastle)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DATASETS = {
    "18V_55N_Wireless": {"short": "Wireless", "label": "Wireless (18 alarm types, 55 devices)"},
    "24V_439N_Microwave": {"short": "Microwave-24", "label": "Microwave-24 (24 alarm types, 439 devices)"},
    "25V_474N_Microwave": {"short": "Microwave-25", "label": "Microwave-25 (25 alarm types, 474 devices)"},
}


@dataclass
class AlarmDataset:
    """One alarm log with its expert answer key.

    alarms:   columns alarm_id, device_id, start, end (seconds since the start of observation)
    true_dag: true_dag[i, j] = 1 means alarm type i causes alarm type j
    topology: device adjacency matrix, or None when the dataset has no topology
    """

    name: str
    alarms: pd.DataFrame
    true_dag: np.ndarray
    topology: np.ndarray | None

    @property
    def n_types(self) -> int:
        return self.true_dag.shape[0]

    @property
    def n_devices(self) -> int:
        return int(self.alarms["device_id"].max()) + 1

    @property
    def n_events(self) -> int:
        return len(self.alarms)

    @property
    def span_days(self) -> float:
        return float(self.alarms["start"].max() - self.alarms["start"].min()) / 86400.0

    @property
    def short(self) -> str:
        return DATASETS.get(self.name, {}).get("short", self.name)

    @property
    def label(self) -> str:
        return DATASETS.get(self.name, {}).get("label", self.name)


def load_dataset(name: str, data_dir: Path | str | None = None) -> AlarmDataset:
    root = Path(data_dir) if data_dir else DATA_DIR
    folder = root / name
    df = pd.read_csv(folder / "Alarm.csv").rename(
        columns={"start_timestamp": "start", "end_timestamp": "end"}
    )
    df = df.sort_values(["start", "device_id", "alarm_id"], kind="stable").reset_index(drop=True)
    dag = (np.load(folder / "DAG.npy") != 0).astype(int)
    np.fill_diagonal(dag, 0)
    topo_path = folder / "Topology.npy"
    topo = (np.load(topo_path) != 0).astype(int) if topo_path.exists() else None
    return AlarmDataset(name=name, alarms=df, true_dag=dag, topology=topo)


def list_datasets() -> list[str]:
    return list(DATASETS)
