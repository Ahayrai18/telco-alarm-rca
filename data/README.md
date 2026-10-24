# Data

Three real telecom alarm datasets from Huawei Noah's Ark Lab, released through the gCastle project and
used in the PCIC 2021 Causal Discovery Competition. The files are copied unchanged from
<https://github.com/gcastle-hub/dataset/tree/master/alarm> (Apache-2.0, see `NOTICE`).

| Folder | Alarms | Alarm types | Devices | Span | Topology |
|---|---|---|---|---|---|
| `raw/18V_55N_Wireless` | 34,838 | 18 | 55 | 240 days | none |
| `raw/24V_439N_Microwave` | 64,598 | 24 | 439 | 6 days | yes (395 links) |
| `raw/25V_474N_Microwave` | 48,572 | 25 | 474 | 6 days | yes (440 links) |

Files in each folder:

- `Alarm.csv`: `alarm_id`, `device_id`, `start_timestamp`, `end_timestamp` (seconds since the start of observation).
- `DAG.npy`: the answer key. `DAG[i, j] = 1` means alarm type *i* causes alarm type *j* (expert annotation).
- `Topology.npy`: symmetric adjacency matrix between devices (Microwave datasets only).

Alarm type and device names are anonymized by the data owner.
