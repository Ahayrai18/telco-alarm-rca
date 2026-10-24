#!/usr/bin/env python
"""Run all experiments and save the results to results/.

    python scripts/run_experiments.py                  # all datasets (about 10-15 min on 1 CPU)
    python scripts/run_experiments.py --datasets 18V_55N_Wireless --force
    python scripts/run_experiments.py --aggregate      # only merge the existing per-dataset results

Per-dataset results are written to results/parts/, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alarmrca.baseline import lagged_cooccurrence, select_edges  # noqa: E402
from alarmrca.data import list_datasets, load_dataset  # noqa: E402
from alarmrca.metrics import graph_metrics, random_dag, random_reference  # noqa: E402
from alarmrca.pc_method import run_pc  # noqa: E402
from alarmrca.rca import explain_alarms, root_agreement  # noqa: E402
from alarmrca.ttpm_fast import fit_ttpm  # noqa: E402

RESULTS = ROOT / "results"

# Main configuration: library defaults or common practice, NOT tuned with the answer key.
CFG = {
    "lift": {"window": 60, "min_lift": 3.0, "min_conf": 0.05, "min_support": 20},
    "pc": {"window": 600, "per_device": True, "alpha": 0.05},
    "ttpm": {"delta": 0.1, "epsilon": 1.0},
    "triage_window": 60,
}


def timed(fn, *a, **k):
    t0 = time.time()
    out = fn(*a, **k)
    return out, time.time() - t0


def run_lift(ds, hops, window, min_lift, min_conf, min_support):
    sc = lagged_cooccurrence(ds.alarms, ds.n_types, ds.topology, ds.n_devices, window=window, hops=hops)
    return select_edges(sc, min_lift, min_conf, min_support)


def run_dataset(name: str, force: bool = False) -> None:
    part = RESULTS / "parts" / f"{name}.json"
    if part.exists() and not force:
        print(f"[{name}] already done, skipped (use --force to redo)", flush=True)
        return
    ds = load_dataset(name)
    hops = 0 if ds.topology is None else 2
    truth = ds.true_dag
    print(f"[{name}] {ds.n_events} alarms, {ds.n_types} types, {ds.n_devices} devices, hops={hops}", flush=True)
    metrics, sweeps, graphs = [], [], {"truth": truth}

    def add(method, role, g, runtime, note="", **extra):
        m = graph_metrics(g, truth)
        metrics.append({"dataset": name, "method": method, "role": role, "hops": extra.pop("hops", hops),
                        **m, "runtime_s": round(runtime, 2), "note": note})

    # 0. random reference
    ref = random_reference(truth)
    metrics.append({"dataset": name, "method": "Random", "role": "main", "hops": 0, **{
        k: (round(v, 4) if isinstance(v, float) else v) for k, v in ref.items()},
        "edges_true": int(truth.sum()), "runtime_s": 0.0,
        "note": "mean of 300 random DAGs with as many edges as the answer key"})

    # 1. lift baseline
    g, t = timed(run_lift, ds, hops, **CFG["lift"])
    graphs["lift"] = g
    add("Lift", "main", g, t, f"W={CFG['lift']['window']}s lift>={CFG['lift']['min_lift']}")
    print(f"  lift  F1={metrics[-1]['f1']}", flush=True)

    # 2. PC
    (g, info), t = timed(run_pc, ds.alarms, ds.n_types, **CFG["pc"])
    graphs["pc"] = g
    add("PC", "main", g, t, f"{CFG['pc']['window']}s windows per device, n={info['n_samples']}")
    print(f"  PC    F1={metrics[-1]['f1']}", flush=True)

    # 3. TTPM
    res = fit_ttpm(ds.alarms, ds.topology, ds.n_types, max_hop=hops, **CFG["ttpm"])
    graphs["ttpm"] = res.graph
    add("TTPM", "main", res.graph, res.runtime_s, f"delta={CFG['ttpm']['delta']} hops<={hops} iterations={res.n_iter}")
    metrics[-1]["bic_score"] = round(res.score, 1)
    print(f"  TTPM  F1={metrics[-1]['f1']}  ({res.runtime_s:.0f}s)", flush=True)

    # 4. topology ablation (datasets with a topology only)
    if ds.topology is not None:
        for h in (0, 1):
            r = fit_ttpm(ds.alarms, ds.topology, ds.n_types, max_hop=h, **CFG["ttpm"])
            graphs[f"ttpm_hop{h}"] = r.graph
            add(f"TTPM hop<={h}", "ablation_topology", r.graph, r.runtime_s, f"delta={CFG['ttpm']['delta']}", hops=h)
            print(f"  TTPM hop<={h} F1={metrics[-1]['f1']}", flush=True)

    # 5. sensitivity sweeps
    def sw(method, params, g, runtime, **extra):
        sweeps.append({"dataset": name, "method": method, "params": params, **graph_metrics(g, truth),
                       "runtime_s": round(runtime, 2), **extra})

    for d in (0.01, 0.03, 0.1, 0.3, 1.0):
        r = fit_ttpm(ds.alarms, ds.topology, ds.n_types, max_hop=hops, delta=d, epsilon=1.0)
        sw("TTPM", f"delta={d}", r.graph, r.runtime_s, delta=d, bic_score=round(r.score, 1))
    print("  TTPM sweep done", flush=True)
    for per_device in (True, False):
        for window in (600, 3600):
            for alpha in (0.05, 0.001):
                (g, info), t = timed(run_pc, ds.alarms, ds.n_types, window=window, per_device=per_device, alpha=alpha)
                sw("PC", f"{'per device' if per_device else 'whole network'}, {window}s, alpha={alpha}", g, t,
                   window=window, per_device=per_device, alpha=alpha, n_samples=info["n_samples"])
    for window in (60, 600, 3600):
        scores = lagged_cooccurrence(ds.alarms, ds.n_types, ds.topology, ds.n_devices, window=window, hops=hops)
        for min_lift, min_conf in ((2.0, 0.02), (3.0, 0.05), (5.0, 0.05)):
            g = select_edges(scores, min_lift, min_conf, 20)
            sw("Lift", f"W={window}s, lift>={min_lift}, conf>={min_conf}", g, 0.0,
               window=window, min_lift=min_lift, min_conf=min_conf)
    print("  PC and Lift sweeps done", flush=True)

    # 6. triage over the whole log
    W = CFG["triage_window"]
    triage, ref_roots = [], None
    for label, g in (("Expert", truth), ("TTPM", graphs["ttpm"]), ("PC", graphs["pc"]), ("Lift", graphs["lift"])):
        ex = explain_alarms(ds.alarms, g, ds.topology, ds.n_devices, window=W, hops=hops)
        rnd = [1 - explain_alarms(ds.alarms, random_dag(ds.n_types, int((g != 0).sum()), 100 + s), ds.topology,
                                  ds.n_devices, window=W, hops=hops)["is_root"].mean() for s in range(3)]
        row = {"dataset": name, "graph": label, "edges": int((g != 0).sum()), "window_s": W, "hops": hops,
               "explained_frac": round(float(1 - ex["is_root"].mean()), 4),
               "explained_frac_random": round(float(np.mean(rnd)), 4)}
        if label == "Expert":
            ref_roots = ex["is_root"].to_numpy()
        else:
            row.update({k: round(v, 4) for k, v in root_agreement(ref_roots, ex["is_root"].to_numpy()).items()})
        triage.append(row)
    print("  triage done", flush=True)

    (RESULTS / "graphs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "graphs" / f"{name}.npz", **graphs)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps({"metrics": metrics, "sweeps": sweeps, "triage": triage}, indent=1))
    print(f"[{name}] done", flush=True)


def aggregate() -> None:
    parts = sorted((RESULTS / "parts").glob("*.json"))
    order = {n: i for i, n in enumerate(list_datasets())}
    parts.sort(key=lambda p: order.get(p.stem, 99))
    data = [json.loads(p.read_text()) for p in parts]
    met = pd.DataFrame([r for d in data for r in d["metrics"]])
    # fair chance level: F1 of random DAGs with as many edges as the method returned
    met["f1_chance"] = met["f1"]
    for i, row in met.iterrows():
        if row["method"] != "Random":
            truth = np.load(RESULTS / "graphs" / f"{row['dataset']}.npz")["truth"]
            met.loc[i, "f1_chance"] = round(random_reference(truth, n_draws=100, n_edges=int(row["edges_pred"]))["f1"], 4)
    met.to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame([r for d in data for r in d["sweeps"]]).to_csv(RESULTS / "sensitivity.csv", index=False)
    pd.DataFrame([r for d in data for r in d["triage"]]).to_csv(RESULTS / "triage.csv", index=False)
    import castle
    info = {"config": CFG, "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "gcastle": getattr(castle, "__version__", "?"),
            "datasets": [p.stem for p in parts]}
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=1))
    print("results merged:", [p.stem for p in parts], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list_datasets())
    ap.add_argument("--force", action="store_true", help="recompute even if results exist")
    ap.add_argument("--aggregate", action="store_true", help="only merge existing results")
    a = ap.parse_args()
    if not a.aggregate:
        for n in a.datasets:
            run_dataset(n, a.force)
    aggregate()
