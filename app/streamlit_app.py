"""Dashboard: root-cause analysis of telecom alarms (Huawei alarm datasets via gCastle).

Run with:  streamlit run app/streamlit_app.py
It only reads the raw data and the files in results/, so gCastle and PyTorch are not needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alarmrca.data import DATASETS, load_dataset  # noqa: E402
from alarmrca.metrics import edge_status, graph_metrics, random_dag  # noqa: E402
from alarmrca.rca import explain_alarms, root_ranking, topo_rank  # noqa: E402

RESULTS = ROOT / "results"
INK, MUTED = "#1F2933", "#8A94A6"
STATUS_COLOR = {"correct": "#0E7C66", "wrong": "#D64933", "reversed": "#E09F3E", "missed": "#B8C0CC"}
STATUS_TEXT = {"correct": "Correct", "wrong": "Wrong", "reversed": "Reversed", "missed": "Missed"}
METHOD_COLOR = {"Random": "#C5CCD6", "Lift": "#E09F3E", "PC": "#5B7DB1", "TTPM": "#0E7C66"}
GRAPH_LABEL = {
    "truth": "Expert graph (answer key)",
    "ttpm": "TTPM",
    "pc": "PC",
    "lift": "Lift baseline",
    "ttpm_hop1": "TTPM, topology up to 1 hop",
    "ttpm_hop0": "TTPM, no topology",
}

st.set_page_config(page_title="Root-cause alarms in a telecom network", page_icon="📡", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
    .stApp { font-family: 'Atkinson Hyperlegible', system-ui, sans-serif; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    .block-container { padding-top: 2.2rem; max-width: 1200px; }
    .legend span { display:inline-block; margin-right:1.1rem; font-size:0.9rem; }
    .legend i { display:inline-block; width:0.8rem; height:0.8rem; border-radius:2px; margin-right:0.35rem; vertical-align:-1px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data loading
@st.cache_resource(show_spinner=False)
def get_dataset(name: str):
    return load_dataset(name)


@st.cache_resource(show_spinner=False)
def get_graphs(name: str) -> dict[str, np.ndarray]:
    z = np.load(RESULTS / "graphs" / f"{name}.npz")
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def get_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / filename)


def default_hops(ds) -> int:
    return 0 if ds.topology is None else 2


def fmt(x: float) -> str:
    return f"{x:.2f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"day {seconds // 86400 + 1}, {seconds % 86400 // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


if not (RESULTS / "metrics.csv").exists():
    st.error("The results/ folder has no results yet. Run `python scripts/run_experiments.py` first.")
    st.stop()

metrics = get_table("metrics.csv")
sens = get_table("sensitivity.csv")
triage = get_table("triage.csv")
SHORT = {k: v["short"] for k, v in DATASETS.items()}

# ----------------------------------------------------------------------------- header and dataset choice
st.title("Finding the root-cause alarm in an alarm flood")
st.markdown(
    "A single fault in a telecom network, such as a power outage or a failed microwave link, can trigger dozens of "
    "alarms across many devices at once. Operators have to guess which alarm is the cause and which ones are only "
    "consequences. This project learns the cause-and-effect relations between alarm types directly from the logs, "
    "then uses them to filter the flood. Data: three real alarm logs released by Huawei Noah's Ark Lab (gCastle)."
)

ds_name = st.selectbox("Dataset", list(DATASETS), format_func=lambda n: DATASETS[n]["label"], key="dataset")
ds = get_dataset(ds_name)
graphs = get_graphs(ds_name)
hops_default = default_hops(ds)

tab_overview, tab_data, tab_graph, tab_triage, tab_notes = st.tabs(
    ["Results overview", "Alarm data", "Causal graph", "Triage simulation", "Methods and limitations"]
)

# ----------------------------------------------------------------------------- 1. results overview
with tab_overview:
    main = metrics[metrics["role"] == "main"]
    f1 = main.pivot(index="method", columns="dataset", values="f1")
    f_t, f_p, f_l = f1.loc["TTPM"], f1.loc["PC"], f1.loc["Lift"]
    st.subheader("How close are the learned causal graphs to the expert graph?")
    st.markdown(
        f"The answer key is a cause-and-effect graph between alarm types drawn by domain experts. On the three "
        f"datasets, TTPM (a topological Hawkes-process model that uses timing and topology) reaches an F1 of "
        f"**{fmt(f_t.iloc[0])}**, **{fmt(f_t.iloc[1])}** and **{fmt(f_t.iloc[2])}**. PC, a classic algorithm that only "
        f"sees a table of which alarms occurred, reaches {fmt(f_p.iloc[0])}, {fmt(f_p.iloc[1])} and {fmt(f_p.iloc[2])}. "
        f"A simple time-association baseline reaches {fmt(f_l.iloc[0])}, {fmt(f_l.iloc[1])} and {fmt(f_l.iloc[2])}."
    )

    chart_df = main.copy()
    chart_df["Dataset"] = chart_df["dataset"].map(SHORT)
    bars = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Dataset:N", sort=list(SHORT.values()), axis=alt.Axis(labelAngle=0, title=None)),
            xOffset=alt.XOffset("method:N", sort=list(METHOD_COLOR)),
            y=alt.Y("f1:Q", title="Directed-edge F1", scale=alt.Scale(domain=[0, 0.8])),
            color=alt.Color("method:N", sort=list(METHOD_COLOR), title="Method",
                            scale=alt.Scale(domain=list(METHOD_COLOR), range=list(METHOD_COLOR.values()))),
            tooltip=["Dataset", "method", alt.Tooltip("f1:Q", format=".3f"), "precision", "recall", "shd"],
        )
        .properties(height=320, width="container")
    )
    labels_ = bars.mark_text(dy=-6, fontSize=11).encode(text=alt.Text("f1:Q", format=".2f"), color=alt.value(INK))
    st.altair_chart(bars + labels_, width="stretch")

    st.caption("Random is the mean of 300 random DAGs with as many edges as the answer key. The answer-key graphs "
               "are dense (about 22 to 25 percent of all pairs), so an F1 around 0.2 is already chance level. "
               "Methods that return fewer edges have a lower chance level; see the Chance F1 column in the table below.")
    near = main[main["method"].isin(["Lift", "PC"]) & ((main["f1"] - main["f1_chance"]) < 0.05)]
    if len(near):
        st.markdown("Note: " + ", ".join(f"{r.method} on {SHORT[r.dataset]}" for r in near.itertuples())
                    + " only match chance level (F1 within 0.05 of a random DAG with the same number of edges).")

    st.subheader(f"Details for {SHORT[ds_name]}")
    sub = main[main["dataset"] == ds_name].set_index("method").loc[["Random", "Lift", "PC", "TTPM"]]
    table = sub[["edges_pred", "tp", "fp", "reversed", "precision", "recall", "f1", "f1_chance", "shd",
                 "runtime_s"]].rename(
        columns={"edges_pred": "Edges found", "tp": "Correct", "fp": "Wrong", "reversed": "Reversed",
                 "precision": "Precision", "recall": "Recall", "f1": "F1", "f1_chance": "Chance F1",
                 "shd": "SHD", "runtime_s": "Seconds"})
    st.dataframe(table.round(3), width="stretch")
    st.markdown(
        "**How to read it.** Precision: out of every 10 relations reported, how many are in the expert graph. "
        "Recall: out of all relations in the expert graph, how many were found. SHD: the number of corrections "
        "(add, remove, reverse) still needed; lower is better."
    )

# ----------------------------------------------------------------------------- 2. alarm data
with tab_data:
    a = ds.alarms
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alarms", f"{ds.n_events:,}")
    c2.metric("Devices", ds.n_devices)
    c3.metric("Alarm types", ds.n_types)
    c4.metric("Time span", f"{ds.span_days:.0f} days")

    unit = 86400 if ds.span_days > 30 else 3600
    unit_name = "day" if unit == 86400 else "hour"
    rate = ((a["start"] - a["start"].min()) // unit).value_counts().sort_index()
    rate = rate.reindex(range(int(rate.index.max()) + 1), fill_value=0).rename_axis("t").reset_index(name="alarms")
    st.subheader(f"Alarms per {unit_name}")
    area = (
        alt.Chart(rate).mark_area(line={"color": "#5B7DB1"}, color="#5B7DB1", opacity=0.25, interpolate="step-after")
        .encode(x=alt.X("t:Q", title=f"{unit_name}s since start of observation"),
                y=alt.Y("alarms:Q", title="number of alarms"), tooltip=["t", "alarms"])
        .properties(height=230, width="container")
    )
    st.altair_chart(area, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Alarms per type")
        per_type = a["alarm_id"].value_counts().rename_axis("type").reset_index(name="count")
        per_type["type"] = "A" + per_type["type"].astype(str)
        bar = (
            alt.Chart(per_type).mark_bar(color="#5B7DB1")
            .encode(y=alt.Y("type:N", sort="-x", title=None), x=alt.X("count:Q", title="number of alarms"),
                    tooltip=["type", "count"])
            .properties(height=max(220, 17 * ds.n_types), width="container")
        )
        st.altair_chart(bar, width="stretch")
    with right:
        st.subheader("Data characteristics")
        dur = a["end"] - a["start"]
        lines = [
            f"Alarms with zero duration: **{pct((dur == 0).mean())}**.",
            f"Alarms recorded in the same second as the previous alarm: **{pct((a['start'].diff() == 0).mean())}**.",
            "Alarm type names are anonymized (A0, A1, ...), so relations can only be read from timing patterns.",
        ]
        if ds.topology is not None:
            deg = ds.topology.sum(axis=1)
            lines.insert(0, f"Topology: **{int(ds.topology.sum() // 2)}** links between devices, {deg.mean():.1f} "
                            f"neighbors per device on average (maximum {int(deg.max())}).")
        else:
            lines.insert(0, "This dataset has no device topology.")
        for line in lines:
            st.markdown("- " + line)

# ----------------------------------------------------------------------------- 3. causal graph
with tab_graph:
    options = [k for k in ("ttpm", "pc", "lift", "ttpm_hop1", "ttpm_hop0", "truth") if k in graphs]
    cc1, cc2, cc3 = st.columns([2, 2, 2])
    which = cc1.selectbox("Graph to compare with the answer key", options, format_func=GRAPH_LABEL.get)
    view = cc2.radio("View", ["Graph", "Edge matrix"], index=0 if ds.n_types <= 18 else 1, horizontal=True)
    show_missed = cc3.checkbox("Show missed edges", value=True)
    est, truth = graphs[which], graphs["truth"]
    m = graph_metrics(est, truth)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("F1", fmt(m["f1"]))
    k2.metric("Precision", fmt(m["precision"]))
    k3.metric("Recall", fmt(m["recall"]))
    k4.metric("SHD", m["shd"])
    k5.metric("Edges found", f"{m['edges_pred']} of {m['edges_true']}")

    status = edge_status(est, truth)
    st.markdown(
        '<div class="legend">'
        + "".join(f'<span><i style="background:{STATUS_COLOR[s]}"></i>{STATUS_TEXT[s]}</span>' for s in STATUS_COLOR)
        + "</div>",
        unsafe_allow_html=True,
    )
    if view == "Graph":
        dot = ["digraph G {", 'rankdir=LR; nodesep=0.22; ranksep=1.0; bgcolor="transparent"; splines=true;',
               f'node [shape=circle, style=filled, fillcolor="white", color="{INK}", fontname="Helvetica", '
               'fontsize=11, width=0.45, fixedsize=true];', "edge [arrowsize=0.7];"]
        dot += [f'A{i} [label="A{i}"];' for i in range(ds.n_types)]
        # invisible answer-key edges keep the node layout identical across methods
        dot += [f"A{i} -> A{j} [style=invis];" for i, j in zip(*np.nonzero(truth))]
        look = {"correct": f'color="{STATUS_COLOR["correct"]}", penwidth=2.2',
                "wrong": f'color="{STATUS_COLOR["wrong"]}", penwidth=1.2',
                "reversed": f'color="{STATUS_COLOR["reversed"]}", penwidth=1.8',
                "missed": f'color="{STATUS_COLOR["missed"]}", style=dashed, penwidth=1'}
        for (i, j), s in status.items():
            if s == "missed" and not show_missed:
                continue
            dot.append(f"A{i} -> A{j} [{look[s]}, constraint=false];")
        dot.append("}")
        st.graphviz_chart("\n".join(dot), width="stretch")
    else:
        rows = [{"cause": f"A{i}", "effect": f"A{j}", "status": STATUS_TEXT[s]}
                for (i, j), s in status.items() if s != "missed" or show_missed]
        order = [f"A{i}" for i in range(ds.n_types)]
        heat = (
            alt.Chart(pd.DataFrame(rows)).mark_rect(stroke="white", strokeWidth=1.2)
            .encode(x=alt.X("effect:N", sort=order, title="effect alarm type"),
                    y=alt.Y("cause:N", sort=order, title="cause alarm type"),
                    color=alt.Color("status:N", legend=None,
                                    scale=alt.Scale(domain=[STATUS_TEXT[s] for s in STATUS_COLOR],
                                                    range=list(STATUS_COLOR.values()))),
                    tooltip=["cause", "effect", "status"])
            .properties(height=max(360, 19 * ds.n_types), width=max(360, 19 * ds.n_types))
        )
        st.altair_chart(heat)

    with st.expander("Edge list"):
        edge_df = pd.DataFrame([{"cause": f"A{i}", "effect": f"A{j}", "status": STATUS_TEXT[s]}
                                for (i, j), s in sorted(status.items())])
        pick = st.multiselect("Status", list(STATUS_TEXT.values()), default=list(STATUS_TEXT.values()))
        st.dataframe(edge_df[edge_df["status"].isin(pick)], hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- 4. triage simulation
with tab_triage:
    st.markdown(
        "An alarm counts as **explained** when the graph gives its type a parent and an alarm of that parent type "
        "appeared a short time earlier on the same device or a neighboring one. Alarms that cannot be explained are "
        "**root alarms**, the ones an operator should look at first."
    )
    c1, c2, c3 = st.columns(3)
    source = c1.selectbox("Graph to use", [k for k in ("truth", "ttpm", "pc", "lift") if k in graphs],
                          format_func=GRAPH_LABEL.get, key="tri_graph")
    window = c2.slider("Time gap from parent to child (seconds)", 10, 600, 60, step=10)
    if ds.topology is not None:
        hops = c3.slider("Topology hops", 0, 2, hops_default)
    else:
        hops = 0
        c3.markdown("This dataset has no topology: same device only.")

    width_min = st.slider("Burst window width (minutes)", 2, 30, 10)
    width_s = width_min * 60

    @st.cache_data(show_spinner=False)
    def storm_table(name: str, width: int) -> pd.DataFrame:
        a_ = get_dataset(name).alarms
        b = ((a_["start"] - a_["start"].min()) // width).value_counts().head(8)
        out = pd.DataFrame({"start": b.index * width + a_["start"].min(), "alarms": b.values})
        return out.sort_values("alarms", ascending=False).reset_index(drop=True)

    storms = storm_table(ds_name, width_s)
    pick_i = st.selectbox("Alarm burst", range(len(storms)),
                          format_func=lambda i: f"{clock(storms.loc[i, 'start'])}, {storms.loc[i, 'alarms']} alarms")
    t0 = int(storms.loc[pick_i, "start"])

    @st.cache_data(show_spinner=False)
    def explain_window(name: str, which_graph: str, t_start: int, width: int, w: int, h: int):
        d = get_dataset(name)
        g = get_graphs(name)[which_graph]
        # include `w` seconds of history so parents just before the window can explain alarms inside it
        sl = d.alarms[(d.alarms["start"] >= t_start - w) & (d.alarms["start"] < t_start + width)]
        ex_ = explain_alarms(sl, g, d.topology, d.n_devices, window=w, hops=h)
        keep = ex_["start"] >= t_start
        draws = [1 - explain_alarms(sl, random_dag(d.n_types, int((g != 0).sum()), 11 + s), d.topology, d.n_devices,
                                    window=w, hops=h).loc[keep, "is_root"].mean() for s in range(5)]
        return ex_, keep.to_numpy(), float(np.mean(draws))

    ex, in_window, rnd_frac = explain_window(ds_name, source, t0, width_s, window, hops)
    cur = ex[in_window]
    n_all, n_root = len(cur), int(cur["is_root"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alarms in window", n_all)
    m2.metric("Root alarms", n_root)
    m3.metric("Alarms explained", pct(1 - n_root / max(n_all, 1)))
    m4.metric("Random graph, same size (mean of 5)", pct(rnd_frac),
              help="Chance level: random DAGs with the same number of edges are used to explain the same alarms. "
                   "The mean of 5 draws is shown because a single draw varies a lot.")
    st.caption("Be careful reading these numbers: explained does not mean correct. In a single burst a random graph "
               "can also seem to explain many alarms, and a correct graph can explain few. Compare several graphs "
               "on the same burst, and see Methods and limitations.")

    if n_all > 1500:
        st.warning("The window contains many alarms. The chart only shows the first 1,500.")
    plot = cur.head(1500).copy()
    rank = topo_rank(graphs[source])
    lane_pos = {t: i for i, t in enumerate(np.argsort(rank))}
    plot["x"] = plot["start"] - t0
    plot["y"] = plot["alarm_id"].map(lane_pos)
    plot["Role"] = np.where(plot["is_root"], "Root alarm", "Explained")
    plot["Type"] = "A" + plot["alarm_id"].astype(str)
    label_expr = "{" + ",".join(f"{v}:'A{k}'" for k, v in lane_pos.items()) + "}[datum.value]"
    y_axis = alt.Axis(tickCount=ds.n_types, labelExpr=label_expr, title="alarm type (causes on top)",
                      grid=True, labelOverlap=False, labelFontSize=10)
    y_enc = alt.Y("y:Q", axis=y_axis, scale=alt.Scale(domain=[-0.5, ds.n_types - 0.5], reverse=True))
    layers = []
    kids = plot[~plot["is_root"] & plot["parent_row"].isin(ex.index[in_window])].copy()
    if len(kids) and len(kids) <= 700:
        par = ex.loc[kids["parent_row"].to_numpy()]
        seg = pd.DataFrame({"x": par["start"].to_numpy() - t0, "y": par["alarm_id"].map(lane_pos).to_numpy(),
                            "x2": kids["x"].to_numpy(), "y2": kids["y"].to_numpy()})
        layers.append(alt.Chart(seg).mark_rule(color="#D64933", opacity=0.25, strokeWidth=1)
                      .encode(x="x:Q", x2="x2:Q", y=alt.Y("y:Q", scale=alt.Scale(reverse=True)), y2="y2:Q"))
    layers.append(
        alt.Chart(plot).mark_point(filled=True)
        .encode(x=alt.X("x:Q", title="seconds since window start"), y=y_enc,
                size=alt.Size("Role:N", scale=alt.Scale(domain=["Root alarm", "Explained"], range=[70, 22]), legend=None),
                color=alt.Color("Role:N", scale=alt.Scale(domain=["Root alarm", "Explained"], range=[INK, "#B8C0CC"]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Type", alt.Tooltip("device_id:Q", title="Device"), "Role",
                         alt.Tooltip("lag:Q", title="Gap from parent (seconds)")])
    )
    st.altair_chart(alt.layer(*layers).properties(height=max(320, 18 * ds.n_types), width="container"), width="stretch")
    st.caption("Red lines connect a parent alarm to the alarms it explains.")

    st.subheader("Top root alarms")
    rk = root_ranking(cur)
    if len(rk):
        top = rk.head(15).copy()
        top["Type"] = "A" + top["alarm_id"].astype(str)
        top["Time"] = top["start"].map(clock)
        st.dataframe(top[["Type", "device_id", "Time", "n_explained"]].rename(
            columns={"device_id": "Device", "n_explained": "Alarms explained"}),
            hide_index=True, width="stretch")
    full = triage[(triage["dataset"] == ds_name)].set_index("graph")
    st.info(
        f"Over the whole log ({SHORT[ds_name]}, {int(full['window_s'].iloc[0])}-second gap), the expert graph explains "
        f"{pct(full.loc['Expert', 'explained_frac'])} of alarms and TTPM {pct(full.loc['TTPM', 'explained_frac'])}. "
        f"A random graph with as many edges as TTPM explains only {pct(full.loc['TTPM', 'explained_frac_random'])}. "
        "These percentages are not accuracy: the dataset has no cause labels for individual events, so this triage "
        "demonstrates how to use a graph and does not claim correctness."
    )

# ----------------------------------------------------------------------------- 5. methods and limitations
with tab_notes:
    st.subheader("Three ways to find the graph")
    st.markdown(
        "**Lift baseline.** For every pair of alarm types (a, b), count how often b appears within 60 seconds after a "
        "on the same device or a neighbor, then divide by the chance level. Pairs with lift of at least 3, confidence "
        "of at least 0.05 and support of at least 20 events become edges. Because alarms are rare, lift easily reaches "
        "the thousands; confidence and support are the thresholds that actually filter.\n\n"
        "**PC (gCastle).** A constraint-based algorithm built on independence tests. The log becomes a 0/1 table: one "
        "row per device per 10-minute window, one column per alarm type. PC sees neither the order of events inside a "
        "window nor the topology.\n\n"
        "**TTPM (gCastle).** A topological Hawkes-process model: every alarm raises the rate of other alarms on the "
        "same device and its neighbors, with an influence that decays over time. The graph is searched by hill "
        "climbing on the BIC score. The search is sped up with a per-alarm-type score cache; the result is identical "
        "to the original implementation (tested in tests/)."
    )
    st.subheader("How sensitive are the results?")
    ttpm_s = sens[(sens["dataset"] == ds_name) & (sens["method"] == "TTPM")][
        ["params", "delta", "f1", "precision", "recall"]]
    line = (
        alt.Chart(ttpm_s).mark_line(point=True, color=METHOD_COLOR["TTPM"])
        .encode(x=alt.X("delta:Q", scale=alt.Scale(type="log"), title="delta (decay per second)"),
                y=alt.Y("f1:Q", title="F1", scale=alt.Scale(zero=False)),
                tooltip=["params", "f1", "precision", "recall"])
        .properties(height=220, width="container")
    )
    st.altair_chart(line, width="stretch")
    pc_s = sens[(sens["dataset"] == ds_name) & (sens["method"] == "PC")].sort_values("f1", ascending=False)
    lift_s = sens[(sens["dataset"] == ds_name) & (sens["method"] == "Lift")].sort_values("f1", ascending=False)
    st.markdown(
        f"Across a 100-fold range of delta, TTPM's F1 ranges from {fmt(ttpm_s['f1'].min())} to {fmt(ttpm_s['f1'].max())}. "
        f"The best PC configuration out of 8 variants (chosen using the answer key, so over-optimistic) reaches "
        f"{fmt(pc_s['f1'].iloc[0])}; the best lift baseline out of 9 variants reaches {fmt(lift_s['f1'].iloc[0])}."
    )
    st.subheader("Limitations")
    st.markdown(
        "- The answer key comes from expert annotation at the alarm-type level. It can be incomplete, so precision may "
        "look lower than it really is.\n"
        "- TTPM (the gCastle implementation) only counts influence from alarms recorded in an earlier second, so "
        "relations recorded in the same second are invisible to it. Example: in Microwave-24, 98% of A7 alarms are "
        "recorded on the same device and in the same second as an A8 alarm (the expert graph has A8 causing A7), and "
        "TTPM does not find that edge.\n"
        "- Only three datasets, and one data realization per dataset. Small differences between methods should not "
        "be read as definite.\n"
        "- The main configuration uses defaults without tuning. Huawei's reference scores for the same datasets use "
        "a protocol that is not fully known, so they are context only.\n"
        "- The data come from networks of unknown origin supplied by Huawei; results may not transfer to other networks.\n"
        "- Triage only demonstrates how to use a graph; without per-event cause labels its accuracy cannot be measured."
    )

st.divider()
st.caption("Author: Raihan Muzhaffar Athallah. Data: Huawei Noah's Ark Lab via gCastle (Apache-2.0). "
           "This project is not affiliated with Huawei.")
