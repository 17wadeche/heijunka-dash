# heijunka-dash.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
DEFAULT_DATA_PATH = Path(r"C:\Users\wadec8\OneDrive - Medtronic PLC\metrics_aggregate.xlsx")
DATA_URL = st.secrets.get("HEIJUNKA_DATA_URL", os.environ.get("HEIJUNKA_DATA_URL"))
st.set_page_config(page_title="Heijunka Metrics", layout="wide")
if hasattr(st, "autorefresh"):
    st.autorefresh(interval=720 * 60 * 1000, key="auto-refresh")
@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_data(data_path: str | None, data_url: str | None):
    if data_url:
        if data_url.lower().endswith(".json"):
            df = pd.read_json(data_url)
        else:
            df = pd.read_csv(data_url)
        return _postprocess(df)
    if not data_path:
        return pd.DataFrame()
    p = Path(data_path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        df = pd.read_excel(p, sheet_name="All Metrics")
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".json":
        df = pd.read_json(p)
    else:
        return pd.DataFrame()
    return _postprocess(df)
def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "period_date" in df.columns:
        df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce").dt.normalize()
    for col in ["Total Available Hours", "Completed Hours", "Target Output", "Actual Output",
                "Target UPLH", "Actual UPLH"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"Actual Output", "Target Output"}.issubset(df.columns):
        df["Efficiency vs Target"] = (df["Actual Output"] / df["Target Output"]).replace([np.inf, -np.inf], np.nan)
    if {"Completed Hours", "Total Available Hours"}.issubset(df.columns):
        df["Capacity Utilization"] = (df["Completed Hours"] / df["Total Available Hours"]).replace([np.inf, -np.inf], np.nan)
    return df
data_path = None if DATA_URL else str(DEFAULT_DATA_PATH)
mtime_key = 0
if data_path:
    p = Path(data_path)
    mtime_key = p.stat().st_mtime if p.exists() else 0
df = load_data(data_path, DATA_URL)
st.title("Heijunka Metrics Dashboard")
if df.empty:
    st.warning("No data found yet. Make sure metrics_aggregate.xlsx exists and has the 'All Metrics' sheet.")
    st.stop()
teams = sorted([t for t in df["team"].dropna().unique()])
default_teams = teams
col1, col2, col3 = st.columns([2,2,6], gap="large")
with col1:
    selected_teams = st.multiselect("Teams", teams, default=default_teams)
with col2:
    min_date = pd.to_datetime(df["period_date"].min()).date() if df["period_date"].notna().any() else None
    max_date = pd.to_datetime(df["period_date"].max()).date() if df["period_date"].notna().any() else None
    if min_date and max_date:
        start, end = st.date_input("Date range", (min_date, max_date))
    else:
        start, end = None, None
f = df.copy()
if selected_teams:
    f = f[f["team"].isin(selected_teams)]
if start and end:
    f = f[(f["period_date"] >= pd.to_datetime(start)) & (f["period_date"] <= pd.to_datetime(end))]

if f.empty:
    st.info("No rows match your filters.")
    st.stop()
latest = (f.sort_values(["team", "period_date"])
            .groupby("team", as_index=False)
            .tail(1))
if len(selected_teams) == 2:
    teamA, teamB = selected_teams[0], selected_teams[1]
    latest_2 = (f.sort_values(["team", "period_date"])
                  .groupby("team", as_index=False)
                  .tail(1)
                  .set_index("team"))
    end_dt = pd.to_datetime(f["period_date"].max())
    start_dt = end_dt - pd.Timedelta(weeks=4)
    window = f[(f["period_date"] > start_dt) & (f["period_date"] <= end_dt)]
    avg4 = (window.groupby("team", as_index=False)[
        ["Target Output","Actual Output","Total Available Hours","Completed Hours",
         "Target UPLH","Actual UPLH","Efficiency vs Target","Capacity Utilization"]
    ].mean(numeric_only=True).set_index("team"))
    def val(df, t, col):
        return df.loc[t, col] if (t in df.index and col in df.columns) else np.nan
    kpis = []
    metrics = [
        ("Actual Output", "{:,.0f}"),
        ("Target Output", "{:,.0f}"),
        ("Efficiency vs Target", "{:.2f}x"),
        ("Actual UPLH", "{:.2f}"),
        ("Target UPLH", "{:.2f}"),
        ("Capacity Utilization", "{:.0%}"),
    ]
    for label, fmt in metrics:
        latest_A = val(latest_2, teamA, label)
        latest_B = val(latest_2, teamB, label)
        avg_A = val(avg4, teamA, label)
        avg_B = val(avg4, teamB, label)
        kpis.append({
            "Metric": label,
            "Latest – " + teamA: latest_A, "Latest – " + teamB: latest_B,
            "Δ Latest (A–B)": (latest_A - latest_B) if pd.notna(latest_A) and pd.notna(latest_B) else np.nan,
            "4-wk avg – " + teamA: avg_A, "4-wk avg – " + teamB: avg_B,
            "Δ 4-wk (A–B)": (avg_A - avg_B) if pd.notna(avg_A) and pd.notna(avg_B) else np.nan,
            "_fmt": fmt
        })
    kpi_df = pd.DataFrame(kpis)
    st.markdown("### Head-to-Head")
    def _fmt(x, fmt):
        return "—" if pd.isna(x) else fmt.format(x)
    show = kpi_df.drop(columns=["_fmt"]).copy()
    for i, row in kpi_df.iterrows():
        fmt = row["_fmt"]
        for c in show.columns:
            if c != "Metric":
                show.loc[i, c] = _fmt(row[c], fmt)
    st.dataframe(show, use_container_width=True)
    latest_pairs = []
    rowA = latest_2.loc[teamA] if teamA in latest_2.index else None
    rowB = latest_2.loc[teamB] if teamB in latest_2.index else None
    if rowA is not None and rowB is not None:
        def add_pair(name, a, b):
            latest_pairs.extend([
                {"Metric": name, "Team": teamA, "Value": a},
                {"Metric": name, "Team": teamB, "Value": b},
            ])
        add_pair("Actual Output", rowA.get("Actual Output", np.nan), rowB.get("Actual Output", np.nan))
        add_pair("Target Output", rowA.get("Target Output", np.nan), rowB.get("Target Output", np.nan))
        add_pair("Efficiency vs Target", rowA.get("Efficiency vs Target", np.nan), rowB.get("Efficiency vs Target", np.nan))
        add_pair("Actual UPLH", rowA.get("Actual UPLH", np.nan), rowB.get("Actual UPLH", np.nan))
        add_pair("Capacity Utilization", rowA.get("Capacity Utilization", np.nan), rowB.get("Capacity Utilization", np.nan))
        latest_bar_df = pd.DataFrame(latest_pairs).dropna(subset=["Value"])
        if not latest_bar_df.empty:
            latest_bars = (
                alt.Chart(latest_bar_df)
                .mark_bar()
                .encode(
                    x=alt.X("Team:N", title=None),
                    y=alt.Y("Value:Q", title=None),
                    color="Team:N",
                    column=alt.Column("Metric:N", title="Latest Week", header=alt.Header(labelLimit=100))
                )
                .properties(height=220)
            )
            st.altair_chart(latest_bars, use_container_width=True)
    st.markdown("#### Gap over time (A – B)")
    gap_metric = st.selectbox(
        "Metric for gap line",
        ["Actual Output", "Target Output", "Actual UPLH", "Target UPLH", "Efficiency vs Target", "Capacity Utilization"],
        index=2,
        key="gap_metric"
    )
    piv = f[f["team"].isin([teamA, teamB])][["period_date","team",gap_metric]] \
            .dropna().pivot_table(index="period_date", columns="team", values=gap_metric, aggfunc="mean")
    if teamA in piv.columns and teamB in piv.columns:
        piv = piv.sort_index()
        piv = piv.assign(Gap=lambda d: d[teamA] - d[teamB]).reset_index()
        gap_line = (
            alt.Chart(piv)
            .mark_line(point=True)
            .encode(
                x=alt.X("period_date:T", title="Week"),
                y=alt.Y("Gap:Q", title=f"{gap_metric}: {teamA} – {teamB}"),
                tooltip=[alt.Tooltip("period_date:T", title="Week"), alt.Tooltip("Gap:Q", format=",.2f")]
            )
            .properties(height=240)
        )
        zero_line = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(strokeDash=[4,3]).encode(y="y:Q")
        st.altair_chart(gap_line + zero_line, use_container_width=True)
kpi_cols = st.columns(4)
def kpi(col, label, value, fmt="{:,.2f}"):
    if pd.isna(value):
        col.metric(label, "—")
    else:
        try:
            col.metric(label, fmt.format(value))
        except Exception:
            col.metric(label, str(value))
tot_target = latest["Target Output"].sum(skipna=True)
tot_actual = latest["Actual Output"].sum(skipna=True)
tot_tahl  = latest["Total Available Hours"].sum(skipna=True)
tot_chl   = latest["Completed Hours"].sum(skipna=True)
with kpi_cols[0]:
    st.subheader("Latest (All Selected Teams)")
kpi(kpi_cols[1], "Target Output", tot_target, "{:,.0f}")
kpi(kpi_cols[2], "Actual Output", tot_actual, "{:,.0f}")
kpi(kpi_cols[3], "Actual vs Target", (tot_actual/tot_target if tot_target else np.nan), "{:.2f}x")
kpi_cols2 = st.columns(3)
kpi(kpi_cols2[0], "Target UPLH", (tot_target/tot_tahl if tot_tahl else np.nan), "{:.2f}")
kpi(kpi_cols2[1], "Actual UPLH", (tot_actual/tot_chl if tot_chl else np.nan), "{:.2f}")
kpi(kpi_cols2[2], "Capacity Utilization", (tot_chl/tot_tahl if tot_tahl else np.nan), "{:.0%}")
st.markdown("---")
left, right = st.columns(2)
base = alt.Chart(f).transform_calculate(
    week="toDate(datum.period_date)"
).encode(
    x=alt.X("period_date:T", title="Week")
)
with left:
    st.subheader("Output Trend")
    out_long = f.melt(
        id_vars=["team", "period_date"],
        value_vars=["Target Output", "Actual Output"],
        var_name="Metric", value_name="Value"
    ).dropna(subset=["Value"])
    if len(selected_teams) == 2:
        out_chart = (
            alt.Chart(out_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("period_date:T", title="Week"),
                y=alt.Y("Value:Q", title="Output"),
                color=alt.Color("team:N", title="Team"),
                strokeDash=alt.StrokeDash("Metric:N", title="Series"),
                tooltip=["team:N", "period_date:T", "Metric:N", alt.Tooltip("Value:Q", format=",.0f")]
            )
            .properties(height=280)
        )
    else:
        out_chart = (
            alt.Chart(out_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("period_date:T", title="Week"),
                y=alt.Y("Value:Q", title="Output"),
                color=alt.Color("Metric:N", title="Series"),
                detail="team:N",
                tooltip=["team:N", "period_date:T", "Metric:N", alt.Tooltip("Value:Q", format=",.0f")]
            )
            .properties(height=280)
        )
    st.altair_chart(out_chart, use_container_width=True)
with right:
    st.subheader("UPLH Trend")
    have_target_uplh = "Target UPLH" in f.columns
    uplh_vars = ["Actual UPLH"] + (["Target UPLH"] if have_target_uplh else [])
    uplh_long = f.melt(
        id_vars=["team", "period_date"],
        value_vars=uplh_vars,
        var_name="Metric", value_name="Value"
    ).dropna(subset=["Value"])
    if len(selected_teams) == 2:
        uplh_chart = (
            alt.Chart(uplh_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("period_date:T", title="Week"),
                y=alt.Y("Value:Q", title="UPLH"),
                color=alt.Color("team:N", title="Team"),
                strokeDash=alt.StrokeDash("Metric:N", title="Series"),
                tooltip=["team:N", "period_date:T", "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
            )
            .properties(height=280)
        )
    else:
        uplh_chart = (
            alt.Chart(uplh_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("period_date:T", title="Week"),
                y=alt.Y("Value:Q", title="UPLH"),
                color=alt.Color("Metric:N", title="Series"),
                detail="team:N",
                tooltip=["team:N", "period_date:T", "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
            )
            .properties(height=280)
        )
    st.altair_chart(uplh_chart, use_container_width=True)
st.subheader("Efficiency vs Target (Actual / Target)")
eff = f.assign(Efficiency=lambda d: (d["Actual Output"] / d["Target Output"]))
eff = eff.replace([np.inf, -np.inf], np.nan).dropna(subset=["Efficiency"])
if len(selected_teams) == 2:
    eff_bar = (
        alt.Chart(eff)
        .mark_bar()
        .encode(
            x=alt.X("period_date:T", title="Week"),
            y=alt.Y("Efficiency:Q", title="x of Target"),
            color=alt.condition("datum.Efficiency >= 1", alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=["team:N","period_date:T",
                     alt.Tooltip("Actual Output:Q", format=",.0f"),
                     alt.Tooltip("Target Output:Q", format=",.0f"),
                     alt.Tooltip("Efficiency:Q", format=".2f")]
        )
        .facet(column=alt.Column("team:N", title=None))
        .resolve_scale(y="shared")  # same y across teams
        .properties(columns=2)
    )
    ref = alt.Chart(pd.DataFrame({"y":[1.0]})).mark_rule(strokeDash=[4,3]).encode(y="y:Q")
    st.altair_chart(eff_bar & ref, use_container_width=True)
else:
    color_scale = alt.Scale(domain=[0,1], range=["#d62728", "#2ca02c"])
    eff_bar = (
        alt.Chart(eff)
        .mark_bar()
        .encode(
            x=alt.X("period_date:T", title="Week"),
            y=alt.Y("Efficiency:Q", title="x of Target"),
            color=alt.condition("datum.Efficiency >= 1", alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=["team:N","period_date:T",
                     alt.Tooltip("Actual Output:Q", format=",.0f"),
                     alt.Tooltip("Target Output:Q", format=",.0f"),
                     alt.Tooltip("Efficiency:Q", format=".2f")]
        )
    )
    ref_line = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[4,3]).encode(y="y:Q")
    st.altair_chart((eff_bar + ref_line).properties(height=260), use_container_width=True)
st.markdown("---")
st.subheader("Detailed Rows")
hide_cols = {"source_file", "fallback_used", "error"}
drop_these = [c for c in f.columns if c in hide_cols or c.startswith("Unnamed:")]
f_table = f.drop(columns=drop_these, errors="ignore").sort_values(["team", "period_date"])
st.dataframe(f_table, use_container_width=True)
