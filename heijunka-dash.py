# heijunka_dash_v2.py
# A modernized, insight-forward Streamlit app for your Heijunka metrics
# - Overview/Teams/Anomalies/Data tabs
# - Smarter KPIs with deltas vs. prior period
# - Interactive Altair charts with hover/legend filtering & team focus
# - Small-multiple heatmap + per-team sparklines in the table
# - Cleaner embed mode for SharePoint/Teams via ?embed=true

from __future__ import annotations
import os
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Heijunka Metrics",
    layout="wide",
)

# Hide chrome in embed mode (works even outside Streamlit Cloud)
params = st.query_params
if str(params.get("embed", "")).lower() == "true":
    st.markdown(
        """
        <style>
        header, footer, .stDeployButton {display: none !important;}
        .block-container {padding-top: 0.75rem; padding-bottom: 0.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Altair defaults (crisp tooltips, nicer fonts)
alt.themes.enable("opaque")

DEFAULT_DATA_PATH = Path(r"C:\\Users\\wadec8\\OneDrive - Medtronic PLC\\metrics_aggregate.xlsx")
DATA_URL = st.secrets.get("HEIJUNKA_DATA_URL", os.environ.get("HEIJUNKA_DATA_URL"))

# -----------------------------
# Data loading & prep
# -----------------------------
@st.cache_data(show_spinner=True, ttl=15 * 60)
def load_data(data_path: str | None, data_url: str | None) -> pd.DataFrame:
    def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # normalize dates
        if "period_date" in df.columns:
            df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce").dt.normalize()
        # numeric coercions
        for col in [
            "Total Available Hours",
            "Completed Hours",
            "Target Output",
            "Actual Output",
            "Target UPLH",
            "Actual UPLH",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # engineered features
        if {"Actual Output", "Target Output"}.issubset(df.columns):
            df["Efficiency vs Target"] = (
                df["Actual Output"] / df["Target Output"]
            ).replace([np.inf, -np.inf], np.nan)
        if {"Completed Hours", "Total Available Hours"}.issubset(df.columns):
            df["Capacity Utilization"] = (
                df["Completed Hours"] / df["Total Available Hours"]
            ).replace([np.inf, -np.inf], np.nan)
        return df

    # prefer URL if provided
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

# memoize once; auto-refresh on a timer if supported (Streamlit Cloud)
if hasattr(st, "autorefresh"):
    st.autorefresh(interval=720 * 60 * 1000, key="auto-refresh")

data_path = None if DATA_URL else str(DEFAULT_DATA_PATH)
df = load_data(data_path, DATA_URL)

st.title("Heijunka Metrics Dashboard")

if df.empty:
    st.warning("No data found yet. Ensure the source exists and has an 'All Metrics' sheet.")
    st.stop()

# -----------------------------
# Filters
# -----------------------------
with st.container():
    c1, c2, c3, c4 = st.columns([2, 2, 2, 6], gap="large")
    teams = sorted([t for t in df.get("team", pd.Series(dtype=str)).dropna().unique()])
    default_teams = teams
    selected_teams = c1.multiselect("Teams", teams, default=default_teams)

    # date range
    if df["period_date"].notna().any():
        min_date = pd.to_datetime(df["period_date"].min()).date()
        max_date = pd.to_datetime(df["period_date"].max()).date()
        start, end = c2.date_input("From/To", (min_date, max_date))
    else:
        start, end = None, None

    # team focus for charts
    focus_team = c3.selectbox("Focus team (for highlights)", ["All"] + teams, index=0)

    # quick actions
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    c4.download_button("⬇️ Download full CSV", csv_buf.getvalue(), file_name="heijunka_all.csv")

# apply filters
f = df.copy()
if selected_teams:
    f = f[f["team"].isin(selected_teams)]
if start and end:
    f = f[(f["period_date"] >= pd.to_datetime(start)) & (f["period_date"] <= pd.to_datetime(end))]

if f.empty:
    st.info("No rows match your filters.")
    st.stop()

# -----------------------------
# Helper calcs
# -----------------------------
# latest and previous per team for delta
f_sorted = f.sort_values(["team", "period_date"]).copy()
latest_by_team = f_sorted.groupby("team", as_index=False).tail(1)
prev_by_team = (
    f_sorted.groupby("team", as_index=False).nth(-2).reset_index(drop=True)
    if f_sorted.groupby("team").size().min() >= 2
    else pd.DataFrame(columns=f_sorted.columns)
)

# totals (latest snapshot across selected teams)
agg_cols = [
    "Target Output",
    "Actual Output",
    "Total Available Hours",
    "Completed Hours",
]
latest_totals = latest_by_team[agg_cols].sum(numeric_only=True)

# deltas vs previous snapshot
if not prev_by_team.empty:
    prev_totals = prev_by_team[agg_cols].sum(numeric_only=True)
else:
    prev_totals = pd.Series({c: np.nan for c in agg_cols})

# KPI values
tot_target = latest_totals.get("Target Output", np.nan)
tot_actual = latest_totals.get("Actual Output", np.nan)
tot_tahl = latest_totals.get("Total Available Hours", np.nan)
tot_chl = latest_totals.get("Completed Hours", np.nan)

# Deltas
delta_actual = (
    (tot_actual - prev_totals.get("Actual Output", np.nan)) if not np.isnan(prev_totals.get("Actual Output", np.nan)) else None
)

def fmt_delta(x: float | None, fmt: str = "+,.0f"):
    if x is None or pd.isna(x):
        return None
    try:
        return format(x, fmt)
    except Exception:
        return str(x)

# -----------------------------
# Tabs
# -----------------------------
ov, teams_tab, anomalies, data_tab = st.tabs([
    "Overview",
    "Teams",
    "Anomalies",
    "Data",
])

# -----------------------------
# Overview
# -----------------------------
with ov:
    k1, k2, k3, k4 = st.columns(4)

    def metric_safe(col, label, value, fmt="{:,.0f}", delta: float | None = None, delta_fmt="+,.0f"):
        if pd.isna(value):
            col.metric(label, "—")
        else:
            d = fmt_delta(delta, delta_fmt) if delta is not None else None
            try:
                col.metric(label, fmt.format(value), delta=d)
            except Exception:
                col.metric(label, str(value), delta=d)

    metric_safe(k1, "Actual Output", tot_actual, "{:,.0f}", delta_actual)
    metric_safe(k2, "Target Output", tot_target, "{:,.0f}",
                delta=(tot_target - prev_totals.get("Target Output", np.nan)) if not np.isnan(prev_totals.get("Target Output", np.nan)) else None)
    metric_safe(k3, "Actual vs Target", (tot_actual / tot_target if tot_target else np.nan), "{:.2f}×")
    metric_safe(k4, "Capacity Utilization", (tot_chl / tot_tahl if tot_tahl else np.nan), "{:.0%}")

    st.markdown("---")

    cl, cr = st.columns(2)

    # Output Trend (interactive)
    out_long = f.melt(
        id_vars=["team", "period_date"],
        value_vars=["Target Output", "Actual Output"],
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"]) 

    sel = alt.selection_point(fields=["team"], bind="legend")
# Build a valid Altair predicate (avoid mixing Python booleans with Altair expressions)
if focus_team == "All":
    cond = sel
else:
    cond = sel | (alt.datum.team == focus_team)

highlight = alt.condition(cond, alt.value(1), alt.value(0.2))

    out_chart = (
        alt.Chart(out_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("period_date:T", title="Week"),
            y=alt.Y("Value:Q", title="Output"),
            color=alt.Color("Metric:N", title="Series"),
            detail="team:N",
            opacity=highlight,
            tooltip=["team:N", "period_date:T", "Metric:N", alt.Tooltip("Value:Q", format=",.0f")],
        )
        .add_params(sel)
        .properties(height=300)
    )

    cl.subheader("Output Trend")
    cl.altair_chart(out_chart, use_container_width=True)

    # UPLH Trend
    uplh_vars = [v for v in ["Actual UPLH", "Target UPLH"] if v in f.columns]
    uplh_long = f.melt(
        id_vars=["team", "period_date"],
        value_vars=uplh_vars,
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"]) 

    uplh_chart = (
        alt.Chart(uplh_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("period_date:T", title="Week"),
            y=alt.Y("Value:Q", title="UPLH"),
            color=alt.Color("Metric:N", title="Series"),
            detail="team:N",
            opacity=highlight,
            tooltip=["team:N", "period_date:T", "Metric:N", alt.Tooltip("Value:Q", format=",.2f")],
        )
        .add_params(sel)
        .properties(height=300)
    )

    cr.subheader("UPLH Trend")
    cr.altair_chart(uplh_chart, use_container_width=True)

    st.markdown("---")

    # Heatmap of efficiency by team/week (small-multiples feel)
    if {"Efficiency vs Target", "team"}.issubset(f.columns):
        st.subheader("Efficiency vs Target — Heatmap")
        heat = (
            alt.Chart(f.dropna(subset=["Efficiency vs Target"]))
            .mark_rect()
            .encode(
                x=alt.X("period_date:T", title="Week"),
                y=alt.Y("team:N", title="Team"),
                color=alt.Color("Efficiency vs Target:Q", title="× of Target", scale=alt.Scale(scheme="greenblue")),
                tooltip=[
                    "team:N",
                    "period_date:T",
                    alt.Tooltip("Actual Output:Q", format=",.0f"),
                    alt.Tooltip("Target Output:Q", format=",.0f"),
                    alt.Tooltip("Efficiency vs Target:Q", format=".2f"),
                ],
            )
            .properties(height=max(200, 20 * max(1, len(selected_teams) or len(teams))))
        )
        st.altair_chart(heat, use_container_width=True)

# -----------------------------
# Teams tab (per-team mini dashboards)
# -----------------------------
with teams_tab:
    st.caption("Hover a sparkline to see values; use the search box to filter the table.")

    # Build per-team latest metrics + 8-week sparklines
    last_n = 8
    roll = (
        f.sort_values(["team", "period_date"]).groupby("team").tail(last_n)
    )

    # sparkline series for Actual Output
    spark = (
        roll.groupby(["team"])  # last N for sparkline
        .apply(lambda d: d.set_index("period_date")["Actual Output"].tolist())
        .rename("Actual Output (last N)")
        .reset_index()
    )

    latest_subset = latest_by_team[[
        "team",
        "Actual Output",
        "Target Output",
        "Efficiency vs Target",
        "Actual UPLH" if "Actual UPLH" in latest_by_team.columns else latest_by_team.columns[0],
    ]].copy()

    # merge
    ttable = latest_subset.merge(spark, on="team", how="left")

    # display with column config (sparkline)
    st.dataframe(
        ttable.sort_values("Efficiency vs Target", ascending=False),
        use_container_width=True,
        column_config={
            "Actual Output (last N)": st.column_config.LineChartColumn(
                "Actual Output — trend",
                width="medium",
                help=f"Last {last_n} points",
                y_min=0,
            ),
            "Actual Output": st.column_config.NumberColumn(format=",.0f"),
            "Target Output": st.column_config.NumberColumn(format=",.0f"),
            "Efficiency vs Target": st.column_config.NumberColumn(format=".2f×"),
            "Actual UPLH": st.column_config.NumberColumn(format=",.2f") if "Actual UPLH" in ttable.columns else None,
        },
        hide_index=True,
    )

# -----------------------------
# Simple anomaly surface (z-score on efficiency)
# -----------------------------
with anomalies:
    if "Efficiency vs Target" in f.columns and f["Efficiency vs Target"].notna().sum() > 5:
        g = f.dropna(subset=["Efficiency vs Target"]).copy()
        g["z_eff"] = (
            (g["Efficiency vs Target"] - g["Efficiency vs Target"].mean()) / g["Efficiency vs Target"].std(ddof=0)
        )
        alert_thresh = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.5)
        flagged = g.loc[g["z_eff"].abs() >= alert_thresh]
        st.write(f"Flagged rows: {len(flagged)}")
        if not flagged.empty:
            chart = (
                alt.Chart(g)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("period_date:T", title="Week"),
                    y=alt.Y("Efficiency vs Target:Q", title="× of Target"),
                    color=alt.condition(alt.datum.z_eff.abs() >= alert_thresh, alt.value("crimson"), alt.value("steelblue")),
                    tooltip=["team:N", "period_date:T", alt.Tooltip("Efficiency vs Target:Q", format=".2f"), alt.Tooltip("z_eff:Q", format=".2f")],
                )
                .properties(height=340)
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(flagged.sort_values(["period_date", "team"]).drop(columns=["z_eff"]), use_container_width=True)
        else:
            st.info("No anomalies at the current threshold.")
    else:
        st.info("Not enough data to compute anomalies yet.")

# -----------------------------
# Data tab
# -----------------------------
with data_tab:
    st.caption("All filtered rows. Use the filter controls above to narrow the set.")
    hide_cols = {"source_file", "fallback_used", "error"}
    drop_these = [c for c in f.columns if c in hide_cols or c.startswith("Unnamed:")]
    f_table = f.drop(columns=drop_these, errors="ignore").sort_values(["team", "period_date"])
    st.dataframe(
        f_table,
        use_container_width=True,
        column_config={
            "Target Output": st.column_config.NumberColumn(format=",.0f"),
            "Actual Output": st.column_config.NumberColumn(format=",.0f"),
            "Total Available Hours": st.column_config.NumberColumn(format=",.0f"),
            "Completed Hours": st.column_config.NumberColumn(format=",.0f"),
            "Actual UPLH": st.column_config.NumberColumn(format=",.2f") if "Actual UPLH" in f_table.columns else None,
            "Target UPLH": st.column_config.NumberColumn(format=",.2f") if "Target UPLH" in f_table.columns else None,
            "Efficiency vs Target": st.column_config.NumberColumn(format=".2f×") if "Efficiency vs Target" in f_table.columns else None,
            "Capacity Utilization": st.column_config.NumberColumn(format=".0%") if "Capacity Utilization" in f_table.columns else None,
        },
    )

# -----------------------------
# End
# -----------------------------
