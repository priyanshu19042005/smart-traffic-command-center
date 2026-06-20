"""
app.py — Smart Traffic Command Center (Streamlit)
=================================================
Enterprise dashboard with 8 pages:

    1. Executive Overview        5. Forecasting Center
    2. Live Incident Analytics   6. Resource Allocation
    3. Road Health Monitoring    7. ML Predictions
    4. Traffic Hotspots          8. Data Quality Monitoring

Run::

    streamlit run dashboard/app.py

The app reads artifacts produced by ``python -m src.run_pipeline``. If an
artifact is missing it shows a friendly prompt instead of crashing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- make 'src' importable when run via `streamlit run` --------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.theme import (inject_css, kpi_card, register_plotly_template,
                             CATEGORY_COLORS, ACCENT, GOOD, WARN, BAD)
from src.utils import load_config, get_path

st.set_page_config(page_title="Smart Traffic Command Center",
                   page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")
inject_css()
TEMPLATE = register_plotly_template()
CFG = load_config()
OUT = get_path("outputs_dir", cfg=CFG)
MAP_CENTER = dict(lat=12.9716, lon=77.5946)


# ===========================================================================
# Cached loaders
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    p = get_path("features", cfg=CFG)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_geojson(name: str) -> dict:
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def download_btn(df: pd.DataFrame, label: str, fname: str) -> None:
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv", width="stretch")


def need_pipeline(msg: str = "Artifact not found.") -> None:
    st.warning(f"⚠️ {msg}\n\nRun the pipeline first:\n\n```\npython -m src.run_pipeline\n```")


# ===========================================================================
# Sidebar — branding + global filters
# ===========================================================================
events = load_events()

st.sidebar.markdown("## 🛰️ TRAFFIC COMMAND")
st.sidebar.caption(f"{CFG.project.city} · Smart City Operations")

PAGES = [
    "① Executive Overview", "② Live Incident Analytics", "③ Road Health Monitoring",
    "④ Traffic Hotspots", "⑤ Forecasting Center", "⑥ Resource Allocation",
    "⑦ ML Predictions", "⑧ Data Quality Monitoring",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()


def global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters shared by analytics pages."""
    if df.empty:
        return df
    st.sidebar.markdown("### 🔎 Filters")
    dmin, dmax = df["date"].min(), df["date"].max()
    dr = st.sidebar.date_input("Date range", (dmin, dmax),
                               min_value=dmin, max_value=dmax)
    corr = st.sidebar.multiselect("Corridor", sorted(df["corridor"].dropna().unique()))
    zone = st.sidebar.multiselect("Zone", sorted(df["zone"].dropna().unique()))
    cause = st.sidebar.multiselect("Cause", sorted(df["event_cause"].dropna().unique()))
    prio = st.sidebar.multiselect("Priority", sorted(df["priority"].dropna().unique()))

    out = df.copy()
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        out = out[(out["date"] >= pd.Timestamp(dr[0])) & (out["date"] <= pd.Timestamp(dr[1]))]
    if corr: out = out[out["corridor"].isin(corr)]
    if zone: out = out[out["zone"].isin(zone)]
    if cause: out = out[out["event_cause"].isin(cause)]
    if prio: out = out[out["priority"].isin(prio)]
    st.sidebar.caption(f"**{len(out):,}** of {len(df):,} events selected")
    return out


# ===========================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ===========================================================================
def page_overview(df: pd.DataFrame) -> None:
    st.markdown("<div class='cc-title'>Executive Command Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='cc-sub'>City-wide situational awareness across incidents, "
                "road health and resources</div>", unsafe_allow_html=True)
    if df.empty:
        return need_pipeline("No processed events found.")

    fdf = global_filters(df)
    active = (fdf["status"] == "active").sum()
    closure = fdf.get("requires_road_closure", pd.Series(dtype=bool)).sum()
    rhi_zone = load_csv("road_health_zone.csv")
    mean_rhi = rhi_zone["health_score"].mean() if not rhi_zone.empty else float("nan")

    c = st.columns(5)
    cards = [
        ("Total Incidents", f"{len(fdf):,}", None, ACCENT),
        ("Active Now", f"{active:,}", "live", WARN),
        ("Road Closures", f"{int(closure):,}", None, BAD),
        ("Avg Risk Score", f"{fdf['risk_score'].mean():.0f}", "/100", ACCENT),
        ("City Road Health", f"{mean_rhi:.0f}" if mean_rhi == mean_rhi else "—", "/100",
         GOOD if mean_rhi >= 60 else WARN),
    ]
    for col, (l, v, d, clr) in zip(c, cards):
        col.markdown(kpi_card(l, v, d, clr), unsafe_allow_html=True)

    st.markdown("### ")
    left, right = st.columns([2, 1])
    with left:
        daily = fdf.groupby("date").size().reset_index(name="incidents")
        fig = px.area(daily, x="date", y="incidents", template=TEMPLATE,
                      title="Daily Incident Volume")
        fig.update_traces(line_color=ACCENT, fillcolor="rgba(0,229,255,0.12)")
        st.plotly_chart(fig, width="stretch")
    with right:
        top = fdf["event_cause"].value_counts().head(7).reset_index()
        top.columns = ["cause", "count"]
        fig = px.bar(top, x="count", y="cause", orientation="h", template=TEMPLATE,
                     title="Top Incident Causes", color="count",
                     color_continuous_scale="Tealgrn")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        zc = fdf.groupby("zone").size().reset_index(name="incidents")
        zc = zc[zc["zone"] != "unknown"].sort_values("incidents", ascending=False)
        fig = px.bar(zc, x="zone", y="incidents", template=TEMPLATE,
                     title="Incidents by Zone", color="incidents",
                     color_continuous_scale="Tealgrn")
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")
    with b:
        heat = (fdf.assign(dow=fdf["start_datetime"].dt.day_name())
                .pivot_table(index="dow", columns="hour", values="id", aggfunc="count")
                .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                          "Friday", "Saturday", "Sunday"]))
        fig = px.imshow(heat, template=TEMPLATE, title="When Incidents Happen (Day × Hour)",
                        color_continuous_scale="Turbo", aspect="auto")
        st.plotly_chart(fig, width="stretch")
    download_btn(fdf, "⬇️ Download filtered events (CSV)", "incidents_filtered.csv")


# ===========================================================================
# PAGE 2 — LIVE INCIDENT ANALYTICS
# ===========================================================================
def page_live(df: pd.DataFrame) -> None:
    st.markdown("<div class='cc-title'>Live Incident Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='cc-sub'>Drill into the incident stream by cause, vehicle, "
                "time and location</div>", unsafe_allow_html=True)
    if df.empty:
        return need_pipeline()
    fdf = global_filters(df)

    c = st.columns(4)
    c[0].markdown(kpi_card("Selected", f"{len(fdf):,}"), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Unique Corridors", fdf["corridor"].nunique()), unsafe_allow_html=True)
    c[2].markdown(kpi_card("High Priority", f"{(fdf['priority']=='High').mean()*100:.0f}%",
                           color=BAD), unsafe_allow_html=True)
    rush = fdf["is_rush_hour"].mean() * 100 if "is_rush_hour" in fdf else 0
    c[3].markdown(kpi_card("In Rush Hour", f"{rush:.0f}%", color=WARN), unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🗺️ Incident Map", "📊 Breakdowns", "🧾 Records"])
    with t1:
        sample = fdf.sample(min(3000, len(fdf)), random_state=1)
        fig = px.scatter_mapbox(
            sample, lat="latitude", lon="longitude", color="event_cause",
            hover_data=["corridor", "zone", "priority", "risk_score"],
            zoom=10, height=560, center=MAP_CENTER)
        fig.update_layout(mapbox_style="carto-darkmatter", template=TEMPLATE,
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch")
    with t2:
        a, b = st.columns(2)
        with a:
            vt = fdf[fdf["veh_type"] != "not_applicable"]["veh_type"].value_counts().head(8)
            fig = px.pie(values=vt.values, names=vt.index, hole=0.55, template=TEMPLATE,
                         title="Vehicle Type Mix")
            st.plotly_chart(fig, width="stretch")
        with b:
            seg = fdf.groupby("day_segment").size().reindex(
                ["morning", "afternoon", "evening", "night"]).reset_index(name="n")
            fig = px.bar(seg, x="day_segment", y="n", template=TEMPLATE,
                         title="Incidents by Day Segment", color="n",
                         color_continuous_scale="Tealgrn")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        hr = fdf.groupby("hour").size().reset_index(name="n")
        fig = px.line(hr, x="hour", y="n", markers=True, template=TEMPLATE,
                      title="Hourly Incident Profile")
        fig.update_traces(line_color=ACCENT)
        st.plotly_chart(fig, width="stretch")
    with t3:
        cols = ["id", "start_datetime", "event_cause", "corridor", "zone",
                "priority", "status", "risk_score", "veh_type"]
        st.dataframe(fdf[[c for c in cols if c in fdf]].sort_values("start_datetime",
                     ascending=False).head(500), width="stretch", height=460)
        download_btn(fdf, "⬇️ Download records", "incident_records.csv")


# ===========================================================================
# PAGE 3 — ROAD HEALTH MONITORING
# ===========================================================================
def page_road_health(df: pd.DataFrame) -> None:
    st.markdown("<div class='cc-title'>Road Health Monitoring</div>", unsafe_allow_html=True)
    st.markdown("<div class='cc-sub'>Road Health Index (0–100) per corridor & zone with "
                "explainable degradation drivers</div>", unsafe_allow_html=True)
    level = st.radio("Level", ["corridor", "zone"], horizontal=True)
    rhi = load_csv(f"road_health_{level}.csv")
    if rhi.empty:
        return need_pipeline("Road health not computed.")

    order = ["Excellent", "Good", "Moderate", "Poor", "Critical"]
    cnt = rhi["health_category"].value_counts().reindex(order).fillna(0).astype(int)
    cols = st.columns(5)
    for col, cat in zip(cols, order):
        col.markdown(kpi_card(cat, cnt[cat], "segments", CATEGORY_COLORS[cat]),
                     unsafe_allow_html=True)

    a, b = st.columns([3, 2])
    with a:
        s = rhi.sort_values("health_score")
        fig = px.bar(s, x="health_score", y="segment", orientation="h", template=TEMPLATE,
                     color="health_category", color_discrete_map=CATEGORY_COLORS,
                     title=f"Road Health Score by {level.title()}",
                     hover_data=["top_factor", "total_events"])
        fig.update_layout(height=max(400, 22 * len(s)))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.pie(values=cnt.values, names=cnt.index, hole=0.5, template=TEMPLATE,
                     title="Health Category Distribution",
                     color=cnt.index, color_discrete_map=CATEGORY_COLORS)
        st.plotly_chart(fig, width="stretch")
        st.markdown("##### 🔧 Worst segments")
        st.dataframe(rhi.nsmallest(6, "health_score")[
            ["segment", "health_score", "health_category", "top_factor"]],
            width="stretch", hide_index=True)

    st.markdown("##### Degradation driver breakdown")
    pts_cols = [c for c in rhi.columns if c.startswith("points_lost_")]
    if pts_cols:
        melt = rhi.melt(id_vars="segment", value_vars=pts_cols,
                        var_name="cause", value_name="points_lost")
        melt["cause"] = melt["cause"].str.replace("points_lost_", "")
        fig = px.bar(melt, x="segment", y="points_lost", color="cause", template=TEMPLATE,
                     title="Health Points Lost per Cause (stacked)")
        fig.update_layout(xaxis_tickangle=-35, height=440)
        st.plotly_chart(fig, width="stretch")
    download_btn(rhi, "⬇️ Download Road Health report", f"road_health_{level}.csv")


# ===========================================================================
# PAGE 4 — TRAFFIC HOTSPOTS
# ===========================================================================
def page_hotspots(df: pd.DataFrame) -> None:
    st.markdown("<div class='cc-title'>Traffic Hotspots</div>", unsafe_allow_html=True)
    st.markdown("<div class='cc-sub'>DBSCAN density clusters by cause with risk scoring "
                "and GIS export</div>", unsafe_allow_html=True)
    causes = ["all"] + list(CFG.hotspots.causes)
    cause = st.selectbox("Hotspot layer", causes)
    hs = load_csv(f"hotspots_{cause}.csv")
    if hs.empty:
        return need_pipeline("Hotspots not detected.")

    c = st.columns(4)
    c[0].markdown(kpi_card("Hotspots", len(hs)), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Incidents Clustered", int(hs["incident_count"].sum())),
                  unsafe_allow_html=True)
    c[2].markdown(kpi_card("Top Risk Score", f"{hs['risk_score'].max():.0f}", "/100", BAD),
                  unsafe_allow_html=True)
    c[3].markdown(kpi_card("Largest Cluster", int(hs["incident_count"].max())),
                  unsafe_allow_html=True)

    fig = px.scatter_mapbox(
        hs, lat="center_lat", lon="center_lon", size="incident_count",
        color="risk_score", color_continuous_scale="Inferno",
        size_max=40, zoom=10, height=560, center=MAP_CENTER,
        hover_data=["hotspot_id", "dominant_cause", "incident_count", "risk_score"])
    fig.update_layout(mapbox_style="carto-darkmatter", template=TEMPLATE)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns([3, 2])
    with a:
        st.dataframe(hs[["hotspot_id", "center_lat", "center_lon", "incident_count",
                         "dominant_cause", "risk_score", "top_zone"]].head(20),
                     width="stretch", hide_index=True)
    with b:
        top = hs.nlargest(10, "risk_score")
        fig = px.bar(top, x="risk_score", y="hotspot_id", orientation="h", template=TEMPLATE,
                     title="Highest-Risk Hotspots", color="risk_score",
                     color_continuous_scale="Inferno")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    gj = load_geojson(f"hotspots_{cause}.geojson")
    if gj:
        st.download_button("⬇️ Download GeoJSON (GIS-ready)",
                           json.dumps(gj, indent=2).encode("utf-8"),
                           file_name=f"hotspots_{cause}.geojson",
                           mime="application/geo+json", width="stretch")


# ===========================================================================
# PAGE 5 — FORECASTING CENTER
# ===========================================================================
def page_forecast(df: pd.DataFrame) -> None:
    st.markdown("<div class='cc-title'>Forecasting Center</div>", unsafe_allow_html=True)
    st.markdown("<div class='cc-sub'>Next day / week / month incident forecasts with model "
                "back-test leaderboard</div>", unsafe_allow_html=True)
    fc = load_csv("forecasts.csv")
    summary = load_csv("forecast_summary.csv")
    metrics = load_csv("forecast_metrics.csv")
    if fc.empty:
        return need_pipeline("Forecasts not generated.")
    fc["date"] = pd.to_datetime(fc["date"])

    scope = st.selectbox("Scope", fc["scope"].unique())
    sub = fc[fc["scope"] == scope]
    srow = summary[summary["scope"] == scope]

    c = st.columns(4)
    if not srow.empty:
        c[0].markdown(kpi_card("Next Day", f"{srow['day'].iloc[0]:.0f}", "incidents", ACCENT),
                      unsafe_allow_html=True)
        c[1].markdown(kpi_card("Next Week", f"{srow['week'].iloc[0]:.0f}", "incidents", ACCENT),
                      unsafe_allow_html=True)
        c[2].markdown(kpi_card("Next Month", f"{srow['month'].iloc[0]:.0f}", "incidents", ACCENT),
                      unsafe_allow_html=True)
        c[3].markdown(kpi_card("Best Model", srow["best_model"].iloc[0].upper(), color=GOOD),
                      unsafe_allow_html=True)

    hist = df[df["corridor"] == scope] if scope not in ("CITY",) and "corridor" in df else df
    daily = hist.groupby("date").size().reset_index(name="y") if not hist.empty else pd.DataFrame()
    fig = go.Figure()
    if not daily.empty:
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["y"], name="History",
                                 line=dict(color="#5b6b8c")))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["yhat_upper"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["yhat_lower"], fill="tonexty",
                             fillcolor="rgba(0,229,255,0.15)", line=dict(width=0),
                             name="80% interval"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["yhat"], name="Forecast",
                             line=dict(color=ACCENT, width=3)))
    fig.update_layout(template=TEMPLATE, height=460,
                      title=f"Incident Forecast — {scope}")
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("##### 🏁 Model back-test leaderboard")
        ms = metrics[metrics["scope"] == scope] if not metrics.empty else metrics
        st.dataframe(ms[["model", "MAE", "RMSE", "MAPE"]].round(2) if not ms.empty else ms,
                     width="stretch", hide_index=True)
    with b:
        st.markdown("##### 📅 Forecast table")
        st.dataframe(sub[["date", "yhat", "yhat_lower", "yhat_upper"]].round(1),
                     width="stretch", hide_index=True, height=260)
    download_btn(fc, "⬇️ Download all forecasts", "forecasts.csv")


# ===========================================================================
# PAGE 6 — RESOURCE ALLOCATION
# ===========================================================================
def page_resources(df: pd.DataFrame) -> None:
    st.markdown("<div class='cc-title'>Resource Allocation</div>", unsafe_allow_html=True)
    st.markdown("<div class='cc-sub'>Recommended deployment of tow trucks, traffic police "
                "and maintenance teams by zone</div>", unsafe_allow_html=True)
    alloc = load_csv("resource_allocation.csv")
    if alloc.empty:
        return need_pipeline("Resource allocation not computed.")

    fleet = CFG.resources.fleet.to_dict()
    c = st.columns(4)
    c[0].markdown(kpi_card("Zones", len(alloc)), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Tow Trucks", int(alloc["tow_trucks"].sum()),
                           f"/{fleet['tow_trucks']} fleet"), unsafe_allow_html=True)
    c[2].markdown(kpi_card("Traffic Police", int(alloc["traffic_police"].sum()),
                           f"/{fleet['traffic_police']} fleet"), unsafe_allow_html=True)
    c[3].markdown(kpi_card("Maintenance Teams", int(alloc["maintenance_teams"].sum()),
                           f"/{fleet['maintenance_teams']} fleet"), unsafe_allow_html=True)

    melt = alloc.melt(id_vars="zone",
                      value_vars=["tow_trucks", "traffic_police", "maintenance_teams"],
                      var_name="resource", value_name="units")
    fig = px.bar(melt, x="zone", y="units", color="resource", barmode="group",
                 template=TEMPLATE, title="Recommended Units per Zone")
    fig.update_layout(xaxis_tickangle=-30, height=440)
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### 🧭 Deployment recommendations")
    show = alloc[["zone", "priority_rank", "tow_trucks", "traffic_police",
                  "maintenance_teams", "rationale"]].sort_values("priority_rank")
    st.dataframe(show, width="stretch", hide_index=True)

    st.markdown("##### Highest-priority zone")
    top = alloc.sort_values("priority_rank").iloc[0]
    st.success(f"**{top['zone']}** → 🚛 Tow Trucks: **{int(top['tow_trucks'])}**, "
               f"👮 Traffic Police: **{int(top['traffic_police'])}**, "
               f"🛠️ Maintenance Teams: **{int(top['maintenance_teams'])}**")
    download_btn(alloc, "⬇️ Download allocation plan", "resource_allocation.csv")


# ===========================================================================
# PAGE 7 — ML PREDICTIONS
# ===========================================================================
def page_ml(df: pd.DataFrame) -> None:
    st.markdown("<div class='cc-title'>ML Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='cc-sub'>Priority, Road-Closure & Risk models — live scoring "
                "and evaluation diagnostics</div>", unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Live Scoring", "📈 Model Performance"])
    with tabs[0]:
        st.markdown("##### Score a hypothetical incident")
        if df.empty:
            return need_pipeline()
        col = st.columns(4)
        cause = col[0].selectbox("Cause", sorted(df["event_cause"].unique()))
        corr = col[1].selectbox("Corridor", sorted(df["corridor"].unique()))
        veh = col[2].selectbox("Vehicle", sorted(df["veh_type"].unique()))
        hour = col[3].slider("Hour", 0, 23, 9)
        if st.button("🚀 Predict", width="stretch"):
            try:
                from src.models.predict import Predictor
                row = df.iloc[[0]].copy()
                row["event_cause"] = cause; row["corridor"] = corr
                row["veh_type"] = veh; row["hour"] = hour
                row["is_rush_hour"] = int(hour in {8, 9, 10, 18, 19, 20})
                scored = Predictor(CFG).predict_frame(row)
                r = scored.iloc[0]
                m = st.columns(3)
                if "priority_label" in scored:
                    m[0].markdown(kpi_card("Priority", r["priority_label"].upper(),
                                  f"p={r['priority_proba']:.2f}",
                                  BAD if r["priority_label"] == "High" else GOOD),
                                  unsafe_allow_html=True)
                if "closure_label" in scored:
                    m[1].markdown(kpi_card("Road Closure",
                                  "REQUIRED" if r["closure_pred"] == 1 else "No",
                                  f"p={r['closure_proba']:.2f}",
                                  BAD if r["closure_pred"] == 1 else GOOD),
                                  unsafe_allow_html=True)
                if "risk_pred" in scored:
                    m[2].markdown(kpi_card("Risk Score", f"{r['risk_pred']:.0f}", "/100",
                                  WARN), unsafe_allow_html=True)
            except FileNotFoundError:
                need_pipeline("Models not trained. Run `python -m src.models.train`.")

    with tabs[1]:
        for task in CFG.models.tasks:
            rep = load_json(f"eval_{task}.json")
            if not rep:
                st.info(f"No evaluation for **{task}** yet.")
                continue
            st.markdown(f"#### `{task}` — {rep.get('type')}")
            mt = rep.get("metrics", {})
            cols = st.columns(len(mt))
            for cc, (k, v) in zip(cols, mt.items()):
                cc.markdown(kpi_card(k.upper(), f"{v:.3f}"), unsafe_allow_html=True)
            if rep.get("type") == "classification" and "roc_curve" in rep:
                a, b = st.columns(2)
                with a:
                    roc = rep["roc_curve"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name="ROC",
                                             line=dict(color=ACCENT, width=3)))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash",
                                  color="#5b6b8c"), name="chance"))
                    fig.update_layout(template=TEMPLATE, title="ROC Curve",
                                      xaxis_title="FPR", yaxis_title="TPR", height=360)
                    st.plotly_chart(fig, width="stretch")
                with b:
                    cm = np.array(rep["confusion_matrix"])
                    fig = px.imshow(cm, text_auto=True, template=TEMPLATE,
                                    color_continuous_scale="Tealgrn",
                                    labels=dict(x="Predicted", y="Actual"),
                                    title="Confusion Matrix")
                    st.plotly_chart(fig, width="stretch")
            elif rep.get("type") == "regression" and "scatter" in rep:
                sc = rep["scatter"]
                fig = px.scatter(x=sc["actual"], y=sc["predicted"], template=TEMPLATE,
                                 labels={"x": "Actual", "y": "Predicted"},
                                 title="Predicted vs Actual", opacity=0.5)
                lo, hi = min(sc["actual"]), max(sc["actual"])
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(dash="dash", color=ACCENT), name="ideal"))
                st.plotly_chart(fig, width="stretch")
            st.divider()


# ===========================================================================
# PAGE 8 — DATA QUALITY MONITORING
# ===========================================================================
def page_quality(df: pd.DataFrame) -> None:
    st.markdown("<div class='cc-title'>Data Quality Monitoring</div>", unsafe_allow_html=True)
    st.markdown("<div class='cc-sub'>Validation gates, completeness and freshness of the "
                "ingest pipeline</div>", unsafe_allow_html=True)
    rep = load_json("validation_report.json")
    if not rep:
        return need_pipeline("Validation report not found.")

    checks = pd.DataFrame(rep.get("checks", []))
    n_err = rep.get("n_errors", 0)
    c = st.columns(4)
    c[0].markdown(kpi_card("Overall", "PASS" if rep.get("passed") else "FAIL",
                           color=GOOD if rep.get("passed") else BAD), unsafe_allow_html=True)
    c[1].markdown(kpi_card("Checks Run", len(checks)), unsafe_allow_html=True)
    c[2].markdown(kpi_card("Errors", n_err, color=BAD if n_err else GOOD), unsafe_allow_html=True)
    warn = int((checks["severity"] == "WARN").sum()) if not checks.empty else 0
    c[3].markdown(kpi_card("Warnings", warn, color=WARN), unsafe_allow_html=True)

    if not checks.empty:
        sev_color = {"OK": GOOD, "WARN": WARN, "ERROR": BAD}
        checks["icon"] = checks["severity"].map({"OK": "✅", "WARN": "⚠️", "ERROR": "❌"})
        st.markdown("##### Validation checks")
        st.dataframe(checks[["icon", "name", "severity", "message"]],
                     width="stretch", hide_index=True, height=340)

    if not df.empty:
        st.markdown("##### Column completeness (processed features)")
        comp = (1 - df.isna().mean()).sort_values().reset_index()
        comp.columns = ["column", "completeness"]
        fig = px.bar(comp.tail(25), x="completeness", y="column", orientation="h",
                     template=TEMPLATE, color="completeness",
                     color_continuous_scale="Tealgrn", title="Completeness by Column")
        fig.update_layout(height=560, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    download_btn(checks, "⬇️ Download validation report", "validation_checks.csv")


# ===========================================================================
# Router
# ===========================================================================
ROUTES = {
    PAGES[0]: page_overview, PAGES[1]: page_live, PAGES[2]: page_road_health,
    PAGES[3]: page_hotspots, PAGES[4]: page_forecast, PAGES[5]: page_resources,
    PAGES[6]: page_ml, PAGES[7]: page_quality,
}
ROUTES[page](events)

st.sidebar.divider()
st.sidebar.caption("🛰️ Smart Traffic Command Center · v1.0\nBuilt on the Astram event dataset")
