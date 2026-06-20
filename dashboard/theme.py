"""
theme.py
========
Shared styling for the Smart City Command Center dashboard: injectable CSS,
a KPI-card renderer and the Plotly template used by every chart.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Command-center palette.
ACCENT = "#00e5ff"
ACCENT2 = "#7c4dff"
GOOD = "#00e676"
WARN = "#ffb300"
BAD = "#ff5252"
BG = "#0a0e1a"
PANEL = "#121a2e"

CATEGORY_COLORS = {
    "Excellent": "#00e676", "Good": "#64dd17", "Moderate": "#ffb300",
    "Poor": "#ff7043", "Critical": "#ff1744",
}


def inject_css() -> None:
    """Global CSS for the command-center look."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: radial-gradient(1200px 600px at 20% -10%, #16213e 0%, {BG} 55%); }}
        section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid #1e2a44; }}
        h1, h2, h3 {{ color: #e6edf3; letter-spacing: .3px; }}
        .cc-title {{
            font-size: 1.9rem; font-weight: 800;
            background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .cc-sub {{ color: #8aa0c0; margin-top:-8px; font-size:.9rem; }}
        .kpi {{
            background: linear-gradient(160deg, #16213e, #0d1426);
            border: 1px solid #243352; border-radius: 16px; padding: 16px 18px;
            box-shadow: 0 6px 24px rgba(0,0,0,.35);
        }}
        .kpi .label {{ color:#8aa0c0; font-size:.78rem; text-transform:uppercase; letter-spacing:1px; }}
        .kpi .value {{ color:#fff; font-size:1.8rem; font-weight:800; margin-top:2px; }}
        .kpi .delta {{ font-size:.8rem; margin-top:2px; }}
        .pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem;
                 font-weight:700; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{ background:{PANEL}; border-radius:10px 10px 0 0; padding:6px 14px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value, delta: str | None = None, color: str = ACCENT) -> str:
    """Return HTML for a KPI card (use inside ``st.markdown(..., unsafe_allow_html=True)``)."""
    d = f"<div class='delta' style='color:{color}'>{delta}</div>" if delta else ""
    return (f"<div class='kpi'><div class='label'>{label}</div>"
            f"<div class='value' style='color:{color}'>{value}</div>{d}</div>")


def register_plotly_template() -> str:
    """Register and return a dark command-center Plotly template name."""
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#cdd9ec", family="Inter, sans-serif"),
        colorway=[ACCENT, ACCENT2, GOOD, WARN, BAD, "#40c4ff", "#ff80ab"],
        xaxis=dict(gridcolor="#1e2a44", zerolinecolor="#1e2a44"),
        yaxis=dict(gridcolor="#1e2a44", zerolinecolor="#1e2a44"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    pio.templates["command_center"] = tmpl
    return "command_center"
