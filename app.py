"""
╔══════════════════════════════════════════════════════════════════════╗
║       PRECICOST INTELLIGENCE                                         ║
║       Manufacturing Variance & Cost Analytics Dashboard              ║
║       AI Recommendations · AI Chatbot · Dark Theme                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import cohere
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 🔑 API KEY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PreciCost Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# DARK THEME CSS (upgraded from Manufacturing BI System)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
div[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px; }

.kpi-card {
    background: linear-gradient(145deg,#1c2128 0%,#161b22 100%);
    border: 1px solid #30363d; border-radius: 14px;
    padding: 22px 18px; text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.5);
    transition: transform .2s, box-shadow .2s;
    height: 130px; display: flex; flex-direction: column; justify-content: center;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 32px rgba(0,0,0,0.6); }
.kpi-card.red   { border-left: 4px solid #f85149; }
.kpi-card.green { border-left: 4px solid #3fb950; }
.kpi-card.yellow{ border-left: 4px solid #d29922; }
.kpi-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #f0f6fc; line-height: 1.1; }
.kpi-sub   { font-size: 12px; margin-top: 6px; }
.kpi-sub.good    { color: #3fb950; }
.kpi-sub.bad     { color: #f85149; }
.kpi-sub.neutral { color: #d29922; }

.alert-red    { background:#2d1117; border-left:4px solid #f85149; border-radius:8px; padding:12px 16px; margin:5px 0; color:#ff7b72; font-size:13px; }
.alert-yellow { background:#2d2208; border-left:4px solid #d29922; border-radius:8px; padding:12px 16px; margin:5px 0; color:#e3b341; font-size:13px; }
.alert-green  { background:#0d2a1a; border-left:4px solid #3fb950; border-radius:8px; padding:12px 16px; margin:5px 0; color:#56d364; font-size:13px; }
.alert-blue   { background:#0d1b2e; border-left:4px solid #388bfd; border-radius:8px; padding:12px 16px; margin:5px 0; color:#79c0ff; font-size:13px; }

.ai-rec-container {
    background: linear-gradient(135deg, #0d1b2e 0%, #111827 100%);
    border: 1px solid #1d4ed8;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 10px 0;
    position: relative;
    overflow: hidden;
}
.ai-rec-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #388bfd, #a78bfa, #3fb950);
}
.ai-rec-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; font-weight: 600; color: #79c0ff;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 12px;
}
.ai-rec-content {
    color: #c9d1d9; font-size: 13px; line-height: 1.7;
}

.section-title {
    font-size: 16px; font-weight: 700; color: #f0f6fc;
    border-bottom: 2px solid #21262d; padding-bottom: 8px;
    margin: 18px 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;
}

.page-banner {
    background: linear-gradient(90deg,#1f2d3d 0%,#1c2128 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 18px 24px; margin-bottom: 20px;
}
.page-banner h1 { font-size: 22px; font-weight: 700; color: #f0f6fc; margin: 0; }
.page-banner p  { font-size: 13px; color: #8b949e; margin: 4px 0 0 0; }

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9", family="Segoe UI"),
    xaxis=dict(gridcolor="#21262d", showgrid=True, linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", showgrid=True, linecolor="#30363d"),
    margin=dict(l=30, r=20, t=50, b=30),
    legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1)
)

COLORS = ["#388bfd","#3fb950","#d29922","#f85149","#a5d6ff","#56d364",
          "#e3b341","#ff7b72","#79c0ff","#7ee787","#ffa657","#cae8ff"]

LABOR_RATE = 25  # ₹25/hr


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def fmt_currency(v):
    if v >= 10_000_000: return f"₹{v/10_000_000:.2f}Cr"
    if v >= 100_000:    return f"₹{v/100_000:.2f}L"
    if v >= 1_000:      return f"₹{v/1_000:.1f}K"
    return f"₹{v:,.0f}"


def kpi_card(label, value, sub="", sub_class="neutral", border_class=""):
    cls = f"kpi-card {border_class}".strip()
    return f"""<div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub {sub_class}">{sub}</div>
    </div>"""


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def alert(msg, kind="blue"):
    st.markdown(f'<div class="alert-{kind}">{msg}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING (PreciCost original structure)
# ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    ac = pd.read_csv(Actual_Costs.csv")
    bm = pd.read_csv(Budget_Master.csv")
    dt = pd.read_csv(Date.csv")
    mc = pd.read_csv(Machines.csv")
    pl = pd.read_csv(Production_Logs.csv")
    rr = pd.read_csv(Rework_Registry.csv")
    sh = pd.read_csv(Shifts.csv")

    dt["date"] = pd.to_datetime(dt["date"])
    dt["date_key"] = dt["date_key"].astype(int)
    for df in [ac, pl, rr]:
        df["date_key"] = df["date_key"].astype(int)

    return ac, bm, dt, mc, pl, rr, sh

ac, bm, dt, mc, pl, rr, sh = load_data()


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS (PreciCost original)
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px 0;'>
        <div style='font-size:36px;'>🏭</div>
        <div style='font-size:15px;font-weight:700;color:#f0f6fc;'>PreciCost Intelligence</div>
        <div style='font-size:11px;color:#8b949e;'>Manufacturing Variance Analytics</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    page = st.selectbox("📌 Navigation", [
        "📊 Executive Variance",
        "♻️ Waste & Quality",
        "🗺️ Stage Heatmap",
        "🤖 AI Chatbot"
    ])

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#8b949e;text-transform:uppercase;'
                'letter-spacing:1px;margin-bottom:8px;">🔑 API KEY</div>', unsafe_allow_html=True)
    _api_input = st.text_input(
        "Cohere API Key",
        value=st.session_state.get("cohere_api_key", COHERE_API_KEY),
        type="password",
        placeholder="Paste your Cohere API key…",
        label_visibility="collapsed",
        key="cohere_api_key"
    )
    if not _api_input:
        st.warning("⚠️ Enter a Cohere API key to enable AI features.")

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#8b949e;text-transform:uppercase;'
                'letter-spacing:1px;margin-bottom:8px;">🎛️ FILTERS</div>', unsafe_allow_html=True)

    years = sorted(dt["year"].unique())
    sel_years = st.multiselect("📅 Year", years, default=years)

    quarters = sorted(dt["quarter"].unique())
    sel_quarters = st.multiselect("📆 Quarter", quarters, default=quarters)

    lines = sorted(mc["production_line"].unique())
    sel_lines = st.multiselect("🏗️ Production Line", lines, default=lines)

    stages = sorted(bm["stage_name"].unique())
    sel_stages = st.multiselect("🔧 Stage", stages, default=stages)

    shifts_list = sorted(sh["shift_name"].unique())
    sel_shifts = st.multiselect("🕐 Shift", shifts_list, default=shifts_list)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:11px;color:#6e7681;padding:6px 0;line-height:1.8;'>
        📁 7-CSV Star Schema<br>
        🔗 PreciCost v2.0<br>
        🤖 AI-Powered Analytics
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# APPLY FILTERS (PreciCost original filter logic)
# ─────────────────────────────────────────────────────────────────────
dt_f = dt[dt["year"].isin(sel_years) & dt["quarter"].isin(sel_quarters)]
valid_keys = set(dt_f["date_key"])

mc_f = mc[mc["production_line"].isin(sel_lines)]
sh_f = sh[sh["shift_name"].isin(sel_shifts)]
bm_f = bm[bm["stage_name"].isin(sel_stages)]

ac_f = ac[ac["date_key"].isin(valid_keys) & ac["stage_id"].isin(bm_f["stage_id"])]
pl_f = pl[
    pl["date_key"].isin(valid_keys)
    & pl["machine_id"].isin(mc_f["machine_id"])
    & pl["shift_id"].isin(sh_f["shift_id"])
    & pl["stage_id"].isin(bm_f["stage_id"])
]
rr_f = rr[rr["date_key"].isin(valid_keys) & rr["stage_id"].isin(bm_f["stage_id"])]


# ─────────────────────────────────────────────────────────────────────
# KPI CALCULATIONS — PRECICOST ORIGINAL (DO NOT MODIFY)
# ─────────────────────────────────────────────────────────────────────
actual_cost = (
    ac_f["material_cost"].sum()
    + ac_f["utility_cost"].sum()
    + ac_f["payroll_hours"].sum() * LABOR_RATE
)

units_produced = pl_f["units_produced"].sum()
scrap_count = pl_f["scrap_count"].sum()

budgeted_cost = sum(
    pl_f[pl_f["stage_id"] == row.stage_id]["units_produced"].sum() * row.budgeted_unit_cost
    for row in bm_f.itertuples()
)

variance_pct = (actual_cost - budgeted_cost) / budgeted_cost if budgeted_cost else 0
avg_unit_cost = actual_cost / units_produced if units_produced else 0
waste_dollars = scrap_count * avg_unit_cost
rework_leak = rr_f["parts_cost_lost"].sum() + rr_f["man_hours_lost"].sum() * LABOR_RATE
cpu = actual_cost / max(units_produced - scrap_count, 1)

eff_color = "red" if variance_pct > 0.10 else ("yellow" if variance_pct > 0 else "green")
eff_label = "🔴 Over Budget" if variance_pct > 0.10 else ("🟡 Slightly Over" if variance_pct > 0 else "🟢 Under Budget")
scrap_rate = (scrap_count / units_produced * 100) if units_produced else 0


# ─────────────────────────────────────────────────────────────────────
# AI RECOMMENDATIONS ENGINE (from Manufacturing BI System)
# ─────────────────────────────────────────────────────────────────────
def render_ai_recommendations_section(context: dict, page_type: str, section_key: str):
    """5 buttons — each shows ONE card. Translation row below active card."""
    st.markdown('<div class="section-title">🤖 GENERATE WITH AI</div>', unsafe_allow_html=True)

    active_key  = f"active_card_{section_key}"
    content_key = f"card_content_{section_key}"
    trans_key   = f"translation_{section_key}"
    lang_key    = f"lang_{section_key}"

    for k, default in [(active_key, None), (content_key, {}), (trans_key, ""), (lang_key, "Telugu")]:
        if k not in st.session_state:
            st.session_state[k] = default

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: clicked_analysis = st.button("📊 AI Analysis",      key=f"btn_a_{section_key}", use_container_width=True)
    with b2: clicked_causes   = st.button("🔍 Problem Causes",   key=f"btn_c_{section_key}", use_container_width=True)
    with b3: clicked_recs     = st.button("✅ Recommendations",  key=f"btn_r_{section_key}", use_container_width=True)
    with b4: clicked_plan     = st.button("🗓️ Efficiency Plan",  key=f"btn_p_{section_key}", use_container_width=True)
    with b5: clicked_outcomes = st.button("🎯 Expected Outcomes", key=f"btn_o_{section_key}", use_container_width=True)

    def build_data_summary():
        if page_type == "executive":
            return (
                f"Executive Dashboard Data (all monetary values in Indian Rupees ₹):\n"
                f"- Total Actual Cost: {context['total_actual']}\n"
                f"- Total Budgeted Cost: {context['total_budget']}\n"
                f"- Overall Variance: {context['variance_pct']:+.1f}%\n"
                f"- Cost Per Unit: {context['cpu']}\n"
                f"- Total Units Produced: {context['units_produced']:,}\n"
                f"- Scrap Count: {context['scrap_count']:,}\n"
                f"- Worst Variance Stage: {context['worst_stage']}"
            )
        elif page_type == "waste":
            return (
                f"Waste & Quality Tracker Data (all monetary values in Indian Rupees ₹):\n"
                f"- Total Waste Dollars (Scrap): {context['waste_dollars']}\n"
                f"- Rework Financial Leak: {context['rework_leak']}\n"
                f"- Scrap Count: {context['scrap_count']:,}\n"
                f"- Scrap Rate: {context['scrap_rate']:.1f}%\n"
                f"- Top Rework Reason: {context['top_reason']}\n"
                f"- Worst Stage for Waste: {context.get('worst_waste_stage', 'N/A')}"
            )
        elif page_type == "heatmap":
            return (
                f"Stage × Shift Heatmap Data (all monetary values in Indian Rupees ₹):\n"
                f"- Worst Stage-Shift Combination: {context['worst_combo']}\n"
                f"- Highest Variance Stage: {context['worst_stage']}\n"
                f"- Highest Variance Shift: {context['worst_shift']}\n"
                f"- Overall Variance: {context['variance_pct']:+.1f}%\n"
                f"- Total Actual Cost: {context['total_actual']}\n"
                f"- Total Budgeted Cost: {context['total_budget']}"
            )
        return str(context)

    CARD_CONFIG = {
        "analysis": {
            "spinner":      "📊 Generating AI Analysis…",
            "preamble":     (
                "You are a manufacturing analyst. "
                "Write 3-4 sentences summarizing the overall performance situation based on the data. "
                "Be specific with numbers. No bullet points — clear paragraph form only. "
                "All monetary values are in Indian Rupees (₹)."
            ),
            "title":        "📊 AI ANALYSIS",
            "header_color": "#79c0ff",
            "border_color": "#1d4ed8",
        },
        "causes": {
            "spinner":      "🔍 Identifying Problem Causes…",
            "preamble":     (
                "You are a manufacturing analyst. "
                "List 3-4 bullet points identifying root causes visible in the data. "
                "Format: **Cause title:** one sentence explanation. "
                "All monetary values are in Indian Rupees (₹)."
            ),
            "title":        "🔍 PROBLEM CAUSES",
            "header_color": "#e3b341",
            "border_color": "#d29922",
        },
        "recs": {
            "spinner":      "✅ Generating Recommendations…",
            "preamble":     (
                "You are a manufacturing analyst. "
                "List 4-5 bullet points of specific, actionable recommendations prioritized by impact. "
                "Format: **Action title:** one sentence on what to do. "
                "All monetary values are in Indian Rupees (₹)."
            ),
            "title":        "✅ RECOMMENDATIONS",
            "header_color": "#56d364",
            "border_color": "#3fb950",
        },
        "plan": {
            "spinner":      "🗓️ Building Efficiency Improvement Plan…",
            "preamble":     (
                "You are a manufacturing operations expert. "
                "Based on the recommendations already known for this data, create a detailed "
                "0-30 Day Efficiency Improvement Plan. "
                "Structure it as:\n"
                "**Week 1 (Days 1-7):** 2 immediate actions\n"
                "**Week 2 (Days 8-14):** 2 short-term actions\n"
                "**Week 3-4 (Days 15-30):** 2 medium-term actions\n"
                "Each action should have a clear owner role (e.g. Shift Supervisor, Quality Engineer) "
                "and expected impact. Be specific and practical. "
                "All monetary values are in Indian Rupees (₹)."
            ),
            "title":        "🗓️ EFFICIENCY IMPROVEMENT PLAN (0–30 DAYS)",
            "header_color": "#ffa657",
            "border_color": "#e06c00",
        },
        "outcomes": {
            "spinner":      "🎯 Projecting Expected Outcomes…",
            "preamble":     (
                "You are a manufacturing analyst. "
                "Based on the recommendations for this data, project the Expected Outcomes "
                "if those recommendations are implemented within 30 days. "
                "List 4-5 bullet points. Each bullet: **Metric name:** expected improvement with a % or ₹ estimate. "
                "Be realistic and data-driven. All monetary values are in Indian Rupees (₹)."
            ),
            "title":        "🎯 EXPECTED OUTCOMES",
            "header_color": "#c9a7ff",
            "border_color": "#7c3aed",
        },
    }

    triggered = None
    if clicked_analysis:   triggered = "analysis"
    elif clicked_causes:   triggered = "causes"
    elif clicked_recs:     triggered = "recs"
    elif clicked_plan:     triggered = "plan"
    elif clicked_outcomes: triggered = "outcomes"

    if triggered:
        st.session_state[active_key] = triggered
        st.session_state[trans_key]  = ""
        if triggered not in st.session_state[content_key]:
            cfg      = CARD_CONFIG[triggered]
            base_msg = build_data_summary()
            if triggered in ("plan", "outcomes") and "recs" in st.session_state[content_key]:
                base_msg += f"\n\nExisting Recommendations:\n{st.session_state[content_key]['recs']}"
            with st.spinner(cfg["spinner"]):
                try:
                    _key = st.session_state.get("cohere_api_key", COHERE_API_KEY)
                    if not _key:
                        st.session_state[content_key][triggered] = "❌ No API key provided. Please enter your Cohere API key in the sidebar."
                        st.stop()
                    client = cohere.Client(api_key=_key)
                    response = client.chat(
                        model="command-r-08-2024",
                        message=base_msg,
                        preamble=cfg["preamble"],
                        max_tokens=500,
                    )
                    st.session_state[content_key][triggered] = response.text
                except Exception as e:
                    st.session_state[content_key][triggered] = f"❌ Error: {str(e)}"

    active = st.session_state[active_key]
    if active:
        cfg  = CARD_CONFIG[active]
        text = st.session_state[content_key].get(active, "")
        html = text.replace("\n", "<br>")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ai-rec-container" style="min-height:180px;border-color:{cfg['border_color']};">
            <div class="ai-rec-header" style="color:{cfg['header_color']};">{cfg['title']}</div>
            <div class="ai-rec-content">{html}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🌐 TRANSLATE THIS REPORT</div>', unsafe_allow_html=True)

        lang_col, btn_col, clear_col = st.columns([2, 1, 1])
        with lang_col:
            lang_choice = st.selectbox(
                "Select Language",
                ["Telugu", "Hindi", "Tamil", "Kannada", "Marathi", "Bengali", "Gujarati",
                 "Punjabi", "Urdu", "Spanish", "French", "German", "Arabic", "Chinese", "Japanese"],
                index=["Telugu", "Hindi", "Tamil", "Kannada", "Marathi", "Bengali", "Gujarati",
                       "Punjabi", "Urdu", "Spanish", "French", "German", "Arabic", "Chinese", "Japanese"
                       ].index(st.session_state[lang_key]),
                key=f"lang_select_{section_key}",
                label_visibility="collapsed"
            )
            st.session_state[lang_key] = lang_choice

        with btn_col:
            translate_clicked = st.button(
                f"🌐 Translate to {lang_choice}",
                key=f"btn_translate_{section_key}",
                use_container_width=True
            )
        with clear_col:
            if st.button("🗑️ Clear", key=f"clear_{section_key}", use_container_width=True):
                st.session_state[active_key]  = None
                st.session_state[content_key] = {}
                st.session_state[trans_key]   = ""
                st.rerun()

        if translate_clicked:
            with st.spinner(f"🌐 Translating to {lang_choice}…"):
                try:
                    _key = st.session_state.get("cohere_api_key", COHERE_API_KEY)
                    if not _key:
                        st.session_state[trans_key] = "❌ No API key provided. Please enter your Cohere API key in the sidebar."
                        st.stop()
                    client = cohere.Client(api_key=_key)
                    trans_response = client.chat(
                        model="command-r-08-2024",
                        message=(
                            f"Translate the following manufacturing report text into {lang_choice}. "
                            f"Preserve all formatting, bullet points, and bold markers. "
                            f"Keep the ₹ symbol as-is for Indian Rupees. "
                            f"Only return the translated text, nothing else.\n\n{text}"
                        ),
                        max_tokens=600,
                    )
                    st.session_state[trans_key] = trans_response.text
                except Exception as e:
                    st.session_state[trans_key] = f"❌ Translation error: {str(e)}"

        if st.session_state[trans_key]:
            trans_html = st.session_state[trans_key].replace("\n", "<br>")
            st.markdown(f"""
            <div class="ai-rec-container" style="min-height:140px;border-color:#a78bfa;margin-top:8px;">
                <div class="ai-rec-header" style="color:#c9a7ff;">🌐 {lang_choice.upper()} TRANSLATION</div>
                <div class="ai-rec-content">{trans_html}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# GUARD: empty data
# ─────────────────────────────────────────────────────────────────────
if pl_f.empty and ac_f.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""<div class="page-banner">
    <h1>🏭 PreciCost Intelligence</h1>
    <p>Manufacturing Variance & Cost Analytics Dashboard · AI-Powered Insights</p>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 – EXECUTIVE VARIANCE
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Executive Variance":

    # ── KPI Cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Actual Cost",    fmt_currency(actual_cost),   fmt_currency(budgeted_cost) + " budget",
         "bad" if variance_pct > 0.1 else "good",  eff_color),
        (c2, "Budgeted Cost",  fmt_currency(budgeted_cost), f"{len(dt_f)} production days",
         "neutral", ""),
        (c3, "Cost Variance",  f"{variance_pct:+.1%}",      eff_label,
         "bad" if variance_pct > 0.1 else ("neutral" if variance_pct > 0 else "good"), eff_color),
        (c4, "Cost Per Unit",  fmt_currency(cpu),           "Good units only",
         "neutral", ""),
        (c5, "Total Units",    f"{units_produced:,}",       f"Scrap: {scrap_count:,}",
         "bad" if scrap_count / max(units_produced, 1) > 0.08 else "neutral", ""),
    ]
    for col, label, val, sub, sub_cls, border_cls in kpis:
        with col:
            st.markdown(kpi_card(label, val, sub, sub_cls, border_cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Stage Bar ─────────────────────────────────────────────
    col_g, col_b = st.columns([1, 2])

    with col_g:
        section("📉 VARIANCE GAUGE")
        gauge_color = "#f85149" if variance_pct > 0.10 else ("#d29922" if variance_pct > 0 else "#3fb950")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=variance_pct * 100,
            delta={"reference": 0, "valueformat": ".1f", "suffix": "%"},
            number={"suffix": "%", "font": {"color": gauge_color, "size": 32}},
            title={"text": "Overall Variance %", "font": {"color": "#c9d1d9", "size": 13}},
            gauge={
                "axis": {"range": [-15, 30], "tickcolor": "#8b949e"},
                "bar":  {"color": gauge_color},
                "bgcolor": "#161b22", "borderwidth": 1, "bordercolor": "#30363d",
                "steps": [
                    {"range": [-15, 0],  "color": "#0d2a1a"},
                    {"range": [0,  10],  "color": "#2d2208"},
                    {"range": [10, 30],  "color": "#2d1117"},
                ],
                "threshold": {"line": {"color": "#79c0ff","width": 3}, "thickness": 0.8, "value": 10}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="#0d1117", font_color="#c9d1d9",
                                height=300, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_b:
        section("📊 VARIANCE % BY STAGE")
        stage_var = []
        for _, row in bm_f.iterrows():
            sid  = row["stage_id"]
            ac_s = ac_f[ac_f["stage_id"] == sid]
            pl_s = pl_f[pl_f["stage_id"] == sid]
            a_cost = ac_s["material_cost"].sum() + ac_s["utility_cost"].sum() + ac_s["payroll_hours"].sum() * LABOR_RATE
            b_cost = pl_s["units_produced"].sum() * row["budgeted_unit_cost"]
            var    = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
            stage_var.append({"Stage": row["stage_name"], "Variance %": var,
                               "Actual": a_cost, "Budget": b_cost})
        df_sv = pd.DataFrame(stage_var).sort_values("Variance %", ascending=False)
        bar_colors = ["#f85149" if v > 10 else ("#d29922" if v > 0 else "#3fb950")
                      for v in df_sv["Variance %"]]

        fig_bar = go.Figure(go.Bar(
            x=df_sv["Stage"], y=df_sv["Variance %"],
            marker_color=bar_colors,
            text=[f"{v:+.1f}%" for v in df_sv["Variance %"]],
            textposition="outside", textfont=dict(size=11, color="#c9d1d9"),
            hovertemplate="<b>%{x}</b><br>Variance: %{y:.1f}%<extra></extra>"
        ))
        fig_bar.add_hline(y=0,  line_color="#79c0ff", line_dash="dash",  line_width=1)
        fig_bar.add_hline(y=10, line_color="#f85149", line_dash="dot",   line_width=1,
                          annotation_text="10% Threshold", annotation_font_color="#f85149")
        fig_bar.update_layout(**DARK_LAYOUT, height=300, showlegend=False,
                              title=dict(text="Stage-Level Cost Variance", font=dict(color="#c9d1d9",size=13)),
                              yaxis_title="Variance %")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Actual vs Budget Trend ────────────────────────────────────────
    section("📈 ACTUAL vs BUDGET TREND")

    ac_trend = ac_f.merge(dt_f[["date_key","date","month","quarter"]], on="date_key", how="left")
    ac_trend["actual_cost"] = ac_trend["material_cost"] + ac_trend["utility_cost"] + ac_trend["payroll_hours"] * LABOR_RATE
    monthly_actual = ac_trend.groupby("month")["actual_cost"].sum().reset_index()

    pl_trend = pl_f.merge(dt_f[["date_key","month"]], on="date_key", how="left")
    pl_bm    = pl_trend.merge(bm[["stage_id","budgeted_unit_cost"]], on="stage_id", how="left")
    pl_bm["budget_cost"] = pl_bm["units_produced"] * pl_bm["budgeted_unit_cost"]
    monthly_budget = pl_bm.groupby("month")["budget_cost"].sum().reset_index()

    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    monthly = monthly_actual.merge(monthly_budget, on="month", how="outer").fillna(0)
    monthly["month_num"] = monthly["month"].apply(lambda m: month_order.index(m) if m in month_order else 99)
    monthly = monthly.sort_values("month_num")

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["budget_cost"],
        name="Budgeted Cost", fill="tozeroy",
        fillcolor="rgba(56,139,253,0.12)", line=dict(color="#388bfd", width=2),
        hovertemplate="Month: %{x}<br>Budget: ₹%{y:,.0f}<extra></extra>"
    ))
    fig_area.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["actual_cost"],
        name="Actual Cost", fill="tozeroy",
        fillcolor="rgba(248,81,73,0.12)", line=dict(color="#f85149", width=2),
        hovertemplate="Month: %{x}<br>Actual: ₹%{y:,.0f}<extra></extra>"
    ))
    fig_area.update_layout(**DARK_LAYOUT, height=300,
                           title=dict(text="Monthly Actual vs Budgeted Cost (₹)", font=dict(color="#c9d1d9",size=13)))
    st.plotly_chart(fig_area, use_container_width=True)

    # ── AI Recommendations ────────────────────────────────────────────
    worst_stage = df_sv.sort_values("Variance %", ascending=False).iloc[0]["Stage"] if not df_sv.empty else "N/A"
    exec_context = {
        "total_actual":   fmt_currency(actual_cost),
        "total_budget":   fmt_currency(budgeted_cost),
        "variance_pct":   variance_pct * 100,
        "cpu":            fmt_currency(cpu),
        "units_produced": units_produced,
        "scrap_count":    scrap_count,
        "worst_stage":    worst_stage,
    }
    render_ai_recommendations_section(exec_context, "executive", "exec")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 – WASTE & QUALITY
# ══════════════════════════════════════════════════════════════════════
elif page == "♻️ Waste & Quality":

    # ── KPI Cards ─────────────────────────────────────────────────────
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown(kpi_card("Waste Dollars (Scrap)", fmt_currency(waste_dollars),
                             "Scrap × Avg Unit Cost",
                             "bad" if waste_dollars > 500_000 else "neutral", "red"), unsafe_allow_html=True)
    with w2:
        st.markdown(kpi_card("Rework Financial Leak", fmt_currency(rework_leak),
                             "Parts lost + Labor lost",
                             "bad" if rework_leak > 500_000 else "neutral", "yellow"), unsafe_allow_html=True)
    with w3:
        st.markdown(kpi_card("Scrap Units", f"{scrap_count:,}",
                             f"Rate: {scrap_rate:.1f}%",
                             "bad" if scrap_rate > 8 else "neutral"), unsafe_allow_html=True)
    with w4:
        avg_scrap = pl_f["scrap_count"].mean() if not pl_f.empty else 0
        st.markdown(kpi_card("Avg Scrap / Log", f"{avg_scrap:.1f}",
                             "Units scrapped per entry",
                             "bad" if avg_scrap > 25 else "good"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        section("🍩 REWORK REASON DISTRIBUTION")
        rr_reason = rr_f.groupby("rework_reason").agg(
            total_cost=pd.NamedAgg("parts_cost_lost", "sum"),
            hours_lost=pd.NamedAgg("man_hours_lost", "sum")
        ).reset_index()
        rr_reason["labor_cost"] = rr_reason["hours_lost"] * LABOR_RATE
        rr_reason["total"] = rr_reason["total_cost"] + rr_reason["labor_cost"]

        if not rr_reason.empty:
            fig_donut = go.Figure(go.Pie(
                labels=rr_reason["rework_reason"],
                values=rr_reason["total"],
                hole=0.55,
                marker=dict(colors=COLORS[:len(rr_reason)], line=dict(color="#0d1117", width=2)),
                textinfo="label+percent",
                textfont=dict(size=11, color="#e6edf3"),
                hovertemplate="<b>%{label}</b><br>Cost: ₹%{value:,.0f}<br>%{percent}<extra></extra>"
            ))
            fig_donut.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"), height=340,
                margin=dict(l=20,r=20,t=50,b=20),
                legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1),
                title=dict(text="Rework Cost by Reason (₹)", font=dict(color="#c9d1d9",size=13)),
                annotations=[dict(text=f"<b>{fmt_currency(rework_leak)}</b>",
                                  font=dict(size=16,color="#f0f6fc"), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    with d2:
        section("🗺️ WASTE ₹ TREEMAP — STAGE → REASON")
        rr_stage = rr_f.merge(bm[["stage_id","stage_name"]], on="stage_id", how="left")
        rr_stage["total_rework"] = rr_stage["parts_cost_lost"] + rr_stage["man_hours_lost"] * LABOR_RATE
        treemap_data = rr_stage.groupby(["stage_name","rework_reason"])["total_rework"].sum().reset_index()
        treemap_data = treemap_data[treemap_data["total_rework"] > 0]

        if not treemap_data.empty:
            fig_tree = px.treemap(
                treemap_data, path=["stage_name","rework_reason"],
                values="total_rework", color="total_rework",
                color_continuous_scale=[[0,"#0d2a1a"],[0.5,"#d29922"],[1,"#f85149"]]
            )
            fig_tree.update_traces(
                texttemplate="<b>%{label}</b><br>₹%{value:,.0f}",
                hovertemplate="<b>%{label}</b><br>Rework: ₹%{value:,.0f}<extra></extra>"
            )
            fig_tree.update_layout(
                paper_bgcolor="#0d1117", font_color="#c9d1d9",
                height=340, margin=dict(l=0,r=0,t=50,b=0),
                title=dict(text="Waste ₹: Stage → Reason", font=dict(color="#c9d1d9",size=13)),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_tree, use_container_width=True)

    # ── Scatter: Scrap vs Labor Hours ─────────────────────────────────
    section("⚡ SCRAP COUNT vs LABOR HOURS")
    scatter_data = (
        pl_f.merge(dt_f[["date_key","month"]], on="date_key", how="left")
            .merge(ac_f[["date_key","stage_id","payroll_hours"]], on=["date_key","stage_id"], how="left")
            .merge(bm[["stage_id","stage_name"]], on="stage_id", how="left")
            .dropna(subset=["payroll_hours"])
    )
    if not scatter_data.empty:
        fig_sc = px.scatter(
            scatter_data.sample(min(len(scatter_data), 1200), random_state=42),
            x="payroll_hours", y="scrap_count",
            color="stage_name", size="units_produced",
            size_max=20, color_discrete_sequence=COLORS,
            hover_data=["month"],
            labels={"payroll_hours":"Labor Hours","scrap_count":"Scrap Count","stage_name":"Stage"}
        )
        fig_sc.update_layout(**DARK_LAYOUT, height=360,
                             title=dict(text="Scrap Count vs Labor Hours  (bubble = units produced)",
                                        font=dict(color="#c9d1d9",size=13)))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Rework Registry Table ─────────────────────────────────────────
    section("📋 REWORK REGISTRY — TOP 20 ENTRIES")
    if not rr_f.empty:
        rr_display = rr_f.merge(bm[["stage_id","stage_name"]], on="stage_id", how="left")
        rr_display = rr_display.merge(dt_f[["date_key","date"]], on="date_key", how="left")
        rr_display["rework_cost"] = rr_display["parts_cost_lost"] + rr_display["man_hours_lost"] * LABOR_RATE
        rd = rr_display[["date","stage_name","rework_reason","man_hours_lost","parts_cost_lost","rework_cost"]].copy()
        rd["date"]            = pd.to_datetime(rd["date"]).dt.strftime("%Y-%m-%d")
        rd["rework_cost"]     = rd["rework_cost"].map("₹{:,.0f}".format)
        rd["parts_cost_lost"] = rd["parts_cost_lost"].map("₹{:,.0f}".format)
        rd.columns = ["Date","Stage","Reason","Hours Lost","Parts Cost","Rework Cost"]
        rd["Action"] = rd["Rework Cost"].apply(
            lambda v: "🔴 Fix Immediately"
            if float(v.replace("₹","").replace(",","")) > 2000 else "🟡 Monitor"
        )
        st.dataframe(rd.head(20), use_container_width=True, hide_index=True)

    # ── AI Recommendations ────────────────────────────────────────────
    top_reason = (rr_f.groupby("rework_reason").apply(
        lambda g: g["parts_cost_lost"].sum() + g["man_hours_lost"].sum() * LABOR_RATE
    ).idxmax() if not rr_f.empty else "N/A")

    worst_waste_stage = (
        rr_f.merge(bm[["stage_id","stage_name"]], on="stage_id", how="left")
        .assign(total_rework=lambda d: d["parts_cost_lost"] + d["man_hours_lost"] * LABOR_RATE)
        .groupby("stage_name")["total_rework"].sum()
        .sort_values(ascending=False).index[0]
        if not rr_f.empty else "N/A"
    )

    waste_context = {
        "waste_dollars":     fmt_currency(waste_dollars),
        "rework_leak":       fmt_currency(rework_leak),
        "scrap_count":       scrap_count,
        "scrap_rate":        scrap_rate,
        "top_reason":        top_reason,
        "worst_waste_stage": worst_waste_stage,
    }
    render_ai_recommendations_section(waste_context, "waste", "waste")


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 – STAGE HEATMAP
# ══════════════════════════════════════════════════════════════════════
elif page == "🗺️ Stage Heatmap":

    section("🌡️ STAGE × SHIFT VARIANCE % HEATMAP  ( 🔴 > 10%  |  🟡 > 0%  |  🟢 ≤ 0% )")

    hm_data = []
    for _, s_row in bm_f.iterrows():
        for _, sh_row in sh_f.iterrows():
            sid, shid = s_row["stage_id"], sh_row["shift_id"]
            ac_s  = ac_f[ac_f["stage_id"] == sid]
            pl_s  = pl_f[(pl_f["stage_id"] == sid) & (pl_f["shift_id"] == shid)]
            a_cost = ac_s["material_cost"].sum() + ac_s["utility_cost"].sum() + ac_s["payroll_hours"].sum() * LABOR_RATE
            b_cost = pl_s["units_produced"].sum() * s_row["budgeted_unit_cost"]
            var    = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
            hm_data.append({"Stage": s_row["stage_name"], "Shift": sh_row["shift_name"],
                             "Variance %": round(var, 1), "actual": a_cost, "budget": b_cost})

    df_hm = pd.DataFrame(hm_data)
    pivot = df_hm.pivot(index="Stage", columns="Shift", values="Variance %")

    colorscale = [
        [0.00, "#1a4731"],
        [0.35, "#3fb950"],
        [0.50, "#d29922"],
        [0.65, "#f85149"],
        [1.00, "#8b1a1a"],
    ]
    z_vals = pivot.values
    x_labs = pivot.columns.tolist()
    y_labs = pivot.index.tolist()

    annotations = []
    for i, yl in enumerate(y_labs):
        for j, xl in enumerate(x_labs):
            val = z_vals[i][j]
            if np.isnan(val): continue
            fc = "#f85149" if val > 10 else ("#d29922" if val > 0 else "#3fb950")
            annotations.append(dict(
                x=xl, y=yl,
                text=f"<b>{val:+.1f}%</b>",
                font=dict(size=14, color=fc),
                showarrow=False
            ))

    fig_hm = go.Figure(go.Heatmap(
        z=z_vals, x=x_labs, y=y_labs,
        colorscale=colorscale, zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in z_vals],
        hovertemplate="Stage: <b>%{y}</b><br>Shift: <b>%{x}</b><br>Variance: <b>%{text}</b><extra></extra>",
        colorbar=dict(
            title=dict(text="Variance %", font=dict(color="#c9d1d9")),
            tickfont=dict(color="#c9d1d9"),
            bgcolor="#161b22", bordercolor="#30363d"
        )
    ))
    fig_hm.update_layout(
        paper_bgcolor="#0d1117", font=dict(color="#c9d1d9", family="Segoe UI"),
        xaxis=dict(title="Shift",  tickfont=dict(color="#c9d1d9", size=13), side="bottom"),
        yaxis=dict(title="Stage",  tickfont=dict(color="#c9d1d9", size=13), autorange="reversed"),
        annotations=annotations,
        height=480, margin=dict(l=160,r=60,t=60,b=60),
        title=dict(text="Variance % by Stage × Shift", font=dict(color="#f0f6fc",size=15))
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Machine × Shift Production Volume ─────────────────────────────
    section("🏭 MACHINE × SHIFT PRODUCTION VOLUME")
    pl_mc  = pl_f.merge(mc_f[["machine_id","machine_name","production_line"]], on="machine_id", how="left")
    pl_mc  = pl_mc.merge(sh_f[["shift_id","shift_name"]], on="shift_id", how="left")
    mc_hm  = pl_mc.groupby(["machine_name","shift_name"])["units_produced"].sum().reset_index()
    mc_pivot = mc_hm.pivot(index="machine_name", columns="shift_name", values="units_produced").fillna(0)

    fig_mc = go.Figure(go.Heatmap(
        z=mc_pivot.values,
        x=mc_pivot.columns.tolist(),
        y=mc_pivot.index.tolist(),
        colorscale="Blues",
        text=mc_pivot.values.astype(int),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Units", tickfont=dict(color="#9aa0b5"))
    ))
    fig_mc.update_layout(
        **DARK_LAYOUT, height=500,
        title=dict(text="Units Produced by Machine × Shift", font=dict(color="#c9d1d9",size=13)),
        xaxis_title="Shift", yaxis_title="Machine"
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── Stage Performance Summary Table ───────────────────────────────
    section("📋 STAGE PERFORMANCE SUMMARY")
    summary_rows = []
    for _, row in bm_f.iterrows():
        sid   = row["stage_id"]
        ac_s  = ac_f[ac_f["stage_id"] == sid]
        pl_s  = pl_f[pl_f["stage_id"] == sid]
        a_cost = ac_s["material_cost"].sum() + ac_s["utility_cost"].sum() + ac_s["payroll_hours"].sum() * LABOR_RATE
        b_cost = pl_s["units_produced"].sum() * row["budgeted_unit_cost"]
        var    = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
        scrap  = pl_s["scrap_count"].sum()
        units  = pl_s["units_produced"].sum()
        summary_rows.append({
            "Stage":           row["stage_name"],
            "Category":        row.get("category",""),
            "Actual Cost":     fmt_currency(a_cost),
            "Budget":          fmt_currency(b_cost),
            "Variance %":      f"{var:+.1f}%",
            "Units Produced":  f"{units:,}",
            "Scrap Units":     f"{scrap:,}",
            "Status":          "🔴 Over" if var > 10 else ("🟡 Warning" if var > 0 else "🟢 Good"),
        })
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # ── AI Recommendations ────────────────────────────────────────────
    worst_combo = (df_hm.sort_values("Variance %", ascending=False)
                   .apply(lambda r: f"{r['Stage']} × {r['Shift']}", axis=1)
                   .iloc[0] if not df_hm.empty else "N/A")

    shift_v = df_hm.groupby("Shift").agg(actual=("actual","sum"), budget=("budget","sum")).reset_index()
    shift_v["var_pct"] = (shift_v["actual"] - shift_v["budget"]) / shift_v["budget"].replace(0, np.nan) * 100
    worst_shift = shift_v.sort_values("var_pct", ascending=False).iloc[0]["Shift"] if not shift_v.empty else "N/A"

    stage_v = df_hm.groupby("Stage").agg(actual=("actual","sum"), budget=("budget","sum")).reset_index()
    stage_v["var_pct"] = (stage_v["actual"] - stage_v["budget"]) / stage_v["budget"].replace(0, np.nan) * 100
    worst_stage = stage_v.sort_values("var_pct", ascending=False).iloc[0]["Stage"] if not stage_v.empty else "N/A"

    heatmap_context = {
        "worst_combo":  worst_combo,
        "worst_stage":  worst_stage,
        "worst_shift":  worst_shift,
        "variance_pct": variance_pct * 100,
        "total_actual": fmt_currency(actual_cost),
        "total_budget": fmt_currency(budgeted_cost),
    }
    render_ai_recommendations_section(heatmap_context, "heatmap", "heatmap")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 – AI CHATBOT
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":
    st.markdown("""<div class="page-banner">
        <h1>🤖 AI Analytics Chatbot</h1>
        <p>Ask questions about production data in natural language — rule-based + data-driven intelligence</p>
    </div>""", unsafe_allow_html=True)

    # Pre-compute chatbot metrics
    stage_var2 = []
    for _, row in bm_f.iterrows():
        sid   = row["stage_id"]
        ac_s  = ac_f[ac_f["stage_id"] == sid]
        pl_s  = pl_f[pl_f["stage_id"] == sid]
        a_cost = ac_s["material_cost"].sum() + ac_s["utility_cost"].sum() + ac_s["payroll_hours"].sum() * LABOR_RATE
        b_cost = pl_s["units_produced"].sum() * row["budgeted_unit_cost"]
        var    = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
        stage_var2.append({"stage_name": row["stage_name"], "var_pct": var})
    sv2 = pd.DataFrame(stage_var2)

    worst_stage_c = sv2.sort_values("var_pct", ascending=False).iloc[0]["stage_name"] if not sv2.empty else "N/A"
    best_stage_c  = sv2.sort_values("var_pct").iloc[0]["stage_name"]                  if not sv2.empty else "N/A"

    top_reason_c = "N/A"
    if not rr_f.empty:
        rr_agg = rr_f.groupby("rework_reason").apply(
            lambda g: g["parts_cost_lost"].sum() + g["man_hours_lost"].sum() * LABOR_RATE
        )
        top_reason_c = rr_agg.idxmax()

    shift_hm = []
    for _, sh_row in sh_f.iterrows():
        pl_s  = pl_f[pl_f["shift_id"] == sh_row["shift_id"]]
        ac_sh = ac_f.copy()  # costs are not shift-split in source data
        b_cost = sum(
            pl_s[pl_s["stage_id"] == r.stage_id]["units_produced"].sum() * r.budgeted_unit_cost
            for r in bm_f.itertuples()
        )
        a_cost = (ac_sh["material_cost"].sum() + ac_sh["utility_cost"].sum()
                  + ac_sh["payroll_hours"].sum() * LABOR_RATE) * (len(pl_s) / max(len(pl_f), 1))
        var = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
        shift_hm.append({"shift_name": sh_row["shift_name"], "var_pct": var})
    shift_df = pd.DataFrame(shift_hm)
    worst_shift_c = shift_df.sort_values("var_pct", ascending=False).iloc[0]["shift_name"] if not shift_df.empty else "N/A"

    def generate_response(msg_raw: str) -> str:
        msg = msg_raw.lower()

        if any(k in msg for k in ["highest variance","worst variance","most variance","max variance"]):
            return (f"📊 **Highest Variance Stage: {worst_stage_c}**\n\n"
                    f"This stage has the largest positive budget overrun. "
                    f"Overall system variance is **{variance_pct*100:+.1f}%**.\n\n"
                    f"**Recommendation:** Review material procurement and labor allocation for `{worst_stage_c}`.")

        if any(k in msg for k in ["best stage","lowest variance","best performing"]):
            return (f"✅ **Best Performing Stage: {best_stage_c}**\n\n"
                    f"Operating at or below budgeted cost — a benchmark for efficiency.\n\n"
                    f"**Tip:** Identify and replicate its processes across other stages.")

        if "variance" in msg and "all" in msg:
            lines = [f"- **{r['stage_name']}**: {r['var_pct']:+.1f}%" for _, r in sv2.iterrows()]
            return "📋 **Variance % by Stage:**\n\n" + "\n".join(lines)

        if "variance" in msg:
            status = ("⚠️ Significantly over budget — immediate action needed." if variance_pct > 0.10 else
                      "📌 Moderately over budget — keep monitoring." if variance_pct > 0 else
                      "✅ Within budget — excellent performance!")
            return (f"📉 **Overall Variance: {variance_pct*100:+.1f}%**\n\n"
                    f"- Actual Cost: **{fmt_currency(actual_cost)}**\n"
                    f"- Budgeted Cost: **{fmt_currency(budgeted_cost)}**\n"
                    f"- Worst Stage: **{worst_stage_c}**\n\n{status}")

        if any(k in msg for k in ["why is cost high","cost high","expensive","over budget"]):
            return (f"💰 **Cost Analysis:**\n\n"
                    f"Total actual cost **{fmt_currency(actual_cost)}** vs budget **{fmt_currency(budgeted_cost)}** "
                    f"= **{variance_pct*100:+.1f}%** variance.\n\n"
                    f"**Root Cause Drivers:**\n"
                    f"- Highest-variance stage: **{worst_stage_c}**\n"
                    f"- Top rework reason: **{top_reason_c}**\n"
                    f"- Total waste spend: **{fmt_currency(waste_dollars)}**\n\n"
                    f"**Actions:** Audit material suppliers, review payroll-to-output ratio, "
                    f"reduce scrap in `{worst_stage_c}`.")

        if any(k in msg for k in ["total cost","actual cost","overall cost"]):
            return (f"💵 **Cost Summary:**\n\n"
                    f"- Total Actual: **{fmt_currency(actual_cost)}**\n"
                    f"- Total Budget: **{fmt_currency(budgeted_cost)}**\n"
                    f"- Variance: **{variance_pct*100:+.1f}%**\n"
                    f"- Cost Per Unit: **{fmt_currency(cpu)}**")

        if any(k in msg for k in ["waste","scrap"]):
            return (f"🗑️ **Waste & Scrap Analysis:**\n\n"
                    f"- Total Waste ₹: **{fmt_currency(waste_dollars)}**\n"
                    f"- Scrap Count: **{scrap_count:,} units**\n"
                    f"- Scrap Rate: **{scrap_rate:.1f}%**\n\n"
                    f"**Recommendation:** Implement SPC (Statistical Process Control) and "
                    f"increase inspection frequency at high-scrap stages.")

        if any(k in msg for k in ["rework","repair","fix process"]):
            return (f"🔧 **Rework Registry:**\n\n"
                    f"- Total Rework Cost: **{fmt_currency(rework_leak)}**\n"
                    f"- Top Rework Reason: **{top_reason_c}**\n\n"
                    f"**Recommendation:** Root-cause analysis for `{top_reason_c}`. "
                    f"Provide targeted operator training and update SOPs.")

        if any(k in msg for k in ["shift","supervisor","night","morning","afternoon"]):
            lines = [f"- **{r['shift_name']}**: {r['var_pct']:+.1f}% variance" for _, r in shift_df.iterrows()]
            return (f"🕐 **Shift Performance:**\n\n" + "\n".join(lines) +
                    f"\n\n**Worst Shift:** `{worst_shift_c}`\n\n"
                    f"Review staffing levels and supervisor effectiveness for the `{worst_shift_c}` shift.")

        if any(k in msg for k in ["recommend","suggestion","improve","what should","action"]):
            recs = []
            if variance_pct > 0.10:
                recs.append(f"🔴 **Reduce material & labor costs** — variance is {variance_pct*100:+.1f}%")
            if scrap_rate > 8:
                recs.append("🔍 **Improve quality control** — scrap rate is elevated")
            if rework_leak > 300_000:
                recs.append("🔧 **Fix process inefficiencies** — rework cost is high")
            if waste_dollars > 500_000:
                recs.append("💰 **Investigate waste sources** — waste spend is significant")
            if not recs:
                recs.append("✅ **System performing well** — maintain current practices")
            return "🧠 **AI Recommendations:**\n\n" + "\n\n".join(recs)

        if any(k in msg for k in ["summary","overview","report","status","how is"]):
            return (f"📋 **System Summary:**\n\n"
                    f"| Metric | Value |\n|---|---|\n"
                    f"| Total Actual Cost | {fmt_currency(actual_cost)} |\n"
                    f"| Total Budgeted Cost | {fmt_currency(budgeted_cost)} |\n"
                    f"| Overall Variance | {variance_pct*100:+.1f}% |\n"
                    f"| Cost Per Unit | {fmt_currency(cpu)} |\n"
                    f"| Total Units | {units_produced:,} |\n"
                    f"| Scrap Count | {scrap_count:,} |\n"
                    f"| Scrap Rate | {scrap_rate:.1f}% |\n"
                    f"| Total Waste ₹ | {fmt_currency(waste_dollars)} |\n"
                    f"| Rework Leak | {fmt_currency(rework_leak)} |\n"
                    f"| Worst Stage | {worst_stage_c} |")

        if any(k in msg for k in ["hi","hello","hey","help","what can"]):
            return ("👋 **Hello! I'm your PreciCost Analytics AI.**\n\n"
                    "You can ask me:\n\n"
                    "- 📊 *Which stage has highest variance?*\n"
                    "- 💰 *Why is cost high?*\n"
                    "- 🗑️ *Show waste analysis*\n"
                    "- 🔧 *Tell me about rework*\n"
                    "- 🕐 *How are shifts performing?*\n"
                    "- 🧠 *Give recommendations*\n"
                    "- 📋 *Give me a summary*")

        return (f"🤖 Quick snapshot — Variance: **{variance_pct*100:+.1f}%** | "
                f"Waste: **{fmt_currency(waste_dollars)}** | Rework: **{fmt_currency(rework_leak)}**\n\n"
                f"Try: *'Which stage has highest variance?'* or *'Give recommendations'*")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"assistant",
            "content": ("👋 **Welcome to the PreciCost Analytics Chatbot!**\n\n"
                        "I can answer questions about production costs, variance, waste, rework, and more.\n\n"
                        "Try asking: *'Which stage has highest variance?'* or *'Give recommendations'*")}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    st.markdown("**⚡ Quick Questions:**")
    q_cols = st.columns(4)
    quick_qs = [
        "Which stage has highest variance?",
        "Give me a summary",
        "Give recommendations",
        "Show waste analysis"
    ]
    for i, qc in enumerate(q_cols):
        with qc:
            if st.button(quick_qs[i], key=f"qb_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":quick_qs[i]})
                st.session_state.chat_history.append({"role":"assistant","content":generate_response(quick_qs[i])})
                st.rerun()

    if prompt := st.chat_input("Ask anything about your manufacturing data…"):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        st.session_state.chat_history.append({"role":"assistant","content":generate_response(prompt)})
        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
