"""
╔══════════════════════════════════════════════════════════════════════╗
║       PRECICOST INTELLIGENCE                                         ║
║       Manufacturing Variance & Cost Analytics Dashboard              ║
║       AI Recommendations · AI Chatbot · Dark Theme                  ║
║       AI Engine: Groq (llama-3.3-70b-versatile) — Ultra Fast        ║
╚══════════════════════════════════════════════════════════════════════╝

SECURITY NOTE:
  - API key is loaded ONLY from environment variable GROQ_API_KEY
  - Never paste your key into this file
  - Local dev  : create a .env file (see .env.example) — it is git-ignored
  - Streamlit  : add GROQ_API_KEY under Settings → Secrets
  - Free key   : https://console.groq.com
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from groq import Groq
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# AUTO-LOAD .env FOR LOCAL DEVELOPMENT
# python-dotenv is optional — silently skipped if not installed
# ─────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # rely on system environment variables

# ─────────────────────────────────────────────────────────────────────
# 🔑 API KEY — from environment ONLY, never hardcoded
# ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PreciCost Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# DARK THEME CSS
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
.kpi-card.red    { border-left: 4px solid #f85149; }
.kpi-card.green  { border-left: 4px solid #3fb950; }
.kpi-card.yellow { border-left: 4px solid #d29922; }
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
    border: 1px solid #1d4ed8; border-radius: 12px;
    padding: 20px 24px; margin: 10px 0;
    position: relative; overflow: hidden;
}
.ai-rec-container::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #388bfd, #a78bfa, #3fb950);
}
.ai-rec-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; font-weight: 600; color: #79c0ff;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;
}
.ai-rec-content { color: #c9d1d9; font-size: 13px; line-height: 1.7; }

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
    border: none !important; color: white !important;
    font-weight: 600 !important; letter-spacing: 0.5px !important;
}
.groq-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, #0d1b2e, #111827);
    border: 1px solid #388bfd; border-radius: 20px;
    padding: 4px 12px; font-size: 11px; color: #79c0ff;
    font-weight: 600; letter-spacing: 0.5px;
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
    legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1),
)

COLORS = [
    "#388bfd","#3fb950","#d29922","#f85149","#a5d6ff","#56d364",
    "#e3b341","#ff7b72","#79c0ff","#7ee787","#ffa657","#cae8ff",
]

LABOR_RATE = 25                           # ₹25 / hr
GROQ_MODEL = "llama-3.3-70b-versatile"   # fastest free-tier model


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def fmt_currency(v: float) -> str:
    if v >= 10_000_000: return f"₹{v/10_000_000:.2f}Cr"
    if v >= 100_000:    return f"₹{v/100_000:.2f}L"
    if v >= 1_000:      return f"₹{v/1_000:.1f}K"
    return f"₹{v:,.0f}"


def kpi_card(label, value, sub="", sub_class="neutral", border_class="") -> str:
    cls = f"kpi-card {border_class}".strip()
    return (
        f'<div class="{cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub {sub_class}">{sub}</div>'
        f"</div>"
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def call_groq(system_prompt: str, user_message: str, max_tokens: int = 500) -> str:
    """
    Call Groq API with llama-3.3-70b-versatile.
    GROQ_API_KEY is read from the module-level constant which is populated
    exclusively from os.getenv() — it is NEVER stored in source code.
    """
    if not GROQ_API_KEY:
        return (
            "❌ **Groq API key not found.**\n\n"
            "**Local dev:** add `GROQ_API_KEY=gsk_...` to your `.env` file.\n\n"
            "**Streamlit Cloud:** go to Settings → Secrets and add the key.\n\n"
            "Get a free key at https://console.groq.com"
        )
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"❌ Groq API Error: {exc}"


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@st.cache_data
def load_data():
    ac = pd.read_csv(f"{DATA_DIR}/Actual_Costs.csv")
    bm = pd.read_csv(f"{DATA_DIR}/Budget_Master.csv")
    dt = pd.read_csv(f"{DATA_DIR}/Date.csv")
    mc = pd.read_csv(f"{DATA_DIR}/Machines.csv")
    pl = pd.read_csv(f"{DATA_DIR}/Production_Logs.csv")
    rr = pd.read_csv(f"{DATA_DIR}/Rework_Registry.csv")
    sh = pd.read_csv(f"{DATA_DIR}/Shifts.csv")
    dt["date"]     = pd.to_datetime(dt["date"])
    dt["date_key"] = dt["date_key"].astype(int)
    for df in [ac, pl, rr]:
        df["date_key"] = df["date_key"].astype(int)
    return ac, bm, dt, mc, pl, rr, sh


ac, bm, dt, mc, pl, rr, sh = load_data()


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR — filters only, NO API key input in UI
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px 0;'>
        <div style='font-size:36px;'>🏭</div>
        <div style='font-size:15px;font-weight:700;color:#f0f6fc;'>PreciCost Intelligence</div>
        <div style='font-size:11px;color:#8b949e;'>Manufacturing Variance Analytics</div>
        <br><span class="groq-badge">⚡ Powered by Groq</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    page = st.selectbox("📌 Navigation", [
        "📊 Executive Variance",
        "♻️ Waste & Quality",
        "🗺️ Stage Heatmap",
        "🤖 AI Chatbot",
    ])

    st.markdown("---")

    # Read-only AI status badge — key is never entered in the UI
    if GROQ_API_KEY:
        st.markdown(
            '<div style="background:#0d2a1a;border:1px solid #3fb950;border-radius:8px;'
            'padding:8px 12px;font-size:12px;color:#56d364;">✅ Groq AI Active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#2d2208;border:1px solid #d29922;border-radius:8px;'
            'padding:8px 12px;font-size:12px;color:#e3b341;">'
            '⚠️ Set GROQ_API_KEY in .env or Streamlit Secrets</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#8b949e;text-transform:uppercase;'
        'letter-spacing:1px;margin-bottom:8px;">🎛️ FILTERS</div>',
        unsafe_allow_html=True,
    )

    years        = sorted(dt["year"].unique())
    sel_years    = st.multiselect("📅 Year",            years,       default=years)
    quarters     = sorted(dt["quarter"].unique())
    sel_quarters = st.multiselect("📆 Quarter",         quarters,    default=quarters)
    lines        = sorted(mc["production_line"].unique())
    sel_lines    = st.multiselect("🏗️ Production Line", lines,       default=lines)
    stages       = sorted(bm["stage_name"].unique())
    sel_stages   = st.multiselect("🔧 Stage",           stages,      default=stages)
    shifts_list  = sorted(sh["shift_name"].unique())
    sel_shifts   = st.multiselect("🕐 Shift",           shifts_list, default=shifts_list)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#6e7681;padding:6px 0;line-height:1.8;'>
        📁 7-CSV Star Schema<br>
        🔗 PreciCost v2.0<br>
        ⚡ Groq llama-3.3-70b-versatile
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────
dt_f       = dt[dt["year"].isin(sel_years) & dt["quarter"].isin(sel_quarters)]
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
# FILTER FINGERPRINT — auto-clear AI cache whenever filters change
# ─────────────────────────────────────────────────────────────────────
_filter_fingerprint = (
    str(sorted(sel_years))
    + str(sorted(sel_quarters))
    + str(sorted(sel_lines))
    + str(sorted(sel_stages))
    + str(sorted(sel_shifts))
)

if st.session_state.get("_last_filter_fingerprint") != _filter_fingerprint:
    for _sk in ["exec", "waste", "heatmap"]:
        st.session_state[f"active_card_{_sk}"]  = None
        st.session_state[f"card_content_{_sk}"] = {}
        st.session_state[f"translation_{_sk}"]  = ""
    st.session_state["_last_filter_fingerprint"] = _filter_fingerprint


# ─────────────────────────────────────────────────────────────────────
# KPI CALCULATIONS
# ─────────────────────────────────────────────────────────────────────
actual_cost = (
    ac_f["material_cost"].sum()
    + ac_f["utility_cost"].sum()
    + ac_f["payroll_hours"].sum() * LABOR_RATE
)
units_produced = pl_f["units_produced"].sum()
scrap_count    = pl_f["scrap_count"].sum()
budgeted_cost  = sum(
    pl_f[pl_f["stage_id"] == row.stage_id]["units_produced"].sum() * row.budgeted_unit_cost
    for row in bm_f.itertuples()
)
variance_pct  = (actual_cost - budgeted_cost) / budgeted_cost if budgeted_cost else 0
avg_unit_cost = actual_cost / units_produced if units_produced else 0
waste_dollars = scrap_count * avg_unit_cost
rework_leak   = rr_f["parts_cost_lost"].sum() + rr_f["man_hours_lost"].sum() * LABOR_RATE
cpu           = actual_cost / max(units_produced - scrap_count, 1)
scrap_rate    = (scrap_count / units_produced * 100) if units_produced else 0
eff_color     = "red" if variance_pct > 0.10 else ("yellow" if variance_pct > 0 else "green")


# ─────────────────────────────────────────────────────────────────────
# AI RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────────────────────────────
def render_ai_recommendations_section(context: dict, page_type: str, section_key: str) -> None:
    """5 AI buttons — every click always regenerates fresh from Groq."""

    section("⚡ GENERATE WITH AI (Groq)")

    active_key  = f"active_card_{section_key}"
    content_key = f"card_content_{section_key}"
    trans_key   = f"translation_{section_key}"
    lang_key    = f"lang_{section_key}"

    for k, default in [
        (active_key, None), (content_key, {}), (trans_key, ""), (lang_key, "Telugu"),
    ]:
        if k not in st.session_state:
            st.session_state[k] = default

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: clicked_analysis = st.button("📊 AI Analysis",       key=f"btn_a_{section_key}", use_container_width=True)
    with b2: clicked_causes   = st.button("🔍 Problem Causes",    key=f"btn_c_{section_key}", use_container_width=True)
    with b3: clicked_recs     = st.button("✅ Recommendations",   key=f"btn_r_{section_key}", use_container_width=True)
    with b4: clicked_plan     = st.button("🗓️ Efficiency Plan",   key=f"btn_p_{section_key}", use_container_width=True)
    with b5: clicked_outcomes = st.button("🎯 Expected Outcomes", key=f"btn_o_{section_key}", use_container_width=True)

    def build_data_summary() -> str:
        if page_type == "executive":
            return (
                "Executive Dashboard Data (all monetary values in Indian Rupees ₹):\n"
                f"- Total Actual Cost: {context['total_actual']}\n"
                f"- Total Budgeted Cost: {context['total_budget']}\n"
                f"- Overall Variance: {context['variance_pct']:+.1f}%\n"
                f"- Cost Per Unit: {context['cpu']}\n"
                f"- Total Units Produced: {context['units_produced']:,}\n"
                f"- Scrap Count: {context['scrap_count']:,}\n"
                f"- Worst Variance Stage: {context['worst_stage']}"
            )
        if page_type == "waste":
            return (
                "Waste & Quality Tracker Data (all monetary values in Indian Rupees ₹):\n"
                f"- Total Waste Dollars (Scrap): {context['waste_dollars']}\n"
                f"- Rework Financial Leak: {context['rework_leak']}\n"
                f"- Scrap Count: {context['scrap_count']:,}\n"
                f"- Scrap Rate: {context['scrap_rate']:.1f}%\n"
                f"- Top Rework Reason: {context['top_reason']}\n"
                f"- Worst Stage for Waste: {context.get('worst_waste_stage', 'N/A')}"
            )
        if page_type == "heatmap":
            return (
                "Stage × Shift Heatmap Data (all monetary values in Indian Rupees ₹):\n"
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
            "spinner": "⚡ Groq is generating AI Analysis…",
            "system": (
                "You are a manufacturing analyst. "
                "Write 3-4 sentences summarizing the overall performance situation based on the data. "
                "Be specific with numbers. No bullet points — clear paragraph form only. "
                "All monetary values are in Indian Rupees (₹)."
            ),
            "title": "📊 AI ANALYSIS", "header_color": "#79c0ff", "border_color": "#1d4ed8",
        },
        "causes": {
            "spinner": "⚡ Groq is identifying Problem Causes…",
            "system": (
                "You are a manufacturing analyst. "
                "List 3-4 bullet points identifying root causes visible in the data. "
                "Format: **Cause title:** one sentence explanation. "
                "All monetary values are in Indian Rupees (₹)."
            ),
            "title": "🔍 PROBLEM CAUSES", "header_color": "#e3b341", "border_color": "#d29922",
        },
        "recs": {
            "spinner": "⚡ Groq is generating Recommendations…",
            "system": (
                "You are a manufacturing analyst. "
                "List 4-5 bullet points of specific, actionable recommendations prioritized by impact. "
                "Format: **Action title:** one sentence on what to do. "
                "All monetary values are in Indian Rupees (₹)."
            ),
            "title": "✅ RECOMMENDATIONS", "header_color": "#56d364", "border_color": "#3fb950",
        },
        "plan": {
            "spinner": "⚡ Groq is building the Efficiency Plan…",
            "system": (
                "You are a manufacturing operations expert. "
                "Create a detailed 0-30 Day Efficiency Improvement Plan. "
                "Structure it as:\n"
                "**Week 1 (Days 1-7):** 2 immediate actions\n"
                "**Week 2 (Days 8-14):** 2 short-term actions\n"
                "**Week 3-4 (Days 15-30):** 2 medium-term actions\n"
                "Each action should have a clear owner role (e.g. Shift Supervisor, Quality Engineer) "
                "and expected impact. Be specific and practical. "
                "All monetary values are in Indian Rupees (₹)."
            ),
            "title": "🗓️ EFFICIENCY IMPROVEMENT PLAN (0–30 DAYS)", "header_color": "#ffa657", "border_color": "#e06c00",
        },
        "outcomes": {
            "spinner": "⚡ Groq is projecting Expected Outcomes…",
            "system": (
                "You are a manufacturing analyst. "
                "Project the Expected Outcomes if recommendations are implemented within 30 days. "
                "List 4-5 bullet points. Each bullet: **Metric name:** expected improvement with a % or ₹ estimate. "
                "Be realistic and data-driven. All monetary values are in Indian Rupees (₹)."
            ),
            "title": "🎯 EXPECTED OUTCOMES", "header_color": "#c9a7ff", "border_color": "#7c3aed",
        },
    }

    triggered = None
    if clicked_analysis:   triggered = "analysis"
    elif clicked_causes:   triggered = "causes"
    elif clicked_recs:     triggered = "recs"
    elif clicked_plan:     triggered = "plan"
    elif clicked_outcomes: triggered = "outcomes"

    # Always regenerate on every click — no stale cache ever served
    if triggered:
        st.session_state[active_key] = triggered
        st.session_state[trans_key]  = ""
        cfg      = CARD_CONFIG[triggered]
        base_msg = build_data_summary()
        if triggered in ("plan", "outcomes") and "recs" in st.session_state[content_key]:
            base_msg += f"\n\nExisting Recommendations:\n{st.session_state[content_key]['recs']}"
        with st.spinner(cfg["spinner"]):
            st.session_state[content_key][triggered] = call_groq(
                cfg["system"], base_msg, max_tokens=500
            )

    active = st.session_state[active_key]
    if active:
        cfg  = CARD_CONFIG[active]
        text = st.session_state[content_key].get(active, "")
        html = text.replace("\n", "<br>")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="ai-rec-container" style="min-height:180px;border-color:{cfg["border_color"]};">'
            f'<div class="ai-rec-header" style="color:{cfg["header_color"]};">{cfg["title"]}</div>'
            f'<div class="ai-rec-content">{html}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;color:#8b949e;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:8px;">🌐 TRANSLATE THIS REPORT</div>',
            unsafe_allow_html=True,
        )

        LANGUAGES = [
            "Telugu","Hindi","Tamil","Kannada","Marathi","Bengali",
            "Gujarati","Punjabi","Urdu","Spanish","French","German",
            "Arabic","Chinese","Japanese",
        ]
        lang_col, btn_col, clear_col = st.columns([2, 1, 1])
        with lang_col:
            lang_choice = st.selectbox(
                "Select Language", LANGUAGES,
                index=LANGUAGES.index(st.session_state[lang_key]),
                key=f"lang_select_{section_key}",
                label_visibility="collapsed",
            )
            st.session_state[lang_key] = lang_choice
        with btn_col:
            translate_clicked = st.button(
                f"🌐 Translate to {lang_choice}",
                key=f"btn_translate_{section_key}",
                use_container_width=True,
            )
        with clear_col:
            if st.button("🗑️ Clear", key=f"clear_{section_key}", use_container_width=True):
                st.session_state[active_key]  = None
                st.session_state[content_key] = {}
                st.session_state[trans_key]   = ""
                st.rerun()

        if translate_clicked:
            with st.spinner(f"⚡ Groq is translating to {lang_choice}…"):
                st.session_state[trans_key] = call_groq(
                    system_prompt=(
                        "You are a professional translator. Translate the given manufacturing report text "
                        "exactly as instructed. Preserve all formatting, bullet points, and bold markers (**). "
                        "Keep the ₹ symbol as-is. Only return the translated text, nothing else."
                    ),
                    user_message=f"Translate the following manufacturing report text into {lang_choice}:\n\n{text}",
                    max_tokens=600,
                )

        if st.session_state[trans_key]:
            trans_html = st.session_state[trans_key].replace("\n", "<br>")
            st.markdown(
                f'<div class="ai-rec-container" style="min-height:140px;border-color:#a78bfa;margin-top:8px;">'
                f'<div class="ai-rec-header" style="color:#c9a7ff;">🌐 {lang_choice.upper()} TRANSLATION</div>'
                f'<div class="ai-rec-content">{trans_html}</div></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────
# GUARD: empty data
# ─────────────────────────────────────────────────────────────────────
if pl_f.empty and ac_f.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-banner">'
    "<h1>🏭 PreciCost Intelligence</h1>"
    "<p>Manufacturing Variance & Cost Analytics Dashboard · ⚡ AI-Powered by Groq (llama-3.3-70b-versatile)</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 – EXECUTIVE VARIANCE
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Executive Variance":

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Actual Cost",   fmt_currency(actual_cost),
         fmt_currency(budgeted_cost) + " budget",
         "bad" if variance_pct > 0.1 else "good", eff_color),
        (c2, "Budgeted Cost", fmt_currency(budgeted_cost),
         f"{len(dt_f)} production days", "neutral", ""),
        (c3, "Cost Variance", f"{abs(variance_pct):.1%}",
         "Under Budget" if variance_pct < 0 else ("Over Budget" if variance_pct > 0 else "On Budget"),
         "good" if variance_pct < 0 else ("bad" if variance_pct > 0.1 else "neutral"), eff_color),
        (c4, "Cost Per Unit", fmt_currency(cpu), "Good units only", "neutral", ""),
        (c5, "Total Units",   f"{units_produced:,}", f"Scrap: {scrap_count:,}",
         "bad" if scrap_count / max(units_produced, 1) > 0.08 else "neutral", ""),
    ]
    for col, label, val, sub, sub_cls, border_cls in kpis:
        with col:
            st.markdown(kpi_card(label, val, sub, sub_cls, border_cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_g, col_b = st.columns([1, 2])

    with col_g:
        section("📉 VARIANCE GAUGE")
        gauge_color = (
            "#3fb950" if variance_pct < 0
            else ("#f85149" if variance_pct > 0.10 else "#d29922")
        )
        variance_status = (
            "Under Budget" if variance_pct < 0
            else ("Over Budget" if variance_pct > 0 else "On Budget")
        )
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(variance_pct * 100),
            number={"suffix": "%", "font": {"color": gauge_color, "size": 32}},
            title={
                "text": (
                    f"Overall Variance %<br>"
                    f"<span style='font-size:11px;color:{gauge_color}'>{variance_status}</span>"
                ),
                "font": {"color": "#c9d1d9", "size": 13},
            },
            gauge={
                "axis": {"range": [0, 50], "tickcolor": "#8b949e"},
                "bar":  {"color": gauge_color},
                "bgcolor": "#161b22", "borderwidth": 1, "bordercolor": "#30363d",
                "steps": [
                    {"range": [0,  10], "color": "#0d2a1a"},
                    {"range": [10, 20], "color": "#2d2208"},
                    {"range": [20, 50], "color": "#2d1117"},
                ],
                "threshold": {"line": {"color": "#79c0ff", "width": 3}, "thickness": 0.8, "value": 10},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d1117", font_color="#c9d1d9",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_b:
        section("📊 VARIANCE % BY STAGE")
        stage_var = []
        for _, row in bm_f.iterrows():
            sid   = row["stage_id"]
            ac_s  = ac_f[ac_f["stage_id"] == sid]
            pl_s  = pl_f[pl_f["stage_id"] == sid]
            a_cost = (
                ac_s["material_cost"].sum()
                + ac_s["utility_cost"].sum()
                + ac_s["payroll_hours"].sum() * LABOR_RATE
            )
            b_cost = pl_s["units_produced"].sum() * row["budgeted_unit_cost"]
            var    = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
            stage_var.append({
                "Stage": row["stage_name"], "Variance %": var, "Display %": abs(var),
                "Status": "Under Budget" if var < 0 else ("Over Budget" if var > 0 else "On Budget"),
                "Actual": a_cost, "Budget": b_cost,
            })

        df_sv = pd.DataFrame(stage_var).sort_values("Variance %", ascending=False)
        bar_colors = [
            "#3fb950" if v < 0 else ("#f85149" if v > 10 else "#d29922")
            for v in df_sv["Variance %"]
        ]
        fig_bar = go.Figure(go.Bar(
            x=df_sv["Stage"], y=df_sv["Variance %"],
            marker_color=bar_colors,
            text=[f"{abs(v):.1f}% {'Under' if v < 0 else 'Over'}".strip() for v in df_sv["Variance %"]],
            textposition="outside", textfont=dict(size=11, color="#c9d1d9"),
            hovertemplate=(
                "<b>%{x}</b><br>Variance: %{customdata[0]:.1f}%<br>"
                "Status: %{customdata[1]}<br>Actual: ₹%{customdata[2]:,.0f}<br>"
                "Budget: ₹%{customdata[3]:,.0f}<extra></extra>"
            ),
            customdata=df_sv[["Display %", "Status", "Actual", "Budget"]].values,
        ))
        fig_bar.add_hline(y=0,  line_color="#79c0ff", line_dash="dash", line_width=1)
        fig_bar.add_hline(
            y=10, line_color="#f85149", line_dash="dot", line_width=1,
            annotation_text="10% Over Budget Threshold", annotation_font_color="#f85149",
        )
        fig_bar.update_layout(
            **DARK_LAYOUT, height=300, showlegend=False,
            title=dict(text="Stage-Level Cost Variance", font=dict(color="#c9d1d9", size=13)),
            yaxis_title="Variance %",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    section("📈 ACTUAL vs BUDGET TREND")
    ac_trend = ac_f.merge(dt_f[["date_key","date","month","quarter"]], on="date_key", how="left")
    ac_trend["actual_cost"] = (
        ac_trend["material_cost"] + ac_trend["utility_cost"] + ac_trend["payroll_hours"] * LABOR_RATE
    )
    monthly_actual = ac_trend.groupby("month")["actual_cost"].sum().reset_index()
    pl_trend = pl_f.merge(dt_f[["date_key","month"]], on="date_key", how="left")
    pl_bm    = pl_trend.merge(bm[["stage_id","budgeted_unit_cost"]], on="stage_id", how="left")
    pl_bm["budget_cost"] = pl_bm["units_produced"] * pl_bm["budgeted_unit_cost"]
    monthly_budget = pl_bm.groupby("month")["budget_cost"].sum().reset_index()

    month_order = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December",
    ]
    monthly = monthly_actual.merge(monthly_budget, on="month", how="outer").fillna(0)
    monthly["month_num"] = monthly["month"].apply(
        lambda m: month_order.index(m) if m in month_order else 99
    )
    monthly = monthly.sort_values("month_num")

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["budget_cost"], name="Budgeted Cost",
        fill="tozeroy", fillcolor="rgba(56,139,253,0.12)", line=dict(color="#388bfd", width=2),
        hovertemplate="Month: %{x}<br>Budget: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_area.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["actual_cost"], name="Actual Cost",
        fill="tozeroy", fillcolor="rgba(248,81,73,0.12)", line=dict(color="#f85149", width=2),
        hovertemplate="Month: %{x}<br>Actual: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_area.update_layout(
        **DARK_LAYOUT, height=300,
        title=dict(text="Monthly Actual vs Budgeted Cost (₹)", font=dict(color="#c9d1d9", size=13)),
    )
    st.plotly_chart(fig_area, use_container_width=True)

    worst_stage = (
        df_sv.sort_values("Variance %", ascending=False).iloc[0]["Stage"]
        if not df_sv.empty else "N/A"
    )
    render_ai_recommendations_section(
        context={
            "total_actual":   fmt_currency(actual_cost),
            "total_budget":   fmt_currency(budgeted_cost),
            "variance_pct":   variance_pct * 100,
            "cpu":            fmt_currency(cpu),
            "units_produced": units_produced,
            "scrap_count":    scrap_count,
            "worst_stage":    worst_stage,
        },
        page_type="executive", section_key="exec",
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 – WASTE & QUALITY
# ══════════════════════════════════════════════════════════════════════
elif page == "♻️ Waste & Quality":

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown(
            kpi_card("Waste Dollars (Scrap)", fmt_currency(waste_dollars), "Scrap × Avg Unit Cost",
                     "bad" if waste_dollars > 500_000 else "neutral", "red"),
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown(
            kpi_card("Rework Financial Leak", fmt_currency(rework_leak), "Parts lost + Labor lost",
                     "bad" if rework_leak > 500_000 else "neutral", "yellow"),
            unsafe_allow_html=True,
        )
    with w3:
        st.markdown(
            kpi_card("Scrap Units", f"{scrap_count:,}", f"Rate: {scrap_rate:.1f}%",
                     "bad" if scrap_rate > 8 else "neutral"),
            unsafe_allow_html=True,
        )
    with w4:
        avg_scrap = pl_f["scrap_count"].mean() if not pl_f.empty else 0
        st.markdown(
            kpi_card("Avg Scrap / Log", f"{avg_scrap:.1f}", "Units scrapped per entry",
                     "bad" if avg_scrap > 25 else "good"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    with d1:
        section("🍩 REWORK REASON DISTRIBUTION")
        rr_reason = rr_f.groupby("rework_reason").agg(
            total_cost=pd.NamedAgg("parts_cost_lost", "sum"),
            hours_lost=pd.NamedAgg("man_hours_lost",  "sum"),
        ).reset_index()
        rr_reason["labor_cost"] = rr_reason["hours_lost"] * LABOR_RATE
        rr_reason["total"]      = rr_reason["total_cost"] + rr_reason["labor_cost"]
        if not rr_reason.empty:
            fig_donut = go.Figure(go.Pie(
                labels=rr_reason["rework_reason"], values=rr_reason["total"], hole=0.55,
                marker=dict(colors=COLORS[:len(rr_reason)], line=dict(color="#0d1117", width=2)),
                textinfo="label+percent", textfont=dict(size=11, color="#e6edf3"),
                hovertemplate="<b>%{label}</b><br>Cost: ₹%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            fig_donut.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"), height=340,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(bgcolor="#1c2128", bordercolor="#30363d", borderwidth=1),
                title=dict(text="Rework Cost by Reason (₹)", font=dict(color="#c9d1d9", size=13)),
                annotations=[dict(text=f"<b>{fmt_currency(rework_leak)}</b>",
                                  font=dict(size=16, color="#f0f6fc"), showarrow=False)],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    with d2:
        section("🗺️ WASTE ₹ TREEMAP — STAGE → REASON")
        rr_stage = rr_f.merge(bm[["stage_id","stage_name"]], on="stage_id", how="left")
        rr_stage["total_rework"] = rr_stage["parts_cost_lost"] + rr_stage["man_hours_lost"] * LABOR_RATE
        treemap_data = (
            rr_stage.groupby(["stage_name","rework_reason"])["total_rework"]
            .sum().reset_index()
        )
        treemap_data = treemap_data[treemap_data["total_rework"] > 0]
        if not treemap_data.empty:
            fig_tree = px.treemap(
                treemap_data, path=["stage_name","rework_reason"],
                values="total_rework", color="total_rework",
                color_continuous_scale=[[0,"#0d2a1a"],[0.5,"#d29922"],[1,"#f85149"]],
            )
            fig_tree.update_traces(
                texttemplate="<b>%{label}</b><br>₹%{value:,.0f}",
                hovertemplate="<b>%{label}</b><br>Rework: ₹%{value:,.0f}<extra></extra>",
            )
            fig_tree.update_layout(
                paper_bgcolor="#0d1117", font_color="#c9d1d9",
                height=340, margin=dict(l=0, r=0, t=50, b=0),
                title=dict(text="Waste ₹: Stage → Reason", font=dict(color="#c9d1d9", size=13)),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_tree, use_container_width=True)

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
            labels={"payroll_hours":"Labor Hours","scrap_count":"Scrap Count","stage_name":"Stage"},
        )
        fig_sc.update_layout(
            **DARK_LAYOUT, height=360,
            title=dict(text="Scrap Count vs Labor Hours  (bubble = units produced)",
                       font=dict(color="#c9d1d9", size=13)),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    section("📋 REWORK REGISTRY — TOP 20 ENTRIES")
    if not rr_f.empty:
        rr_display = (
            rr_f.merge(bm[["stage_id","stage_name"]], on="stage_id", how="left")
                .merge(dt_f[["date_key","date"]], on="date_key", how="left")
        )
        rr_display["rework_cost"] = rr_display["parts_cost_lost"] + rr_display["man_hours_lost"] * LABOR_RATE
        rd = rr_display[
            ["date","stage_name","rework_reason","man_hours_lost","parts_cost_lost","rework_cost"]
        ].copy()
        rd["date"]            = pd.to_datetime(rd["date"]).dt.strftime("%Y-%m-%d")
        rd["rework_cost"]     = rd["rework_cost"].map("₹{:,.0f}".format)
        rd["parts_cost_lost"] = rd["parts_cost_lost"].map("₹{:,.0f}".format)
        rd.columns = ["Date","Stage","Reason","Hours Lost","Parts Cost","Rework Cost"]
        rd["Action"] = rd["Rework Cost"].apply(
            lambda v: "🔴 Fix Immediately"
            if float(v.replace("₹","").replace(",","")) > 2000 else "🟡 Monitor"
        )
        st.dataframe(rd.head(20), use_container_width=True, hide_index=True)

    top_reason = (
        rr_f.groupby("rework_reason")
             .apply(lambda g: g["parts_cost_lost"].sum() + g["man_hours_lost"].sum() * LABOR_RATE)
             .idxmax()
        if not rr_f.empty else "N/A"
    )
    worst_waste_stage = (
        rr_f.merge(bm[["stage_id","stage_name"]], on="stage_id", how="left")
            .assign(total_rework=lambda d: d["parts_cost_lost"] + d["man_hours_lost"] * LABOR_RATE)
            .groupby("stage_name")["total_rework"].sum()
            .sort_values(ascending=False).index[0]
        if not rr_f.empty else "N/A"
    )

    render_ai_recommendations_section(
        context={
            "waste_dollars":     fmt_currency(waste_dollars),
            "rework_leak":       fmt_currency(rework_leak),
            "scrap_count":       scrap_count,
            "scrap_rate":        scrap_rate,
            "top_reason":        top_reason,
            "worst_waste_stage": worst_waste_stage,
        },
        page_type="waste", section_key="waste",
    )


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
            a_cost = (
                ac_s["material_cost"].sum()
                + ac_s["utility_cost"].sum()
                + ac_s["payroll_hours"].sum() * LABOR_RATE
            )
            b_cost = pl_s["units_produced"].sum() * s_row["budgeted_unit_cost"]
            var    = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
            hm_data.append({
                "Stage": s_row["stage_name"], "Shift": sh_row["shift_name"],
                "Variance %": round(var, 1), "actual": a_cost, "budget": b_cost,
            })

    df_hm  = pd.DataFrame(hm_data)
    pivot  = df_hm.pivot(index="Stage", columns="Shift", values="Variance %")
    z_vals = pivot.values
    x_labs = pivot.columns.tolist()
    y_labs = pivot.index.tolist()

    colorscale = [
        [0.00,"#0d47a1"],[0.20,"#1976d2"],[0.38,"#4fc3f7"],
        [0.48,"#e0f7fa"],[0.52,"#fff9c4"],[0.62,"#ffb300"],
        [0.75,"#e53935"],[0.88,"#b71c1c"],[1.00,"#4a0000"],
    ]

    annotations = []
    for i, yl in enumerate(y_labs):
        for j, xl in enumerate(x_labs):
            val = z_vals[i][j]
            if np.isnan(val):
                continue
            txt_color = (
                "#ffffff" if val <= -50
                else ("#0d1117" if val <= 5
                      else ("#1a1a1a" if val <= 80 else "#ffffff"))
            )
            annotations.append(dict(
                x=xl, y=yl, text=f"<b>{val:+.1f}%</b>",
                font=dict(size=14, color=txt_color), showarrow=False,
            ))

    fig_hm = go.Figure(go.Heatmap(
        z=z_vals, x=x_labs, y=y_labs,
        colorscale=colorscale, zmid=0, zmin=-300, zmax=300,
        text=[[f"{v:+.1f}%" for v in row] for row in z_vals],
        hovertemplate="Stage: <b>%{y}</b><br>Shift: <b>%{x}</b><br>Variance: <b>%{text}</b><extra></extra>",
        colorbar=dict(
            title=dict(text="Variance %", font=dict(color="#c9d1d9", size=13)),
            tickfont=dict(color="#c9d1d9"),
            tickvals=[-300,-200,-100,0,100,200,300],
            ticktext=["-300%","-200%","-100%","0%","+100%","+200%","+300%"],
            bgcolor="#161b22", bordercolor="#30363d", thickness=18,
        ),
    ))
    fig_hm.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="Segoe UI"),
        xaxis=dict(title="Shift",  tickfont=dict(color="#c9d1d9", size=13), side="bottom", showgrid=False),
        yaxis=dict(title="Stage",  tickfont=dict(color="#c9d1d9", size=13), autorange="reversed", showgrid=False),
        annotations=annotations, height=480,
        margin=dict(l=160, r=80, t=60, b=60),
        title=dict(text="Variance % by Stage × Shift", font=dict(color="#f0f6fc", size=15)),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    section("🏭 MACHINE × SHIFT PRODUCTION VOLUME")
    pl_mc = (
        pl_f.merge(mc_f[["machine_id","machine_name","production_line"]], on="machine_id", how="left")
            .merge(sh_f[["shift_id","shift_name"]], on="shift_id", how="left")
    )
    mc_hm    = pl_mc.groupby(["machine_name","shift_name"])["units_produced"].sum().reset_index()
    mc_pivot = mc_hm.pivot(index="machine_name", columns="shift_name", values="units_produced").fillna(0)

    fig_mc = go.Figure(go.Heatmap(
        z=mc_pivot.values, x=mc_pivot.columns.tolist(), y=mc_pivot.index.tolist(),
        colorscale="Blues",
        text=mc_pivot.values.astype(int), texttemplate="%{text}", textfont={"size": 10},
        colorbar=dict(title="Units", tickfont=dict(color="#9aa0b5")),
    ))
    fig_mc.update_layout(
        **DARK_LAYOUT, height=500,
        title=dict(text="Units Produced by Machine × Shift", font=dict(color="#c9d1d9", size=13)),
        xaxis_title="Shift", yaxis_title="Machine",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    section("📋 STAGE PERFORMANCE SUMMARY")
    summary_rows = []
    for _, row in bm_f.iterrows():
        sid    = row["stage_id"]
        ac_s   = ac_f[ac_f["stage_id"] == sid]
        pl_s   = pl_f[pl_f["stage_id"] == sid]
        a_cost = (
            ac_s["material_cost"].sum()
            + ac_s["utility_cost"].sum()
            + ac_s["payroll_hours"].sum() * LABOR_RATE
        )
        b_cost = pl_s["units_produced"].sum() * row["budgeted_unit_cost"]
        var    = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
        summary_rows.append({
            "Stage":          row["stage_name"],
            "Category":       row.get("category",""),
            "Actual Cost":    fmt_currency(a_cost),
            "Budget":         fmt_currency(b_cost),
            "Variance %":     f"{var:+.1f}%",
            "Units Produced": f"{pl_s['units_produced'].sum():,}",
            "Scrap Units":    f"{pl_s['scrap_count'].sum():,}",
            "Status":         "🔴 Over" if var > 10 else ("🟡 Warning" if var > 0 else "🟢 Good"),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    worst_combo = (
        df_hm.sort_values("Variance %", ascending=False)
             .apply(lambda r: f"{r['Stage']} × {r['Shift']}", axis=1).iloc[0]
        if not df_hm.empty else "N/A"
    )
    shift_v = df_hm.groupby("Shift").agg(actual=("actual","sum"), budget=("budget","sum")).reset_index()
    shift_v["var_pct"] = (shift_v["actual"]-shift_v["budget"]) / shift_v["budget"].replace(0,np.nan) * 100
    worst_shift = shift_v.sort_values("var_pct", ascending=False).iloc[0]["Shift"] if not shift_v.empty else "N/A"

    stage_v = df_hm.groupby("Stage").agg(actual=("actual","sum"), budget=("budget","sum")).reset_index()
    stage_v["var_pct"] = (stage_v["actual"]-stage_v["budget"]) / stage_v["budget"].replace(0,np.nan) * 100
    worst_stage = stage_v.sort_values("var_pct", ascending=False).iloc[0]["Stage"] if not stage_v.empty else "N/A"

    render_ai_recommendations_section(
        context={
            "worst_combo":  worst_combo, "worst_stage": worst_stage, "worst_shift": worst_shift,
            "variance_pct": variance_pct * 100,
            "total_actual": fmt_currency(actual_cost), "total_budget": fmt_currency(budgeted_cost),
        },
        page_type="heatmap", section_key="heatmap",
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 – AI CHATBOT
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":
    st.markdown(
        '<div class="page-banner">'
        "<h1>🤖 AI Analytics Chatbot</h1>"
        "<p>⚡ Powered by Groq · llama-3.3-70b-versatile · Ask anything about your manufacturing data</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Pre-compute metrics for chatbot context
    stage_var2 = []
    for _, row in bm_f.iterrows():
        sid    = row["stage_id"]
        ac_s   = ac_f[ac_f["stage_id"] == sid]
        pl_s   = pl_f[pl_f["stage_id"] == sid]
        a_cost = (
            ac_s["material_cost"].sum()
            + ac_s["utility_cost"].sum()
            + ac_s["payroll_hours"].sum() * LABOR_RATE
        )
        b_cost = pl_s["units_produced"].sum() * row["budgeted_unit_cost"]
        var    = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
        stage_var2.append({"stage_name": row["stage_name"], "var_pct": var})
    sv2 = pd.DataFrame(stage_var2)

    worst_stage_c = sv2.sort_values("var_pct", ascending=False).iloc[0]["stage_name"] if not sv2.empty else "N/A"
    best_stage_c  = sv2.sort_values("var_pct").iloc[0]["stage_name"]                  if not sv2.empty else "N/A"
    top_reason_c  = (
        rr_f.groupby("rework_reason")
             .apply(lambda g: g["parts_cost_lost"].sum() + g["man_hours_lost"].sum() * LABOR_RATE)
             .idxmax()
        if not rr_f.empty else "N/A"
    )

    shift_hm = []
    for _, sh_row in sh_f.iterrows():
        pl_s   = pl_f[pl_f["shift_id"] == sh_row["shift_id"]]
        b_cost = sum(
            pl_s[pl_s["stage_id"] == r.stage_id]["units_produced"].sum() * r.budgeted_unit_cost
            for r in bm_f.itertuples()
        )
        a_cost = (
            (ac_f["material_cost"].sum() + ac_f["utility_cost"].sum() + ac_f["payroll_hours"].sum() * LABOR_RATE)
            * (len(pl_s) / max(len(pl_f), 1))
        )
        var = (a_cost - b_cost) / b_cost * 100 if b_cost else 0
        shift_hm.append({"shift_name": sh_row["shift_name"], "var_pct": var})
    shift_df      = pd.DataFrame(shift_hm)
    worst_shift_c = (
        shift_df.sort_values("var_pct", ascending=False).iloc[0]["shift_name"]
        if not shift_df.empty else "N/A"
    )

    # Groq system prompt — injects live dashboard numbers
    DASHBOARD_SYSTEM = (
        "You are an expert manufacturing analyst assistant for PreciCost Intelligence dashboard.\n"
        "Answer questions about this manufacturing data concisely. Use ₹ for Indian Rupees.\n"
        "Format important numbers in bold. Use bullet points for lists.\n\n"
        "Current Dashboard Data:\n"
        f"- Total Actual Cost: {fmt_currency(actual_cost)}\n"
        f"- Total Budgeted Cost: {fmt_currency(budgeted_cost)}\n"
        f"- Overall Variance: {variance_pct*100:+.1f}%\n"
        f"- Cost Per Unit (good units): {fmt_currency(cpu)}\n"
        f"- Total Units Produced: {units_produced:,}\n"
        f"- Scrap Count: {scrap_count:,}\n"
        f"- Scrap Rate: {scrap_rate:.1f}%\n"
        f"- Total Waste (₹): {fmt_currency(waste_dollars)}\n"
        f"- Rework Financial Leak: {fmt_currency(rework_leak)}\n"
        f"- Worst Variance Stage: {worst_stage_c}\n"
        f"- Best Performing Stage: {best_stage_c}\n"
        f"- Top Rework Reason: {top_reason_c}\n"
        f"- Worst Shift: {worst_shift_c}\n\n"
        "Stage Variances:\n"
        + "\n".join(f"  - {r['stage_name']}: {r['var_pct']:+.1f}%" for _, r in sv2.iterrows())
    )

    def generate_response_groq(user_msg: str) -> str:
        return call_groq(DASHBOARD_SYSTEM, user_msg, max_tokens=400)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": (
            "👋 **Welcome to the PreciCost Analytics Chatbot!**\n\n"
            "⚡ Powered by **Groq** (llama-3.3-70b-versatile) — ultra-fast AI responses.\n\n"
            "I can answer questions about production costs, variance, waste, rework, and more.\n\n"
            "Try asking: *'Which stage has highest variance?'* or *'Give recommendations'*"
        )}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="⚡" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    st.markdown("**⚡ Quick Questions:**")
    quick_qs = [
        "Which stage has highest variance?",
        "Give me a summary",
        "Give recommendations",
        "Show waste analysis",
    ]
    q_cols = st.columns(4)
    for i, qc in enumerate(q_cols):
        with qc:
            if st.button(quick_qs[i], key=f"qb_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": quick_qs[i]})
                with st.spinner("⚡ Groq is thinking…"):
                    answer = generate_response_groq(quick_qs[i])
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    if prompt := st.chat_input("Ask anything about your manufacturing data…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("⚡ Groq is thinking…"):
            answer = generate_response_groq(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
