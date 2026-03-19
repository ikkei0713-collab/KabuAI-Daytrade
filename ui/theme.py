"""KabuAI-Daytrade dark theme styling for Streamlit."""

import streamlit as st


def load_theme() -> None:
    """Inject custom CSS for the professional dark trading terminal theme."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


_THEME_CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────── */
:root {
    --bg-primary:   #0e1117;
    --bg-secondary: #1a1a2e;
    --bg-card:      #16213e;
    --bg-hover:     #1f2b47;
    --border:       #2a2a4a;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted:   #64748b;
    --accent:       #6366f1;
    --profit:       #00d4aa;
    --loss:         #ff4757;
    --warning:      #f59e0b;
    --info:         #3b82f6;
}

/* Force dark background everywhere */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: var(--bg-primary) !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
}

/* ── Typography ───────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--text-primary);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── KPI / Metric Cards ───────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
}

[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary) !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetricDelta"] svg { display: none; }

[data-testid="stMetricDelta"][style*="color: rgb(9, 171, 59)"],
[data-testid="stMetricDelta"] span[style*="color: rgb(9, 171, 59)"] {
    color: var(--profit) !important;
}

[data-testid="stMetricDelta"][style*="color: rgb(255, 43, 43)"],
[data-testid="stMetricDelta"] span[style*="color: rgb(255, 43, 43)"] {
    color: var(--loss) !important;
}

/* ── Card container utility ───────────────────────────────── */
div[data-testid="stVerticalBlock"] > div.card-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}

/* ── Custom card class (via markdown) ─────────────────────── */
.card {
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

.card-header {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.card-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text-primary);
}

.card-value.profit { color: var(--profit); }
.card-value.loss   { color: var(--loss); }

.card-delta {
    font-size: 0.85rem;
    margin-top: 4px;
}

.card-delta.profit { color: var(--profit); }
.card-delta.loss   { color: var(--loss); }

/* ── Tables / DataFrames ──────────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: 8px;
    overflow: hidden;
}

[data-testid="stDataFrame"] table {
    background-color: var(--bg-secondary) !important;
}

[data-testid="stDataFrame"] th {
    background-color: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em;
    border-bottom: 2px solid var(--border) !important;
}

[data-testid="stDataFrame"] td {
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
}

/* Danger button override */
.danger-btn .stButton > button {
    background: linear-gradient(135deg, var(--loss), #dc2626) !important;
    box-shadow: 0 2px 8px rgba(255, 71, 87, 0.25) !important;
}

/* Success button override */
.success-btn .stButton > button {
    background: linear-gradient(135deg, var(--profit), #059669) !important;
    box-shadow: 0 2px 8px rgba(0, 212, 170, 0.25) !important;
}

/* ── Selectbox / Inputs ───────────────────────────────────── */
[data-testid="stSelectbox"],
[data-testid="stDateInput"],
[data-testid="stNumberInput"],
[data-testid="stTextInput"] {
    color: var(--text-primary);
}

.stSelectbox > div > div,
.stDateInput > div > div,
.stTextInput > div > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* ── Status badges ────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-active  { background: rgba(0,212,170,0.15); color: var(--profit); }
.badge-inactive { background: rgba(100,116,139,0.2); color: var(--text-muted); }
.badge-pending { background: rgba(245,158,11,0.15); color: var(--warning); }
.badge-filled  { background: rgba(59,130,246,0.15); color: var(--info); }
.badge-rejected { background: rgba(255,71,87,0.15); color: var(--loss); }

/* ── Dividers ─────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Sidebar branding ─────────────────────────────────────── */
.sidebar-brand {
    text-align: center;
    padding: 16px 0 24px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}

.sidebar-brand h2 {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--profit), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.sidebar-brand p {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 4px 0 0 0;
}

/* ── Section headers ──────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
}

.section-header h3 {
    margin: 0 !important;
    font-size: 1.1rem !important;
}

/* ── System status dots ───────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

.status-dot.online  { background: var(--profit); box-shadow: 0 0 6px var(--profit); }
.status-dot.offline { background: var(--loss);   box-shadow: 0 0 6px var(--loss); }
.status-dot.warning { background: var(--warning); box-shadow: 0 0 6px var(--warning); }

/* ── Plotly chart container ───────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* ── Toast / Alert overrides ──────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
    border-left: 4px solid var(--accent) !important;
}

/* ── Hide Streamlit footer & menu ─────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
"""
