"""
SentimentIQ — Political Sentiment Intelligence Platform
app.py  |  Main Streamlit application

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentimentIQ — Political Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — dark navy / purple design system
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #d1d5db;
}

/* ---------- Background ---------- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, section.main, .block-container,
[data-testid="block-container"] {
    background-color: #0a0e1a !important;
    max-width: 1400px;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: #0a0e1a !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1529 0%, #111827 100%) !important;
    border-right: 1px solid rgba(167,139,250,0.12) !important;
}
[data-testid="stSidebar"] * { color: #c4b5fd !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stTextArea label { color: #a78bfa !important; font-weight: 600; }

/* ---------- Hide streamlit chrome ---------- */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #0f1529 0%, #161f3d 50%, #0f1529 100%);
    border-radius: 20px;
    padding: 48px 44px 40px;
    margin-bottom: 32px;
    border: 1px solid rgba(167,139,250,0.20);
    box-shadow: 0 8px 48px rgba(0,0,0,0.7), 0 0 80px rgba(109,40,217,0.06);
    text-align: center;
}
.hero h1 {
    font-size: 2.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #818cf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    letter-spacing: -0.02em;
}
.hero .subtitle {
    color: #6b7280;
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0 auto 18px;
    line-height: 1.6;
}
.hero .tagline {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 999px;
    background: rgba(167,139,250,0.10);
    border: 1px solid rgba(167,139,250,0.25);
    color: #a78bfa;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ---------- Section title ---------- */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #a78bfa;
    margin: 32px 0 16px;
    border-left: 3px solid #6d28d9;
    padding-left: 12px;
    letter-spacing: 0.02em;
}

/* ---------- Metric cards ---------- */
.card-grid { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 130px;
    background: linear-gradient(135deg, #111827 0%, #131b2e 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(167,139,250,0.25);
    box-shadow: 0 8px 32px rgba(109,40,217,0.12);
}
.metric-value { font-size: 2.6rem; font-weight: 800; line-height: 1; margin-bottom: 6px; }
.metric-pct   { font-size: 1.0rem; font-weight: 600; opacity: 0.75; margin-bottom: 4px; }
.metric-label { font-size: 0.75rem; color: #6b7280; font-weight: 600;
                letter-spacing: 0.09em; text-transform: uppercase; }
.c-total    { color: #818cf8; }
.c-positive { color: #34d399; }
.c-negative { color: #f87171; }
.c-neutral  { color: #fbbf24; }

/* ---------- Post cards ---------- */
.post-card {
    background: linear-gradient(135deg, #111827 0%, #131b2e 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}
.post-card:hover { border-color: rgba(167,139,250,0.25); }
.post-text { font-size: 0.96rem; color: #e2e8f0; margin-bottom: 12px; line-height: 1.6; }
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 10px;
}
.badge-Positive { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.28); }
.badge-Negative { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.28); }
.badge-Neutral  { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.28); }

.meta-row { display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.79rem; color: #6b7280; align-items: center; }
.meta-item span { font-weight: 600; color: #9ca3af; }
.prob-bar-wrap { margin-top: 10px; }
.prob-label { font-size: 0.75rem; color: #6b7280; margin-bottom: 3px; }
.prob-bar-outer { background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; width: 100%; }
.prob-bar-inner { border-radius: 4px; height: 6px; }

/* ---------- Info / Warning boxes ---------- */
.info-box {
    background: rgba(129,140,248,0.08);
    border: 1px solid rgba(129,140,248,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    color: #818cf8;
    font-size: 0.92rem;
    margin-bottom: 16px;
}
.warn-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    color: #fbbf24;
    font-size: 0.92rem;
    margin-bottom: 16px;
}
.err-box {
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    color: #f87171;
    font-size: 0.92rem;
    margin-bottom: 16px;
}
.success-box {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    color: #34d399;
    font-size: 0.92rem;
    margin-bottom: 16px;
}
.demo-banner {
    background: linear-gradient(90deg, rgba(251,191,36,0.10), rgba(251,191,36,0.05));
    border: 1px solid rgba(251,191,36,0.35);
    border-radius: 12px;
    padding: 14px 20px;
    color: #fbbf24;
    font-size: 0.9rem;
    margin-bottom: 20px;
    text-align: center;
    font-weight: 600;
}

/* ---------- Metric info table ---------- */
.metric-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.metric-table th {
    color: #a78bfa; font-weight: 700; text-align: left;
    padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.06);
}
.metric-table td {
    color: #d1d5db; padding: 8px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.metric-table tr:last-child td { border-bottom: none; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.92rem;
    padding: 0.55rem 1.6rem;
    transition: opacity 0.2s, transform 0.15s;
    letter-spacing: 0.02em;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* ---------- Slider & inputs ---------- */
.stSlider > div > div > div { background: #6d28d9 !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #111827 !important;
    border: 1px solid rgba(167,139,250,0.20) !important;
    color: #d1d5db !important;
    border-radius: 8px !important;
}
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid rgba(167,139,250,0.20) !important;
    color: #d1d5db !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------
from src import x_api, predict, visualizations
from src.preprocessing import preprocess_dataframe
from data.demo_posts import DEMO_POSTS


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "results_df":     None,
        "fetch_meta":     None,   # dict with query, fetched_at, source
        "model_metadata": None,
        "data_source":    "demo", # "live" | "demo"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _posts_to_df(posts: list) -> pd.DataFrame:
    """Convert a list of XPost (or demo dict) objects to a DataFrame."""
    rows = []
    for p in posts:
        if hasattr(p, "post_id"):          # XPost dataclass
            rows.append({
                "post_id":    p.post_id,
                "text":       p.text,
                "created_at": p.created_at,
                "likes":      p.likes,
                "retweets":   p.retweets,
                "replies":    p.replies,
                "lang":       p.lang,
                "url":        p.url,
            })
        else:                               # demo dict
            created = p.get("created_at")
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    created = None
            rows.append({
                "post_id":    p.get("post_id", ""),
                "text":       p.get("text", ""),
                "created_at": created,
                "likes":      p.get("likes", 0),
                "retweets":   p.get("retweets", 0),
                "replies":    p.get("replies", 0),
                "lang":       p.get("lang", "en"),
                "url":        f"https://x.com/i/web/status/{p.get('post_id', '')}",
            })
    return pd.DataFrame(rows)


def _run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment predictions to the dataframe. Raises FileNotFoundError if no model."""
    pred_df = predict.predict_batch(df["text"].tolist())
    for col in ["predicted_sentiment", "prob_negative", "prob_neutral", "prob_positive"]:
        df[col] = pred_df[col].values
    return df


def _df_to_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _pct(num, total) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * num / total:.1f}%"


def _prob_bar(label: str, prob: float, color: str) -> str:
    w = int(prob * 100)
    return (
        f'<div class="prob-bar-wrap">'
        f'<div class="prob-label">{label}: {prob:.1%}</div>'
        f'<div class="prob-bar-outer">'
        f'<div class="prob-bar-inner" style="width:{w}%;background:{color};"></div>'
        f'</div></div>'
    )


SENTIMENT_COLORS = {
    "Positive": "#34d399",
    "Negative": "#f87171",
    "Neutral":  "#fbbf24",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Controls")
    st.markdown("---")

    # Data source selector
    data_source = st.selectbox(
        "Data Source",
        options=["Demo Mode (sample data)", "Live X API"],
        index=0 if st.session_state.data_source == "demo" else 1,
        help="Live mode requires a valid X_BEARER_TOKEN in your .env file.",
        key="sb_data_source",
    )
    is_live = "Live" in data_source
    st.session_state.data_source = "live" if is_live else "demo"

    st.markdown("---")

    if is_live:
        st.markdown("**🔍 Search Settings**")
        query = st.text_input(
            "Search Query",
            value="election OR democracy OR politics OR government",
            key="sb_query",
            help="Use X query operators. -is:retweet and lang:en are added automatically.",
        )
        max_results = st.slider(
            "Max Results",
            min_value=10, max_value=100, value=20, step=10,
            key="sb_max_results",
            help="Free tier: up to 10 posts per API call. Basic tier: up to 100.",
        )
    else:
        st.markdown("**📂 Demo Mode**")
        st.markdown(
            '<div style="font-size:0.82rem; color:#6b7280; line-height:1.6;">'
            "Demo mode uses 15 pre-written sample posts to demonstrate all "
            "dashboard features without an X API key."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Sentiment filter (post-collection)
    sentiment_filter = st.selectbox(
        "Filter Posts by Sentiment",
        options=["All", "Positive", "Negative", "Neutral"],
        key="sb_sentiment_filter",
    )

    st.markdown("---")
    st.markdown("**ℹ️ Model Status**")
    if predict.model_is_trained():
        st.markdown(
            '<div style="color:#34d399; font-size:0.85rem;">✅ Model artefacts found</div>',
            unsafe_allow_html=True,
        )
        meta = predict.load_metadata()
        if meta:
            st.markdown(
                f'<div style="color:#6b7280; font-size:0.78rem;">'
                f'Accuracy: <strong style="color:#a78bfa">{meta.get("accuracy", "—")}</strong><br>'
                f'Macro F1: <strong style="color:#a78bfa">{meta.get("macro_f1", "—")}</strong><br>'
                f'Trained: {meta.get("train_date", "—")[:10]}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="color:#f87171; font-size:0.85rem;">❌ Model not trained<br>'
            '<span style="color:#6b7280; font-size:0.78rem;">Run: python -m src.train_model</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem; color:#4b5563; line-height:1.6;">'
        "⚠️ Results are automated model estimates. "
        "Keyword search may introduce sampling bias. "
        "Do not interpret as population-level opinion."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <span class="tagline">Political Sentiment Intelligence</span>
    <h1>🧠 SentimentIQ</h1>
    <div class="subtitle">
        Real-time political post analysis powered by TF-IDF feature extraction
        and Logistic Regression — trained on the TweetEval sentiment dataset.
        Collect live posts via the official X API and explore automated sentiment predictions.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model readiness check
# ---------------------------------------------------------------------------
if not predict.model_is_trained():
    st.markdown("""
<div class="err-box">
    <strong>⚠️ Machine-Learning Model Not Found</strong><br><br>
    The trained classifier artefacts are missing. You must train the model before
    predictions can be made.<br><br>
    <strong>Run this command in your terminal:</strong><br>
    <code style="background:rgba(0,0,0,0.3);padding:4px 8px;border-radius:6px;">
    python -m src.train_model
    </code><br><br>
    This downloads the TweetEval sentiment dataset, trains a TF-IDF + Logistic Regression
    classifier, and saves the artefacts to <code>models/</code>.
    It typically takes 1–3 minutes.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API token status banner (live mode)
# ---------------------------------------------------------------------------
if is_live and not x_api.bearer_token_configured():
    st.markdown("""
<div class="err-box">
    <strong>🔑 X API Token Not Configured</strong><br><br>
    To fetch live posts you need a valid X Developer bearer token.<br><br>
    <strong>Steps:</strong><br>
    1. Go to <a href="https://developer.x.com/en/portal/dashboard" style="color:#f87171">developer.x.com</a>
       and create or sign in to your account.<br>
    2. Create a new App in the Developer Portal.<br>
    3. Copy the <strong>Bearer Token</strong> from the "Keys and Tokens" page.<br>
    4. Create a file named <code>.env</code> in the project root (see <code>.env.example</code>).<br>
    5. Add: <code>X_BEARER_TOKEN=your_actual_token_here</code><br>
    6. Restart the Streamlit application.<br><br>
    ⚠️ Switch to <strong>Demo Mode</strong> in the sidebar to explore the dashboard
    without an API token.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Collect / Load button
# ---------------------------------------------------------------------------
col_btn, col_refresh = st.columns([2, 10])
with col_btn:
    if is_live:
        fetch_label = "🔴 Fetch Live Posts"
    else:
        fetch_label = "▶️  Load Demo Data"
    run_clicked = st.button(fetch_label, key="btn_run")

with col_refresh:
    if st.session_state.results_df is not None:
        if st.button("🔄 Clear & Reset", key="btn_reset"):
            st.session_state.results_df = None
            st.session_state.fetch_meta = None
            st.rerun()


# ---------------------------------------------------------------------------
# Data collection pipeline
# ---------------------------------------------------------------------------
if run_clicked:
    if not predict.model_is_trained():
        st.markdown(
            '<div class="err-box">❌ Cannot analyse posts — model artefacts not found. '
            'Run <code>python -m src.train_model</code> first.</div>',
            unsafe_allow_html=True,
        )
    elif is_live and not x_api.bearer_token_configured():
        st.markdown(
            '<div class="err-box">❌ X_BEARER_TOKEN not configured. '
            'See the configuration guide above.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Collecting posts …" if is_live else "Loading demo data …"):
            # ---- Live mode ----
            if is_live:
                result = x_api.fetch_recent_posts(
                    query=st.session_state.sb_query,
                    max_results=st.session_state.sb_max_results,
                )
                if result.error:
                    st.markdown(
                        f'<div class="err-box">❌ <strong>X API Error</strong><br>{result.error}</div>',
                        unsafe_allow_html=True,
                    )
                elif not result.posts:
                    st.markdown(
                        '<div class="warn-box">⚠️ The search returned no posts. '
                        'Try a different query or broader keywords.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    raw_df = _posts_to_df(result.posts)
                    meta = {
                        "source":     "live",
                        "query":      result.query,
                        "fetched_at": result.fetched_at,
                        "n_raw":      result.total_fetched,
                        "warning":    result.warning,
                    }
                    if result.warning:
                        st.markdown(
                            f'<div class="warn-box">{result.warning}</div>',
                            unsafe_allow_html=True,
                        )

                    with st.spinner("Running sentiment predictions …"):
                        try:
                            df = _run_predictions(raw_df)
                            st.session_state.results_df = df
                            st.session_state.fetch_meta = meta
                        except FileNotFoundError as exc:
                            st.markdown(
                                f'<div class="err-box">❌ {exc}</div>',
                                unsafe_allow_html=True,
                            )

            # ---- Demo mode ----
            else:
                raw_df = _posts_to_df(DEMO_POSTS)
                meta = {
                    "source":     "demo",
                    "query":      "N/A — Demo Mode",
                    "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "n_raw":      len(DEMO_POSTS),
                    "warning":    None,
                }
                with st.spinner("Running sentiment predictions …"):
                    try:
                        df = _run_predictions(raw_df)
                        st.session_state.results_df = df
                        st.session_state.fetch_meta = meta
                    except FileNotFoundError as exc:
                        st.markdown(
                            f'<div class="err-box">❌ {exc}</div>',
                            unsafe_allow_html=True,
                        )


# ---------------------------------------------------------------------------
# Dashboard — render results
# ---------------------------------------------------------------------------
if st.session_state.results_df is not None:
    df: pd.DataFrame = st.session_state.results_df.copy()
    meta: dict = st.session_state.fetch_meta or {}

    # ---- Demo banner ----
    if meta.get("source") == "demo":
        st.markdown(
            '⚠️ <strong>DEMO MODE</strong> — The posts below are synthetic sample data '
            'for demonstration purposes only. They are NOT real X posts.',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="demo-banner">'
            '⚠️  DEMO MODE — These are synthetic sample posts, NOT real X data.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ---- Collection metadata ----
    st.markdown('<div class="section-title">📡 Collection Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Source</div>'
            f'<div style="color:#818cf8;font-weight:700;font-size:1rem;margin-top:8px;">'
            f'{"🔴 Live X API" if meta.get("source") == "live" else "🟡 Demo Mode"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Fetched At</div>'
            f'<div style="color:#818cf8;font-weight:700;font-size:0.9rem;margin-top:8px;">'
            f'{meta.get("fetched_at", "—")}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Query Used</div>'
            f'<div style="color:#818cf8;font-weight:600;font-size:0.82rem;margin-top:8px;word-break:break-word;">'
            f'{meta.get("query", "—")[:120]}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ---- Sentiment counts ----
    total   = len(df)
    pos_cnt = (df["predicted_sentiment"] == "Positive").sum()
    neg_cnt = (df["predicted_sentiment"] == "Negative").sum()
    neu_cnt = (df["predicted_sentiment"] == "Neutral").sum()

    st.markdown('<div class="section-title">📊 Overview</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="card-grid">
    <div class="metric-card">
        <div class="metric-value c-total">{total}</div>
        <div class="metric-label">Total Posts</div>
    </div>
    <div class="metric-card">
        <div class="metric-value c-positive">{pos_cnt}</div>
        <div class="metric-pct c-positive">{_pct(pos_cnt, total)}</div>
        <div class="metric-label">Positive</div>
    </div>
    <div class="metric-card">
        <div class="metric-value c-negative">{neg_cnt}</div>
        <div class="metric-pct c-negative">{_pct(neg_cnt, total)}</div>
        <div class="metric-label">Negative</div>
    </div>
    <div class="metric-card">
        <div class="metric-value c-neutral">{neu_cnt}</div>
        <div class="metric-pct c-neutral">{_pct(neu_cnt, total)}</div>
        <div class="metric-label">Neutral</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ---- Charts ----
    st.markdown('<div class="section-title">📈 Sentiment Visualizations</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1, 1], gap="large")
    counts = {"Positive": int(pos_cnt), "Negative": int(neg_cnt), "Neutral": int(neu_cnt)}

    with chart_col1:
        st.markdown("**Sentiment Distribution**")
        if any(v > 0 for v in counts.values()):
            st.plotly_chart(
                visualizations.donut_chart(counts),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with chart_col2:
        st.markdown("**Sentiment Counts**")
        st.plotly_chart(
            visualizations.bar_chart(counts),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Trend chart (full width)
    st.markdown("**Sentiment Trend Over Time**")
    st.plotly_chart(
        visualizations.trend_chart(df),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ---- Post Explorer ----
    st.markdown('<div class="section-title">🔍 Political Post Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">🤖 Sentiment labels below are <strong>automated model estimates</strong> '
        'produced by the TF-IDF + Logistic Regression classifier. '
        'They may not perfectly reflect real-world sentiment, especially for sarcasm, '
        'mixed opinions, or regional political idioms.</div>',
        unsafe_allow_html=True,
    )

    # Apply sidebar filter
    display_df = df.copy()
    if sentiment_filter != "All":
        display_df = display_df[display_df["predicted_sentiment"] == sentiment_filter]

    if display_df.empty:
        st.markdown(
            f'<div class="warn-box">No posts matching "{sentiment_filter}" sentiment found.</div>',
            unsafe_allow_html=True,
        )
    else:
        for _, row in display_df.iterrows():
            sentiment = row["predicted_sentiment"]
            color = SENTIMENT_COLORS.get(sentiment, "#9ca3af")
            created_str = ""
            if pd.notna(row.get("created_at")) and row["created_at"] is not None:
                try:
                    created_str = pd.to_datetime(row["created_at"]).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    created_str = str(row["created_at"])

            post_url = row.get("url", "")
            url_html = (
                f'<a href="{post_url}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#818cf8;font-size:0.78rem;">🔗 View Post</a>'
                if post_url and not row.get("post_id", "").startswith("demo_")
                else ""
            )

            prob_pos = row.get("prob_positive", 0)
            prob_neg = row.get("prob_negative", 0)
            prob_neu = row.get("prob_neutral", 0)

            st.markdown(f"""
<div class="post-card">
    <div class="badge badge-{sentiment}">{sentiment} (estimated)</div>
    <div class="post-text">"{row['text']}"</div>
    <div class="meta-row">
        <div class="meta-item">🕐 <span>{created_str or "Unknown"}</span></div>
        <div class="meta-item">❤️ <span>{int(row.get('likes', 0)):,}</span></div>
        <div class="meta-item">🔁 <span>{int(row.get('retweets', 0)):,}</span></div>
        <div class="meta-item">💬 <span>{int(row.get('replies', 0)):,}</span></div>
        {f'<div class="meta-item">{url_html}</div>' if url_html else ''}
    </div>
    <div style="margin-top:10px;">
        {_prob_bar("Positive", prob_pos, "#34d399")}
        {_prob_bar("Neutral",  prob_neu, "#fbbf24")}
        {_prob_bar("Negative", prob_neg, "#f87171")}
    </div>
</div>
""", unsafe_allow_html=True)

    # ---- Data table & export ----
    st.markdown('<div class="section-title">📋 Full Results Table</div>', unsafe_allow_html=True)
    export_cols = [
        "post_id", "text", "predicted_sentiment",
        "prob_positive", "prob_neutral", "prob_negative",
        "created_at", "likes", "retweets", "replies",
    ]
    export_df = df[[c for c in export_cols if c in df.columns]]
    st.dataframe(
        export_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="📥 Download Results as CSV",
        data=_df_to_csv(export_df),
        file_name=f"sentimentiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="btn_download",
    )


# ---------------------------------------------------------------------------
# Model Performance Section
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)

meta_loaded = predict.load_metadata()

if meta_loaded:
    st.markdown(
        '<div class="info-box">📌 These metrics were computed on a <strong>held-out test set</strong> '
        'that the model never saw during training. They reflect generalisation performance, '
        'not training performance.</div>',
        unsafe_allow_html=True,
    )

    info_col, chart_col = st.columns([1, 1], gap="large")

    with info_col:
        st.markdown("**Algorithm & Training Details**")
        st.markdown(f"""
<table class="metric-table">
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Algorithm</td><td>{meta_loaded.get("algorithm", "—")}</td></tr>
<tr><td>Dataset</td><td>{meta_loaded.get("dataset_source", "—")}</td></tr>
<tr><td>Label Mapping</td><td>0→Negative, 1→Neutral, 2→Positive</td></tr>
<tr><td>Total Examples</td><td>{meta_loaded.get("n_total", "—"):,}</td></tr>
<tr><td>Training Examples</td><td>{meta_loaded.get("n_train", "—"):,}</td></tr>
<tr><td>Test Examples</td><td>{meta_loaded.get("n_test", "—"):,}</td></tr>
<tr><td>Test Split</td><td>{int(meta_loaded.get("test_size", 0.2)*100)}%</td></tr>
<tr><td>Random Seed</td><td>{meta_loaded.get("random_seed", "—")}</td></tr>
<tr><td>Trained On</td><td>{meta_loaded.get("train_date", "—")}</td></tr>
</table>
""", unsafe_allow_html=True)

        st.markdown("<br>**Held-Out Test Metrics**", unsafe_allow_html=True)
        st.markdown(f"""
<table class="metric-table">
<tr><th>Metric</th><th>Value</th><th>Avg</th></tr>
<tr><td>Accuracy</td><td><strong style="color:#a78bfa">{meta_loaded.get("accuracy", "—")}</strong><td>—</td></tr>
<tr><td>Precision</td><td><strong style="color:#a78bfa">{meta_loaded.get("macro_precision", "—")}</strong></td><td>macro</td></tr>
<tr><td>Recall</td><td><strong style="color:#a78bfa">{meta_loaded.get("macro_recall", "—")}</strong></td><td>macro</td></tr>
<tr><td>F1-Score</td><td><strong style="color:#a78bfa">{meta_loaded.get("macro_f1", "—")}</strong></td><td>macro</td></tr>
</table>
""", unsafe_allow_html=True)

    with chart_col:
        cm = meta_loaded.get("confusion_matrix")
        cm_labels = meta_loaded.get("confusion_matrix_labels", ["Negative", "Neutral", "Positive"])
        if cm:
            st.markdown("**Confusion Matrix (Test Set)**")
            st.plotly_chart(
                visualizations.confusion_matrix_chart(cm, cm_labels),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    st.markdown("**Per-Class Classification Report (Test Set)**")
    cr_text = meta_loaded.get("classification_report", "")
    if cr_text:
        st.code(cr_text, language=None)

else:
    st.markdown(
        '<div class="warn-box">⚠️ Model artefacts not found. '
        'Run <code>python -m src.train_model</code> to train the model and see evaluation results here.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Limitations & Responsible AI notice
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">⚠️ Limitations & Responsible Use</div>', unsafe_allow_html=True)
st.markdown("""
<div class="warn-box">
<strong>Important caveats for this educational project:</strong><br><br>
• <strong>Sampling bias:</strong> Keyword-based post collection overrepresents users who engage with
  those specific terms and does not represent the general population.<br>
• <strong>Model limitations:</strong> The classifier may misunderstand sarcasm, political irony,
  mixed sentiment, regional idioms, and cultural context.<br>
• <strong>No political affiliation inference:</strong> Sentiment about a topic is not the same as
  a user's political affiliation or stance.<br>
• <strong>No causal claims:</strong> Sentiment trends do not imply causation of real-world events.<br>
• <strong>Data privacy:</strong> Follow X Developer Policy for storing, displaying, and deleting
  collected posts. This project stores only the minimum necessary data in session state.<br>
• <strong>Educational purpose only:</strong> Results should not be used to make claims about
  broader public opinion, voter behaviour, or electoral outcomes.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin-top:48px;">
<p style="text-align:center;color:#374151;font-size:0.78rem;padding-bottom:20px;">
    SentimentIQ &nbsp;·&nbsp; TF-IDF + Logistic Regression &nbsp;·&nbsp;
    TweetEval Dataset &nbsp;·&nbsp; Official X API v2 &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp;
    Educational College Project
</p>
""", unsafe_allow_html=True)
