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
    page_icon="◼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — monochrome design system lives in assets/style.css
# ---------------------------------------------------------------------------
_CSS = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def _html(s: str) -> None:
    """Render an HTML snippet (strips indentation so Markdown never treats it as code)."""
    st.markdown("\n".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


def _section(kicker: str, title: str, blurb: str = "") -> None:
    p = f"<p>{blurb}</p>" if blurb else ""
    _html(f'<div class="section"><div class="kicker">{kicker}</div><h2>{title}</h2>{p}</div>')



# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------
from html import escape

from src import x_api, predict, hate_speech, visualizations, engine, transformer_models
from data.demo_posts import DEMO_POSTS


# ---------------------------------------------------------------------------
# Input sources
# ---------------------------------------------------------------------------
SOURCES = {
    "Post link or ID":   "link",
    "Paste text":        "text",
    "Search X (live)":   "live",
    "Demo mode":         "demo",
}
SOURCE_BUTTON = {
    "link": "Analyse post",
    "text": "Analyse text",
    "live": "Fetch live posts",
    "demo": "Load demo data",
}
SOURCE_LABEL = {
    "link": "X post link",
    "text": "Pasted text",
    "live": "Live X search",
    "demo": "Demo mode",
}

if "results_df" not in st.session_state:
    st.session_state.results_df = None
    st.session_state.fetch_meta = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _posts_to_df(posts: list) -> pd.DataFrame:
    """Convert XPost objects or plain dicts to a DataFrame."""
    rows = []
    for p in posts:
        if hasattr(p, "post_id"):          # XPost dataclass
            rows.append({
                "post_id": p.post_id, "text": p.text, "created_at": p.created_at,
                "likes": p.likes, "retweets": p.retweets, "replies": p.replies,
                "lang": p.lang, "url": p.url, "username": p.username, "via": p.via,
            })
        else:                               # dict (demo or pasted text)
            created = p.get("created_at")
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    created = None
            rows.append({
                "post_id": p.get("post_id", ""), "text": p.get("text", ""), "created_at": created,
                "likes": p.get("likes", 0), "retweets": p.get("retweets", 0),
                "replies": p.get("replies", 0), "lang": p.get("lang", "en"),
                "url": "", "username": "", "via": p.get("via", "demo"),
            })
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def _warm_models() -> bool:
    """Load the transformer models once per server process (first run downloads them)."""
    if transformer_models.available():
        transformer_models.sentiment_probs(["warm up"])
        transformer_models.hate_probs(["warm up"])
    return True


def _run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Run the fuzzy ensemble engine and append its columns."""
    out = engine.analyze(df["text"].tolist())
    for col in out.columns:
        df[col] = out[col].values
    return df


def _df_to_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _pct(num, total) -> str:
    return "0.0%" if total == 0 else f"{100 * num / total:.1f}%"


def _prob_bar(label: str, prob: float, color: str) -> str:
    w = int(prob * 100)
    return (
        f'<div><div class="prob-label"><span>{label}</span><b>{prob:.0%}</b></div>'
        f'<div class="prob-bar-outer">'
        f'<div class="prob-bar-inner" style="width:{w}%;background:{color};"></div>'
        f'</div></div>'
    )


def _notice(text: str, kind: str = "") -> None:
    _html(f'<div class="notice {kind}">{text}</div>')


def _stat(label: str, value: str, sub: str = "", dark: bool = False) -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="stat{" dark" if dark else ""}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{sub_html}</div>')


def _info_card(label: str, value: str) -> str:
    return f'<div class="stat"><div class="label">{label}</div><div class="text-value">{value}</div></div>'


def _collect(source: str):
    """
    Gather posts for the chosen source.
    Returns (raw_df, meta) on success, or None after showing a notice.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if source == "demo":
        return _posts_to_df(DEMO_POSTS), {
            "source": "demo", "query": "N/A", "fetched_at": now, "warning": None}

    if source == "text":
        lines = [ln.strip() for ln in st.session_state.get("in_text", "").splitlines() if ln.strip()]
        if not lines:
            _notice("Type or paste some text first. Put each post on its own line.", "outline")
            return None
        posts = [{"post_id": f"text_{i:03d}", "text": t, "via": "pasted", "lang": "en"}
                 for i, t in enumerate(lines, 1)]
        return _posts_to_df(posts), {
            "source": "text", "query": f"{len(lines)} pasted line(s)", "fetched_at": now, "warning": None}

    if source == "link":
        result = x_api.lookup_posts(st.session_state.get("in_links", ""))
    else:  # live search
        if not x_api.bearer_token_configured():
            _notice("<strong>X API token not configured.</strong> Live search needs a bearer token in "
                    "<code>.env</code> (see <code>.env.example</code>). Post links work without one.", "alert")
            return None
        result = x_api.fetch_recent_posts(
            query=st.session_state.sb_query, max_results=st.session_state.sb_max_results)

    if result.error:
        _notice(f"<strong>Could not fetch posts.</strong><br>{escape(result.error)}", "alert")
        return None
    if not result.posts:
        _notice("No posts were returned. Try a different link or query.", "outline")
        return None
    if result.warning:
        _notice(escape(result.warning), "outline")
    return _posts_to_df(result.posts), {
        "source": source, "query": result.query, "fetched_at": result.fetched_at,
        "warning": result.warning}


def _post_card(row) -> str:
    sentiment = row["predicted_sentiment"]
    has_hate = "is_hate" in row.index and pd.notna(row.get("is_hate"))
    flagged = bool(row["is_hate"]) if has_hate else False

    created_str = ""
    if pd.notna(row.get("created_at")) and row["created_at"] is not None:
        try:
            dt = pd.to_datetime(row["created_at"])
            created_str = dt.strftime("%d %b %Y") if row.get("via") == "oembed" else dt.strftime("%d %b %Y, %H:%M UTC")
        except Exception:
            created_str = str(row["created_at"])

    via = row.get("via", "")
    meta_bits = []
    if created_str:
        meta_bits.append(f"<span>{escape(created_str)}</span>")
    if row.get("username"):
        meta_bits.append(f"<span><b>@{escape(str(row['username']))}</b></span>")
    if via in ("api", "demo"):
        meta_bits += [f"<span><b>{int(row.get('likes', 0)):,}</b> likes</span>",
                      f"<span><b>{int(row.get('retweets', 0)):,}</b> reposts</span>",
                      f"<span><b>{int(row.get('replies', 0)):,}</b> replies</span>"]
    if via in ("api", "oembed") and row.get("url"):
        meta_bits.append(f'<a href="{escape(str(row["url"]), quote=True)}" target="_blank" '
                         f'rel="noopener noreferrer">View on X</a>')
    if via == "pasted":
        meta_bits.append("<span>Pasted text</span>")

    badges = f'<span class="badge badge-{sentiment}"><span class="pfx">Sentiment</span>{sentiment}</span>'
    if row.get("sentiment_confidence"):
        badges += (f'<span class="badge badge-Clear"><span class="pfx">Confidence</span>'
                   f'{escape(str(row["sentiment_confidence"]))}</span>')
    probs = [_prob_bar("Positive", row["prob_positive"], "#1d1d1f"),
             _prob_bar("Neutral",  row["prob_neutral"],  "#8e8e93"),
             _prob_bar("Negative", row["prob_negative"], "#c7c7cc")]
    severity = str(row.get("hate_severity") or ("Hate" if flagged else "None"))
    if has_hate:
        if flagged:
            badges += f'<span class="badge badge-Hate"><span class="pfx">Hate speech</span>{escape(severity)}</span>'
        elif severity == "Offensive":
            badges += '<span class="badge badge-Negative"><span class="pfx">Hate speech</span>Offensive, not hate</span>'
        else:
            badges += '<span class="badge badge-Clear"><span class="pfx">Hate speech</span>None detected</span>'
        probs.append(_prob_bar("Hate score", row["prob_hate"], "#000000"))

    reasons = []
    for key in ("sentiment_notes", "hate_notes"):
        val = row.get(key)
        if isinstance(val, (list, tuple)):
            reasons += [str(v) for v in val]
    why = ""
    if reasons:
        why = ('<div class="why"><span>Why</span><ul>'
               + "".join(f"<li>{escape(r)}</li>" for r in reasons[:4]) + "</ul></div>")

    text_html = escape(str(row["text"])).replace("\n", "<br>")
    return (
        f'<div class="post-card{" flagged" if flagged else ""}">'
        f'<div class="badge-row">{badges}</div>'
        f'<div class="post-text">&ldquo;{text_html}&rdquo;</div>'
        f'<div class="meta-row">{"".join(meta_bits)}</div>'
        f'<div class="probs{" four" if has_hate else ""}">{"".join(probs)}</div>'
        f'{why}</div>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    _html('<div class="side-title">Input</div>')
    source_name = st.selectbox(
        "Input source", options=list(SOURCES), index=0, key="sb_source",
        label_visibility="collapsed",
    )
    source = SOURCES[source_name]

    if source == "live":
        st.markdown("---")
        _html('<div class="side-title">Search</div>')
        st.text_input(
            "Search Query", value="election OR democracy OR politics OR government", key="sb_query",
            help="Use X query operators. -is:retweet and lang:en are added automatically.",
        )
        st.slider(
            "Max Results", min_value=10, max_value=100, value=20, step=10, key="sb_max_results",
            help="Free tier: up to 10 posts per API call. Basic tier: up to 100.",
        )
    elif source == "link":
        _html('<div class="side-note" style="margin-top:10px">Works without an API token via X\'s public '
              'embed endpoint. With a token you also get like / repost / reply counts.</div>')
    elif source == "demo":
        _html('<div class="side-note" style="margin-top:10px">15 pre-written sample posts, so every '
              'feature can be explored without any X access.</div>')

    st.markdown("---")
    _html('<div class="side-title">Filter results</div>')
    sentiment_filter = st.selectbox(
        "Sentiment", options=["All", "Positive", "Negative", "Neutral"], key="sb_sentiment_filter",
    )
    hate_filter = st.selectbox(
        "Hate speech", options=["All", "Flagged only", "Not flagged"], key="sb_hate_filter",
    )

    st.markdown("---")
    _html('<div class="side-title">Engine</div>')
    eng = engine.status()
    sent_ok, hate_ok = eng["ready"], eng["hate_ready"]

    def _row(ok: bool, name: str, hint: str = "") -> str:
        h = f'<div class="side-note" style="margin:0 0 6px 16px">{hint}</div>' if hint and not ok else ""
        return (f'<div class="side-status" style="margin-bottom:4px"><span class="dot{"" if ok else " off"}">'
                f'</span>{name}</div>{h}')

    _html(
        _row(eng["transformer"], "RoBERTa transformers", "<code>pip install -r requirements.txt</code>")
        + _row(eng["vader"], "VADER lexicon", "<code>pip install vaderSentiment</code>")
        + _row(eng["tfidf_sentiment"], "TF-IDF sentiment", "<code>python -m src.train_model</code>")
        + _row(eng["tfidf_hate"], "TF-IDF hate speech", "<code>python -m src.hate_speech</code>")
        + _row(True, "Fuzzy-logic fusion")
    )

    st.markdown("---")
    _html('<div class="side-note">Results are automated model estimates and can be wrong. '
          'Do not use them to judge individuals or as population-level opinion.</div>')


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
_html("""
<div class="hero">
    <div class="eyebrow">Sentiment &amp; Hate-Speech Intelligence</div>
    <h1>SentimentIQ.<br><span>Read the room.</span></h1>
    <div class="subtitle">
        Paste any X post link and see its sentiment and whether it contains
        hate speech &mdash; in seconds.
    </div>
</div>
""")


# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------
if not sent_ok:
    _notice(
        "<strong>No model available yet.</strong><br>"
        "Install the dependencies with <code>pip install -r requirements.txt</code> (adds the RoBERTa "
        "transformers), or train the lightweight model with <code>python -m src.train_model</code>.",
        "alert",
    )
elif not eng["transformer"]:
    _notice(
        "<strong>Running in lightweight mode.</strong> The RoBERTa transformers aren't installed, so "
        "results use TF-IDF + VADER only and are less accurate. Run "
        "<code>pip install -r requirements.txt</code> and restart for the full engine.",
        "outline",
    )
if sent_ok and not hate_ok:
    _notice(
        "<strong>Hate-speech detection is off.</strong> Install the transformers or train it with "
        "<code>python -m src.hate_speech</code>, then refresh.",
        "outline",
    )


# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
if source == "link":
    _html('<div class="hint">Paste one or more X / Twitter post links (or numeric post IDs), '
          'separated by spaces or new lines.</div>')
    st.text_area("Post links or IDs", key="in_links", height=110, label_visibility="collapsed",
                 placeholder="https://x.com/username/status/1234567890123456789")
elif source == "text":
    _html('<div class="hint">Type or paste any text. Put each post on its own line to score several at once.</div>')
    st.text_area("Text to analyse", key="in_text", height=130, label_visibility="collapsed",
                 placeholder="Paste a post, a reply, or any sentence ...")

col_btn, col_refresh, _ = st.columns([3, 3, 6])
with col_btn:
    run_clicked = st.button(SOURCE_BUTTON[source], key="btn_run")
with col_refresh:
    if st.session_state.results_df is not None:
        if st.button("Clear results", key="btn_reset", type="secondary"):
            st.session_state.results_df = None
            st.session_state.fetch_meta = None
            st.rerun()


# ---------------------------------------------------------------------------
# Collection + prediction
# ---------------------------------------------------------------------------
if run_clicked:
    if not sent_ok:
        _notice("Cannot analyse yet &mdash; no model is available (see the notice above).", "alert")
    else:
        with st.spinner("Fetching ..."):
            collected = _collect(source)
        if collected is not None:
            raw_df, meta = collected
            if eng["transformer"] and not transformer_models.is_loaded():
                with st.spinner("Loading RoBERTa models (the first run downloads about 1 GB) ..."):
                    _warm_models()
            with st.spinner("Running the fuzzy ensemble ..."):
                try:
                    st.session_state.results_df = _run_predictions(raw_df)
                    st.session_state.fetch_meta = meta
                except FileNotFoundError as exc:
                    _notice(escape(str(exc)), "alert")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def _render_posts(df: pd.DataFrame, has_hate: bool, heading: tuple):
    _section(*heading)
    shown = df
    if sentiment_filter != "All":
        shown = shown[shown["predicted_sentiment"] == sentiment_filter]
    if has_hate and hate_filter != "All":
        shown = shown[shown["is_hate"] == (hate_filter == "Flagged only")]
    if shown.empty:
        _notice("No posts match the current filters.", "outline")
        return
    for _, row in shown.iterrows():
        _html(_post_card(row))


def _render_overview(df: pd.DataFrame, meta: dict, has_hate: bool, counts: dict):
    total = len(df)
    _section("Overview", "The mood, at a glance.")
    cards = [_stat("Total posts", f"{total}", "analysed", dark=True),
             _stat("Positive", f"{counts['Positive']}", _pct(counts["Positive"], total)),
             _stat("Neutral",  f"{counts['Neutral']}",  _pct(counts["Neutral"], total)),
             _stat("Negative", f"{counts['Negative']}", _pct(counts["Negative"], total))]
    if has_hate:
        n_hate = int(df["is_hate"].sum())
        cards.append(_stat("Hate flagged", f"{n_hate}", _pct(n_hate, total), dark=True))
    _html(f'<div class="card-grid">{"".join(cards)}</div>')

    _html(f"""
<div class="card-grid" style="margin-top:16px">
    {_info_card("Source", SOURCE_LABEL.get(meta.get("source"), "—"))}
    {_info_card("Fetched at", escape(str(meta.get("fetched_at", "—"))))}
    {_info_card("Query", escape(str(meta.get("query", "—"))[:120]))}
</div>
""")


def _render_charts(df: pd.DataFrame, has_hate: bool, counts: dict):
    _section("Visualizations", "Where the signals sit.")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        _html('<div class="chart-title">Sentiment distribution</div>')
        st.plotly_chart(visualizations.donut_chart(counts), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        if has_hate:
            n_hate = int(df["is_hate"].sum())
            _html('<div class="chart-title">Hate speech</div>')
            st.plotly_chart(visualizations.hate_donut(n_hate, len(df) - n_hate),
                            width="stretch", config={"displayModeBar": False})
        else:
            _html('<div class="chart-title">Sentiment counts</div>')
            st.plotly_chart(visualizations.bar_chart(counts), width="stretch",
                            config={"displayModeBar": False})
    if len(df) > 5 and df["created_at"].notna().any():
        _html('<div class="chart-title" style="margin-top:24px">Trend over time</div>')
        st.plotly_chart(visualizations.trend_chart(df), width="stretch",
                        config={"displayModeBar": False})


if st.session_state.results_df is not None:
    df: pd.DataFrame = st.session_state.results_df.copy()
    meta: dict = st.session_state.fetch_meta or {}
    has_hate = "is_hate" in df.columns

    if meta.get("source") == "demo":
        _notice("<strong>Demo mode.</strong> These are synthetic sample posts &mdash; not real X data.", "outline")
    if any(df["via"] == "oembed"):
        _notice("Fetched via X's public embed endpoint, so engagement counts are not shown. "
                "Add an X bearer token to <code>.env</code> to include them.", "outline")

    counts = {k: int((df["predicted_sentiment"] == k).sum()) for k in ("Positive", "Neutral", "Negative")}
    small = len(df) <= 5          # one or a few posts: show the verdict first

    if small:
        _render_posts(df, has_hate, ("Result", "Here's the verdict."))
        _render_overview(df, meta, has_hate, counts)
    else:
        _render_overview(df, meta, has_hate, counts)
        _render_charts(df, has_hate, counts)
        _render_posts(df, has_hate, ("Post explorer", "Every post, scored.",
                                     "Labels are automated model estimates. Sarcasm, mixed opinions and "
                                     "coded language may be misread."))

    _section("Data", "Take it with you.")
    export_cols = ["post_id", "username", "text", "predicted_sentiment",
                   "prob_positive", "prob_neutral", "prob_negative", "sentiment_confidence",
                   "hate_label", "hate_severity", "prob_hate", "created_at", "likes", "retweets", "replies"]
    export_df = df[[c for c in export_cols if c in df.columns]]
    st.dataframe(export_df, width="stretch", hide_index=True)
    st.download_button(
        label="Download CSV", data=_df_to_csv(export_df),
        file_name=f"sentimentiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv", key="btn_download",
    )


# ---------------------------------------------------------------------------
# Model Performance Section
# ---------------------------------------------------------------------------
def _spec_table(head: tuple, rows: list) -> str:
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="spec-table"><tr><th>{head[0]}</th><th>{head[1]}</th></tr>{body}</table>'


def _fmt(v):
    return f"{v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 999 else v


def _render_model_panel(title: str, m: dict, rows: list, labels_default: list):
    _html(f'<div class="chart-title" style="font-size:1.15rem;margin-top:8px">{title}</div>')
    left, right = st.columns([1, 1], gap="large")
    with left:
        _html(_spec_table(("Metric", "Value"), rows))
        _html(f'<div class="hint" style="margin-top:14px">{escape(m.get("algorithm", ""))}<br>'
              f'{escape(m.get("dataset_source", ""))}<br>'
              f'{_fmt(m.get("n_train", "—"))} train / {_fmt(m.get("n_test", "—"))} test examples</div>')
    with right:
        cm = m.get("confusion_matrix")
        if cm:
            st.plotly_chart(
                visualizations.confusion_matrix_chart(cm, m.get("confusion_matrix_labels", labels_default)),
                width="stretch", config={"displayModeBar": False})
    if m.get("caveat"):
        _notice(f"<strong>Caveat.</strong> {escape(m['caveat'])}", "outline")


_section("Model performance", "How well it works.",
         "Every engine scored on the same untouched TweetEval test sets.")

bench = engine.load_benchmark()
if bench:
    NAMES = [("old_tfidf", "Original TF-IDF model"), ("vader_only", "VADER lexicon only"),
             ("roberta_only", "RoBERTa only"), ("fuzzy_ensemble", "Fuzzy ensemble (current)")]

    def _bench_table(res: dict, cols: list) -> str:
        head = "".join(f"<th>{c[1]}</th>" for c in cols)
        body = ""
        for key, name in NAMES:
            if key not in res:
                continue
            cells = "".join(f"<td>{res[key].get(c[0], 0):.1%}</td>" for c in cols)
            strong = ' style="font-weight:700"' if key == "fuzzy_ensemble" else ""
            body += f"<tr{strong}><td>{name}</td>{cells}</tr>"
        return (f'<table class="spec-table bench"><tr><th>Engine</th>{head}</tr>{body}</table>')

    b1, b2 = st.columns([1, 1], gap="large")
    with b1:
        s = bench["sentiment"]
        _html(f'<div class="chart-title">Sentiment &middot; {s["n_test"]:,} test tweets</div>')
        _html(_bench_table(s, [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("macro_recall", "Macro recall")]))
    with b2:
        h = bench["hate"]
        _html(f'<div class="chart-title">Hate speech &middot; {h["n_test"]:,} test tweets</div>')
        _html(_bench_table(h, [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("recall_hate", "Hate recall")]))
    _html(f'<div class="hint" style="margin-top:14px">{escape(bench.get("protocol", ""))} '
          f'Run <code>python -m src.evaluate</code> to reproduce.</div>')
    _html('<div class="chart-title" style="font-size:1.15rem;margin-top:40px">Lightweight TF-IDF models</div>')

sm_, hm_ = predict.load_metadata(), hate_speech.load_metadata()
if sm_:
    _render_model_panel("Sentiment model", sm_, [
        ("Accuracy", sm_.get("accuracy", "—")), ("Precision (macro)", sm_.get("macro_precision", "—")),
        ("Recall (macro)", sm_.get("macro_recall", "—")), ("F1-score (macro)", sm_.get("macro_f1", "—")),
    ], ["Negative", "Neutral", "Positive"])
if hm_:
    _render_model_panel("Hate-speech model", hm_, [
        ("Accuracy", hm_.get("accuracy", "—")), ("Precision (hate)", hm_.get("precision_hate", "—")),
        ("Recall (hate)", hm_.get("recall_hate", "—")), ("F1-score (hate)", hm_.get("f1_hate", "—")),
    ], ["Not hate", "Hate"])
if not sm_ and not hm_:
    _notice("No trained models found. Run <code>python -m src.train_model</code> and "
            "<code>python -m src.hate_speech</code> to see evaluation results here.", "outline")


# ---------------------------------------------------------------------------
# Limitations & Responsible AI notice
# ---------------------------------------------------------------------------
_section("Responsible use", "Know the limits.")
_notice("""
<strong>Still imperfect.</strong> The RoBERTa + fuzzy ensemble handles context, negation and some sarcasm far better than word counts, but subtle irony and new coded language can still fool it.<br>
<strong>Explainable, not infallible.</strong> The "Why" lines show which fuzzy rules fired; use them to sanity-check a verdict.<br>
<strong>Estimates, not verdicts.</strong> A flag is a probability, not a finding. Never use it to judge or report an individual.<br>
<strong>Sampling bias.</strong> Keyword search overrepresents people engaging with those terms.<br>
<strong>Data privacy.</strong> Follow the X Developer Policy for storing, displaying and deleting posts.<br>
<strong>Educational purpose.</strong> Do not use for claims about public opinion, voter behaviour or elections.
""")

_html("""
<div class="footer">
    SentimentIQ &nbsp;·&nbsp; TF-IDF + Logistic Regression &nbsp;·&nbsp; TweetEval &nbsp;·&nbsp;
    X API v2 &nbsp;·&nbsp; Streamlit<br>Educational college project
</div>
""")
