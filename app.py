"""
Senti-Q — sentiment and hate-speech analysis for posts on X.

Run:   streamlit run app.py      (or double-click run.bat on Windows)

Open a post by pasting its link, or send one straight from X with the
"Senti-Q for X" browser extension, which opens  http://localhost:8501/?post=<link>.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Senti-Q",
    page_icon=str(ROOT / "assets" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src import engine, transformer_models, x_links  # noqa: E402  (after page config)

st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


def _html(s: str) -> None:
    """Render an HTML snippet (indentation stripped so Markdown never treats it as code)."""
    st.markdown("\n".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


BOOKMARKLET = (
    "javascript:(()=>{const A='http://localhost:8501',R=/^https:\\/\\/(?:www\\.)?(?:x|twitter)\\.com\\/(?:i\\/web|[^/]+)"
    "\\/status\\/\\d+/;let u=(location.href.match(R)||[])[0];if(!u){let b=null,d=1e9;document.querySelectorAll"
    "('article').forEach(a=>{const r=a.getBoundingClientRect();if(r.bottom<0||r.top>innerHeight)return;const "
    "l=[...a.querySelectorAll('a[href*=\"/status/\"]')].find(x=>x.querySelector('time'));const m=Math.abs(r.top"
    "+r.height/2-innerHeight/2);if(l&&m<d){d=m;b=l.href}});u=b&&(b.match(R)||[])[0]}if(!u){alert('Senti-Q: "
    "open or scroll to a post on X first.');return}window.open(A+'/?post='+encodeURIComponent(u),'_blank')})()"
)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _warm_models() -> bool:
    """Load the language models once per server process (first run downloads ~1 GB)."""
    if transformer_models.available():
        transformer_models.sentiment_probs(["warm up"])
        transformer_models.hate_probs(["warm up"])
    return True


def _analyse(raw: str) -> None:
    """Open the linked posts, run the engine and store the results in the session."""
    st.session_state.pop("pick", None)          # reset the post picker for the new batch
    with st.spinner("Opening the post ..."):
        opened = x_links.open_posts(raw)
    st.session_state.errors = opened.errors
    if not opened.posts:
        st.session_state.results = None
        return
    if transformer_models.available() and not transformer_models.is_loaded():
        with st.spinner("Loading the language models. The first time downloads about 1 GB ..."):
            _warm_models()
    with st.spinner("Reading the room ..."):
        df = pd.DataFrame([{
            "post_id": p.post_id, "text": p.text, "created_at": p.created_at, "username": p.username,
            "display_name": p.display_name, "url": p.url, "embed_html": p.embed_html,
        } for p in opened.posts])
        out = engine.analyze(df["text"].tolist())
        for col in out.columns:
            df[col] = out[col].values
    st.session_state.results = df


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _post_html(row) -> str:
    """The post being analysed, styled like a post on X."""
    name = str(row.get("display_name") or row.get("username") or "Unknown")
    handle = str(row.get("username") or "")
    when = ""
    if pd.notna(row.get("created_at")) and row["created_at"] is not None:
        when = pd.to_datetime(row["created_at"]).strftime("%d %b %Y")
    text_html = escape(str(row["text"])).replace("\n", "<br>")
    return (
        '<div class="xpost">'
        '<div class="xpost-head">'
        f'<div class="avatar">{escape(name.strip()[:1].upper() or "?")}</div>'
        f'<div class="who"><b>{escape(name)}</b>{f"<span>@{escape(handle)}</span>" if handle else ""}</div>'
        '<div class="xmark">𝕏</div></div>'
        f'<div class="xpost-text">{text_html}</div>'
        '<div class="xpost-foot">'
        f'<span>{escape(when)}</span>'
        f'<a href="{escape(str(row["url"]), quote=True)}" target="_blank" rel="noopener noreferrer">'
        'Open on X &#8599;</a></div>'
        '</div>'
    )


def _bar_row(label: str, value: float, strong: bool) -> str:
    return (f'<div class="bar-row{" strong" if strong else ""}"><span>{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{value * 100:.1f}%"></div></div>'
            f'<span class="num">{value:.0%}</span></div>')


def _dashboard_html(row) -> str:
    """The analysis shown beside the post."""
    sentiment = row["predicted_sentiment"]
    probs = {"Positive": row["prob_positive"], "Neutral": row["prob_neutral"], "Negative": row["prob_negative"]}
    conf = str(row.get("sentiment_confidence") or "")
    has_hate = "is_hate" in row.index and pd.notna(row.get("is_hate"))
    flagged = bool(row["is_hate"]) if has_hate else False
    severity = str(row.get("hate_severity") or ("Hate" if flagged else "None"))

    tiles = ('<div class="tile"><div class="label">Sentiment</div>'
             f'<div class="verdict">{escape(sentiment)}</div>'
             f'<div class="sub"><span class="num">{probs[sentiment]:.0%}</span> likely'
             f'{f" &middot; {escape(conf.lower())} confidence" if conf else ""}</div></div>')
    if has_hate:
        word = {"None": "None found", "Offensive": "Offensive", "Hate": "Hate speech",
                "Severe": "Severe hate"}.get(severity, severity)
        tiles += (f'<div class="tile{" dark" if flagged else ""}"><div class="label">Hate speech</div>'
                  f'<div class="verdict">{escape(word)}</div>'
                  f'<div class="sub">{"Flagged" if flagged else "Not flagged"} &middot; '
                  f'score <span class="num">{row["prob_hate"]:.1%}</span></div></div>')
    html = f'<div class="dash"><div class="dash-row">{tiles}</div>'

    html += ('<div class="tile"><div class="label">How it reads</div>'
             + "".join(_bar_row(k, v, k == sentiment) for k, v in probs.items()) + "</div>")

    if has_hate:
        thr = engine.hate_threshold()
        score = float(row["prob_hate"])
        html += ('<div class="tile"><div class="label">Hate score</div>'
                 f'<div class="meter"><div class="meter-fill" style="width:{score * 100:.1f}%"></div>'
                 f'<div class="meter-thr" style="left:{thr * 100:.0f}%"></div></div>'
                 f'<div class="meter-scale"><span>0</span><span style="left:{thr * 100:.0f}%">'
                 f'flagged above {thr:.0%}</span><span>100</span></div></div>')

    reasons = []
    for key in ("sentiment_notes", "hate_notes"):
        val = row.get(key)
        if isinstance(val, (list, tuple)):
            reasons += [str(v) for v in val]
    if reasons:
        html += ('<div class="tile"><div class="label">Why</div><ul class="reasons">'
                 + "".join(f"<li>{escape(r)}</li>" for r in reasons[:5]) + "</ul></div>")
    return html + "</div>"


@st.cache_data(show_spinner=False, ttl=3600)
def _embed_for(post_id: str, handle: str) -> str:
    p = x_links.fetch_post(post_id, handle)
    return p.embed_html if p else ""


def _render_result(df: pd.DataFrame) -> None:
    if len(df) > 1:
        labels = {i: f"{i + 1} · @{df.iloc[i]['username'] or 'post'}" for i in range(len(df))}
        pick = st.pills("Posts", options=list(labels), format_func=labels.get, default=0,
                        key="pick", label_visibility="collapsed")
        row = df.iloc[pick if pick is not None and pick < len(df) else 0]
    else:
        row = df.iloc[0]

    left, right = st.columns([5, 6], gap="large")
    with left:
        _html('<div class="col-label">The post</div>')
        _html(_post_html(row))
        if st.toggle("Show it exactly as it appears on X", key=f"embed_{row['post_id']}"):
            embed = row.get("embed_html") or _embed_for(str(row["post_id"]), str(row.get("username", "")))
            if embed:
                import streamlit.components.v1 as components
                components.html(embed + '<script async src="https://platform.twitter.com/widgets.js" '
                                'charset="utf-8"></script>', height=640, scrolling=True)
            else:
                st.caption("X didn't return an embeddable version of this post.")
    with right:
        _html('<div class="col-label">The analysis</div>')
        _html(_dashboard_html(row))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.errors = []

with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    _html('<div class="brand"><span class="brand-mark">Q</span>Senti-Q</div>')
    with st.popover("Use it on X", width="content"):
        st.markdown("**Analyse posts straight from X**")
        st.markdown(
            "Stay signed in to X in your own browser. Senti-Q never asks for your X password.\n\n"
            "1. Open `chrome://extensions` (or `edge://extensions`), switch on **Developer mode**, "
            "click **Load unpacked** and choose the `browser-extension` folder in this project.\n"
            "2. On X, click the **Senti-Q** icon in your toolbar. It analyses the post in the middle "
            "of your screen. Or right-click any post's timestamp and choose **Analyse with Senti-Q**."
        )
        st.markdown("**No install?** Save a bookmark with this as its address, then click it on X:")
        st.code(BOOKMARKLET, language=None, wrap_lines=True)

_html("""
<div class="hero">
  <div class="eyebrow">Sentiment &amp; hate-speech analysis for X</div>
  <h1>Read the <em>room.</em></h1>
  <p>Paste a link to any public post on X. Senti-Q tells you how it feels, and whether it crosses the line.</p>
</div>
""")

status = engine.status()
if not status["ready"]:
    _html('<div class="notice dark"><b>Setup isn\'t finished.</b> Close this tab and double-click '
          '<code>run.bat</code> in the project folder. It installs everything the first time.</div>')

# Deep link from the browser extension / bookmarklet:  /?post=<link>
deep_link = st.query_params.get("post")
run_deep_link = bool(deep_link) and st.session_state.get("_handled_link") != deep_link
if run_deep_link:
    st.session_state["_handled_link"] = deep_link
    st.session_state["link_input"] = deep_link

with st.form("open_post", border=False):
    c_in, c_btn = st.columns([5, 1], vertical_alignment="bottom")
    with c_in:
        st.text_input("Post link", key="link_input", label_visibility="collapsed",
                      placeholder="Paste a post link, e.g. https://x.com/username/status/1234567890")
    with c_btn:
        submitted = st.form_submit_button("Analyse", width="stretch", disabled=not status["ready"])
_html('<div class="hint">Paste several links separated by spaces to compare posts.</div>')

link = (st.session_state.get("link_input") or "").strip()
if status["ready"] and ((submitted and link) or run_deep_link):
    _analyse(link)
    first = x_links.extract_post_refs(link)
    if first:
        pid, handle = first[0]
        st.query_params["post"] = f"https://x.com/{handle or 'i'}/status/{pid}"
        st.session_state["_handled_link"] = st.query_params["post"]

for err in st.session_state.errors:
    _html(f'<div class="notice">{escape(err)}</div>')

if st.session_state.results is not None:
    _render_result(st.session_state.results)

_html('<div class="footer">Senti-Q &middot; Automated estimates, not verdicts. '
      'Sarcasm and coded language can still fool it.</div>')
