"""
Senti-Q — how people are reacting to a post on X.

Run:   streamlit run app.py      (or double-click run.bat on Windows)

Senti-Q's main job is the *reactions*: it reads the replies to a post and shows how
people responded (the share of positive, neutral and negative replies), whether the
post drew hate speech, and the post's own tone. X only shows replies to signed-in
users, so the "Senti-Q for X" browser extension reads them in the user's own browser
and opens   /?post=<post link>&r=<packed replies>   (see src/reactions.py).
"""

from __future__ import annotations

import hashlib
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

from src import engine, reactions, transformer_models, x_links  # noqa: E402  (after page config)

st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

READ_MARK = "#senti-q"          # links ending in this make the extension read the replies


def _html(s: str) -> None:
    """Render an HTML snippet (indentation stripped so Markdown never treats it as code)."""
    st.markdown("\n".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


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


def _ensure_models() -> None:
    if transformer_models.available() and not transformer_models.is_loaded():
        with st.spinner("Loading the language models. The first time downloads about 1 GB ..."):
            _warm_models()


def _canonical(link: str) -> tuple[str, str] | None:
    refs = x_links.extract_post_refs(link)
    if not refs:
        return None
    pid, handle = refs[0]
    return pid, f"https://x.com/{handle or 'i'}/status/{pid}"


def _build(link: str, collected: reactions.Collected | None) -> dict:
    """Open and score the post and, when the extension supplied them, its replies."""
    found = _canonical(link)
    if not found:
        return {"error": "That doesn't look like a link to an X post. It should look like "
                         "https://x.com/username/status/1234567890."}
    pid, _ = found
    handle = x_links.extract_post_refs(link)[0][1]

    post = None
    if collected and collected.post["id"] == pid and collected.post["text"]:
        cp = collected.post
        post = {"id": pid, "handle": cp["handle"] or handle, "name": cp["name"], "text": cp["text"],
                "created": cp["time"], "likes": cp["likes"], "replies": cp["replies"],
                "reposts": cp["reposts"], "has_counts": True}
    else:
        with st.spinner("Opening the post ..."):
            p = x_links.fetch_post(pid, handle)
        if p:
            post = {"id": pid, "handle": p.username, "name": p.display_name, "text": p.text,
                    "created": p.created_at, "has_counts": False}
    if post is None:
        return {"error": f"Couldn't open post {pid}. It may be private, deleted or age-restricted."}
    post["url"] = f"https://x.com/{post['handle'] or 'i'}/status/{pid}"

    _ensure_models()
    with st.spinner("Reading the post ..."):
        post_scores = engine.analyze([post["text"]]).iloc[0].to_dict()

    result = {"post": post, "post_scores": post_scores, "collected": collected,
              "replies": None, "summary": None, "skipped_lang": 0}
    if collected and collected.replies:
        with st.spinner(f"Reading {len(collected.replies)} replies ..."):
            df, skipped = reactions.score_replies(collected)
        result.update(replies=df, summary=reactions.summarize(df), skipped_lang=skipped)
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _num(n: int) -> str:
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1000:.1f}K" if n >= 10_000 else f"{n:,}"


def _post_html(post: dict) -> str:
    name = str(post.get("name") or post.get("handle") or "Unknown")
    handle = str(post.get("handle") or "")
    when = ""
    if post.get("created"):
        try:
            when = pd.to_datetime(post["created"]).strftime("%d %b %Y")
        except (ValueError, TypeError):
            when = ""
    counts = ""
    if post.get("has_counts"):
        counts = ('<div class="xpost-counts">'
                  f'<span><b>{_num(post.get("replies", 0))}</b> replies</span>'
                  f'<span><b>{_num(post.get("reposts", 0))}</b> reposts</span>'
                  f'<span><b>{_num(post.get("likes", 0))}</b> likes</span></div>')
    return (
        '<div class="xpost"><div class="xpost-head">'
        f'<div class="avatar">{escape(name.strip()[:1].upper() or "?")}</div>'
        f'<div class="who"><b>{escape(name)}</b>{f"<span>@{escape(handle)}</span>" if handle else ""}</div>'
        '<div class="xmark">𝕏</div></div>'
        f'<div class="xpost-text">{escape(str(post["text"])).replace(chr(10), "<br>")}</div>'
        f'{counts}'
        f'<div class="xpost-foot"><span>{escape(when)}</span>'
        f'<a href="{escape(post["url"], quote=True)}" target="_blank" rel="noopener noreferrer">Open on X &#8599;</a>'
        '</div></div>'
    )


def _post_tone_html(ps: dict) -> str:
    sentiment = ps["predicted_sentiment"]
    prob = ps[f"prob_{sentiment.lower()}"]
    hate = str(ps.get("hate_severity") or "None")
    hate_word = {"None": "None", "Offensive": "Offensive", "Hate": "Hate speech", "Severe": "Severe"}.get(hate, hate)
    return ('<div class="tone">'
            f'<div><span class="label">The post itself</span><b>{escape(sentiment)}</b> '
            f'<span class="num">{prob:.0%}</span></div>'
            f'<div><span class="label">Hate in the post</span><b>{escape(hate_word)}</b></div></div>')


SW = {"Positive": "pos", "Neutral": "neu", "Negative": "neg"}


def _reactions_html(s: reactions.Summary, a: dict) -> str:
    collected: reactions.Collected = a["collected"]
    margin = f" &middot; &plusmn;<span class='num'>{s.margin * 100:.0f}</span> pts" if s.margin else " &middot; small sample"
    html = ('<div class="dash">'
            '<div class="tile"><div class="label">Overall response</div>'
            f'<div class="verdict">{escape(s.verdict)}</div>'
            f'<div class="sub">from <span class="num">{s.n}</span> replies{margin}</div></div>')

    cells = "".join(
        f'<div><div class="big">{s.shares[k]:.0%}</div>'
        f'<div class="mix-name"><i class="sw {SW[k]}"></i>{k} <span class="num">{s.counts[k]}</span></div></div>'
        for k in reactions.LABELS)
    segs = "".join(f'<div class="seg {SW[k]}" style="width:{s.shares[k] * 100:.2f}%"></div>' for k in reactions.LABELS)
    liked = " &middot; ".join(f"{s.liked_shares[k]:.0%} {k.lower()}" for k in reactions.LABELS)
    html += ('<div class="tile"><div class="label">How people reacted</div>'
             f'<div class="mix3">{cells}</div><div class="stack">{segs}</div>'
             f'<div class="sub">Weighted by likes: {liked}</div></div>')

    flagged = s.hate_n > 0
    if flagged:
        head = f'{s.hate_n} {"reply" if s.hate_n == 1 else "replies"}'
        detail = f'<span class="num">{s.hate_share:.0%}</span> of replies'
        if s.severe_n:
            detail += f' &middot; {s.severe_n} severe'
    else:
        head, detail = "None found", f'0 of {s.n} replies'
    if s.offensive_n:
        detail += f' &middot; {s.offensive_n} offensive'
    html += (f'<div class="tile{" dark" if flagged else ""}"><div class="label">Hate speech received</div>'
             f'<div class="verdict">{head}</div><div class="sub">{detail}</div></div>')

    line = reactions.reception(a["post_scores"]["predicted_sentiment"], s)
    if line:
        html += f'<div class="insight">{escape(line)}</div>'

    notes = []
    if collected.collected_at is not None:
        notes.append(f"Read from X in your browser at {collected.collected_at.astimezone().strftime('%H:%M')}")
    if a["skipped_lang"]:
        notes.append(f"{a['skipped_lang']} replies in other languages not scored")
    if collected.media_only:
        notes.append(f"{collected.media_only} image-only replies")
    if collected.self_replies:
        notes.append(f"{collected.self_replies} by the author left out")
    if collected.trimmed:
        notes.append(f"{collected.trimmed} more not included")
    if notes:
        html += f'<div class="foot">{" &middot; ".join(notes)}</div>'
    return html + "</div>"


def _cta(post: dict, text: str, title: str = "See how people reacted") -> None:
    _html(f'<div class="tile cta"><div class="label">The reactions</div>'
          f'<div class="cta-title">{escape(title)}</div><p>{text}</p></div>')
    st.link_button("Read the replies on X  ↗", post["url"] + READ_MARK, type="primary")
    st.caption("This needs the Senti-Q for X extension (see **Get the extension** at the top). "
               "The replies are read in your own signed-in browser and brought back here.")


def _reply_cards(df: pd.DataFrame) -> str:
    cards = []
    for _, r in df.iterrows():
        sentiment = r["predicted_sentiment"]
        tags = f'<span class="tag"><i class="sw {SW[sentiment]}"></i>{sentiment}</span>'
        if bool(r.get("is_hate", False)):
            severe = r.get("hate_severity") == "Severe"
            tags += f'<span class="tag hate">{"Hate &middot; Severe" if severe else "Hate speech"}</span>'
        elif r.get("hate_severity") == "Offensive":
            tags += '<span class="tag">Offensive</span>'
        handle = str(r.get("handle") or "")
        likes = int(r.get("likes", 0))
        cards.append(
            '<div class="reply"><div class="reply-head">'
            f'<span class="avatar sm">{escape((handle[:1] or "?").upper())}</span>'
            f'<b>{"@" + escape(handle) if handle else "Reply"}</b>'
            f'{f"<span class=num>{_num(likes)} likes</span>" if likes else ""}</div>'
            f'<div class="reply-text">{escape(str(r["text"])).replace(chr(10), "<br>")}</div>'
            f'<div class="tags">{tags}</div></div>')
    return f'<div class="replies">{"".join(cards)}</div>'


def _render_replies(df: pd.DataFrame) -> None:
    _html('<div class="section-title">What people are saying</div>')
    hate_n = int(df["is_hate"].sum()) if "is_hate" in df.columns else 0
    options = {"All": len(df)}
    options.update({k: int((df["predicted_sentiment"] == k).sum()) for k in reactions.LABELS})
    if "is_hate" in df.columns:
        options["Hate speech"] = hate_n
    choice = st.pills("Filter", options=list(options), default="All", key="reply_filter",
                      format_func=lambda o: f"{o}  {options[o]}", label_visibility="collapsed") or "All"
    if choice == "Hate speech":
        shown = df[df["is_hate"]]
    elif choice in reactions.LABELS:
        shown = df[df["predicted_sentiment"] == choice]
    else:
        shown = df
    if shown.empty:
        _html('<div class="notice">No replies in this group.</div>')
        return
    limit = 24
    if len(shown) > limit and not st.toggle(f"Show all {len(shown)}", key="show_all_replies"):
        shown = shown.head(limit)
    _html(_reply_cards(shown))


def _render(a: dict) -> None:
    post, collected, summary = a["post"], a["collected"], a["summary"]
    left, right = st.columns([5, 7], gap="large")
    with left:
        _html('<div class="col-label">The post</div>')
        _html(_post_html(post))
        _html(_post_tone_html(a["post_scores"]))
    with right:
        _html('<div class="col-label">The reactions</div>')
        if collected is not None and collected.error == "login":
            _cta(post, "X asked for a sign-in, so the replies couldn't be read. Sign in to X in this "
                       "browser, then try again.", "Sign in to X first")
        elif collected is not None and collected.error:
            _cta(post, "Senti-Q couldn't find this post on X. It may have been deleted or made "
                       "private.", "Post not found on X")
        elif summary is not None and summary.n > 0:
            _html(_reactions_html(summary, a))
        elif collected is not None:
            extra = f" ({collected.media_only} image-only replies were skipped.)" if collected.media_only else ""
            _cta(post, "No written replies were found on this post yet." + escape(extra), "No replies yet")
        else:
            _cta(post, "X only shows replies to people who are signed in, so Senti-Q reads them in "
                       "your own browser and brings them back here. Nothing is stored.")
    if a["replies"] is not None and len(a["replies"]):
        _render_replies(a["replies"])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
    st.session_state.analysis_id = None
    st.session_state.ingested = None

with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    _html('<div class="brand"><span class="brand-mark">Q</span>Senti-Q</div>')
    with st.popover("Get the extension", width="content"):
        st.markdown("**Senti-Q for X** reads the replies to a post in your own signed-in browser "
                    "and brings them here. It never touches your password or account.")
        st.markdown(
            "1. Open `chrome://extensions` (or `edge://extensions`) and switch on **Developer mode**.\n"
            "2. Click **Load unpacked** and choose the `browser-extension` folder in this project.\n"
            "3. Pin **Senti-Q for X** to your toolbar.\n\n"
            "**Then:** open any post on X and click the Senti-Q icon. Or paste a link here and press "
            "**Read the replies on X**."
        )

_has_result = bool(st.query_params.get("post") or st.query_params.get("r") or st.session_state.analysis)
_html(f"""
<div class="hero{' compact' if _has_result else ''}">
  <div class="eyebrow">Reaction analysis for X</div>
  <h1>Read the <em>room.</em></h1>
  <p>Senti-Q reads the replies to any post on X and shows how people are reacting: how positive,
     negative or neutral, and whether the post is drawing hate.</p>
</div>
""")

status = engine.status()
if not status["ready"]:
    _html('<div class="notice dark"><b>Setup isn\'t finished.</b> Close this tab and double-click '
          '<code>run.bat</code> in the project folder. It installs everything the first time.</div>')

# --- links coming in:  /?post=<link>  (just the post)  or  /?post=<link>&r=<packed replies> ---
qp_post = st.query_params.get("post", "")
qp_r = st.query_params.get("r", "")
if status["ready"] and qp_r:
    digest = hashlib.sha1(qp_r.encode()).hexdigest()
    if st.session_state.ingested != digest:
        st.session_state.ingested = digest
        try:
            collected = reactions.decode(qp_r)
        except ValueError:
            collected = None
            st.session_state.analysis = {"error": "This reactions link is damaged or incomplete. "
                                                  "Click Senti-Q on X again."}
        if collected is not None:
            link = qp_post or collected.post_url or f"https://x.com/i/status/{collected.post['id']}"
            st.session_state.analysis = _build(link, collected)
            found = _canonical(link)
            if found:
                st.session_state.analysis_id = found[0]
                st.session_state["link_input"] = found[1]
                st.query_params.clear()                 # tidy the address bar: keep just the post
                st.query_params["post"] = found[1]
elif status["ready"] and qp_post:
    found = _canonical(qp_post)
    if found and found[0] != st.session_state.analysis_id:
        st.session_state["link_input"] = found[1]
        st.session_state.analysis = _build(qp_post, None)
        st.session_state.analysis_id = found[0]

with st.form("open_post", border=False):
    c_in, c_btn = st.columns([5, 1], vertical_alignment="bottom")
    with c_in:
        st.text_input("Post link", key="link_input", label_visibility="collapsed",
                      placeholder="Paste a post link, e.g. https://x.com/username/status/1234567890")
    with c_btn:
        submitted = st.form_submit_button("Analyse", width="stretch", disabled=not status["ready"])
_html('<div class="hint">Paste a post link, or open the post on X and click the Senti-Q extension.</div>')

if submitted and status["ready"]:
    link = (st.session_state.get("link_input") or "").strip()
    found = _canonical(link)
    if not found:
        st.session_state.analysis = _build(link, None)          # produces the friendly error
        st.session_state.analysis_id = None
    elif found[0] != st.session_state.analysis_id or st.session_state.analysis is None \
            or st.session_state.analysis.get("error"):
        st.session_state.analysis = _build(link, None)
        st.session_state.analysis_id = found[0]
        st.query_params.clear()
        st.query_params["post"] = found[1]

a = st.session_state.analysis
if a and a.get("error"):
    _html(f'<div class="notice">{escape(a["error"])}</div>')
elif a:
    _render(a)

_html('<div class="footer">Senti-Q &middot; Automated estimates, not verdicts. '
      'Sarcasm and coded language can still fool it.</div>')
