"""
src/reactions.py — How people reacted to a post on X.

The "Senti-Q for X" browser extension reads the replies shown on a post's page in the
user's own signed-in browser, packs them (JSON -> raw deflate -> base64url) and opens
Senti-Q with them in the link:  /?post=<post link>&r=<payload>.

This module unpacks that payload safely, scores every reply with the engine and
summarises the response: the share of positive / neutral / negative replies, an
overall verdict from a fuzzy rule base, and how much hate speech the post received.
"""

from __future__ import annotations

import base64
import json
import math
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src import engine, fuzzy

MAX_PAYLOAD_CHARS = 200_000      # base64 text accepted from the URL
MAX_JSON_BYTES = 2_000_000       # decompressed size limit (guards against zip bombs)
MAX_REPLIES = 1000
MAX_TEXT = 2000

# X's language codes the English models can score. "und" = undetermined; q* codes are
# emoji-only / mention-only / hashtag-only posts; "zxx" = no linguistic content.
SCOREABLE_LANGS = {"", "en", "und", "qme", "qam", "qht", "qct", "qst", "zxx"}

_ID_RE = re.compile(r"^\d{1,25}$")
_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,30}$")


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

@dataclass
class Collected:
    """Everything the extension read from the post's page."""
    post_url: str
    post: dict                                  # id, handle, name, text, lang, time, likes, replies, reposts
    replies: list[dict] = field(default_factory=list)   # id, handle, text, likes, lang
    self_replies: int = 0                       # the author's own replies (not reactions)
    media_only: int = 0                         # replies with no text (images, GIFs)
    trimmed: int = 0                            # replies dropped to keep the link short
    logged_out: bool = False                    # X asked the user to sign in
    error: str = ""                             # "login" | "notfound" | ""
    collected_at: Optional[datetime] = None


def _str(v, limit: int = MAX_TEXT) -> str:
    return v[:limit] if isinstance(v, str) else ""


def _int(v) -> int:
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 0


def decode(token: str) -> Collected:
    """Unpack and validate the extension's payload. Raises ValueError on bad input."""
    if not token or len(token) > MAX_PAYLOAD_CHARS:
        raise ValueError("missing or oversized payload")
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        inflater = zlib.decompressobj(-15)                  # raw deflate
        data = inflater.decompress(raw, MAX_JSON_BYTES)
        if inflater.unconsumed_tail:
            raise ValueError("payload too large")
        obj = json.loads(data.decode("utf-8"))
    except (ValueError, zlib.error, UnicodeDecodeError) as exc:
        raise ValueError(f"unreadable payload: {exc}") from exc
    if not isinstance(obj, dict) or obj.get("v") != 1:
        raise ValueError("unsupported payload version")

    p = obj.get("p") if isinstance(obj.get("p"), list) else []
    p = (p + [None] * 9)[:9]
    post = {"id": _str(p[0], 25), "handle": _str(p[1], 30), "name": _str(p[2], 100), "text": _str(p[3]),
            "lang": _str(p[4], 10), "time": _str(p[5], 40), "likes": _int(p[6]), "replies": _int(p[7]),
            "reposts": _int(p[8])}
    if not _ID_RE.match(post["id"]):
        raise ValueError("payload has no valid post id")

    replies, seen = [], set()
    for r in (obj.get("r") or [])[:MAX_REPLIES]:
        if not isinstance(r, list) or len(r) < 3:
            continue
        rid, handle, text = _str(r[0], 25), _str(r[1], 30), _str(r[2]).strip()
        if not _ID_RE.match(rid) or rid in seen or rid == post["id"] or not text:
            continue
        seen.add(rid)
        replies.append({"id": rid, "handle": handle if _HANDLE_RE.match(handle) else "",
                        "text": text, "likes": _int(r[3] if len(r) > 3 else 0),
                        "lang": _str(r[4] if len(r) > 4 else "", 10).lower()})

    at = obj.get("at")
    collected_at = None
    if isinstance(at, (int, float)) and at > 0:
        collected_at = datetime.fromtimestamp(at / 1000, tz=timezone.utc)
    return Collected(
        post_url=_str(obj.get("u"), 300), post=post, replies=replies,
        self_replies=_int(obj.get("s")), media_only=_int(obj.get("m")), trimmed=_int(obj.get("t")),
        logged_out=bool(obj.get("lo")), error=_str(obj.get("err"), 20), collected_at=collected_at,
    )


def encode(obj: dict) -> str:
    """Pack a payload the way the extension does (used by tests and tooling)."""
    deflater = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = deflater.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8")) + deflater.flush()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# ---------------------------------------------------------------------------
# Scoring and summary
# ---------------------------------------------------------------------------

def score_replies(collected: Collected, cache: Optional[dict] = None) -> tuple[pd.DataFrame, int]:
    """
    Run the engine on every scoreable reply.
    `cache` (text -> scores) lets a re-read skip replies that were already scored, so the
    same reply always gets exactly the same result.
    Returns (scored replies, number of replies skipped because of their language).
    """
    rows = [r for r in collected.replies if r["lang"] in SCOREABLE_LANGS]
    skipped = len(collected.replies) - len(rows)
    if not rows:
        return pd.DataFrame(), skipped
    cache = {} if cache is None else cache
    texts = [r["text"] for r in rows]
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        cache.update(zip(todo, engine.analyze(todo).to_dict("records")))
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame([cache[t] for t in texts])], axis=1)
    return df.sort_values("likes", ascending=False, kind="stable").reset_index(drop=True), skipped


def compare(previous_ids: set, ids: set) -> dict:
    """How a new read of a post differs from the last one."""
    return {"same": previous_ids == ids, "new": len(ids - previous_ids), "gone": len(previous_ids - ids)}


@dataclass
class Summary:
    n: int
    counts: dict                 # label -> count
    shares: dict                 # label -> share 0..1 (each reply counts once)
    liked_shares: dict           # label -> share 0..1 (each reply counts 1 + its likes)
    verdict: str                 # fuzzy response verdict
    margin: Optional[float]      # 95% margin of error for the largest share (None if n < 10)
    hate_n: int = 0
    hate_share: float = 0.0
    severe_n: int = 0
    offensive_n: int = 0


LABELS = ["Positive", "Neutral", "Negative"]


def summarize(df: pd.DataFrame) -> Summary:
    n = len(df)
    if n == 0:
        zero = {k: 0 for k in LABELS}
        return Summary(0, zero, {k: 0.0 for k in LABELS}, {k: 0.0 for k in LABELS}, "No reactions", None)

    counts = {k: int((df["predicted_sentiment"] == k).sum()) for k in LABELS}
    shares = {k: counts[k] / n for k in LABELS}
    weight = 1 + df["likes"].clip(lower=0)
    liked = {k: float(weight[df["predicted_sentiment"] == k].sum() / weight.sum()) for k in LABELS}
    verdict, _ = fuzzy.response_verdict(shares["Positive"], shares["Neutral"], shares["Negative"])

    top = max(shares.values())
    margin = 1.96 * math.sqrt(top * (1 - top) / n) if n >= 10 else None

    summary = Summary(n, counts, shares, liked, verdict, margin)
    if "is_hate" in df.columns:
        summary.hate_n = int(df["is_hate"].sum())
        summary.hate_share = summary.hate_n / n
        sev = df.get("hate_severity", pd.Series(dtype=str))
        summary.severe_n = int((sev == "Severe").sum())
        summary.offensive_n = int((sev == "Offensive").sum())
    return summary


def reception(post_sentiment: str, summary: Summary) -> str:
    """One line comparing the post's own tone with how it was received."""
    if summary.n == 0:
        return ""
    net = summary.shares["Positive"] - summary.shares["Negative"]
    if summary.verdict == "Mostly neutral":
        return "Most replies are neutral: people are commenting rather than taking sides."
    if summary.verdict == "Mixed":
        return "Opinion is split: supportive and critical replies are roughly balanced."
    if post_sentiment == "Positive" and net <= -0.15:
        return "Backlash: an upbeat post drew mostly negative replies."
    if post_sentiment == "Negative" and net >= 0.15:
        return "Pushback: the replies are far more positive than the post."
    if post_sentiment == "Negative" and net <= -0.15:
        return "The replies share the post's negativity."
    if post_sentiment == "Positive" and net >= 0.15:
        return "The replies echo the post's positive tone."
    return "The response leans " + ("positive." if net > 0 else "negative.")
