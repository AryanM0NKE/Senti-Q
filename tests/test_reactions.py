"""
tests/test_reactions.py — Reaction payload decoding, summary maths and the fuzzy verdict.
No models or network needed (the engine is stubbed where scoring is involved).
"""

import base64
import sys
import zlib
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import fuzzy, reactions


def _payload(**over):
    obj = {"v": 1, "u": "https://x.com/jack/status/20", "at": 1700000000000,
           "p": ["20", "jack", "Jack", "just setting up my twttr", "en", "2006-03-21T20:50:14.000Z", 5, 3, 1],
           "r": [["101", "amy", "Love this! 🎉", 12, "en"],
                 ["102", "bob", "This is terrible", 3, "en"],
                 ["103", "bob", "", 0, "en"],                      # empty -> dropped
                 ["104", "<script>", "hi", 0, "en"],               # bad handle -> blanked
                 ["20", "jack", "duplicate of the post", 0, "en"]],  # the post itself -> dropped
           "s": 2, "m": 4, "t": 7}
    obj.update(over)
    return obj


# --- payload ----------------------------------------------------------------

def test_roundtrip_decode_validates_and_cleans():
    c = reactions.decode(reactions.encode(_payload()))
    assert c.post["id"] == "20" and c.post["handle"] == "jack" and c.post["likes"] == 5
    assert [r["id"] for r in c.replies] == ["101", "102", "104"]
    assert c.replies[0]["text"] == "Love this! 🎉"
    assert c.replies[2]["handle"] == ""
    assert (c.self_replies, c.media_only, c.trimmed) == (2, 4, 7)
    assert c.collected_at.year == 2023


def test_decode_matches_browser_style_base64url():
    # The extension produces unpadded base64url of raw deflate; check we accept exactly that.
    raw = zlib.compressobj(9, zlib.DEFLATED, -15)
    data = raw.compress(b'{"v":1,"p":["5"],"r":[]}') + raw.flush()
    token = base64.urlsafe_b64encode(data).decode().rstrip("=")
    assert reactions.decode(token).post["id"] == "5"


@pytest.mark.parametrize("bad", ["", "not-base64!!", reactions.encode({"v": 2, "p": ["1"]}),
                                 reactions.encode({"v": 1, "p": ["abc"]})])
def test_decode_rejects_bad_payloads(bad):
    with pytest.raises(ValueError):
        reactions.decode(bad)


def test_decode_rejects_zip_bomb():
    bomb = reactions.encode({"v": 1, "p": ["1"], "r": [], "pad": "a" * 3_000_000})
    with pytest.raises(ValueError):
        reactions.decode(bomb)


# --- summary ----------------------------------------------------------------

def _scored(labels, likes=None, hate=None, severity=None):
    n = len(labels)
    return pd.DataFrame({"predicted_sentiment": labels, "likes": likes or [0] * n,
                         "is_hate": hate or [False] * n, "hate_severity": severity or ["None"] * n})


def test_summary_shares_margin_and_likes_weighting():
    df = _scored(["Negative"] * 6 + ["Positive"] * 2 + ["Neutral"] * 2,
                 likes=[0] * 6 + [100, 0] + [0, 0])
    s = reactions.summarize(df)
    assert s.n == 10 and s.counts == {"Positive": 2, "Neutral": 2, "Negative": 6}
    assert s.shares["Negative"] == pytest.approx(0.6)
    assert s.liked_shares["Positive"] > s.shares["Positive"]       # the liked reply pulls weight
    assert s.margin == pytest.approx(1.96 * (0.6 * 0.4 / 10) ** 0.5)
    assert s.verdict in ("Mostly negative", "Overwhelmingly negative")


def test_summary_counts_hate_and_handles_empty():
    df = _scored(["Negative"] * 4, hate=[True, True, False, False],
                 severity=["Severe", "Hate", "Offensive", "None"])
    s = reactions.summarize(df)
    assert (s.hate_n, s.severe_n, s.offensive_n) == (2, 1, 1) and s.hate_share == 0.5
    assert s.margin is None                                          # too few replies
    assert reactions.summarize(pd.DataFrame()).verdict == "No reactions"


def test_language_filter(monkeypatch):
    c = reactions.decode(reactions.encode(_payload(r=[["1", "a", "great", 0, "en"],
                                                      ["2", "b", "बहुत बढ़िया", 0, "hi"],
                                                      ["3", "c", "😂😂", 0, "qme"]])))
    monkeypatch.setattr(reactions.engine, "analyze", lambda texts: pd.DataFrame(
        {"predicted_sentiment": ["Positive"] * len(texts)}))
    df, skipped = reactions.score_replies(c)
    assert skipped == 1 and sorted(df["id"]) == ["1", "3"]


# --- fuzzy verdict ----------------------------------------------------------

@pytest.mark.parametrize("pos,neu,neg,expected", [
    (0.85, 0.10, 0.05, "Overwhelmingly positive"),
    (0.50, 0.30, 0.20, "Mostly positive"),
    (0.45, 0.10, 0.45, "Mixed"),
    (0.10, 0.80, 0.10, "Mostly neutral"),
    (0.20, 0.30, 0.50, "Mostly negative"),
    (0.05, 0.10, 0.85, "Overwhelmingly negative"),
])
def test_response_verdict(pos, neu, neg, expected):
    assert fuzzy.response_verdict(pos, neu, neg)[0] == expected


def test_reception_lines():
    s = reactions.summarize(_scored(["Negative"] * 7 + ["Positive"] * 2 + ["Neutral"]))
    assert reactions.reception("Positive", s).startswith("Backlash")
    s = reactions.summarize(_scored(["Positive"] * 7 + ["Negative"] * 2 + ["Neutral"]))
    assert reactions.reception("Negative", s).startswith("Pushback")
