"""
tests/test_x_links.py — Offline tests for opening X posts from links.
No network access needed (HTTP calls are mocked).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import x_links


def test_extract_refs_from_links_and_bare_ids():
    raw = ("https://x.com/jack/status/20?s=20  https://twitter.com/nasa/status/1700000000000000000\n"
           "1234567890123456789, https://x.com/jack/status/20 https://x.com/i/web/status/555555")
    assert x_links.extract_post_refs(raw) == [
        ("20", "jack"), ("1700000000000000000", "nasa"), ("1234567890123456789", ""), ("555555", ""),
    ]


def test_extract_ignores_garbage():
    assert x_links.extract_post_ids("hello world https://example.com/status/5") == []
    assert x_links.extract_post_ids("") == []


def test_oembed_text_strips_tags_and_unescapes():
    html_doc = ('<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Tom &amp; Jerry<br>'
                '<a href="https://t.co/x">link</a> #fun</p>&mdash; A (@a) '
                '<a href="https://twitter.com/a/status/1">March 21, 2006</a></blockquote>')
    assert x_links._oembed_text(html_doc) == "Tom & Jerry\nlink #fun"


def _resp(status, payload=None):
    r = MagicMock(status_code=status)
    r.json.return_value = payload or {}
    return r


def test_fetch_post_parses_oembed_payload():
    payload = {"html": '<blockquote><p>hello world</p>&mdash; Jack (@jack) '
                       '<a href="https://twitter.com/jack/status/20">March 21, 2006</a></blockquote>',
               "author_name": "jack", "author_url": "https://twitter.com/jack"}
    with patch.object(x_links.requests, "get", return_value=_resp(200, payload)):
        p = x_links.fetch_post("20", "jack")
    assert p.text == "hello world" and p.username == "jack"
    assert p.created_at.year == 2006 and p.url == "https://x.com/jack/status/20"


def test_fetch_post_handles_http_error():
    with patch.object(x_links.requests, "get", return_value=_resp(404)):
        assert x_links.fetch_post("999") is None


def test_open_posts_reports_unreachable_and_bad_input():
    assert x_links.open_posts("not a link").errors
    with patch.object(x_links, "fetch_post", side_effect=lambda pid, h: None):
        res = x_links.open_posts("https://x.com/a/status/123")
    assert not res.posts and "123" in res.errors[0]
