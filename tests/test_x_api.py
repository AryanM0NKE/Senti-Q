"""
tests/test_x_api.py — Offline unit tests for post-link parsing and oEmbed parsing.
No network access or API token required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import x_api


def test_extract_ids_from_links_and_bare_ids():
    raw = ("https://x.com/jack/status/20?s=20  https://twitter.com/a/status/1700000000000000000\n"
           "1234567890123456789, https://x.com/jack/status/20")
    assert x_api.extract_post_ids(raw) == ["20", "1700000000000000000", "1234567890123456789"]


def test_extract_ids_ignores_garbage():
    assert x_api.extract_post_ids("hello world https://example.com/status/5") == []
    assert x_api.extract_post_ids("") == []


def test_oembed_text_strips_tags_and_unescapes():
    html_doc = ('<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Tom &amp; Jerry<br>'
                '<a href="https://t.co/x">link</a> #fun</p>&mdash; A (@a) '
                '<a href="https://twitter.com/a/status/1">March 21, 2006</a></blockquote>')
    assert x_api._oembed_text(html_doc) == "Tom & Jerry\nlink #fun"


def test_lookup_without_ids_returns_error():
    result = x_api.lookup_posts("no links here")
    assert result.error and not result.posts


def test_lookup_falls_back_to_oembed_when_api_fails(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "dummy")
    monkeypatch.setattr(x_api, "fetch_posts_by_ids",
                        lambda ids: x_api.FetchResult(error="X API lookup failed: no access."))
    post = x_api.XPost(post_id="20", text="hello", username="jack", via="oembed")
    monkeypatch.setattr(x_api, "fetch_post_oembed", lambda pid, *a, **k: post)
    result = x_api.lookup_posts("https://x.com/jack/status/20")
    assert result.posts == [post]
    assert "public fallback" in result.warning


def test_fetch_post_oembed_handles_http_error():
    resp = MagicMock(status_code=404)
    with patch.object(x_api.requests, "get", return_value=resp):
        assert x_api.fetch_post_oembed("999") is None
