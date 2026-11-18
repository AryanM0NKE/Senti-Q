"""
src/x_links.py — Open X (Twitter) posts from their links.

Senti-Q never searches or queries X. It only opens the specific posts whose links
the user pastes into the app (or sends from the "Senti-Q for X" browser extension),
using X's public oEmbed endpoint: the same mechanism any website uses to embed a
post. No account, API key or token is involved, and only public posts can be opened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from typing import Optional

import requests

OEMBED_ENDPOINT = "https://publish.twitter.com/oembed"
MAX_LINKS = 20

# Matches x.com/<handle>/status/<id>, twitter.com/..., and X's own x.com/i/web/status/<id> form
_STATUS_URL_RE = re.compile(r"(?:twitter|x)\.com/(?:i/web|([^/\s?#]+))/status(?:es)?/(\d{1,25})", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"^\d{5,25}$")


@dataclass
class XPost:
    post_id: str
    text: str
    created_at: Optional[datetime] = None
    username: str = ""        # @handle without the "@"
    display_name: str = ""
    embed_html: str = ""      # X's official embed markup

    @property
    def url(self) -> str:
        handle = self.username or "i"
        return f"https://x.com/{handle}/status/{self.post_id}"


@dataclass
class OpenResult:
    posts: list[XPost] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def extract_post_refs(raw: str) -> list[tuple[str, str]]:
    """
    Find post links (x.com / twitter.com, any handle, query strings allowed) or bare
    numeric post IDs in free text. Returns (post_id, handle) pairs in order, without
    duplicates; handle is "" for bare IDs.
    """
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in re.split(r"[\s,]+", (raw or "").strip()):
        if not chunk:
            continue
        m = _STATUS_URL_RE.search(chunk)
        if m:
            handle, pid = m.group(1) or "", m.group(2)
            handle = "" if handle.lower() == "i" else handle
        elif _BARE_ID_RE.match(chunk):
            handle, pid = "", chunk
        else:
            continue
        if pid not in seen:
            seen.add(pid)
            refs.append((pid, handle))
    return refs


def extract_post_ids(raw: str) -> list[str]:
    return [pid for pid, _ in extract_post_refs(raw)]


def _oembed_text(html_doc: str) -> str:
    """Extract the post text from the <p> inside X's oEmbed blockquote."""
    m = re.search(r"<p[^>]*>(.*?)</p>", html_doc, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    inner = re.sub(r"<br\s*/?>", "\n", m.group(1), flags=re.IGNORECASE)
    inner = re.sub(r"<[^>]+>", "", inner)          # drop links/tags, keep their text
    return unescape(inner).strip()


def fetch_post(post_id: str, handle: str = "", *, timeout: int = 15) -> Optional[XPost]:
    """Open one public post by ID. Returns None if it is private, deleted or unreachable."""
    url = f"https://twitter.com/{handle or 'i'}/status/{post_id}"
    try:
        resp = requests.get(OEMBED_ENDPOINT, params={"url": url, "omit_script": "1", "dnt": "true"},
                            timeout=timeout)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    payload = resp.json()
    html_doc = payload.get("html", "")
    text = _oembed_text(html_doc)
    if not text:
        return None

    author_url = payload.get("author_url", "")
    username = author_url.rstrip("/").split("/")[-1] if author_url else handle
    created = None
    date_m = re.search(r">([A-Z][a-z]+ \d{1,2}, \d{4})</a>\s*</blockquote>", html_doc)
    if date_m:
        try:
            created = datetime.strptime(date_m.group(1), "%B %d, %Y").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return XPost(post_id=post_id, text=text, created_at=created, username=username,
                 display_name=payload.get("author_name", ""), embed_html=html_doc)


def open_posts(raw: str) -> OpenResult:
    """Open every post linked in `raw` (up to MAX_LINKS), keeping the order given."""
    refs = extract_post_refs(raw)
    result = OpenResult()
    if not refs:
        result.errors.append("That doesn't look like a link to an X post. It should look like "
                             "https://x.com/username/status/1234567890.")
        return result
    if len(refs) > MAX_LINKS:
        result.errors.append(f"Only the first {MAX_LINKS} links were opened.")
        refs = refs[:MAX_LINKS]
    for pid, handle in refs:
        post = fetch_post(pid, handle)
        if post:
            result.posts.append(post)
        else:
            result.errors.append(f"Couldn't open post {pid}. It may be private, deleted or age-restricted.")
    return result
