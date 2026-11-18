"""
src/x_api.py — Official X API v2 integration for SentimentIQ.

Uses the bearer-token (App-Only) authentication, which is available
on the Free and Basic tiers of the X Developer Portal, via the
recent-search endpoint:

    GET https://api.twitter.com/2/tweets/search/recent

Documentation:
    https://developer.x.com/en/docs/x-api/tweets/search/api-reference/get-tweets-search-recent

IMPORTANT
---------
* Never hard-code credentials.  Read from environment variable X_BEARER_TOKEN.
* Never substitute synthetic data when live collection fails.
* Respect rate limits (15 requests / 15 min on Free tier).
* Store only data needed for this educational project.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env if present

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_ENDPOINT = "https://api.twitter.com/2/tweets/search/recent"

# Fields requested from the API
TWEET_FIELDS = [
    "id",
    "text",
    "created_at",
    "public_metrics",
    "lang",
    "author_id",
]

# X API rate-limit headers
_RATE_LIMIT_REMAINING = "x-rate-limit-remaining"
_RATE_LIMIT_RESET     = "x-rate-limit-reset"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class XPost:
    """Represents a single post fetched from the X API."""
    post_id:    str
    text:       str
    created_at: Optional[datetime] = None
    likes:      int = 0
    retweets:   int = 0
    replies:    int = 0
    impressions: int = 0
    lang:       str = "und"
    author_id:  str = ""
    username:   str = ""     # @handle without the "@", when known
    via:        str = "api"  # "api" (authenticated) | "oembed" (public fallback)

    @property
    def url(self) -> str:
        """Public URL for the post."""
        if self.username:
            return f"https://x.com/{self.username}/status/{self.post_id}"
        return f"https://x.com/i/web/status/{self.post_id}"

    @property
    def created_at_str(self) -> str:
        if self.created_at:
            return self.created_at.strftime("%Y-%m-%d %H:%M UTC")
        return "Unknown"


@dataclass
class FetchResult:
    """Container returned by the search functions."""
    posts:         list[XPost] = field(default_factory=list)
    total_fetched: int = 0
    query:         str = ""
    fetched_at:    str = ""
    error:         Optional[str] = None
    warning:       Optional[str] = None


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def get_bearer_token() -> Optional[str]:
    """
    Return the bearer token from the environment, or None if not set.
    Never logs or prints the actual token value.
    """
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    return token if token else None


def bearer_token_configured() -> bool:
    """True if a non-empty bearer token is available in the environment."""
    return bool(get_bearer_token())


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_recent_posts(
    query: str,
    max_results: int = 100,
    *,
    lang: str = "en",
    timeout: int = 30,
) -> FetchResult:
    """
    Search recent posts using the X API v2 recent-search endpoint.

    Parameters
    ----------
    query : str
        Search query using X query operators.
        Example: ``"election OR politics OR democracy -is:retweet lang:en"``
    max_results : int
        Total number of posts to retrieve (capped at 100 per API call on
        the Free / Basic tier; pagination handled automatically up to the
        maximum).  The Free tier allows up to 10 posts per request; the
        Basic tier allows up to 100.
    lang : str
        BCP-47 language tag appended to the query if not already present.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    FetchResult
    """
    token = get_bearer_token()
    if not token:
        return FetchResult(
            error=(
                "X_BEARER_TOKEN is not configured. "
                "Create a .env file (see .env.example) and add your token."
            )
        )

    headers = {"Authorization": f"Bearer {token}"}

    # Build query — add language filter if not already present
    full_query = query.strip()
    if f"lang:{lang}" not in full_query and "-is:retweet" not in full_query:
        full_query = f"({full_query}) lang:{lang} -is:retweet"
    elif f"lang:{lang}" not in full_query:
        full_query = f"({full_query}) lang:{lang}"

    posts: list[XPost] = []
    next_token: Optional[str] = None
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # X Free tier: max 10 per request; Basic: up to 100
    per_page = min(max_results, 10)   # safe default for Free tier

    while len(posts) < max_results:
        remaining_needed = max_results - len(posts)
        page_size = min(per_page, remaining_needed, 100)
        page_size = max(page_size, 10)   # API minimum is 10

        params: dict = {
            "query":       full_query,
            "max_results": page_size,
            "tweet.fields": ",".join(TWEET_FIELDS),
            "expansions":  "author_id",
            "user.fields": "username",
        }
        if next_token:
            params["next_token"] = next_token

        try:
            resp = requests.get(
                SEARCH_ENDPOINT,
                headers=headers,
                params=params,
                timeout=timeout,
            )
        except requests.exceptions.ConnectionError:
            return FetchResult(
                posts=posts,
                total_fetched=len(posts),
                query=query,
                fetched_at=fetched_at,
                error="Network connection failed. Check your internet connection.",
            )
        except requests.exceptions.Timeout:
            return FetchResult(
                posts=posts,
                total_fetched=len(posts),
                query=query,
                fetched_at=fetched_at,
                error=f"Request timed out after {timeout}s.",
            )

        # Rate-limit information
        remaining = resp.headers.get(_RATE_LIMIT_REMAINING)
        reset_ts   = resp.headers.get(_RATE_LIMIT_RESET)

        if resp.status_code == 429:
            reset_msg = ""
            if reset_ts:
                reset_dt = datetime.fromtimestamp(int(reset_ts), tz=timezone.utc)
                reset_msg = f" Rate limit resets at {reset_dt.strftime('%H:%M UTC')}."
            return FetchResult(
                posts=posts,
                total_fetched=len(posts),
                query=query,
                fetched_at=fetched_at,
                error=f"Rate limit exceeded.{reset_msg}  Please wait before trying again.",
            )

        if resp.status_code == 401:
            return FetchResult(
                error=(
                    "Authentication failed (HTTP 401). "
                    "Your X_BEARER_TOKEN is invalid or has been revoked. "
                    "Please generate a new bearer token in the X Developer Portal."
                )
            )

        if resp.status_code == 403:
            return FetchResult(
                error=(
                    "Access forbidden (HTTP 403). "
                    "Your X Developer App may not have access to the search endpoint. "
                    "Verify your app permissions and tier in the X Developer Portal."
                )
            )

        if resp.status_code not in (200, 201):
            try:
                detail = resp.json().get("detail") or resp.json().get("title") or resp.text
            except Exception:
                detail = resp.text
            return FetchResult(
                posts=posts,
                total_fetched=len(posts),
                query=query,
                fetched_at=fetched_at,
                error=f"X API error {resp.status_code}: {detail}",
            )

        data = resp.json()
        batch = data.get("data", [])

        if not batch:
            break   # No more results

        users = {u["id"]: u.get("username", "") for u in data.get("includes", {}).get("users", [])}
        for item in batch:
            metrics = item.get("public_metrics") or {}
            created_raw = item.get("created_at")
            created_dt: Optional[datetime] = None
            if created_raw:
                try:
                    created_dt = datetime.fromisoformat(
                        created_raw.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            posts.append(XPost(
                post_id    = item["id"],
                text       = item.get("text", ""),
                created_at = created_dt,
                likes      = metrics.get("like_count", 0),
                retweets   = metrics.get("retweet_count", 0),
                replies    = metrics.get("reply_count", 0),
                impressions= metrics.get("impression_count", 0),
                lang       = item.get("lang", "und"),
                author_id  = item.get("author_id", ""),
                username   = users.get(item.get("author_id", ""), ""),
            ))

        # Pagination
        meta = data.get("meta", {})
        next_token = meta.get("next_token")
        if not next_token:
            break

        # Polite pause between paginated requests
        time.sleep(0.5)

    warning = None
    if remaining is not None and int(remaining) <= 1:
        warning = "⚠️ You are close to the X API rate limit. Results may be incomplete."

    return FetchResult(
        posts=posts,
        total_fetched=len(posts),
        query=query,
        fetched_at=fetched_at,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Single-post lookup (paste a link or an ID)
# ---------------------------------------------------------------------------

LOOKUP_ENDPOINT = "https://api.twitter.com/2/tweets"
OEMBED_ENDPOINT = "https://publish.twitter.com/oembed"

_STATUS_URL_RE = re.compile(r"(?:twitter|x)\.com/[^/\s]+/status(?:es)?/(\d{1,25})", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"^\d{5,25}$")


def extract_post_ids(raw: str) -> list[str]:
    """
    Pull post IDs out of free text: full links (x.com / twitter.com, any handle,
    query strings allowed), or bare numeric IDs. Accepts several, separated by
    spaces, commas or new lines. Duplicates are removed, order is kept.
    """
    ids: list[str] = []
    for chunk in re.split(r"[\s,]+", raw.strip()):
        if not chunk:
            continue
        m = _STATUS_URL_RE.search(chunk)
        pid = m.group(1) if m else (chunk if _BARE_ID_RE.match(chunk) else None)
        if pid and pid not in ids:
            ids.append(pid)
    return ids


def _parse_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_posts_by_ids(ids: list[str], *, timeout: int = 30) -> FetchResult:
    """Look up specific posts (max 100) with the authenticated X API v2 endpoint."""
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    token = get_bearer_token()
    if not token:
        return FetchResult(error="X_BEARER_TOKEN is not configured.", fetched_at=fetched_at)

    params = {
        "ids": ",".join(ids[:100]),
        "tweet.fields": ",".join(TWEET_FIELDS),
        "expansions": "author_id",
        "user.fields": "username,name",
    }
    try:
        resp = requests.get(LOOKUP_ENDPOINT, headers={"Authorization": f"Bearer {token}"},
                            params=params, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return FetchResult(error="Network connection failed.", fetched_at=fetched_at)
    except requests.exceptions.Timeout:
        return FetchResult(error=f"Request timed out after {timeout}s.", fetched_at=fetched_at)

    if resp.status_code != 200:
        reasons = {
            401: "the bearer token is invalid or revoked",
            402: "the X developer account has no API credits or an inactive plan",
            403: "the app's access tier does not include reading posts",
            429: "the rate limit was hit",
        }
        why = reasons.get(resp.status_code, f"HTTP {resp.status_code}")
        return FetchResult(error=f"X API lookup failed: {why}.", fetched_at=fetched_at)

    data = resp.json()
    users = {u["id"]: u.get("username", "") for u in data.get("includes", {}).get("users", [])}
    posts = []
    for item in data.get("data", []):
        m = item.get("public_metrics") or {}
        posts.append(XPost(
            post_id=item["id"], text=item.get("text", ""), created_at=_parse_dt(item.get("created_at")),
            likes=m.get("like_count", 0), retweets=m.get("retweet_count", 0),
            replies=m.get("reply_count", 0), impressions=m.get("impression_count", 0),
            lang=item.get("lang", "und"), author_id=item.get("author_id", ""),
            username=users.get(item.get("author_id", ""), ""), via="api",
        ))

    missing = len(ids[:100]) - len(posts)
    warning = (f"{missing} post(s) could not be retrieved (deleted, private or protected)."
               if missing > 0 else None)
    return FetchResult(posts=posts, total_fetched=len(posts), query=", ".join(ids),
                       fetched_at=fetched_at, warning=warning)


def _oembed_text(html_doc: str) -> str:
    """Extract the post text from the <p> inside X's oEmbed blockquote."""
    m = re.search(r"<p[^>]*>(.*?)</p>", html_doc, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    inner = re.sub(r"<br\s*/?>", "\n", m.group(1), flags=re.IGNORECASE)
    inner = re.sub(r"<[^>]+>", "", inner)          # drop links/tags, keep their text
    return unescape(inner).strip()


def fetch_post_oembed(post_id: str, username: str = "i", *, timeout: int = 15) -> Optional[XPost]:
    """
    Public, token-free fallback: X's oEmbed endpoint returns the text of any
    public post. No engagement metrics are available this way. Returns None if
    the post is private, deleted or unreachable.
    """
    url = f"https://twitter.com/{username}/status/{post_id}"
    try:
        resp = requests.get(OEMBED_ENDPOINT, params={"url": url, "omit_script": "1", "dnt": "true"},
                            timeout=timeout)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    payload = resp.json()
    text = _oembed_text(payload.get("html", ""))
    if not text:
        return None

    author_url = payload.get("author_url", "")
    handle = author_url.rstrip("/").split("/")[-1] if author_url else ""
    created = None
    date_m = re.search(r">([A-Z][a-z]+ \d{1,2}, \d{4})</a>\s*</blockquote>", payload.get("html", ""))
    if date_m:
        try:
            created = datetime.strptime(date_m.group(1), "%B %d, %Y").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return XPost(post_id=post_id, text=text, created_at=created, username=handle, via="oembed")


def lookup_posts(raw_refs: str) -> FetchResult:
    """
    Resolve pasted links / IDs to posts. Tries the authenticated API first; for
    anything it could not return (no token, plan without read access, etc.)
    falls back to the public oEmbed endpoint.
    """
    ids = extract_post_ids(raw_refs)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not ids:
        return FetchResult(error="No post link or ID found. Paste a link like "
                                 "https://x.com/user/status/123456789 or just the numeric ID.",
                           fetched_at=fetched_at)

    posts: list[XPost] = []
    notes: list[str] = []
    if bearer_token_configured():
        api = fetch_posts_by_ids(ids)
        if api.error:
            notes.append(f"{api.error} Used the public fallback instead (no engagement counts).")
        else:
            posts = api.posts
            if api.warning:
                notes.append(api.warning)

    found = {p.post_id for p in posts}
    for pid in (i for i in ids if i not in found):
        p = fetch_post_oembed(pid)
        if p:
            posts.append(p)

    if not posts:
        return FetchResult(error="Could not retrieve any of those posts. They may be private, "
                                 "deleted, or X may be blocking the request.",
                           query=", ".join(ids), fetched_at=fetched_at)
    order = {pid: n for n, pid in enumerate(ids)}
    posts.sort(key=lambda p: order.get(p.post_id, 0))
    return FetchResult(posts=posts, total_fetched=len(posts), query=", ".join(ids),
                       fetched_at=fetched_at, warning=" ".join(notes) or None)
