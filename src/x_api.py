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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

    @property
    def url(self) -> str:
        """Construct a public URL for the post (requires author_id)."""
        if self.author_id:
            return f"https://x.com/i/web/status/{self.post_id}"
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
