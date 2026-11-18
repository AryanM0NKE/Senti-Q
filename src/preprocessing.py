"""
src/preprocessing.py — Text cleaning and normalisation for Senti-Q.

Cleans raw tweet / post text for TF-IDF feature extraction while preserving
meaningful sentiment signals (capitalisation, punctuation counts, etc.).
"""

from __future__ import annotations

import re
import string


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_URL_PATTERN     = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")   # keep the word, drop the hash
_MULTI_SPACE     = re.compile(r"\s+")
_NON_ASCII       = re.compile(r"[^\x00-\x7F]+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str, *, keep_hashtag_words: bool = True) -> str:
    """
    Normalise a single tweet / post string for model input.

    Steps (in order):
    1. Lowercase.
    2. Remove URLs.
    3. Remove @mentions.
    4. Replace #hashtag with its word (optional; default True).
    5. Remove non-ASCII characters (emoji, accented).
    6. Remove remaining punctuation (except apostrophes in contractions).
    7. Collapse whitespace.

    Parameters
    ----------
    text : str
        Raw post text.
    keep_hashtag_words : bool
        If True, ``#election`` becomes ``election``.
        If False, the entire token is removed.

    Returns
    -------
    str
        Cleaned text suitable for TF-IDF vectorisation.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _MENTION_PATTERN.sub(" ", text)

    if keep_hashtag_words:
        text = _HASHTAG_PATTERN.sub(r"\1", text)
    else:
        text = _HASHTAG_PATTERN.sub(" ", text)

    text = _NON_ASCII.sub(" ", text)

    # Remove punctuation except apostrophes (don't → dont retains meaning)
    table = str.maketrans(
        "", "",
        string.punctuation.replace("'", ""),
    )
    text = text.translate(table)
    text = text.replace("'", "")   # now remove apostrophes too

    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def preprocess_dataframe(
    df,
    text_col: str = "text",
    *,
    drop_duplicates: bool = True,
    min_length: int = 3,
) -> "pd.DataFrame":   # noqa: F821  (avoid circular import)
    """
    Apply ``clean_text`` to a DataFrame column and remove unusable rows.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Name of the column containing raw post text.
    drop_duplicates : bool
        Drop rows with identical raw text before cleaning.
    min_length : int
        Minimum number of characters in cleaned text for a row to be kept.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``cleaned_text`` column and
        unusable rows removed.
    """
    import pandas as pd

    df = df.copy()

    # Drop rows with missing text
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip() != ""]

    if drop_duplicates:
        df = df.drop_duplicates(subset=[text_col])

    df["cleaned_text"] = df[text_col].astype(str).apply(clean_text)

    # Remove rows where cleaned text is too short to be informative
    df = df[df["cleaned_text"].str.len() >= min_length]
    df = df.reset_index(drop=True)
    return df
