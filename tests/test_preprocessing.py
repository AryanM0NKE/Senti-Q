"""
tests/test_preprocessing.py — Unit tests for src/preprocessing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.preprocessing import clean_text, preprocess_dataframe


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:

    def test_lowercases_input(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_urls(self):
        result = clean_text("Check this out https://example.com and www.foo.bar now")
        assert "http" not in result
        assert "www" not in result

    def test_removes_mentions(self):
        result = clean_text("Hey @JohnDoe, what do you think?")
        assert "@" not in result
        assert "johndoe" not in result

    def test_keeps_hashtag_words_by_default(self):
        result = clean_text("I love #democracy and #freedom")
        assert "democracy" in result
        assert "freedom" in result
        assert "#" not in result

    def test_removes_hashtag_words_when_disabled(self):
        result = clean_text("I love #democracy", keep_hashtag_words=False)
        assert "democracy" not in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! It's a great day.")
        assert "," not in result
        assert "!" not in result
        assert "." not in result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_non_string(self):
        assert clean_text(None) == ""   # type: ignore
        assert clean_text(42) == ""     # type: ignore

    def test_collapses_whitespace(self):
        result = clean_text("hello   world    foo")
        assert "  " not in result
        assert result == "hello world foo"

    def test_removes_non_ascii(self):
        result = clean_text("café résumé naïve")
        # Non-ASCII removed, words may remain as partial ASCII
        for ch in result:
            assert ord(ch) < 128

    def test_preserves_sentiment_words(self):
        result = clean_text("I absolutely love this amazing policy!")
        assert "love" in result
        assert "amazing" in result

    def test_full_pipeline(self):
        raw = "@politician Great job on #ClimatePolicy! https://example.com 🌍"
        result = clean_text(raw)
        assert "climatepolicy" in result
        assert "http" not in result
        assert "@" not in result


# ---------------------------------------------------------------------------
# preprocess_dataframe
# ---------------------------------------------------------------------------

class TestPreprocessDataframe:

    def _make_df(self, texts):
        return pd.DataFrame({"text": texts})

    def test_adds_cleaned_text_column(self):
        df = self._make_df(["Hello World", "Good morning"])
        result = preprocess_dataframe(df)
        assert "cleaned_text" in result.columns

    def test_drops_empty_text_rows(self):
        df = self._make_df(["", "  ", "Valid text here"])
        result = preprocess_dataframe(df)
        assert len(result) == 1

    def test_drops_none_rows(self):
        df = pd.DataFrame({"text": [None, "Valid text", None]})
        result = preprocess_dataframe(df)
        assert len(result) == 1

    def test_drops_duplicate_rows(self):
        df = self._make_df(["Same text", "Same text", "Different text"])
        result = preprocess_dataframe(df, drop_duplicates=True)
        assert len(result) == 2

    def test_keeps_duplicates_when_disabled(self):
        df = self._make_df(["Same text", "Same text"])
        result = preprocess_dataframe(df, drop_duplicates=False)
        assert len(result) == 2

    def test_filters_by_min_length(self):
        df = self._make_df(["Hi", "Hello there friend", "Ok"])
        result = preprocess_dataframe(df, min_length=5)
        assert all(result["cleaned_text"].str.len() >= 5)

    def test_resets_index(self):
        df = self._make_df(["Text one", "Text two", "Text three"])
        result = preprocess_dataframe(df)
        assert list(result.index) == list(range(len(result)))

    def test_custom_text_col(self):
        df = pd.DataFrame({"content": ["Hello world", "Good day"]})
        result = preprocess_dataframe(df, text_col="content")
        assert "cleaned_text" in result.columns

    def test_does_not_modify_original_df(self):
        df = self._make_df(["Hello", "World"])
        _ = preprocess_dataframe(df)
        assert "cleaned_text" not in df.columns
