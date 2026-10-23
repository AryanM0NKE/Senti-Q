"""
tests/test_prediction.py — Unit tests for src/predict.py

Tests that do NOT require trained model artefacts use mocking.
Tests that DO require artefacts are skipped gracefully when models/ is empty.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_clf(classes=None):
    """Return a mock sklearn classifier."""
    if classes is None:
        classes = ["Negative", "Neutral", "Positive"]
    clf = MagicMock()
    clf.classes_ = classes

    def _predict_proba(X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        # Return equal probabilities for simplicity
        row = np.array([1/3, 1/3, 1/3])
        return np.tile(row, (n, 1))

    clf.predict_proba.side_effect = _predict_proba
    return clf


def _make_mock_vectorizer():
    """Return a mock TfidfVectorizer."""
    vec = MagicMock()

    def _transform(texts):
        # Return a 2-D array-like mock with shape attribute
        n = len(texts)
        arr = MagicMock()
        arr.shape = (n, 100)
        return arr

    vec.transform.side_effect = _transform
    return vec


# ---------------------------------------------------------------------------
# model_is_trained
# ---------------------------------------------------------------------------

class TestModelIsTrained:

    def test_returns_false_when_files_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(predict, "VECTORIZER_PATH", tmp_path / "vec.joblib")
        monkeypatch.setattr(predict, "CLASSIFIER_PATH", tmp_path / "clf.joblib")
        assert predict.model_is_trained() is False

    def test_returns_true_when_both_files_present(self, tmp_path, monkeypatch):
        vec_path = tmp_path / "vec.joblib"
        clf_path = tmp_path / "clf.joblib"
        vec_path.touch()
        clf_path.touch()
        monkeypatch.setattr(predict, "VECTORIZER_PATH", vec_path)
        monkeypatch.setattr(predict, "CLASSIFIER_PATH", clf_path)
        assert predict.model_is_trained() is True

    def test_returns_false_when_only_one_file_present(self, tmp_path, monkeypatch):
        vec_path = tmp_path / "vec.joblib"
        vec_path.touch()
        monkeypatch.setattr(predict, "VECTORIZER_PATH", vec_path)
        monkeypatch.setattr(predict, "CLASSIFIER_PATH", tmp_path / "clf.joblib")
        assert predict.model_is_trained() is False


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------

class TestLoadMetadata:

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(predict, "METADATA_PATH", tmp_path / "model_metadata.json")
        monkeypatch.setattr(predict, "_metadata", None)
        assert predict.load_metadata() is None

    def test_loads_valid_json(self, tmp_path, monkeypatch):
        import json
        meta = {"accuracy": 0.72, "macro_f1": 0.68}
        path = tmp_path / "model_metadata.json"
        path.write_text(json.dumps(meta))
        monkeypatch.setattr(predict, "METADATA_PATH", path)
        monkeypatch.setattr(predict, "_metadata", None)
        result = predict.load_metadata()
        assert result == meta


# ---------------------------------------------------------------------------
# predict_batch — with mocked artefacts
# ---------------------------------------------------------------------------

class TestPredictBatch:

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Clear module-level cache before each test."""
        predict._vectorizer = None
        predict._classifier = None
        yield
        predict._vectorizer = None
        predict._classifier = None

    def test_raises_when_artefacts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(predict, "VECTORIZER_PATH", tmp_path / "vec.joblib")
        monkeypatch.setattr(predict, "CLASSIFIER_PATH", tmp_path / "clf.joblib")
        with pytest.raises(FileNotFoundError):
            predict.predict_batch(["some text"])

    def test_returns_dataframe(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        result = predict.predict_batch(["I love democracy", "Terrible policy"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_returns_required_columns(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        result = predict.predict_batch(["Test text here"])
        required = {
            "text", "cleaned_text", "predicted_sentiment",
            "prob_negative", "prob_neutral", "prob_positive",
        }
        assert required.issubset(set(result.columns))

    def test_predicted_sentiment_is_valid_label(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        texts = [
            "I love this government policy",
            "This is a terrible situation",
            "The meeting was held on Tuesday",
        ]
        result = predict.predict_batch(texts)
        for label in result["predicted_sentiment"]:
            assert label in {"Positive", "Negative", "Neutral"}

    def test_probabilities_sum_to_one(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        result = predict.predict_batch(["Political news today"])
        for _, row in result.iterrows():
            total = row["prob_positive"] + row["prob_neutral"] + row["prob_negative"]
            assert abs(total - 1.0) < 0.02, f"Probs don't sum to 1: {total}"

    def test_handles_very_short_text(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        # Very short text cleans to < 3 chars — should default to Neutral
        result = predict.predict_batch(["hi"])
        assert len(result) == 1
        assert result.iloc[0]["predicted_sentiment"] == "Neutral"

    def test_handles_empty_list(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        result = predict.predict_batch([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_probabilities_are_between_0_and_1(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        result = predict.predict_batch(["Election day coverage", "Government policy debate"])
        for col in ["prob_positive", "prob_neutral", "prob_negative"]:
            assert (result[col] >= 0).all()
            assert (result[col] <= 1).all()


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------

class TestPredictSingle:

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        predict._vectorizer = None
        predict._classifier = None
        yield
        predict._vectorizer = None
        predict._classifier = None

    def test_returns_dict(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        result = predict.predict_single("Democracy is important")
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self, monkeypatch):
        monkeypatch.setattr(predict, "_vectorizer", _make_mock_vectorizer())
        monkeypatch.setattr(predict, "_classifier", _make_mock_clf())

        result = predict.predict_single("The election results were announced")
        assert "predicted_sentiment" in result
        assert "prob_positive" in result
        assert "prob_negative" in result
        assert "prob_neutral" in result
