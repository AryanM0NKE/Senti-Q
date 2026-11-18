"""
tests/test_hate_speech.py — Tests for src/hate_speech.py.

The inference tests use a stub classifier, so no trained artefacts are needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import hate_speech


@pytest.fixture
def stub_models(monkeypatch):
    vec = MagicMock()
    vec.transform.side_effect = lambda texts: np.array(texts, dtype=object).reshape(-1, 1)

    clf = MagicMock()
    clf.classes_ = np.array([0, 1])
    # "hateful" texts -> 0.9 hate, everything else -> 0.1
    clf.predict_proba.side_effect = lambda X: np.array(
        [[0.1, 0.9] if "vermin" in str(x[0]) else [0.9, 0.1] for x in X])

    monkeypatch.setattr(hate_speech, "_vectorizer", vec)
    monkeypatch.setattr(hate_speech, "_classifier", clf)


def test_predict_batch_flags_and_scores(stub_models):
    out = hate_speech.predict_batch(["you vermin", "lovely weather today"])
    assert list(out["hate_label"]) == ["Hate", "Not hate"]
    assert list(out["is_hate"]) == [True, False]
    assert out["prob_hate"].iloc[0] == pytest.approx(0.9)


def test_empty_text_scores_zero(stub_models):
    out = hate_speech.predict_batch(["", "!!", "you vermin"])
    assert list(out["prob_hate"][:2]) == [0.0, 0.0]
    assert bool(out["is_hate"].iloc[2]) is True


def test_missing_model_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(hate_speech, "_vectorizer", None)
    monkeypatch.setattr(hate_speech, "VECTORIZER_PATH", tmp_path / "nope.joblib")
    monkeypatch.setattr(hate_speech, "CLASSIFIER_PATH", tmp_path / "nope2.joblib")
    with pytest.raises(FileNotFoundError):
        hate_speech.predict_batch(["hello"])
