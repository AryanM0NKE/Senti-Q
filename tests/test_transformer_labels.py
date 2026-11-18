"""
tests/test_transformer_labels.py — Label mapping across the different pre-trained models.
Pure functions: no models or network needed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import transformer_models as tm


@pytest.mark.parametrize("scores,expected", [
    ({"negative": 0.1, "neutral": 0.2, "positive": 0.7}, [0.1, 0.2, 0.7]),       # cardiffnlp -latest
    ({"label_0": 0.6, "label_1": 0.3, "label_2": 0.1}, [0.6, 0.3, 0.1]),         # cardiffnlp older (LABEL_0..2)
    ({"neg": 0.05, "neu": 0.15, "pos": 0.8}, [0.05, 0.15, 0.8]),                 # pysentimiento BERTweet
    ({"pos": 0.8, "neu": 0.15, "neg": 0.05}, [0.05, 0.15, 0.8]),                 # order must not matter
])
def test_sentiment_row_handles_every_label_style(scores, expected):
    assert tm.sentiment_row(scores) == pytest.approx(expected)


@pytest.mark.parametrize("scores,expected", [
    ({"not-hate": 0.3, "hate": 0.7}, 0.7),            # cardiffnlp hate-latest
    ({"nothate": 0.9, "hate": 0.1}, 0.1),             # facebook dynabench
    ({"label_0": 0.4, "label_1": 0.6}, 0.6),          # generic
])
def test_hate_row_picks_the_hate_probability(scores, expected):
    assert tm.hate_row(scores) == pytest.approx(expected)


def test_ensemble_averages_models(monkeypatch):
    outputs = {"a": [{"negative": 1.0, "neutral": 0.0, "positive": 0.0}],
               "b": [{"negative": 0.0, "neutral": 0.0, "positive": 1.0}]}
    monkeypatch.setattr(tm, "_scores", lambda model, texts: outputs[model])
    monkeypatch.setattr(tm, "SENTIMENT_MODELS", ["a", "b"])
    assert tm.sentiment_probs(["x"]).tolist() == [[0.5, 0.0, 0.5]]
    assert tm.sentiment_probs(["x"], model="a").tolist() == [[1.0, 0.0, 0.0]]   # an explicit model is used alone
