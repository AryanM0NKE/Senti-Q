"""
src/transformer_models.py — Pre-trained Twitter RoBERTa models (CardiffNLP).

* Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
  RoBERTa-base pre-trained on ~124M tweets, fine-tuned on TweetEval sentiment.
* Hate:      cardiffnlp/twitter-roberta-base-hate-latest
  Same backbone, fine-tuned on 13 English hate-speech datasets.

Weights download from the Hugging Face Hub on first use (~500 MB each) and are
cached in ~/.cache/huggingface afterwards. Everything here is optional: if
`torch` / `transformers` are not installed, `available()` is False and the app
falls back to the TF-IDF models.
"""

from __future__ import annotations

import importlib.util
import os
import re

import numpy as np

SENTIMENT_MODEL = os.environ.get("SENTIMENTIQ_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
HATE_MODEL      = os.environ.get("SENTIMENTIQ_HATE_MODEL",      "cardiffnlp/twitter-roberta-base-hate-latest")

BATCH_SIZE = 32
MAX_LENGTH = 128

_pipes: dict = {}


def available() -> bool:
    """True if torch and transformers are importable."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def _prep(text: str) -> str:
    """Model-card preprocessing: anonymise mentions and links."""
    if not isinstance(text, str) or not text.strip():
        return "."
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+", "http", text)
    return text


def _pipe(model_name: str):
    if model_name not in _pipes:
        from transformers import pipeline
        _pipes[model_name] = pipeline(
            "text-classification", model=model_name, tokenizer=model_name,
            top_k=None, truncation=True, max_length=MAX_LENGTH, device=-1,
        )
    return _pipes[model_name]


def _scores(model_name: str, texts: list[str]) -> list[dict]:
    out = _pipe(model_name)([_prep(t) for t in texts], batch_size=BATCH_SIZE)
    return [{d["label"].lower(): float(d["score"]) for d in row} for row in out]


def sentiment_probs(texts: list[str]) -> np.ndarray:
    """(n, 3) array of [negative, neutral, positive] probabilities."""
    if not texts:
        return np.empty((0, 3))
    rows = _scores(SENTIMENT_MODEL, texts)
    return np.array([[r.get("negative", 0), r.get("neutral", 0), r.get("positive", 0)] for r in rows])


def hate_probs(texts: list[str]) -> np.ndarray:
    """(n,) array of P(hate)."""
    if not texts:
        return np.empty(0)
    rows = _scores(HATE_MODEL, texts)
    probs = []
    for r in rows:
        hate = [v for k, v in r.items() if "hate" in k and "not" not in k and "non" not in k]
        if not hate:  # generic LABEL_0 / LABEL_1 naming
            hate = [r.get("label_1", 0.0)]
        probs.append(hate[0])
    return np.array(probs)


def is_loaded() -> bool:
    return SENTIMENT_MODEL in _pipes and HATE_MODEL in _pipes
