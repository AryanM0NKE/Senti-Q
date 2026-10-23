"""
src/predict.py — Inference helpers for SentimentIQ.

Loads the trained TF-IDF vectorizer and Logistic Regression classifier
from disk and runs predictions on new text.

The module caches the loaded artefacts so Streamlit reruns do not
reload from disk on every interaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import clean_text

ROOT = Path(__file__).parent.parent

VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = ROOT / "models" / "lr_classifier.joblib"
METADATA_PATH   = ROOT / "models" / "model_metadata.json"

LABEL_ORDER = ["Negative", "Neutral", "Positive"]


# ---------------------------------------------------------------------------
# Cached loader
# ---------------------------------------------------------------------------

_vectorizer = None
_classifier = None
_metadata   = None


def _load_artefacts() -> tuple:
    """Load vectorizer + classifier from disk (once per process)."""
    global _vectorizer, _classifier
    if _vectorizer is None:
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Vectorizer not found at {VECTORIZER_PATH}.\n"
                "Run training first:  python -m src.train_model"
            )
        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(
                f"Classifier not found at {CLASSIFIER_PATH}.\n"
                "Run training first:  python -m src.train_model"
            )
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _vectorizer, _classifier


def load_metadata() -> Optional[dict]:
    """Return model metadata dict, or None if the file does not exist."""
    global _metadata
    if _metadata is None:
        if METADATA_PATH.exists():
            with open(METADATA_PATH) as fh:
                _metadata = json.load(fh)
    return _metadata


def model_is_trained() -> bool:
    """Return True if both model artefacts exist on disk."""
    return VECTORIZER_PATH.exists() and CLASSIFIER_PATH.exists()


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def predict_batch(texts: list[str]) -> pd.DataFrame:
    """
    Run sentiment prediction on a list of raw text strings.

    Parameters
    ----------
    texts : list[str]
        Raw post texts (URLs, mentions, hashtags will be cleaned internally).

    Returns
    -------
    pd.DataFrame
        Columns: ``text``, ``cleaned_text``, ``predicted_sentiment``,
        ``prob_negative``, ``prob_neutral``, ``prob_positive``.

    Raises
    ------
    FileNotFoundError
        If model artefacts are not present on disk.
    """
    vectorizer, clf = _load_artefacts()

    cleaned = [clean_text(t) for t in texts]

    valid_mask = [len(c) >= 3 for c in cleaned]
    valid_cleaned = [c for c, ok in zip(cleaned, valid_mask) if ok]

    rows = []
    proba_matrix = (
        clf.predict_proba(vectorizer.transform(valid_cleaned))
        if valid_cleaned
        else np.empty((0, 3))
    )
    classes = list(clf.classes_)   # order may differ from LABEL_ORDER

    v_idx = 0
    for raw, clean, ok in zip(texts, cleaned, valid_mask):
        if ok:
            proba = proba_matrix[v_idx]
            v_idx += 1
            label = classes[int(np.argmax(proba))]
            prob_dict = {c: float(p) for c, p in zip(classes, proba)}
        else:
            label = "Neutral"
            prob_dict = {"Negative": 0.0, "Neutral": 1.0, "Positive": 0.0}

        rows.append({
            "text":                raw,
            "cleaned_text":        clean,
            "predicted_sentiment": label,
            "prob_negative":       round(prob_dict.get("Negative", 0.0), 4),
            "prob_neutral":        round(prob_dict.get("Neutral",  0.0), 4),
            "prob_positive":       round(prob_dict.get("Positive", 0.0), 4),
        })

    return pd.DataFrame(rows)


def predict_single(text: str) -> dict:
    """
    Predict sentiment for a single text string.

    Returns
    -------
    dict with keys:
        predicted_sentiment, prob_negative, prob_neutral, prob_positive
    """
    df = predict_batch([text])
    return df.iloc[0].to_dict()
