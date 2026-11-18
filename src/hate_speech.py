# -*- coding: utf-8 -*-
"""
src/hate_speech.py — Hate-speech detection for Senti-Q.

Binary classifier (Not hate / Hate) trained on the TweetEval *hate* task
(SemEval-2019 Task 5: hate speech against immigrants and women), using the same
TF-IDF + Logistic Regression approach as the sentiment model.

Train:     python -m src.hate_speech
Inference: hate_speech.predict_batch(["some text", ...])

Artefacts are written to models/ (hate_vectorizer.joblib, hate_classifier.joblib,
hate_metadata.json).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import clean_text

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "hate_vectorizer.joblib"
CLASSIFIER_PATH = MODELS_DIR / "hate_classifier.joblib"
METADATA_PATH   = MODELS_DIR / "hate_metadata.json"

LABEL_MAP = {0: "Not hate", 1: "Hate"}
HATE_THRESHOLD = 0.5
TEST_SIZE = 0.20
RANDOM_SEED = 42

TFIDF_PARAMS = dict(
    max_features=30_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    analyzer="word",
)
LR_PARAMS = dict(max_iter=1_000, C=1.0, solver="lbfgs", class_weight="balanced", random_state=42)

_vectorizer = None
_classifier = None
_metadata = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def model_is_trained() -> bool:
    return VECTORIZER_PATH.exists() and CLASSIFIER_PATH.exists()


def load_metadata() -> Optional[dict]:
    global _metadata
    if _metadata is None and METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as fh:
            _metadata = json.load(fh)
    return _metadata


def _load_artefacts():
    global _vectorizer, _classifier
    if _vectorizer is None:
        if not model_is_trained():
            raise FileNotFoundError(
                "Hate-speech model not found in models/.\n"
                "Train it with:  python -m src.hate_speech"
            )
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _vectorizer, _classifier


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_batch(texts: list[str]) -> pd.DataFrame:
    """
    Classify each text as hate / not hate.

    Returns a DataFrame with columns ``hate_label`` ("Hate" | "Not hate"),
    ``prob_hate`` (0-1) and ``is_hate`` (bool). Texts that are empty after
    cleaning get ``prob_hate = 0``.
    """
    vectorizer, clf = _load_artefacts()
    cleaned = [clean_text(t) for t in texts]
    valid = [len(c) >= 3 for c in cleaned]

    probs = np.zeros(len(texts))
    valid_texts = [c for c, ok in zip(cleaned, valid) if ok]
    if valid_texts:
        proba = clf.predict_proba(vectorizer.transform(valid_texts))
        hate_idx = list(clf.classes_).index(1)
        probs[np.array(valid)] = proba[:, hate_idx]

    is_hate = probs >= HATE_THRESHOLD
    return pd.DataFrame({
        "hate_label": np.where(is_hate, "Hate", "Not hate"),
        "prob_hate":  np.round(probs, 4),
        "is_hate":    is_hate,
    })


def predict_single(text: str) -> dict:
    return predict_batch([text]).iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_dataset_df() -> pd.DataFrame:
    """Download TweetEval 'hate' (all splits) as a DataFrame with text, label."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package:  pip install datasets") from exc

    print("Downloading TweetEval hate-speech dataset ...")
    ds = load_dataset("cardiffnlp/tweet_eval", "hate")
    frames = [ds[s].to_pandas() for s in ("train", "validation", "test") if s in ds]
    return pd.concat(frames, ignore_index=True)[["text", "label"]]


def train(df: Optional[pd.DataFrame] = None) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore")
    if df is None:
        df = load_dataset_df()

    df = df.copy()
    df["cleaned"] = df["text"].apply(clean_text)
    df = df[df["cleaned"].str.len() >= 3].reset_index(drop=True)
    print(f"Examples: {len(df):,}   hate share: {df['label'].mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["cleaned"].to_numpy(dtype=str), df["label"].to_numpy(),
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["label"],
    )

    vec = TfidfVectorizer(**TFIDF_PARAMS)
    Xtr = vec.fit_transform(X_tr)          # fit on training split only (no leakage)
    Xte = vec.transform(X_te)

    clf = LogisticRegression(**LR_PARAMS).fit(Xtr, y_tr)
    pred = clf.predict(Xte)

    names = [LABEL_MAP[0], LABEL_MAP[1]]
    metrics = {
        "accuracy":        round(float(accuracy_score(y_te, pred)), 4),
        "precision_hate":  round(float(precision_score(y_te, pred, pos_label=1, zero_division=0)), 4),
        "recall_hate":     round(float(recall_score(y_te, pred, pos_label=1, zero_division=0)), 4),
        "f1_hate":         round(float(f1_score(y_te, pred, pos_label=1, zero_division=0)), 4),
        "macro_f1":        round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": names,
        "classification_report": classification_report(y_te, pred, target_names=names, zero_division=0),
        "n_total": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "train_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "TF-IDF + Logistic Regression (balanced class weights)",
        "dataset_source": "cardiffnlp/tweet_eval - hate task (SemEval-2019 Task 5)",
        "threshold": HATE_THRESHOLD, "test_size": TEST_SIZE, "random_seed": RANDOM_SEED,
        "caveat": ("Trained on tweets about immigrants and women only. Performance on other "
                   "topics, sarcasm or coded language will be lower than the test scores suggest."),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vec, VECTORIZER_PATH)
    joblib.dump(clf, CLASSIFIER_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Accuracy {metrics['accuracy']}  |  hate precision {metrics['precision_hate']}  "
          f"recall {metrics['recall_hate']}  F1 {metrics['f1_hate']}")
    print(metrics["classification_report"])
    print(f"Saved to {MODELS_DIR}")
    return metrics


if __name__ == "__main__":
    train()
