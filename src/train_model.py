# -*- coding: utf-8 -*-
"""
src/train_model.py — Model training pipeline for SentimentIQ.

Loads the TweetEval sentiment dataset (or a user-supplied CSV), trains a
TF-IDF + Logistic Regression pipeline, evaluates it on a held-out test
split, and saves the artefacts to disk with Joblib.

Usage (from project root):
    python -m src.train_model
    python -m src.train_model --csv data/my_labels.csv --text-col text --label-col sentiment
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
import os

# Force UTF-8 output on Windows to avoid cp1252 emoji errors
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODELS_DIR / "lr_classifier.joblib"
METADATA_PATH   = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

# TweetEval labels: 0 = Negative, 1 = Neutral, 2 = Positive
TWEETEVAL_LABEL_MAP: dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

SUPPORTED_LABELS = {"Positive", "Negative", "Neutral"}

# TF-IDF hyperparameters (chosen for short social-media text)
TFIDF_PARAMS = dict(
    max_features=30_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    analyzer="word",
)

LR_PARAMS = dict(
    max_iter=1_000,
    C=1.0,
    solver="lbfgs",
    random_state=42,
    n_jobs=-1,
)

TEST_SIZE   = 0.20
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_tweeteval() -> pd.DataFrame:
    """
    Download and return the TweetEval *sentiment* split as a DataFrame with
    columns ``text`` and ``sentiment`` (string labels).

    Raises
    ------
    RuntimeError
        If the ``datasets`` package is unavailable or the download fails.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required to download TweetEval.\n"
            "Install it with:  pip install datasets"
        ) from exc

    print("⬇️  Downloading TweetEval sentiment dataset …")
    try:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download TweetEval: {exc}\n\n"
            "Workaround: supply your own labeled CSV with\n"
            "  python -m src.train_model --csv <path> --text-col <col> --label-col <col>"
        ) from exc

    frames = []
    for split_name in ("train", "validation", "test"):
        if split_name in ds:
            split_df = ds[split_name].to_pandas()
            frames.append(split_df)

    df = pd.concat(frames, ignore_index=True)
    # Map numeric labels → string labels
    df["sentiment"] = df["label"].map(TWEETEVAL_LABEL_MAP)
    df = df.rename(columns={"text": "text"})
    return df[["text", "sentiment"]]


def load_csv_dataset(
    csv_path: str | Path,
    text_col: str = "text",
    label_col: str = "sentiment",
) -> pd.DataFrame:
    """
    Load a user-supplied labeled CSV.

    Validates that the required columns exist, labels are supported,
    there are no empty texts, and there are no excessive duplicates.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Column validation
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(columns={text_col: "text", label_col: "sentiment"})
    df = df[["text", "sentiment"]].copy()

    # Missing / empty text
    before = len(df)
    df = df.dropna(subset=["text", "sentiment"])
    df = df[df["text"].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"⚠️  Dropped {dropped} rows with missing text or label.")

    # Duplicate removal
    dupes = df.duplicated(subset=["text"]).sum()
    if dupes:
        print(f"⚠️  Removing {dupes} duplicate rows.")
        df = df.drop_duplicates(subset=["text"])

    # Label validation
    unsupported = set(df["sentiment"].unique()) - SUPPORTED_LABELS
    if unsupported:
        raise ValueError(
            f"Unsupported labels in dataset: {unsupported}\n"
            f"Supported labels: {SUPPORTED_LABELS}"
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> dict:
    """
    Train and evaluate a TF-IDF + Logistic Regression classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``text`` and ``sentiment`` columns.

    Returns
    -------
    dict
        Evaluation metrics computed on the held-out test set.
    """
    print(f"📊 Dataset size: {len(df):,} examples")
    print(f"      Label distribution:\n{df['sentiment'].value_counts().to_string()}\n")

    # --- Clean text (without data leakage) ---
    print("🧹 Cleaning text …")
    df["cleaned_text"] = df["text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() >= 3].reset_index(drop=True)

    X = df["cleaned_text"].to_numpy(dtype=str)
    y = df["sentiment"].to_numpy(dtype=str)

    # --- Stratified train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"✂️  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # --- Fit vectorizer on TRAINING data only ---
    print("🔠 Fitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)     # transform, NOT fit_transform

    # --- Train classifier ---
    print("🤖 Training Logistic Regression …")
    t0 = time.time()
    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_vec, y_train)
    elapsed = time.time() - t0
    print(f"      Training completed in {elapsed:.1f}s")

    # --- Evaluate on held-out test set ---
    print("\n📈 Evaluating on held-out test set …")
    y_pred = clf.predict(X_test_vec)

    labels_order = ["Negative", "Neutral", "Positive"]
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm    = confusion_matrix(y_test, y_pred, labels=labels_order)
    cr    = classification_report(
        y_test, y_pred, labels=labels_order, zero_division=0
    )

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (macro)")
    print(f"  Recall    : {rec:.4f}  (macro)")
    print(f"  F1-score  : {f1:.4f}  (macro)")
    print(f"\n{cr}")

    # --- Save artefacts ---
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, CLASSIFIER_PATH)

    metrics = {
        "accuracy":               round(float(acc),  4),
        "macro_precision":        round(float(prec), 4),
        "macro_recall":           round(float(rec),  4),
        "macro_f1":               round(float(f1),   4),
        "confusion_matrix":       cm.tolist(),
        "confusion_matrix_labels": labels_order,
        "classification_report":  cr,
        "n_train":                int(len(X_train)),
        "n_test":                 int(len(X_test)),
        "n_total":                int(len(df)),
        "train_date":             time.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":              "TF-IDF + Logistic Regression (multinomial)",
        "dataset_source":         "cardiffnlp/tweet_eval — sentiment task (Cardiff NLP, HuggingFace)",
        "label_mapping":          {"0": "Negative", "1": "Neutral", "2": "Positive"},
        "tfidf_params":           TFIDF_PARAMS,
        "lr_params": {k: v for k, v in LR_PARAMS.items() if k != "n_jobs"},
        "test_size":              TEST_SIZE,
        "random_seed":            RANDOM_SEED,
    }

    with open(METADATA_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n✅ Artefacts saved:")
    print(f"   {VECTORIZER_PATH}")
    print(f"   {CLASSIFIER_PATH}")
    print(f"   {METADATA_PATH}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the SentimentIQ TF-IDF + LogReg classifier."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to a labeled CSV. Omit to download TweetEval automatically.",
    )
    parser.add_argument("--text-col",  default="text",      help="Text column name in the CSV.")
    parser.add_argument("--label-col", default="sentiment", help="Label column name in the CSV.")
    args = parser.parse_args()

    try:
        if args.csv:
            df = load_csv_dataset(args.csv, args.text_col, args.label_col)
        else:
            df = load_tweeteval()

        train(df)

    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\n❌ Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
