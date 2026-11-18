# -*- coding: utf-8 -*-
"""
src/evaluate.py — Benchmark old vs new models on TweetEval's *official* splits.

    python -m src.evaluate              # full test sets (CPU: ~10-20 min first run)
    python -m src.evaluate --limit 2000 # quicker, stratified sample of each test set

Protocol (no leakage):
  * TF-IDF baselines are re-trained here on the official TRAIN split only.
  * The fuzzy hate threshold is tuned on the VALIDATION split only.
  * All numbers are reported on the untouched TEST split.
Transformer outputs are cached in data/cache/ so re-runs are fast.

Writes models/benchmark.json, which the dashboard displays.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src import fuzzy, lexicon, transformer_models
from src.preprocessing import clean_text

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "models" / "benchmark.json"
SENT_LABELS = ["Negative", "Neutral", "Positive"]


def _load(task: str) -> dict:
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", task)
    return {s: ds[s].to_pandas() for s in ("train", "validation", "test")}


def _sample(df: pd.DataFrame, limit: int | None, seed: int = 0) -> pd.DataFrame:
    if not limit or limit >= len(df):
        return df.reset_index(drop=True)
    frac = limit / len(df)
    return (df.groupby("label", group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=seed))
              .reset_index(drop=True))


def _tfidf_probs(train: pd.DataFrame, others: list[pd.DataFrame], balanced: bool) -> list[np.ndarray]:
    vec = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                          strip_accents="unicode")
    X = vec.fit_transform(train["text"].map(clean_text))
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced" if balanced else None)
    clf.fit(X, train["label"])
    return [clf.predict_proba(vec.transform(d["text"].map(clean_text))) for d in others]


def _cached(name: str, fn, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_{len(texts)}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    arr = fn(texts)
    np.save(path, arr)
    print(f"   {name}: {len(texts)} texts in {time.time() - t0:.0f}s")
    return arr


def _m_sent(y, pred) -> dict:
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "macro_recall": round(recall_score(y, pred, average="macro"), 4)}


def _m_hate(y, pred) -> dict:
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "f1_hate": round(f1_score(y, pred, pos_label=1), 4),
            "recall_hate": round(recall_score(y, pred, pos_label=1), 4)}


def sentiment(limit):
    print("\n== Sentiment (TweetEval official test) ==")
    d = _load("sentiment")
    test = _sample(d["test"], limit)
    y = test["label"].to_numpy()
    texts = test["text"].tolist()

    (p_l,) = _tfidf_probs(d["train"], [test], balanced=False)
    p_t = (_cached(f"sent_test_{limit}", transformer_models.sentiment_probs, texts)
           if transformer_models.available() else None)
    cues = [lexicon.extract(t) for t in texts]
    p_v = np.array([fuzzy.vader_to_probs(c.vader_compound) for c in cues])

    res = {"n_test": len(test),
           "old_tfidf": _m_sent(y, p_l.argmax(1)),
           "vader_only": _m_sent(y, p_v.argmax(1))}
    if p_t is not None:
        res["roberta_only"] = _m_sent(y, p_t.argmax(1))
    fused = np.array([fuzzy.fuse_sentiment(p_t[i] if p_t is not None else None, p_l[i], c).probs
                      for i, c in enumerate(cues)])
    res["fuzzy_ensemble"] = _m_sent(y, fused.argmax(1))
    for k, v in res.items():
        print(f"   {k:16s} {v}")
    return res


def hate(limit):
    print("\n== Hate speech (TweetEval official test) ==")
    d = _load("hate")
    val, test = d["validation"], _sample(d["test"], limit)
    p_l_val, p_l_test = (p[:, 1] for p in _tfidf_probs(d["train"], [val, test], balanced=True))
    avail = transformer_models.available()
    p_t_val = _cached("hate_val", transformer_models.hate_probs, val["text"].tolist()) if avail else None
    p_t_test = (_cached(f"hate_test_{limit}", transformer_models.hate_probs, test["text"].tolist())
                if avail else None)

    def scores(df, p_t, p_l):
        return np.array([fuzzy.fuse_hate(p_t[i] if p_t is not None else None, p_l[i],
                                         lexicon.extract(t), threshold=0.5).score
                         for i, t in enumerate(df["text"])])

    # Tune the decision threshold on VALIDATION only
    s_val = scores(val, p_t_val, p_l_val)
    grid = np.round(np.arange(0.2, 0.81, 0.02), 2)
    f1s = [f1_score(val["label"], (s_val >= g).astype(int), average="macro") for g in grid]
    thr = float(grid[int(np.argmax(f1s))])
    print(f"   tuned threshold on validation: {thr}")

    y = test["label"].to_numpy()
    s_test = scores(test, p_t_test, p_l_test)
    res = {"n_test": len(test), "threshold": thr,
           "old_tfidf": _m_hate(y, (p_l_test >= 0.5).astype(int))}
    if p_t_test is not None:
        res["roberta_only"] = _m_hate(y, (p_t_test >= 0.5).astype(int))
    res["fuzzy_ensemble"] = _m_hate(y, (s_test >= thr).astype(int))
    for k, v in res.items():
        print(f"   {k:16s} {v}")
    return res, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stratified sample size per test set")
    args = ap.parse_args()

    sent = sentiment(args.limit)
    hate_res, thr = hate(args.limit)
    out = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "protocol": ("TweetEval official splits. TF-IDF trained on train only; hate threshold tuned "
                     "on validation only; all scores on the held-out test split."),
        "sentiment": sent, "hate": hate_res, "tuned_hate_threshold": thr,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
