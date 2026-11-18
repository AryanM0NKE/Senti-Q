"""
src/engine.py — SentimentIQ analysis engine.

    texts ─┬─► RoBERTa sentiment ─┐
           ├─► VADER lexicon ─────┼─► fuzzy fusion ─► sentiment + confidence + reasons
           ├─► TF-IDF sentiment ──┘
           ├─► RoBERTa hate ──────┐
           ├─► TF-IDF hate ───────┼─► fuzzy fusion ─► hate score + severity + reasons
           └─► linguistic cues ───┘

Uses whichever experts are available, so the app keeps working with only the
TF-IDF models (no torch) or only the transformers (no TF-IDF artefacts).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import fuzzy, hate_speech, lexicon, predict, transformer_models

BENCHMARK_PATH = Path(__file__).parent.parent / "models" / "benchmark.json"


DEFAULT_TUNED = {"sentiment_support_scale": 1.0, "hate_support_scale": 1.0,
                 "hate_cue_scale": 1.0, "hate_threshold": fuzzy.HATE_THRESHOLD}


def _tuned() -> dict:
    """Fuzzy parameters learned on the validation split by src/evaluate.py (else defaults)."""
    try:
        learned = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8")).get("tuned", {})
        return {**DEFAULT_TUNED, **{k: float(v) for k, v in learned.items() if k in DEFAULT_TUNED}}
    except Exception:
        return dict(DEFAULT_TUNED)


def status() -> dict:
    t = transformer_models.available()
    return {
        "transformer": t,
        "vader": lexicon.vader_available(),
        "tfidf_sentiment": predict.model_is_trained(),
        "tfidf_hate": hate_speech.model_is_trained(),
        "ready": t or predict.model_is_trained(),
        "hate_ready": t or hate_speech.model_is_trained(),
        "name": ("Fuzzy ensemble: RoBERTa + VADER + TF-IDF" if t else "TF-IDF + VADER (fuzzy)"),
    }


def analyze(texts: list[str], *, use_transformer: bool = True) -> pd.DataFrame:
    """Score texts for sentiment and hate speech. Returns one row per text."""
    st = status()
    if not st["ready"]:
        raise FileNotFoundError("No sentiment model available. Install the transformer extras "
                                "(pip install -r requirements.txt) or run python -m src.train_model.")
    use_t = use_transformer and st["transformer"]
    n = len(texts)

    t_sent = transformer_models.sentiment_probs(texts) if use_t else None
    t_hate = transformer_models.hate_probs(texts) if use_t else None
    l_sent = (predict.predict_batch(texts)[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy()
              if st["tfidf_sentiment"] else None)
    l_hate = hate_speech.predict_batch(texts)["prob_hate"].to_numpy() if st["tfidf_hate"] else None
    hate_on = use_t or st["tfidf_hate"]
    tp = _tuned()
    # Learned scales only apply when the transformer is present; without it the
    # support experts *are* the model, so use the defaults.
    s_scale = tp["sentiment_support_scale"] if use_t else 1.0
    h_sup = tp["hate_support_scale"] if use_t else 1.0

    rows = []
    for i, text in enumerate(texts):
        cues = lexicon.extract(text)
        s = fuzzy.fuse_sentiment(t_sent[i] if t_sent is not None else None,
                                 l_sent[i] if l_sent is not None else None, cues, s_scale)
        row = {
            "predicted_sentiment": s.label,
            "prob_negative": round(float(s.probs[0]), 4),
            "prob_neutral":  round(float(s.probs[1]), 4),
            "prob_positive": round(float(s.probs[2]), 4),
            "sentiment_confidence": s.confidence,
            "sentiment_notes": s.notes,
        }
        if hate_on:
            h = fuzzy.fuse_hate(float(t_hate[i]) if t_hate is not None else None,
                                float(l_hate[i]) if l_hate is not None else None, cues,
                                threshold=tp["hate_threshold"] if use_t else fuzzy.HATE_THRESHOLD,
                                support_scale=h_sup, cue_scale=tp["hate_cue_scale"])
            row.update({
                "hate_label": "Hate" if h.is_hate else "Not hate",
                "prob_hate": h.score,
                "is_hate": h.is_hate,
                "hate_severity": h.severity,
                "hate_notes": h.notes,
            })
        rows.append(row)

    df = pd.DataFrame(rows, index=range(n))
    df["engine"] = "Fuzzy ensemble (RoBERTa + VADER + TF-IDF)" if use_t else "TF-IDF + VADER (fuzzy)"
    return df


def load_benchmark() -> dict | None:
    try:
        return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
