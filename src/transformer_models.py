"""
src/transformer_models.py — Twitter-RoBERTa models (CardiffNLP, optionally fine-tuned).

* Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
  RoBERTa-base pre-trained on ~124M tweets, fine-tuned on TweetEval sentiment.
* Hate:      cardiffnlp/twitter-roberta-base-hate-latest
  Same backbone, fine-tuned on 13 English hate-speech datasets.

If `python -m src.finetune` produced a better model, it lives in
models/finetuned/<task>/ and is used automatically instead of the Hub model.

Weights download from the Hugging Face Hub on first use (~500 MB each) and are
cached in ~/.cache/huggingface afterwards. Everything here is optional: if
`torch` / `transformers` are not installed, `available()` is False and the app
falls back to the TF-IDF models.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
FINETUNED_DIR = ROOT / "models" / "finetuned"

HUB_SENTIMENT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HUB_HATE      = "cardiffnlp/twitter-roberta-base-hate-latest"


def _resolve(task: str, hub_id: str, env: str) -> str:
    if os.environ.get(env):
        return os.environ[env]
    local = FINETUNED_DIR / task
    return str(local) if (local / "config.json").exists() else hub_id


SENTIMENT_MODEL = _resolve("sentiment", HUB_SENTIMENT, "SENTIQ_SENTIMENT_MODEL")
HATE_MODEL      = _resolve("hate", HUB_HATE, "SENTIQ_HATE_MODEL")

BATCH_SIZE = 32
MAX_LENGTH = 128

_pipes: dict = {}


def available() -> bool:
    """True if torch and transformers are importable."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def is_finetuned(task: str) -> bool:
    return (SENTIMENT_MODEL if task == "sentiment" else HATE_MODEL).startswith(str(FINETUNED_DIR))


def describe(model_name: str) -> str:
    return f"fine-tuned ({Path(model_name).name})" if model_name.startswith(str(FINETUNED_DIR)) else model_name


def _prep(text: str) -> str:
    """Model-card preprocessing: anonymise mentions and links."""
    if not isinstance(text, str) or not text.strip():
        return "."
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+", "http", text)
    return text


def _device() -> int:
    import torch
    return 0 if torch.cuda.is_available() else -1


def _pipe(model_name: str):
    if model_name not in _pipes:
        from transformers import pipeline
        _pipes[model_name] = pipeline(
            "text-classification", model=model_name, tokenizer=model_name,
            top_k=None, truncation=True, max_length=MAX_LENGTH, device=_device(),
        )
    return _pipes[model_name]


def _scores(model_name: str, texts: list[str]) -> list[dict]:
    out = _pipe(model_name)([_prep(t) for t in texts], batch_size=BATCH_SIZE)
    return [{d["label"].lower(): float(d["score"]) for d in row} for row in out]


def sentiment_probs(texts: list[str], model: str | None = None) -> np.ndarray:
    """(n, 3) array of [negative, neutral, positive] probabilities."""
    if not texts:
        return np.empty((0, 3))
    rows = _scores(model or SENTIMENT_MODEL, texts)
    return np.array([[r.get("negative", 0), r.get("neutral", 0), r.get("positive", 0)] for r in rows])


def hate_probs(texts: list[str], model: str | None = None) -> np.ndarray:
    """(n,) array of P(hate)."""
    if not texts:
        return np.empty(0)
    rows = _scores(model or HATE_MODEL, texts)
    probs = []
    for r in rows:
        hate = [v for k, v in r.items() if "hate" in k and "not" not in k and "non" not in k]
        if not hate:  # generic LABEL_0 / LABEL_1 naming
            hate = [r.get("label_1", 0.0)]
        probs.append(hate[0])
    return np.array(probs)


def is_loaded() -> bool:
    return SENTIMENT_MODEL in _pipes and HATE_MODEL in _pipes
