"""
src/transformer_models.py — Twitter-RoBERTa models (CardiffNLP, optionally fine-tuned).

* Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
  RoBERTa-base pre-trained on ~124M tweets, fine-tuned on TweetEval sentiment.
* Hate:      cardiffnlp/twitter-roberta-base-hate-latest
  Same backbone, fine-tuned on 13 English hate-speech datasets.

If `python -m src.finetune` produced a better model, it lives in
models/finetuned/<task>/ and is used automatically instead of the Hub model.

Weights download from the Hugging Face Hub once (~500 MB each) and are cached in
~/.cache/huggingface. Later loads read that cache directly, without contacting the
Hub. Everything here is optional: if `torch` / `transformers` are not installed,
`available()` is False and the app falls back to the TF-IDF models.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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

# models/ensemble.json (written by `python -m src.compare_models`) names the combination of
# models whose probabilities are averaged. Without it, one model per task is used.
ENSEMBLE_PATH = ROOT / "models" / "ensemble.json"


def _ensemble(task: str, single: str) -> list[str]:
    try:
        models = json.loads(ENSEMBLE_PATH.read_text(encoding="utf-8"))[task]["models"]
        if models and not os.environ.get("SENTIQ_SINGLE_MODELS"):
            return list(models)
    except Exception:
        pass
    return [single]


SENTIMENT_MODELS = _ensemble("sentiment", SENTIMENT_MODEL)
HATE_MODELS      = _ensemble("hate", HATE_MODEL)

BATCH_SIZE = 32
MAX_LENGTH = 128

_pipes: dict = {}
_lock = threading.Lock()          # the app loads models in a background thread


def available() -> bool:
    """True if torch and transformers are importable."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def _is_cached(model_name: str) -> bool:
    if Path(model_name).exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        return any(isinstance(try_to_load_from_cache(model_name, f), str)
                   for f in ("model.safetensors", "pytorch_model.bin"))
    except Exception:
        return False


def models_cached() -> bool:
    """True if every model is already on this computer (so nothing will be downloaded)."""
    return all(_is_cached(m) for m in SENTIMENT_MODELS + HATE_MODELS)


def is_finetuned(task: str) -> bool:
    models = SENTIMENT_MODELS if task == "sentiment" else HATE_MODELS
    return any(m.startswith(str(FINETUNED_DIR)) for m in models)


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


def _load(model_name: str, local_only: bool):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_only)
    return pipeline("text-classification", model=model, tokenizer=tok, top_k=None,
                    truncation=True, max_length=MAX_LENGTH, device=_device())


def _pipe(model_name: str):
    with _lock:
        if model_name not in _pipes:
            try:          # already on disk: load it without contacting the Hub
                _pipes[model_name] = _load(model_name, local_only=True)
            except (OSError, ValueError):   # first run: download it (once)
                _pipes[model_name] = _load(model_name, local_only=False)
        return _pipes[model_name]


def _scores(model_name: str, texts: list[str]) -> list[dict]:
    out = _pipe(model_name)([_prep(t) for t in texts], batch_size=BATCH_SIZE)
    return [{d["label"].lower(): float(d["score"]) for d in row} for row in out]


def sentiment_row(r: dict) -> list[float]:
    """[negative, neutral, positive] from a model's label -> score dict, whatever it calls its labels."""
    if any(k.startswith("label_") for k in r):          # generic LABEL_0/1/2 = negative/neutral/positive
        return [r.get("label_0", 0.0), r.get("label_1", 0.0), r.get("label_2", 0.0)]

    def pick(*keys):
        return next((v for k, v in r.items() if any(x in k for x in keys)), 0.0)
    return [pick("neg"), pick("neu"), pick("pos")]


def hate_row(r: dict) -> float:
    """P(hate) from a model's label -> score dict."""
    if any(k.startswith("label_") for k in r):          # generic LABEL_1 = hate
        return r.get("label_1", 0.0)
    return next((v for k, v in r.items() if "hate" in k and "not" not in k and "non" not in k), 0.0)


def sentiment_probs(texts: list[str], model: str | None = None) -> np.ndarray:
    """(n, 3) array of [negative, neutral, positive] probabilities (ensemble average)."""
    if not texts:
        return np.empty((0, 3))
    models = [model] if model else SENTIMENT_MODELS
    return np.mean([np.array([sentiment_row(r) for r in _scores(m, texts)]) for m in models], axis=0)


def hate_probs(texts: list[str], model: str | None = None) -> np.ndarray:
    """(n,) array of P(hate) (ensemble average)."""
    if not texts:
        return np.empty(0)
    models = [model] if model else HATE_MODELS
    return np.mean([np.array([hate_row(r) for r in _scores(m, texts)]) for m in models], axis=0)


def is_loaded() -> bool:
    return all(m in _pipes for m in SENTIMENT_MODELS + HATE_MODELS)
