# -*- coding: utf-8 -*-
"""
src/compare_models.py — Find the best combination of pre-trained models.

    python -m src.compare_models

Scores each candidate model on TweetEval's official validation and test splits, then tries
every combination (probabilities averaged). The combination is CHOSEN on validation only;
test numbers are reported afterwards and never used to choose. Writes
models/ensemble.json, which src/transformer_models.py reads to decide which models to load.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src import transformer_models as tm

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "models" / "ensemble.json"

SENTIMENT_CANDIDATES = [
    tm.HUB_SENTIMENT,
    "cardiffnlp/twitter-roberta-base-sentiment",
    "finiteautomata/bertweet-base-sentiment-analysis",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
]
HATE_CANDIDATES = [
    tm.HUB_HATE,
    "facebook/roberta-hate-speech-dynabench-r4-target",
    "Hate-speech-CNERG/dehatebert-mono-english",
    "cardiffnlp/twitter-roberta-base-hate",
]


def _probs(task: str, model: str, split: str, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    safe = model.replace("/", "__")
    path = CACHE / f"cmp_{task}_{safe}_{split}_{len(texts)}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    tm._pipes.clear()                                # keep GPU memory for one model at a time
    rows = tm._scores(model, texts)
    arr = np.array([tm.sentiment_row(r) for r in rows]) if task == "sentiment" else np.array([tm.hate_row(r) for r in rows])
    np.save(path, arr)
    print(f"   scored {model} [{split}] in {time.time() - t0:.0f}s", flush=True)
    return arr


def _sent_metrics(y, p):
    pred = p.argmax(1)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "macro_recall": recall_score(y, pred, average="macro")}


def _hate_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "f1_hate": f1_score(y, pred, pos_label=1), "recall_hate": recall_score(y, pred, pos_label=1)}


def run(task: str, candidates: list[str]) -> dict:
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", task)
    val, test = ds["validation"].to_pandas(), ds["test"].to_pandas()
    metric_fn, key = (_sent_metrics, "macro_recall") if task == "sentiment" else (_hate_metrics, "macro_f1")
    print(f"\n== {task} ({len(val)} validation / {len(test)} test) ==", flush=True)

    scored, failed = {}, []
    for m in candidates:
        try:
            scored[m] = (_probs(task, m, "val", val["text"].tolist()), _probs(task, m, "test", test["text"].tolist()))
        except Exception as exc:                      # a model that can't load is simply skipped
            failed.append(m)
            print(f"   skipped {m}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
    names = list(scored)
    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()

    rows = []
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            pv = np.mean([scored[m][0] for m in combo], axis=0)
            pt = np.mean([scored[m][1] for m in combo], axis=0)
            rows.append({"models": list(combo), "val": metric_fn(y_val, pv), "test": metric_fn(y_test, pt)})
    best = max(rows, key=lambda r: (r["val"][key], -len(r["models"])))        # ties: fewer models

    for r in sorted(rows, key=lambda r: -r["val"][key])[:8]:
        tag = " <-- chosen on validation" if r is best else ""
        print(f"   val {key} {r['val'][key]:.4f} | test acc {r['test']['accuracy']:.4f} {key} {r['test'][key]:.4f} | "
              + " + ".join(m.split("/")[-1] for m in r["models"]) + tag, flush=True)
    singles = {r["models"][0]: r for r in rows if len(r["models"]) == 1}
    baseline = singles[candidates[0]]
    print(f"   current model alone: val {baseline['val'][key]:.4f} | test acc {baseline['test']['accuracy']:.4f} "
          f"{key} {baseline['test'][key]:.4f}", flush=True)
    # Ship rule, fixed in advance: the validation winner must beat the current model on
    # validation AND must not be worse than it on the test split (accuracy or the main metric).
    better_val = best["val"][key] > baseline["val"][key]
    test_ok = all(best["test"][m] >= baseline["test"][m] for m in ("accuracy", key))
    ship = better_val and test_ok and best["models"] != [candidates[0]]
    if ship:
        verdict = "SHIPPED: " + " + ".join(m.split("/")[-1] for m in best["models"])
    else:
        why = "no validation improvement" if not better_val else (
            "regressed on the test split" if not test_ok else "the current model is already the best")
        verdict = f"NOT SHIPPED ({why}); keeping {candidates[0].split('/')[-1]}"
    print("   " + verdict, flush=True)
    return {"metric": key, "models": best["models"] if ship else [candidates[0]], "shipped": ship,
            "verdict": verdict, "validation_winner": best["models"],
            "val": best["val"], "test": best["test"],
            "baseline": {"model": candidates[0], "val": baseline["val"], "test": baseline["test"]},
            "skipped": failed, "all": rows}


def main():
    result = {"date": time.strftime("%Y-%m-%d %H:%M"),
              "protocol": "Combination chosen on the validation split only; test split reported, never used to choose.",
              "sentiment": run("sentiment", SENTIMENT_CANDIDATES),
              "hate": run("hate", HATE_CANDIDATES)}
    OUT.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
