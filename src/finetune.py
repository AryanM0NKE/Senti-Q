# -*- coding: utf-8 -*-
"""
src/finetune.py — Fine-tune the Twitter-RoBERTa models on labelled tweets.

    python -m src.finetune --task sentiment            # TweetEval + Tweet Sentiment Extraction
    python -m src.finetune --task hate                 # TweetEval hate (HatEval)
    python -m src.finetune --task sentiment --no-extra # TweetEval only

Needs a CUDA GPU for reasonable speed (an RTX 3050 4 GB takes ~10-20 min for
sentiment). Runs on CPU too, just very slowly.

Protocol (decided before running):
  * Start from the current CardiffNLP checkpoints (already TweetEval-tuned), so the
    value added comes from *new* data and continued training at a low learning rate.
  * Evaluate on the validation split every N steps; early stopping; restore the best
    checkpoint.
  * Ship (save to models/finetuned/<task>/, which the app then loads) only if the
    fine-tuned model beats the original on validation AND does not regress on the
    official test split. The test split gates the yes/no decision only; it is never
    used to choose hyperparameters or checkpoints.
A report with all numbers is written to models/finetuned/<task>_report.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.transformer_models import HUB_HATE, HUB_SENTIMENT, _prep

ROOT = Path(__file__).parent.parent
FT_DIR = ROOT / "models" / "finetuned"

TASKS = {
    "sentiment": dict(base=HUB_SENTIMENT, num_labels=3, metric="macro_recall",
                      label_names=["negative", "neutral", "positive"]),
    "hate":      dict(base=HUB_HATE, num_labels=2, metric="macro_f1",
                      label_names=["not-hate", "hate"]),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_splits(task: str, extra: bool):
    from datasets import Dataset, concatenate_datasets, load_dataset

    te = load_dataset("cardiffnlp/tweet_eval", task)
    train = te["train"].select_columns(["text", "label"])
    sources = [f"tweet_eval/{task} train ({len(train):,})"]

    if task == "sentiment" and extra:
        tse = load_dataset("mteb/tweet_sentiment_extraction")["train"]
        # Same label scheme as TweetEval: 0 negative, 1 neutral, 2 positive (checked below)
        mapping = {"negative": 0, "neutral": 1, "positive": 2}
        # That dataset writes apostrophes as backticks ("I`d"); normalise them.
        rows = [(t.replace("`", "'").strip(), mapping[lt]) for t, lt in zip(tse["text"], tse["label_text"])
                if isinstance(t, str) and t.strip() and lt in mapping]
        extra_ds = Dataset.from_dict({"text": [r[0] for r in rows], "label": [r[1] for r in rows]})
        extra_ds = extra_ds.cast(train.features)
        train = concatenate_datasets([train, extra_ds])
        sources.append(f"mteb/tweet_sentiment_extraction train ({len(extra_ds):,})")

    return train.shuffle(seed=42), te["validation"], te["test"], sources


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict:
    pred = logits.argmax(-1)
    out = {"accuracy": accuracy_score(labels, pred),
           "macro_f1": f1_score(labels, pred, average="macro"),
           "macro_recall": recall_score(labels, pred, average="macro")}
    return {k: round(float(v), 4) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fine-tune Senti-Q transformer models")
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--base", default=None, help="starting checkpoint (default: current CardiffNLP model)")
    ap.add_argument("--no-extra", action="store_true", help="sentiment: TweetEval data only")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--eval-steps", type=int, default=None)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    cfg = TASKS[args.task]
    base = args.base or cfg["base"]
    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"Task: {args.task} | base: {base} | device: {device_name}")

    train, val, test, sources = load_splits(args.task, extra=not args.no_extra)
    print("Training data: " + " + ".join(sources))

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=cfg["num_labels"])
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print(f"Model labels: {id2label}")
    # Safety check: the checkpoint's label order must match the dataset's
    for i, name in enumerate(cfg["label_names"]):
        assert name.replace("-", "") in id2label[i].replace("-", "").replace("_", ""), \
            f"label order mismatch at {i}: model={id2label[i]} expected={name}"

    def encode(batch):
        return tok([_prep(t) for t in batch["text"]], truncation=True, max_length=args.max_len)

    enc = {name: ds.map(encode, batched=True, remove_columns=["text"])
           for name, ds in (("train", train), ("val", val), ("test", test))}

    steps_per_epoch = max(1, len(enc["train"]) // args.batch)
    eval_steps = args.eval_steps or max(50, steps_per_epoch // 4)
    run_dir = FT_DIR / f"_runs_{args.task}"

    targs = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=int(0.06 * steps_per_epoch * args.epochs),   # 6% linear warm-up
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric"],
        greater_is_better=True,
        fp16=cuda,
        logging_steps=200,
        disable_tqdm=True,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=enc["train"], eval_dataset=enc["val"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=lambda p: metrics_from_logits(p.predictions, p.label_ids),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    def score(split):
        p = trainer.predict(enc[split])
        return metrics_from_logits(p.predictions, p.label_ids)

    print("\nScoring the ORIGINAL model ...")
    before = {"val": score("val"), "test": score("test")}
    print(f"   val {before['val']}\n   test {before['test']}")

    print(f"\nFine-tuning: {len(enc['train']):,} examples, {args.epochs} epochs, lr {args.lr}, "
          f"batch {args.batch}, eval every {eval_steps} steps ...")
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    print("\nScoring the FINE-TUNED model (best validation checkpoint) ...")
    after = {"val": score("val"), "test": score("test")}
    print(f"   val {after['val']}\n   test {after['test']}")

    m = cfg["metric"]
    val_better = after["val"][m] > before["val"][m]
    test_ok = after["test"][m] >= before["test"][m]
    ship = val_better and test_ok
    out_dir = FT_DIR / args.task

    if ship:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        trainer.save_model(str(out_dir))
        tok.save_pretrained(str(out_dir))
        verdict = f"SHIPPED: {m} improved on validation and did not regress on test. Saved to {out_dir}"
    else:
        reason = "no validation improvement" if not val_better else "regressed on the test split"
        verdict = f"NOT SHIPPED ({reason}); the app keeps the original model."
    print("\n" + verdict)

    report = {
        "task": args.task, "base_model": base, "device": device_name,
        "training_data": sources, "n_train": len(enc["train"]),
        "hyperparameters": {"epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                            "max_len": args.max_len, "eval_steps": eval_steps,
                            "early_stopping_patience": args.patience, "fp16": cuda},
        "selection_metric": m, "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_minutes": round(minutes, 1),
        "before": before, "after": after, "shipped": ship, "verdict": verdict,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    FT_DIR.mkdir(parents=True, exist_ok=True)
    (FT_DIR / f"{args.task}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.rmtree(run_dir, ignore_errors=True)   # intermediate checkpoints
    print(f"Report: {FT_DIR / (args.task + '_report.json')}")


if __name__ == "__main__":
    main()
