# SentimentIQ — Sentiment & Hate-Speech Intelligence Platform

> Paste any X (Twitter) post link, a block of text, or run a live search, and
> SentimentIQ reports each post's **sentiment** (Positive / Neutral / Negative)
> and whether it contains **hate speech**. Two TF-IDF + Logistic Regression
> classifiers trained on the TweetEval benchmark power it, behind a
> black-and-white Streamlit dashboard.

## Quick start

```bash
pip install -r requirements.txt
python -m src.train_model     # sentiment model  (~1-3 min, downloads TweetEval)
python -m src.hate_speech     # hate-speech model (~1 min)
streamlit run app.py          # opens http://localhost:8501
```

**Input modes** (sidebar → Input):

| Mode | Needs X token? | What it does |
|---|---|---|
| Post link or ID | No (optional) | Paste `https://x.com/user/status/123…` (or several). Uses the X API if a token is set, otherwise X's public embed endpoint. |
| Paste text | No | Score any text; one post per line. |
| Search X (live) | Yes | Keyword search of recent posts via X API v2. |
| Demo mode | No | 15 built-in sample posts. |

> Note: X's API access tiers change frequently and reading posts may require a paid
> plan. Post-link mode therefore falls back to the public embed endpoint
> automatically (text and date only, no like/repost counts).

## Hate-speech model

Binary classifier (Hate / Not hate) trained on the TweetEval `hate` task
(SemEval-2019 Task 5). Class weights are balanced, and a post is flagged when the
predicted probability is at least 0.5. **It was trained only on tweets targeting immigrants and women**,
so it is weaker on other targets, sarcasm and coded language. Treat a flag as an
estimate, never as a finding about a person. Swapping in a transformer
(e.g. `cardiffnlp/twitter-roberta-base-hate`) is the natural upgrade.

---

## 📋 Table of Contents

1. [Project Objective](#1-project-objective)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Technologies Used & Why](#3-technologies-used--why)
4. [Dataset Source & Label Mapping](#4-dataset-source--label-mapping)
5. [Installation](#5-installation)
6. [Configuring X_BEARER_TOKEN](#6-configuring-x_bearer_token)
7. [Training the Model](#7-training-the-model)
8. [Running the Application](#8-running-the-application)
9. [Live Mode vs Demo Mode](#9-live-mode-vs-demo-mode)
10. [Model Evaluation Methodology](#10-model-evaluation-methodology)
11. [Limitations & Responsible Use](#11-limitations--responsible-use)
12. [Future Improvements](#12-future-improvements)
13. [🎓 Viva Q&A Guide](#13--viva-qa-guide)

---

## 1. Project Objective

SentimentIQ analyses the **political sentiment** of English-language posts
fetched from the X (Twitter) platform.  It answers the question:
*"What does social-media text reveal about public sentiment toward political
topics at a given moment?"*

The system:
- Collects posts in real time via the **official X API v2**.
- Cleans and normalises the text with a bespoke preprocessing pipeline.
- Predicts sentiment (**Positive / Neutral / Negative**) using a
  **TF-IDF vectorizer + Logistic Regression** classifier.
- The classifier is trained on the **TweetEval** benchmark dataset — a
  real, publicly available, human-labeled corpus of social-media text.
- Presents results through a premium interactive Streamlit dashboard.

---

## 2. Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        OFFLINE TRAINING                          │
│                                                                  │
│  TweetEval Dataset ──► Preprocessing ──► TF-IDF Fit             │
│  (labeled CSV)          clean_text()      TfidfVectorizer        │
│                              │                  │                │
│                              ▼                  ▼                │
│                        Train / Test   LogisticRegression.fit()   │
│                           Split       (training split only)      │
│                              │                  │                │
│                              ▼                  ▼                │
│                        Evaluate on    models/tfidf_vectorizer    │
│                        held-out set   models/lr_classifier       │
│                         metrics       models/model_metadata.json │
└──────────────────────────────────────────────────────────────────┘
                                │
                     (saved artefacts loaded once)
                                │
┌──────────────────────────────────────────────────────────────────┐
│                       STREAMLIT APP                              │
│                                                                  │
│  Sidebar Controls                                                │
│       │                                                          │
│       ├─► Live Mode: X API v2 ──► FetchResult ──► XPost list    │
│       │   (x_api.py)              recent search                 │
│       │                                                          │
│       └─► Demo Mode: DEMO_POSTS (data/demo_posts.py)            │
│                         │                                        │
│                         ▼                                        │
│                  posts_to_df()          DataFrame                │
│                         │               text col                 │
│                         ▼                                        │
│              predict.predict_batch()  ──► cleaned_text           │
│                (vectorizer.transform +   predicted_sentiment     │
│                 clf.predict_proba)       prob_{pos,neu,neg}      │
│                         │                                        │
│                         ▼                                        │
│              Dashboard Rendering:                                │
│              • Metric cards                                      │
│              • Donut + bar + trend charts                        │
│              • Post explorer with probability bars               │
│              • Model performance panel                           │
│              • CSV export                                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Technologies Used & Why

| Library / Tool | Purpose | Why chosen |
|---|---|---|
| **Python 3.10+** | Core language | Standard for ML/data work |
| **Streamlit** | Interactive web dashboard | Pure Python, no JS needed, ideal for college demos |
| **Scikit-learn** | TF-IDF + Logistic Regression | Industry standard, well-documented, reproducible |
| **Pandas / NumPy** | Data manipulation | Universal in data science |
| **Plotly** | Interactive charts | Beautiful, interactive, works in Streamlit |
| **Joblib** | Save/load model artefacts | Scikit-learn recommended serialisation |
| **Tweepy / requests** | X API communication | Official Python client for X API |
| **python-dotenv** | Load `.env` credentials | Keeps secrets out of source code |
| **datasets (HuggingFace)** | Download TweetEval | Reproducible dataset loading |
| **pytest** | Unit testing | Standard Python test framework |

---

## 4. Dataset Source & Label Mapping

**Dataset**: TweetEval — Sentiment Task  
**Source**: [Cardiff NLP / tweet_eval](https://huggingface.co/datasets/tweet_eval)  
**Paper**: Barbieri et al., *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*, EMNLP Findings 2020.

The dataset contains approximately **45,000 human-labeled tweets** across
train / validation / test splits.

| Numeric Label | String Label | Description |
|:---:|:---:|---|
| **0** | **Negative** | Expresses negative sentiment |
| **1** | **Neutral** | Neither positive nor negative |
| **2** | **Positive** | Expresses positive sentiment |

The labels are defined by human annotators and represent the **overall tone**
of the tweet rather than sentiment about a specific entity.

---

## 5. Installation

### Prerequisites
- Python **3.10** or higher
- pip
- Internet connection (for TweetEval download and X API)

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd SentimentIQ

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. Configuring X_BEARER_TOKEN

> ⚠️ Never hard-code your token in source code or commit the `.env` file.

1. Go to [developer.x.com](https://developer.x.com/en/portal/dashboard) and sign in.
2. Create a new **Project** and **App** (or use an existing one).
3. Navigate to **Keys and Tokens** → copy the **Bearer Token**.
4. In the project root, create a file named `.env`:
   ```
   X_BEARER_TOKEN=your_actual_bearer_token_here
   ```
5. The `.env` file is listed in `.gitignore` — it will not be committed.

**API access notes:** X changes its tiers and pricing often, and read access
(search / post lookup) may require a paid plan or credits. Check the current
limits in the X Developer Portal. If a lookup is refused, **Post link** mode
falls back to the public embed endpoint automatically; live **search** has no
such fallback.

---

## 7. Training the Model

Run the training script from the project root:

```bash
python -m src.train_model
```

This will:
1. Download the TweetEval sentiment dataset (~45k labeled tweets).
2. Clean and preprocess all text.
3. Split into **80% train / 20% test** (stratified, seed=42).
4. Fit the **TF-IDF vectorizer** on the training split only.
5. Train **Logistic Regression** on the vectorised training data.
6. Evaluate on the held-out test split.
7. Save artefacts to `models/`.

**Using a custom CSV instead:**
```bash
python -m src.train_model --csv data/my_labels.csv --text-col text --label-col sentiment
```
The CSV must have:
- A text column (default: `text`)
- A label column with values exactly: `Positive`, `Negative`, or `Neutral`

---

## 8. Running the Application

```bash
streamlit run app.py
```

The dashboard opens automatically at **http://localhost:8501**.

---

## 9. Live Mode vs Demo Mode

| Feature | Live Mode | Demo Mode |
|---|---|---|
| Data source | Official X API v2 | 15 hand-crafted sample posts |
| Requires API token | ✅ Yes | ❌ No |
| Shows real posts | ✅ Yes | ❌ No — clearly labeled as demo |
| Analysis | Real TF-IDF + LogReg predictions | Same model — same predictions |
| Banner | None | ⚠️ Yellow DEMO MODE banner |
| Suitable for | Live demonstration | Testing / no API access |

**Demo mode is NEVER presented as live data.** A prominent yellow banner
always distinguishes it from real API results.

---

## 10. Model Evaluation Methodology

### Preventing Data Leakage
The TF-IDF vectorizer is **fit exclusively on the training split**.
The test split is transformed using the already-fitted vectorizer.
This prevents the model from gaining unfair knowledge of test-set vocabulary.

### Metrics Explained
| Metric | Formula | Why we use it |
|---|---|---|
| **Accuracy** | correct / total | Overall correctness |
| **Precision (macro)** | avg per-class TP/(TP+FP) | Avoid false positives |
| **Recall (macro)** | avg per-class TP/(TP+FN) | Avoid missing true positives |
| **Macro F1** | harmonic mean(precision, recall) | Balanced; fair across classes |
| **Confusion Matrix** | Actual vs predicted | Shows which classes are confused |

Macro averaging gives **equal weight to each class**, making it sensitive to
poor performance on minority classes (e.g., Neutral).

### Actual Measured Results
> ⚠️ These results are generated by running `python -m src.train_model`.
> The exact values are written to `models/model_metadata.json` after training.
> Check that file or the **Model Performance** panel in the dashboard for
> the actual computed numbers.

---

## 11. Limitations & Responsible Use

- **Sampling bias**: Keyword-based search overrepresents users who tweet
  about those specific terms — not a random sample of public opinion.
- **Sarcasm & irony**: The model cannot reliably detect sarcasm or political
  irony, which are common in political discourse.
- **Context sensitivity**: The model was trained on general social-media
  sentiment; political context, dog-whistle language, and coded speech may
  be misclassified.
- **No political-affiliation inference**: Post sentiment ≠ political alignment.
- **No causal claims**: Sentiment trends do not imply causation.
- **X Developer Policy**: Comply with X's [developer agreement](https://developer.x.com/en/developer-terms/agreement-and-policy)
  for data storage, display, and deletion.
- **Educational purpose**: Results should not be used for electoral analysis,
  political targeting, or claims about population-level opinion.

---

## 12. Future Improvements

- Fine-tune a pre-trained transformer (e.g., BERTweet) for higher accuracy.
- Add named-entity recognition to identify political figures / parties.
- Implement a streaming pipeline using the X Filtered Stream endpoint.
- Add topic modelling (LDA) to discover emergent political themes.
- Extend to multilingual analysis.
- Add a time-range selector (requires Elevated API access).
- Deploy to Streamlit Cloud or Hugging Face Spaces.

---

## 13. 🎓 Viva Q&A Guide

### What is TF-IDF?

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts text into
numerical feature vectors.

- **TF (Term Frequency)**: How often a word appears in a single document.
- **IDF (Inverse Document Frequency)**: Penalises words that appear in
  many documents (common words like "the", "and" get low IDF).
- The product TF × IDF gives high scores to words that are *frequent in a
  specific document* but *rare across the corpus* — i.e., the distinctive words.

### What is Logistic Regression?

A **supervised classification algorithm** that models the probability of a
categorical outcome.  For three classes (Positive / Neutral / Negative) we
use the **multinomial** variant, which extends the binary sigmoid function
to a softmax over all classes.  The model learns weights for each TF-IDF
feature during training by minimising cross-entropy loss.

### What is Supervised Learning?

Learning from **labeled examples**.  Each training example has an input
(tweet text) and a known output (Positive / Neutral / Negative).
The model adjusts its parameters to minimise prediction error on these known
examples.  Contrast with *unsupervised learning*, which finds patterns in
unlabeled data.

### What is a Train/Test Split?

We hold out a portion of labeled data (20% here) that the model **never sees
during training**.  After training, we evaluate on this held-out test set to
measure how well the model *generalises* to new, unseen examples.

### What is Stratification?

Stratified splitting preserves the **class proportion** of the original
dataset in both splits.  Without stratification, a random split might put
all examples of a rare class into one split by chance, causing misleading
metrics.

### What is Accuracy?

The fraction of all predictions that were correct.
`accuracy = (TP + TN) / total`

### What is Precision?

Of all examples *predicted* as class C, what fraction *actually* belong to C.
High precision means few false positives.

### What is Recall?

Of all examples that *actually* belong to class C, what fraction were
correctly *predicted* as C.  High recall means few false negatives.

### What is a Macro F1-Score?

The **harmonic mean** of precision and recall, averaged equally across all
classes regardless of class size.  It balances both metrics and is sensitive
to poor performance on minority classes.

### What is a Confusion Matrix?

A square matrix where **rows = true labels** and **columns = predicted labels**.
The diagonal contains correct predictions; off-diagonal cells show
misclassifications.  It helps identify *which* classes are confused with each
other (e.g., model confusing Neutral with Positive).

### What is API Authentication?

The X API uses **OAuth 2.0 Bearer Token** (App-Only auth).  The bearer token
is a long string that identifies and authorises the application.
It is passed in the HTTP `Authorization` header as `Bearer <token>`.
We load it from an environment variable — never hard-coded — to keep it secret.

### Training Data vs Live Inference Data

| Aspect | Training Data | Live Inference Data |
|---|---|---|
| Has known labels | ✅ Yes (TweetEval labels) | ❌ No (unlabeled) |
| Used to fit model | ✅ Yes | ❌ No |
| Used for evaluation | Test split only | ❌ No |
| Model sees it during training | Train split only | ❌ No |

Live posts are **inference-only**: we apply the already-trained model to
predict labels.  We do **not** retrain on live posts (no labels available).

---

## 📁 Final Project Structure

```
SentimentIQ/
├── app.py                        ← Main Streamlit dashboard
├── requirements.txt              ← Pinned dependencies
├── README.md                     ← This file
├── .env.example                  ← Token placeholder (safe to commit)
├── .gitignore                    ← Excludes .env, model artefacts, caches
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          ← Text cleaning & normalisation
│   ├── train_model.py            ← Dataset loading, training, evaluation
│   ├── predict.py                ← Inference with saved artefacts
│   ├── x_api.py                  ← Official X API v2 integration
│   └── visualizations.py        ← Plotly chart builders
├── models/
│   ├── .gitkeep
│   ├── tfidf_vectorizer.joblib  ← Generated by train_model.py
│   ├── lr_classifier.joblib     ← Generated by train_model.py
│   └── model_metadata.json      ← Metrics & training details
├── data/
│   ├── .gitkeep
│   └── demo_posts.py            ← Explicit demo data (not real X posts)
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py    ← 20 unit tests for preprocessing
    └── test_prediction.py       ← 15 unit tests for inference module
```

---

## ⚡ Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train the sentiment model (downloads TweetEval automatically)
python -m src.train_model

# Train the hate-speech model
python -m src.hate_speech

# Run tests
pytest tests/ -v

# Start the dashboard
streamlit run app.py
```

---

*SentimentIQ — Educational College Project*  
*TF-IDF + Logistic Regression | TweetEval Dataset | Official X API v2 | Streamlit*
