# Senti-Q

**How are people reacting to a post on X (Twitter)?** Senti-Q reads the replies to a
post and shows:

* the **reaction mix**: what share of replies are positive, neutral and negative
  (also weighted by likes)
* an **overall response** verdict from fuzzy logic: *Overwhelmingly positive*,
  *Mostly positive*, *Mixed*, *Mostly neutral*, *Mostly negative* or
  *Overwhelmingly negative*
* the **hate speech the post received**: how many replies are hateful, and how severe
* the **post's own tone**, plus a one-line comparison ("Backlash: an upbeat post drew
  mostly negative replies")
* every reply, labelled and filterable

Final-year college project.

---

## Run it

This is a [Streamlit](https://streamlit.io) app: Python serves the web page, so there
is no HTML file to open.

**Windows:** double-click **`run.bat`**. The first run creates a virtual environment,
installs everything and trains the small models (5–10 minutes). Your browser then opens
at **http://localhost:8501**. Keep the `run.bat` window open while you use the app.

**Manually (any OS):**

```bash
python -m venv venv
venv\Scripts\activate              # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m src.train_model          # small TF-IDF sentiment model (~1-3 min)
python -m src.hate_speech          # small TF-IDF hate-speech model (~1 min)
streamlit run app.py
```

**About the language models:** the three language models (~1.5 GB) are downloaded **once**,
on the very first run, into `%USERPROFILE%\.cache\huggingface`. After that they're
loaded from disk, without contacting the internet, in the background as soon as Senti-Q
opens. That takes about 10–15 seconds per start, mostly starting PyTorch, and they stay
in memory until you close Senti-Q.

## Use it

First install the **Senti-Q for X** browser extension (one minute; see
[`browser-extension/README.md`](browser-extension/README.md)). Then either:

* **Paste a post link** like `https://x.com/username/status/1234567890` and press
  **Analyse**. Senti-Q shows the post; press **Read the replies on X**. The post opens in
  your signed-in X, the extension reads the replies, and you're brought straight back to
  the results (about 10–30 seconds).
* **Or start on X:** open any post and click the Senti-Q icon in your toolbar. On a
  timeline (or a profile or search), clicking the icon starts **pick mode**: posts highlight
  as you hover, and you click the one you want (Esc cancels). The post opens in the same
  tab, its replies are read, and you're taken back to where you were. You can also
  right-click a post's timestamp → **Read the reactions with Senti-Q**.

Results always appear in **one Senti-Q tab**, which is reused rather than a new tab each
time. Senti-Q **remembers each post it has read** while it's running: going back to a
post shows its last result instantly, and **Read again on X** refreshes it. A re-read
with the same replies gives exactly the same result and says so. If X shows new
replies, Senti-Q tells you what changed.

**Why an extension?** X only shows replies to people who are signed in. Senti-Q doesn't
use the X API, doesn't search X and never asks for your X password. The extension reads
the replies *you* can see on the post's page, in your own browser, and hands them to
Senti-Q inside the link it opens. Nothing is stored. Without the extension, Senti-Q can
still open the post itself (via X's public embed) and show its tone.

**How many replies?** 100 by default (50–300 in the extension's options), in the order
X shows them. Replies hidden behind *Show more replies* / *Show probable spam* are
included. The author's own replies, ads, image-only replies and X's unrelated
*Discover more* posts are left out. Replies that aren't in English are counted but not
scored, because the models are English-only.

---

## How it works

```
X post page (your browser)
   │  Senti-Q for X extension: scrolls the conversation, collects up to N replies,
   │  packs them (JSON → raw deflate → base64url) into the Senti-Q link
   ▼
Senti-Q  /?post=<link>&r=<packed replies>   →  src/reactions.py
   │  decode + validate (size limits, zip-bomb guard) → score every reply → summarise
   ▼
reaction mix % · likes-weighted mix · fuzzy overall response · hate received · post's own tone
```

Each reply, and the post itself, is scored by the engine:

```
text ──────┬─► RoBERTa + BERTweet sentiment models (probabilities averaged) ──────────┐
           ├─► VADER lexicon      (emoji, CAPS, "!!!", intensifiers) ─────────────────┼─► fuzzy fusion ─► sentiment + confidence
           ├─► TF-IDF + Logistic Regression ─────────────────────────────────────────┘
           ├─► RoBERTa hate       (cardiffnlp/twitter-roberta-base-hate-latest) ──────┐
           ├─► TF-IDF hate model ─────────────────────────────────────────────────────┼─► fuzzy fusion ─► hate score + severity
           └─► linguistic cues    (identity groups, dehumanising words, threats, …) ──┘          + "why" for both
```

**Fuzzy-logic layer** (`src/fuzzy.py`): a zero-order Takagi–Sugeno inference system.

1. **Fuzzify** inputs into graded sets, e.g. transformer confidence *low / medium / high*,
   post length *short / medium / long*, VADER intensity *weak / moderate / strong*.
2. **Fire rules** (AND = min, OR = max, NOT = 1 − x), for example:
   * IF the transformer is highly confident THEN trust it most
   * IF the transformer is unsure AND lexical emotion is strong THEN raise VADER's weight
   * IF a sarcasm marker appears AND the post sounds positive THEN shift toward negative
   * IF an identity group is mentioned AND dehumanising / exclusion / threat language is present THEN hate is high
   * IF there is profanity but no targeted group THEN *offensive, not hate*
3. **Defuzzify** with the firing-strength-weighted average.

The rule strengths are learned on the validation split (`python -m src.evaluate`).

**Overall response** (`fuzzy.response_verdict`): a second fuzzy rule base over the
reply shares.

* **Inputs:** *net tone* = %positive − %negative (*very negative … very positive*) and
  the *neutral share* (*high*).
* **Example rules:**
  * IF net tone is very positive AND neutral share is NOT high THEN *Overwhelmingly positive*
  * IF net tone is balanced AND neutral share is NOT high THEN *Mixed*
* **Output:** the strongest-firing label wins.

The app also shows a 95% margin of error for the main share, so small samples aren't
over-read.

## Results

Scored on TweetEval's official, untouched test sets
(`python -m src.evaluate` → `models/benchmark.json`):

| Engine | Sentiment accuracy (12,284 tweets) | Hate-speech accuracy (2,970 tweets) |
|---|---|---|
| Original TF-IDF + Logistic Regression | 59.5% | 48.3% |
| VADER lexicon only | 54.8% | n/a |
| One RoBERTa model (the previous version of Senti-Q) | 72.2% | 86.7% |
| Two-model sentiment combination, no fuzzy layer | 73.2% | 86.7% |
| **Fuzzy ensemble (Senti-Q)** | **73.1%** | **86.7%** |

* **Where the accuracy comes from:** the transformers. That's +13.6 points on sentiment
  and +38.4 on hate over the original model.
* **What the fuzzy layer does to accuracy:** nothing measurable on this benchmark. Its
  value is the confidence level, the severity grading (offensive vs hate vs severe),
  sarcasm handling and the explanations.
* **How far can it go?** These benchmarks have a ceiling: people disagree about the
  sentiment of short tweets, so even very large models score only in the low-to-mid 70s
  on this test set (from published results as I recall them; not re-checked here).
  Expect small gains from here, not big ones.

### Model search (`python -m src.compare_models`)

Each candidate model is scored on the official validation and test sets, then every
combination is tried (probabilities averaged). The rule was fixed **before** looking at
results: a combination ships only if it beats the current model on validation **and** does
not regress on test.

| Task | Winner on validation | Test result | Shipped? |
|---|---|---|---|
| Sentiment | RoBERTa-sentiment + BERTweet | accuracy 72.2 → 73.2, macro recall 73.5 → 74.0 | **Yes** |
| Hate speech | RoBERTa-hate-latest + RoBERTa-hate | accuracy **86.7 → 79.8** | **No** |

* The sentiment gain is real but small (+1 point).
* The hate combination looked best on validation and then collapsed on test (HatEval's
  test set is drawn from a different distribution), so the single model stays. The same
  thing happened to the hate fine-tuning run below.
* BERTweet-sentiment was trained on SemEval-2017 tweets that TweetEval's *validation*
  split comes from, so its validation score (83.8%) is inflated. The test split is the
  fair comparison.
* A rule that treated insults aimed at nobody in particular as "offensive, not hate" was
  also tested. It slightly lowered accuracy on both splits, so it was not added.
* Converting emojis to words before scoring was tested too: accuracy on the emoji-bearing
  test tweets fell from 73.1% to 68.8%, so emojis are passed to the models as they are.

### Fine-tuning experiments

`python -m src.finetune --task sentiment|hate` fine-tunes the RoBERTa models on a GPU
(an RTX 3050 4 GB laptop GPU took 3–18 minutes per run). The rules were fixed **before**
running:

* pick the checkpoint by validation score (with early stopping)
* ship a fine-tuned model only if it beats the original on validation **and** doesn't
  regress on the official test set

| Run | Validation | Test | Shipped? |
|---|---|---|---|
| Sentiment: TweetEval + 26.7k Tweet Sentiment Extraction tweets, lr 1e-5, 2 epochs | macro recall 77.6 → 76.1 | acc 72.2 → 71.3 | No |
| Sentiment: TweetEval only, lr 5e-6, 1 epoch | macro recall 77.6 → 78.1 | macro recall 73.5 → 73.7, acc 72.2 → 71.4 | Yes, on the machine that trained it |
| Hate: HatEval train, lr 1e-5, 2 epochs | macro F1 85.6 → 86.6 | acc **86.7 → 74.4** | No |

What the experiments show:

* The CardiffNLP models were already fine-tuned on this exact data, so further
  fine-tuning is close to a tie. Mixing in a dataset with a different labelling style
  made things worse.
* The hate run improved on validation but collapsed on test. HatEval's test set is
  deliberately drawn from a different distribution, which is exactly why the
  test-regression gate exists.

Reports are in `models/finetuned/*_report.json`. Fine-tuned weights (~480 MB) are too
large for GitHub, so they stay on the machine that trained them. The model combination in
`models/ensemble.json` takes precedence over them, so every machine runs the same models.

---

## Project structure

```
app.py                     Senti-Q web app (Streamlit)
run.bat                    One-click Windows launcher
assets/                    style.css (design system), favicon
browser-extension/         "Senti-Q for X" Chrome/Edge extension
  content.js               Reads the replies on a post's page (only when asked)
  background.js            Packs them into the Senti-Q link, manages tabs
src/
  reactions.py             Decodes the replies, scores them, summarises the response
  x_links.py               Open X posts from their links (oEmbed)
  engine.py                analyze(): runs every model and the fuzzy fusion
  fuzzy.py                 Fuzzy inference system
  lexicon.py               VADER + linguistic cue extraction
  transformer_models.py    RoBERTa models (uses a fine-tuned one if shipped)
  predict.py, train_model.py   TF-IDF sentiment model
  hate_speech.py           TF-IDF hate-speech model
  finetune.py              GPU fine-tuning with the ship/no-ship protocol
  evaluate.py              Benchmark + learning the fuzzy rule strengths
  preprocessing.py         Text cleaning
models/                    Trained artefacts, benchmark and fine-tuning reports
tests/                     pytest suite (pytest tests -q)
```

## Limitations

* Results are model estimates, not verdicts. Subtle irony, new slang and coded language
  can still fool them. Never use a flag to judge or report an individual.
* The hate-speech training data focuses on English posts about immigrants and women.
  Other targets and languages are weaker.
* Only public posts can be opened. Text inside images and videos isn't analysed.
* The reactions are the replies X chose to show you, in X's order. That's a sample, not
  every reply, and X ranks some replies lower or hides them.
* The extension depends on how X's web page is built. If X changes its page structure,
  `browser-extension/content.js` may need updating. It was tested against a replica of
  X's post page, not the live site.

## Troubleshooting

* **"This site can't be reached" at localhost:8501.** The server isn't running yet. Keep
  the `run.bat` window open and wait for your browser to open by itself; the first run
  takes several minutes.
* **"Getting the language models ready…"** appears for a few seconds right after Senti-Q
  starts, while the models load from disk. Nothing is being downloaded.
* **"Read the replies on X" opens X but nothing comes back.** The extension isn't
  installed or enabled, or you aren't signed in to X in that browser. After installing or
  updating the extension, reload any open X tabs.
* **The extension does nothing.** Make sure Senti-Q is running. On a timeline, scroll a
  post into the middle of the screen before clicking. If you host Senti-Q at a different
  address, set it under the extension's **Options**.
* **To demo from a phone on the same Wi-Fi,** set `address = "0.0.0.0"` in
  `.streamlit/config.toml`.

---

## Viva quick reference

* **TF-IDF:** turns text into numbers. It weights words that are frequent in one post
  but rare overall.
* **Logistic Regression:** a linear classifier that outputs class probabilities
  (softmax over three classes).
* **Transformer / RoBERTa:** a neural network that reads the whole sentence with
  self-attention, so it understands context and negation ("not bad"). Twitter-RoBERTa
  was pre-trained on ~124M tweets.
* **Fine-tuning:** continuing to train a pre-trained model on labelled task data, here
  on a GPU with mixed precision, early stopping and checkpoint selection on validation.
* **Fuzzy logic:** reasoning with degrees of truth (0–1) instead of true/false, via
  membership functions, IF-THEN rules and defuzzification.
* **Train / validation / test:** fit on train; choose settings on validation; report on
  test, which is never used for tuning.
* **Accuracy / precision / recall / F1:** overall correctness / how many flagged were
  right / how many real ones were caught / their harmonic mean. *Macro* averages
  classes equally.
* **Data leakage:** letting test information influence training or tuning. Avoided by
  fitting TF-IDF on train only and tuning on validation only.
