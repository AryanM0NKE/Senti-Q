"""
src/fuzzy.py — Fuzzy inference system that fuses SentimentIQ's experts.

Why fuzzy logic?
    Each expert (RoBERTa transformer, VADER lexicon, TF-IDF model) is good in
    different situations: the transformer understands context, VADER is strong on
    short emotive posts with emojis/CAPS, TF-IDF is a cheap second opinion. Rather
    than a fixed average, fuzzy rules decide *how much to trust each expert for
    this particular post*, using graded (0-1) notions like "confidence is HIGH"
    or "post is SHORT". Every decision is a readable IF-THEN rule, so each verdict
    can be explained.

Method: zero-order Takagi-Sugeno inference.
    1. Fuzzify crisp inputs with trapezoidal / triangular membership functions.
    2. Fire rules: AND = min, OR = max, NOT = 1 - x.
    3. Each rule outputs a constant (or a vector of expert weights).
    4. Defuzzify by the firing-strength-weighted average of the rule outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.lexicon import Cues

LABELS = ["Negative", "Neutral", "Positive"]


# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------

def trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid: 0 below a, rises to 1 on [b, c], falls to 0 at d."""
    if x <= a or x >= d:
        return 1.0 if (b <= x <= c) else 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


def tri(x: float, a: float, b: float, c: float) -> float:
    return trap(x, a, b, b, c)


# Linguistic variables ------------------------------------------------------

def conf_sets(p: float) -> dict:
    """Confidence of an expert's top class (0.33 .. 1)."""
    return {"low": trap(p, -1, 0, 0.45, 0.6), "medium": tri(p, 0.45, 0.65, 0.85),
            "high": trap(p, 0.7, 0.88, 1, 2)}


def length_sets(n: int) -> dict:
    return {"short": trap(n, -1, 0, 4, 8), "medium": tri(n, 5, 14, 26), "long": trap(n, 18, 30, 999, 1000)}


def intensity_sets(abs_compound: float) -> dict:
    """Strength of VADER's lexical polarity |compound|."""
    return {"weak": trap(abs_compound, -1, 0, 0.1, 0.3), "moderate": tri(abs_compound, 0.15, 0.4, 0.65),
            "strong": trap(abs_compound, 0.5, 0.7, 1, 2)}


def prob_sets(p: float) -> dict:
    """A probability (e.g. P(hate)) described as low / medium / high."""
    return {"low": trap(p, -1, 0, 0.2, 0.4), "medium": tri(p, 0.25, 0.5, 0.75), "high": trap(p, 0.6, 0.8, 1, 2)}


def vader_to_probs(compound: float) -> np.ndarray:
    """Fuzzify VADER's compound score into a Negative/Neutral/Positive distribution."""
    mu = np.array([trap(compound, -2, -1, -0.5, -0.05),
                   tri(compound, -0.35, 0.0, 0.35),
                   trap(compound, 0.05, 0.5, 1, 2)])
    return mu / mu.sum() if mu.sum() > 0 else np.array([1 / 3] * 3)


# ---------------------------------------------------------------------------
# Rule engine helper
# ---------------------------------------------------------------------------

@dataclass
class RuleBase:
    fired: list = field(default_factory=list)     # (strength, output, description)

    def rule(self, strength: float, output, description: str):
        if strength > 0:
            self.fired.append((float(strength), output, description))

    def defuzzify(self, default):
        if not self.fired:
            return default
        w = np.array([s for s, _, _ in self.fired])
        outs = np.array([np.asarray(o, dtype=float) for _, o, _ in self.fired])
        return (w[:, None] * outs.reshape(len(w), -1)).sum(0) / w.sum()

    def top(self, k: int = 3, min_strength: float = 0.25) -> list[str]:
        ranked = sorted(self.fired, key=lambda r: -r[0])
        return [d for s, _, d in ranked if s >= min_strength][:k]


# ---------------------------------------------------------------------------
# Sentiment fusion
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    probs: np.ndarray            # [neg, neu, pos]
    label: str
    confidence: str              # High / Medium / Low
    weights: dict                # expert -> weight actually used
    notes: list


def fuse_sentiment(p_t: np.ndarray | None, p_l: np.ndarray | None, cues: Cues,
                   support_scale: float = 1.0) -> SentimentResult:
    """
    p_t : transformer probabilities [neg, neu, pos] (None if unavailable)
    p_l : TF-IDF probabilities      [neg, neu, pos] (None if unavailable)
    support_scale : how much the support experts (VADER, TF-IDF) count relative to
        the rule consequents; learned on the validation split by src/evaluate.py.
    """
    p_v = vader_to_probs(cues.vader_compound)
    conf_t = conf_sets(float(p_t.max())) if p_t is not None else {"low": 1, "medium": 0, "high": 0}
    length = length_sets(cues.n_tokens)
    inten = intensity_sets(abs(cues.vader_compound))

    # --- Rule base 1: how much to trust each expert -> weights (T, V, L) ---
    rb = RuleBase()
    rb.rule(conf_t["high"],                          (0.85, 0.10, 0.05), "Transformer is highly confident, so it dominates")
    rb.rule(conf_t["medium"],                        (0.65, 0.20, 0.15), "Transformer is moderately confident")
    rb.rule(min(conf_t["low"], inten["strong"]),     (0.40, 0.45, 0.15), "Transformer unsure but strong lexical emotion, so VADER weighted up")
    rb.rule(min(conf_t["low"], 1 - inten["strong"]), (0.55, 0.20, 0.25), "All signals weak, so experts are blended")
    rb.rule(min(length["short"], inten["strong"]),   (0.45, 0.45, 0.10), "Short emotive post, where lexicon cues are reliable")
    rb.rule(length["long"],                          (0.75, 0.15, 0.10), "Long post, where context matters most")
    w = rb.defuzzify(np.array([0.6, 0.25, 0.15]))

    experts = {"transformer": p_t, "vader": p_v, "tfidf": p_l}
    avail = np.array([experts[k] is not None for k in experts], dtype=float)
    w = w * np.array([1.0, support_scale, support_scale]) * avail
    if w.sum() == 0:                       # transformer missing and support scaled to 0
        w = avail.copy()
    w = w / w.sum()
    fused = sum(wi * (p if p is not None else 0) for wi, p in zip(w, experts.values()))
    notes = rb.top(2)

    # --- Rule base 2: sarcasm (positive surface, negative intent) ---
    pos_dominant = trap(float(fused[2] - fused[0]), 0.0, 0.25, 1, 2)
    sarcasm_strength = min(cues.sarcasm, pos_dominant)
    if sarcasm_strength > 0.2:
        shift = 0.7 * sarcasm_strength * fused[2]
        fused = fused + np.array([shift, 0, -shift])
        notes.insert(0, "Sarcasm cue on a positive-sounding post, so it is shifted toward negative")

    fused = np.clip(fused, 0, None)
    fused = fused / fused.sum()
    label = LABELS[int(fused.argmax())]

    # --- Rule base 3: linguistic confidence of the final verdict ---
    srt = np.sort(fused)
    margin = float(srt[-1] - srt[-2])
    votes = [LABELS[int(p.argmax())] for p in (p_t, p_v, p_l) if p is not None]
    agree = votes.count(label) / len(votes)
    m = {"low": trap(margin, -1, 0, 0.1, 0.25), "medium": tri(margin, 0.1, 0.3, 0.5), "high": trap(margin, 0.35, 0.55, 1, 2)}
    a_hi, a_lo = trap(agree, 0.5, 0.9, 1, 2), trap(agree, -1, 0, 0.4, 0.7)
    conf = {"High": min(m["high"], a_hi),
            "Medium": max(m["medium"], min(m["high"], a_lo)),
            "Low": max(m["low"], min(m["medium"], a_lo))}
    confidence = max(conf, key=conf.get)

    return SentimentResult(fused, label, confidence,
                           dict(zip(experts, np.round(w, 3).tolist())), notes)


# ---------------------------------------------------------------------------
# Hate-speech fusion
# ---------------------------------------------------------------------------

@dataclass
class HateResult:
    score: float
    is_hate: bool
    severity: str                # None / Offensive / Hate / Severe
    notes: list


HATE_THRESHOLD = 0.5   # tuned on the TweetEval validation split by src/evaluate.py


def fuse_hate(p_t: float | None, p_l: float | None, cues: Cues, threshold: float = HATE_THRESHOLD,
              support_scale: float = 1.0, cue_scale: float = 1.0) -> HateResult:
    """
    p_t : transformer P(hate) (None if unavailable)
    p_l : TF-IDF P(hate)      (None if unavailable)
    support_scale, cue_scale : strength multipliers for the TF-IDF-agreement rule and
        the lexical-cue rules; learned on the validation split by src/evaluate.py.
    """
    primary = p_t if p_t is not None else (p_l if p_l is not None else 0.0)
    pt = prob_sets(primary)
    pl = prob_sets(p_l) if p_l is not None else {"low": 0, "medium": 0, "high": 0}
    hostile = max(cues.dehuman, cues.exclusion, cues.threat)

    rb = RuleBase()
    # First-order consequents: the model-confidence rules output the model's own
    # probability, so with no other evidence the fuzzy score equals P(hate). The
    # cue / agreement rules below can then only move a score when they fire.
    rb.rule(pt["high"],                               primary, "Model detects hateful language with high probability")
    rb.rule(pt["medium"],                             primary, "Model gives a moderate hate probability")
    rb.rule(pt["low"],                                primary, "Model finds little hateful language")
    rb.rule(support_scale * min(pt["medium"], pl["high"]), 0.75, "Second model agrees the language is hateful")
    rb.rule(cue_scale * min(cues.group, hostile),          0.90, "Identity group targeted with dehumanising, exclusionary or threatening language")
    rb.rule(cue_scale * min(cues.threat, float(cues.second_person), 1 - pt["low"]),
                                                           0.75, "Threat directed at a person")
    rb.rule(cue_scale * min(cues.profanity, 1 - cues.group, 1 - pt["high"]),
                                                           0.30, "Profanity without a targeted group: offensive, not hate")
    score = float(rb.defuzzify(np.array([primary]))[0])

    is_hate = score >= threshold
    if is_hate and score >= 0.8 and (cues.threat > 0 or cues.dehuman > 0):
        severity = "Severe"
    elif is_hate:
        severity = "Hate"
    elif cues.profanity > 0 or score >= threshold - 0.15:
        severity = "Offensive"
    else:
        severity = "None"

    notes = rb.top(2, 0.3)
    for name in ("identity group", "dehumanising", "exclusion", "threat"):
        if name in cues.matched and is_hate:
            notes.append(f"{name.capitalize()} cue: {', '.join(cues.matched[name][:3])}")
    return HateResult(round(score, 4), bool(is_hate), severity, notes)
