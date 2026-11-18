"""
tests/test_fuzzy.py — Unit tests for the fuzzy inference layer and cue extraction.
Pure functions: no models or network needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import fuzzy, lexicon
from src.lexicon import Cues


# --- membership functions ---------------------------------------------------

def test_trapezoid_shape():
    assert fuzzy.trap(0.5, 0, 0.2, 0.8, 1) == 1.0
    assert fuzzy.trap(0.1, 0, 0.2, 0.8, 1) == pytest.approx(0.5)
    assert fuzzy.trap(0.9, 0, 0.2, 0.8, 1) == pytest.approx(0.5)
    assert fuzzy.trap(1.5, 0, 0.2, 0.8, 1) == 0.0
    assert fuzzy.tri(0.5, 0, 0.5, 1) == 1.0


def test_vader_fuzzification_is_a_distribution():
    for c in (-1, -0.4, 0, 0.2, 0.9):
        p = fuzzy.vader_to_probs(c)
        assert p.sum() == pytest.approx(1.0)
    assert fuzzy.vader_to_probs(0.9).argmax() == 2
    assert fuzzy.vader_to_probs(-0.9).argmax() == 0
    assert fuzzy.vader_to_probs(0.0).argmax() == 1


# --- sentiment fusion -------------------------------------------------------

def test_confident_transformer_dominates():
    p_t = np.array([0.03, 0.05, 0.92])
    p_l = np.array([0.7, 0.2, 0.1])            # TF-IDF disagrees
    r = fuzzy.fuse_sentiment(p_t, p_l, Cues(n_tokens=12, vader_compound=0.3))
    assert r.label == "Positive"
    assert r.weights["transformer"] > 0.6
    assert r.probs.sum() == pytest.approx(1.0)


def test_sarcasm_shifts_positive_toward_negative():
    p_t = np.array([0.2, 0.2, 0.6])
    plain = fuzzy.fuse_sentiment(p_t, None, Cues(n_tokens=8, vader_compound=0.6))
    sarcastic = fuzzy.fuse_sentiment(p_t, None, Cues(n_tokens=8, vader_compound=0.6, sarcasm=1.0))
    assert sarcastic.probs[0] > plain.probs[0]
    assert any("Sarcasm" in n for n in sarcastic.notes)


def test_works_without_transformer():
    r = fuzzy.fuse_sentiment(None, np.array([0.1, 0.2, 0.7]), Cues(n_tokens=5, vader_compound=0.8))
    assert r.weights["transformer"] == 0
    assert r.label == "Positive"


# --- hate fusion ------------------------------------------------------------

def test_targeted_dehumanising_language_is_hate():
    c = Cues(n_tokens=8, group=0.7, dehuman=0.7)
    r = fuzzy.fuse_hate(0.55, None, c, threshold=0.5)
    assert r.is_hate and r.severity in ("Hate", "Severe")


def test_profanity_alone_is_offensive_not_hate():
    c = Cues(n_tokens=6, profanity=0.85)
    r = fuzzy.fuse_hate(0.15, None, c, threshold=0.5)
    assert not r.is_hate
    assert r.severity == "Offensive"


def test_clean_text_is_not_hate():
    r = fuzzy.fuse_hate(0.03, 0.1, Cues(n_tokens=10), threshold=0.5)
    assert not r.is_hate and r.severity == "None"


# --- cue extraction ---------------------------------------------------------

def test_cue_extraction():
    c = lexicon.extract("Send them back, these immigrants are vermin")
    assert c.group > 0 and c.dehuman > 0 and c.exclusion > 0
    assert "identity group" in c.matched

    s = lexicon.extract("Oh great, another Monday 🙄")
    assert s.sarcasm > 0.5 and s.negative_emoji

    # word boundaries: "skill" must not match the threat word "kill"
    assert lexicon.extract("what a skillful player").threat == 0
