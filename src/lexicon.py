"""
src/lexicon.py — Linguistic cue extraction for the fuzzy-logic layer.

Produces crisp, explainable features (0-1 degrees) from raw post text:
VADER lexicon sentiment, sarcasm markers, profanity, identity-group mentions,
dehumanising language, exclusion phrases and threats. The fuzzy inference
system (src/fuzzy.py) turns these into rule firings.

The word lists are intentionally small and readable. They are *cues* that
adjust the transformer's judgement, not a standalone classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

GROUP_TERMS = [
    "immigrant", "immigrants", "migrant", "migrants", "refugee", "refugees", "illegals",
    "illegal aliens", "foreigners", "women", "woman", "girls", "females", "feminists",
    "muslim", "muslims", "islam", "jew", "jews", "jewish", "christians", "hindus", "sikhs",
    "blacks", "black people", "africans", "asians", "chinese", "mexicans", "latinos", "arabs",
    "indians", "pakistanis", "whites", "white people", "gays", "gay people", "lesbians",
    "trans people", "transgender", "lgbt", "lgbtq", "homosexuals", "disabled", "minorities",
]

DEHUMANISING = [
    "vermin", "parasite", "parasites", "subhuman", "sub human", "savage", "savages", "animals",
    "cockroach", "cockroaches", "rats", "filth", "filthy", "scum", "plague", "infestation",
    "invaders", "invasion", "breeders", "apes", "pigs", "lice", "leeches", "trash", "garbage",
    "disease", "inferior",
]

EXCLUSION = [
    "go back to", "go back where", "send them back", "send them all back", "deport them",
    "deport all", "get out of our country", "get out of my country", "don't belong",
    "do not belong", "not welcome", "kick them out", "ban them", "ban all", "build the wall",
    "close the borders",
]

THREAT = [
    "kill", "killed", "killing", "murder", "shoot", "shot", "hang", "lynch", "burn them",
    "exterminate", "wipe out", "wipe them out", "eradicate", "gas them", "should die",
    "deserve to die", "beat them", "rape", "string them up", "die",
]

PROFANITY_STRONG = [
    "fuck", "fucking", "fucked", "motherfucker", "cunt", "bitch", "bitches", "whore", "whores",
    "slut", "sluts", "bastard", "bastards", "asshole", "retard", "retarded", "dickhead",
]
PROFANITY_MILD = [
    "damn", "shit", "crap", "hell", "idiot", "idiots", "stupid", "moron", "morons", "dumb",
    "loser", "losers", "ugly", "pathetic", "disgusting", "worthless",
]

# Strong markers are sarcastic on their own; weak ones ("love how ...") are often
# sincere, so they only count when corroborated (e.g. by an eye-roll emoji).
SARCASM_MARKERS = [
    "yeah right", "oh great", "just great", "thanks a lot", "thanks for nothing",
    "what a surprise", "big surprise", "well done genius", "wow just wow",
    "/s", "#sarcasm", "#not",
]
SARCASM_WEAK = ["love how", "love it when", "i just love", "so glad", "oh wonderful",
                "oh sure", "sure thing", "nice job", "as if"]
NEGATIVE_EMOJI = ["🙄", "😒", "😤", "🤦", "😑", "🙃", "😡", "🤬", "👎"]

_WORD_RE = re.compile(r"[a-z0-9']+")


def _phrase_regex(terms: list[str]) -> re.Pattern:
    alts = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])")


_RX = {
    "group":   _phrase_regex(GROUP_TERMS),
    "dehuman": _phrase_regex(DEHUMANISING),
    "exclude": _phrase_regex(EXCLUSION),
    "threat":  _phrase_regex(THREAT),
    "prof_s":  _phrase_regex(PROFANITY_STRONG),
    "prof_m":  _phrase_regex(PROFANITY_MILD),
    "sarcasm": _phrase_regex([m for m in SARCASM_MARKERS if m[0].isalnum()]),
    "sarc_weak": _phrase_regex(SARCASM_WEAK),
}
_YOU_RE = re.compile(r"(?<![a-z])(you|your|ur|u)(?![a-z])")


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None


def vader_available() -> bool:
    return _vader() is not None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class Cues:
    n_tokens: int = 0
    vader_compound: float = 0.0       # -1 .. 1 (0 if VADER unavailable)
    sarcasm: float = 0.0              # 0 .. 1
    negative_emoji: bool = False
    group: float = 0.0                # identity-group mention degree
    dehuman: float = 0.0
    exclusion: float = 0.0
    threat: float = 0.0
    profanity: float = 0.0            # 0 .. 1 (strong words weigh more)
    second_person: bool = False
    matched: dict = field(default_factory=dict)   # cue name -> matched terms (for explanations)


def _hits(name: str, text: str) -> list[str]:
    return sorted(set(_RX[name].findall(text)))


def extract(text: str) -> Cues:
    raw = text if isinstance(text, str) else ""
    low = raw.lower()
    low = re.sub(r"https?://\S+", " ", low)
    low = re.sub(r"@\w+", " @user ", low)

    c = Cues(n_tokens=len(_WORD_RE.findall(low)))
    v = _vader()
    if v is not None and raw.strip():
        c.vader_compound = float(v.polarity_scores(raw)["compound"])

    m = {k: _hits(k, low) for k in ("group", "dehuman", "exclude", "threat", "prof_s", "prof_m",
                                    "sarcasm", "sarc_weak")}
    symbolic_sarcasm = [s for s in ("/s", "#sarcasm", "#not") if s in low]
    c.negative_emoji = any(e in raw for e in NEGATIVE_EMOJI)

    # Degrees saturate: one hit is already a meaningful signal, two is near-certain.
    sat = lambda n, one=0.7: 0.0 if n == 0 else min(1.0, one + 0.3 * (n - 1))
    c.group = sat(len(m["group"]))
    c.dehuman = sat(len(m["dehuman"]))
    c.exclusion = sat(len(m["exclude"]), 0.85)
    c.threat = sat(len(m["threat"]), 0.6)
    c.profanity = min(1.0, 0.6 * len(m["prof_s"]) + 0.25 * len(m["prof_m"]))
    c.sarcasm = 1.0 if symbolic_sarcasm else sat(len(m["sarcasm"]), 0.6)
    if m["sarc_weak"]:
        c.sarcasm = max(c.sarcasm, 0.15)          # below the rule's 0.2 cut-off on its own
    if c.negative_emoji:
        c.sarcasm = min(1.0, c.sarcasm + 0.3)
    m["sarcasm"] = m["sarcasm"] + (m["sarc_weak"] if c.sarcasm > 0.2 else [])
    c.second_person = bool(_YOU_RE.search(low))

    c.matched = {k: v for k, v in {
        "identity group": m["group"], "dehumanising": m["dehuman"], "exclusion": m["exclude"],
        "threat": m["threat"], "profanity": m["prof_s"] + m["prof_m"],
        "sarcasm marker": m["sarcasm"] + symbolic_sarcasm,
    }.items() if v}
    return c
