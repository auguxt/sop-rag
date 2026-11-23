"""Small, dependency-free text helpers (tokenising, stemming, sentence split).

They are shared by the offline embedding model, the keyword re-ranker,
the limit/tolerance checker and the evaluation code.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "of", "to", "in", "on",
    "at", "by", "for", "with", "from", "into", "onto", "as", "is", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "doing", "done", "have", "has", "had", "it", "its",
    "this", "that", "these", "those", "there", "here", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "their", "can", "could", "should", "would", "will", "shall", "may", "might",
    "must", "please", "about", "any", "all", "each", "every", "some", "such", "no", "not", "nor",
    "only", "own", "same", "too", "very", "just", "also", "per", "after", "before", "during",
    "up", "down", "out", "over", "under", "again", "once", "between", "both", "s", "t", "us",
    "tell", "need", "want", "get", "got", "let", "like", "one", "today", "now", "much", "many",
}

# Words that appear in almost every SOP; they carry little information when
# deciding *what* a question is about.
GENERIC_DOMAIN_WORDS: Set[str] = {
    "inspection", "inspect", "inspected", "procedure", "procedures", "sop", "sops", "quality",
    "step", "steps", "document", "documents", "required", "require", "requirement", "part",
    "parts", "product", "products", "process", "plant", "check", "checks", "follow", "find",
    "found", "use", "used", "using", "value", "values", "rule", "rules", "allowed", "allowable",
    "acceptable", "accept", "maximum", "minimum", "max", "min", "limit", "limits", "tolerance",
    "tolerances", "specification", "spec", "level", "often", "long", "frequency", "interval",
    "assistant", "say", "says", "according", "state", "stated", "give", "show", "list",
}

# Everyday words that never name a documented parameter. They are ignored when
# checking whether a question mentions something that no SOP covers.
FILLER_WORDS: Set[str] = {
    "soon", "quickly", "quick", "fast", "typically", "usually", "normally", "generally", "exactly",
    "really", "actually", "ok", "okay", "fine", "good", "bad", "right", "correct", "wrong", "current",
    "currently", "latest", "new", "old", "kind", "type", "sort", "way", "thing", "things", "someone",
    "somebody", "anyone", "person", "people", "guy", "team", "time", "times", "job", "help", "know",
    "understand", "explain", "describe", "mean", "means", "meaning", "define", "definition", "happen",
    "happens", "happened", "supposed", "regarding", "related", "relevant", "possible", "situation",
    "case", "cases", "example", "something", "anything", "everything", "else", "there", "here",
    "today", "yesterday", "tomorrow", "week", "now", "currently", "came", "come", "coming", "arrived",
    "arrive", "see", "saw", "seen", "look", "looks", "seems", "think", "noticed", "notice", "found",
    "ask", "asked", "question", "answer", "hi", "hello", "thanks", "thank", "sure", "make", "made",
    "take", "taken", "put", "go", "going", "done", "next", "first", "last", "big", "small", "far",
    "hot", "cold", "high", "low", "much", "many", "few", "several", "allowed", "permitted",
}

_TOKEN_RE = re.compile(r"[a-z]{2,4}-\d{2,3}|[a-z0-9]+(?:\.\d+)?")

_SUFFIXES = (
    "izations", "ization", "ational", "ations", "ation", "ional", "ments", "ment", "ings", "ing",
    "ions", "ion", "ated", "ates", "ate", "ers", "er", "edly", "ed", "ies", "es", "al", "ly", "s",
)


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens. Defect / document codes such as 'sc-01' stay intact."""
    return _TOKEN_RE.findall(text.lower())


def stem(word: str) -> str:
    """A tiny, deterministic suffix-stripping stemmer (good enough for SOP text).

    calibration / calibrated / calibrate -> 'calibr'
    scratches / scratch                  -> 'scratch'
    """
    w = word.lower()
    if len(w) <= 3 or any(ch.isdigit() for ch in w):
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf):
            if suf == "s" and w.endswith("ss"):
                continue
            base = w[: -len(suf)]
            if len(base) < 3:
                continue
            if suf == "ies":
                base += "y"
            w = base
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiousl0123456789":
        w = w[:-1]
    if len(w) > 3 and w.endswith("ll"):
        w = w[:-1]
    return w


def content_stems(text: str, drop_generic: bool = True) -> List[str]:
    """Stems of the informative words of a text (stop words removed)."""
    out = []
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        if drop_generic and tok in GENERIC_DOMAIN_WORDS:
            continue
        out.append(stem(tok))
    return out


def stem_set(text: str, drop_generic: bool = False) -> Set[str]:
    return set(content_stems(text, drop_generic=drop_generic))


def split_sentences(text: str) -> List[str]:
    """Split a passage into sentences / list items (keeps numbers like 0.2 intact)."""
    pieces: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(-|\d{1,2}\.\d{1,2})\s+", "", line)
        for sent in re.split(r"(?<=[.;!?])\s+(?=[A-Z])", line):
            sent = sent.strip()
            if sent:
                pieces.append(sent)
    return pieces


def token_overlap(a: str, b: str) -> float:
    """Fraction of the content stems of `a` that also occur in `b` (0..1)."""
    sa = stem_set(a)
    if not sa:
        return 0.0
    sb = stem_set(b)
    return len(sa & sb) / len(sa)


def unique(seq: Iterable) -> list:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
