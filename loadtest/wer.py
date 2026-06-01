"""Word error rate computation, byte-for-byte compatible with the Java load test.

The normalization and edit-distance algorithm match the Java reference
implementation so WER numbers compare directly across languages.
"""

from __future__ import annotations

import string

_PUNCTUATION = set(string.punctuation)


def normalize_words(text: str | None) -> list[str]:
    """Lowercase, replace ASCII punctuation with whitespace, collapse runs."""
    if text is None:
        return []
    buf = []
    for ch in text:
        if ch in _PUNCTUATION:
            buf.append(" ")
        else:
            buf.append(ch.lower())
    out = "".join(buf).strip().split()
    return out


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance using two rolling rows."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    curr = [0] * (len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        curr[0] = i
        ri = ref[i - 1]
        for j in range(1, len(hyp) + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[len(hyp)]


def compute_wer(reference: str | None, hypothesis: str | None) -> float | None:
    """Word error rate = edit_distance(ref_words, hyp_words) / len(ref_words).

    Returns ``None`` when the hypothesis is empty or the reference has no
    words -- mirrors the Java load test so "no transcript" rows show as a
    blank in summary.csv rather than an inflated 100% miss.
    """
    if hypothesis is None or not hypothesis.strip():
        return None
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return None
    return edit_distance(ref, hyp) / len(ref)
