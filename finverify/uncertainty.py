"""Uncertainty signals for one item, plus a small learned confidence model.

Two families of signals:
  * sampling-based   - agreement among the agent's K sampled answers (self-consistency)
  * evidence-based   - a verifier that checks the chosen answer against the evidence:
                       is it supported by a passage about the same concept and period,
                       is it contradicted by one, and is the support from a trusted source?
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .evidence import Item, is_correct, parse_money

_Q = re.compile(r"^What (?:was|were) (?P<co>.+?)'s (?P<concept>.+?) (?P<period>(?:for the fiscal year ended|as of) .+)\?$")
_TRUSTED = ("Form 10-K", "audited financial statements")

FEATURES = ["consistency", "answered_frac", "n_clusters", "supported", "conflict",
            "trusted_support", "n_matching", "bias"]


@dataclass
class Signals:
    answer: float | None
    correct: bool
    feats: dict[str, float]
    group: str               # observable evidence state, used for group-wise calibration

    def vector(self) -> np.ndarray:
        return np.array([self.feats[f] for f in FEATURES], dtype=float)


def _cluster(values: list[float]) -> list[tuple[float, int]]:
    clusters: list[list[float]] = []
    for v in values:
        for c in clusters:
            if is_correct(v, c[0]):
                c.append(v)
                break
        else:
            clusters.append([v])
    return sorted(((c[0], len(c)) for c in clusters), key=lambda t: -t[1])


_SCALED = re.compile(r"\(?\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)\)?\s*(billion|million|thousand)?", re.I)


def _candidates(p: str) -> list[float]:
    """Every dollar figure a reader could take from a real 10-K passage, in USD."""
    scale = 1e6 if "[in millions" in p else 1e3 if "[in thousands" in p else 1e9 if "[in billions" in p else None
    body = p.split("] ", 1)[1] if p.startswith("[") else p
    out = []
    for m in _SCALED.finditer(body):
        num, unit = m.group(1), (m.group(2) or "").lower()
        v = float(num.replace(",", ""))
        if unit:
            out.append(v * {"billion": 1e9, "million": 1e6, "thousand": 1e3}[unit])
        elif scale and "," in num:
            out.append(v * scale)
    return out


def _signals_real(item: Item, answer: float | None) -> tuple[dict, str]:
    from .realtext import KEYWORDS

    kws = KEYWORDS[item.fact.concept]
    grounded = trusted = conflict = 0.0
    n_match = 0
    for p in item.passages:
        is_feed = p.startswith("Data feed")
        concept_hit = any(k in p.lower() for k in kws)
        n_match += concept_hit
        cands = _candidates(p)
        if answer is not None and any(is_correct(c, answer) for c in cands):
            grounded = 1.0
            trusted = max(trusted, float(not is_feed))
        if answer is not None and concept_hit and cands:
            c0 = cands[0]  # first figure in a line item = most recent column
            if not is_correct(c0, answer) and abs(c0 / answer - 1) < 0.35:
                conflict = 1.0
    group = "unsupported" if not grounded else ("supported" if trusted else "untrusted")
    return {"supported": grounded, "conflict": conflict, "trusted_support": trusted,
            "n_matching": min(n_match, 3) / 3}, group


def signals(item: Item, samples: list[float | None]) -> Signals:
    k = len(samples)
    vals = [s for s in samples if s is not None]
    clusters = _cluster(vals)
    answer = clusters[0][0] if clusters else None

    if item.meta.get("source") == "10k":
        ev, group = _signals_real(item, answer)
        feats = {"consistency": clusters[0][1] / k if clusters else 0.0, "answered_frac": len(vals) / k,
                 "n_clusters": len(clusters) / k, **ev, "bias": 1.0}
        return Signals(answer, is_correct(answer, item.fact.value), feats, group)

    m = _Q.match(item.question)
    concept, period = (m.group("concept"), m.group("period")) if m else ("", "")
    matching = [p for p in item.passages if concept in p and period in p]
    supported = trusted = 0.0
    for p in matching:
        v = parse_money(p)
        if answer is not None and v is not None and is_correct(v, answer):
            supported = 1.0
            trusted = max(trusted, float(any(t in p for t in _TRUSTED)))
    # conflict = the relevant evidence itself disagrees (independent of what the agent chose)
    matched_vals = [v for v in (parse_money(p) for p in matching) if v is not None]
    conflict = float(len(_cluster(matched_vals)) > 1)

    feats = {
        "consistency": clusters[0][1] / k if clusters else 0.0,
        "answered_frac": len(vals) / k,
        "n_clusters": len(clusters) / k,
        "supported": supported,
        "conflict": conflict,
        "trusted_support": trusted,
        "n_matching": min(len(matching), 3) / 3,
        "bias": 1.0,
    }
    group = "conflict" if conflict else ("supported" if supported else "unsupported")
    return Signals(answer, is_correct(answer, item.fact.value), feats, group)


class LogisticScorer:
    """Plain L2-regularised logistic regression (numpy only) mapping signals -> P(correct)."""

    def __init__(self, l2: float = 1e-2, steps: int = 2000, lr: float = 0.5):
        self.l2, self.steps, self.lr = l2, steps, lr
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticScorer":
        w = np.zeros(X.shape[1])
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-X @ w))
            grad = X.T @ (p - y) / len(y) + self.l2 * w
            w -= self.lr * grad
        self.w = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X @ self.w))
