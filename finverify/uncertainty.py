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


def signals(item: Item, samples: list[float | None]) -> Signals:
    k = len(samples)
    vals = [s for s in samples if s is not None]
    clusters = _cluster(vals)
    answer = clusters[0][0] if clusters else None

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
