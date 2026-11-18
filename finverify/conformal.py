"""Choosing an accept/abstain threshold with a finite-sample guarantee.

Policy: answer when confidence score >= lambda, otherwise abstain.
Selective risk:  R(lambda) = P(answer wrong | answered).

Learn-then-Test (Angelopoulos, Bates, Candes, Jordan, Lei 2021):
  for each candidate lambda, test H0: R(lambda) > alpha with an exact binomial p-value,
  walking from the most conservative lambda downward (fixed-sequence testing, so no
  multiplicity correction is needed), and stop at the first lambda that is not rejected.
  Guarantee (calibration and test exchangeable):  P( R(lambda_hat) <= alpha ) >= 1 - delta.

Group-wise (Mondrian) LTT runs the same procedure inside each observable group g with
delta / |G| (union bound). Every group's selective risk is then <= alpha simultaneously,
so the overall selective risk stays <= alpha even if the *mix* of groups changes at test
time -- the shift we expect when evidence gets noisier.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

NEVER = np.inf  # threshold that accepts nothing


def make_grid(ref_scores: np.ndarray, top_quantile: float = 0.7, n: int = 100) -> np.ndarray:
    """Candidate thresholds, most conservative first.

    For LTT the grid (and hence the testing order) must not depend on the calibration
    data, so build it from a separate split (we use the scorer's training split).

    Fixed-sequence testing stops at the first hypothesis it cannot reject, and with zero
    errors a binomial test needs n >= log(delta) / log(1 - alpha) accepted points (about 33
    for alpha = 0.1, delta = 0.1/3). Starting at the 70th percentile instead of the maximum
    keeps roughly 30% of points in the first test, so the sequence does not die on step one.
    """
    if len(ref_scores) == 0:
        return np.array([])
    qs = np.unique(np.quantile(ref_scores, np.linspace(0, top_quantile, n)))
    return qs[::-1]


def naive_threshold(scores: np.ndarray, wrong: np.ndarray, alpha: float, grid=None) -> float:
    """Lowest threshold whose *empirical* selective risk is <= alpha. No guarantee."""
    best = NEVER
    for lam in (make_grid(scores, 1.0) if grid is None else grid):
        acc = scores >= lam
        if acc.any() and wrong[acc].mean() <= alpha:
            best = lam
    return best


def ltt_threshold(scores: np.ndarray, wrong: np.ndarray, alpha: float, delta: float, grid=None) -> float:
    best = NEVER
    for lam in (make_grid(scores) if grid is None else grid):
        acc = scores >= lam
        n, k = int(acc.sum()), int(wrong[acc].sum())
        if n == 0:
            continue  # accepting nothing is trivially safe
        p_value = binom.cdf(k, n, alpha)  # P(<= k errors | true risk = alpha); valid for risk >= alpha
        if p_value > delta:
            break
        best = lam
    return best


def mondrian_ltt(scores, wrong, groups, alpha, delta, grids: dict | None = None) -> dict[str, float]:
    labels = sorted(set(groups))
    d = delta / len(labels)
    return {g: ltt_threshold(scores[groups == g], wrong[groups == g], alpha, d,
                             None if grids is None else grids.get(g, np.array([])))
            for g in labels}


def apply_thresholds(scores, groups, thr) -> np.ndarray:
    if isinstance(thr, dict):
        return np.array([s >= thr.get(g, NEVER) for s, g in zip(scores, groups)])
    return scores >= thr
