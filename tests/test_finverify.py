import numpy as np
import pytest

from finverify.agent import ExtractiveAgent
from finverify.conformal import ltt_threshold, make_grid, mondrian_ltt, naive_threshold
from finverify.data import synthetic_facts
from finverify.evidence import build_items, is_correct, money, parse_money
from finverify.experiment import Table, run_agent, run_regimes
from finverify.uncertainty import signals


@pytest.mark.parametrize("v", [394_328_000_000, 1_234_500_000, -2_150_000_000, 87_600_000, 5_000])
def test_money_roundtrip(v):
    for style in ("short", "full"):
        assert is_correct(parse_money(f"reported {money(v, style)} for"), v)


def test_parse_variants():
    assert parse_money("revenue of $12.5bn") == pytest.approx(12.5e9)
    assert parse_money("no numbers here") is None
    assert parse_money("It was 12.34 billion.") == pytest.approx(12.34e9)
    assert parse_money("fiscal year 2022") is None


def _simulated(n, rng):
    # P(wrong | score s) = 0.6 * (1 - s): selective risk falls as the threshold rises
    s = rng.uniform(size=n)
    return s, rng.uniform(size=n) < 0.6 * (1 - s)


def test_ltt_controls_selective_risk():
    rng = np.random.default_rng(1)
    alpha, delta, viol_ltt, viol_naive = 0.1, 0.1, 0, 0
    reps = 300
    for _ in range(reps):
        sc, wc = _simulated(300, rng)
        st, wt = _simulated(20_000, rng)
        grid = make_grid(_simulated(300, rng)[0])  # testing order from independent data
        for thr_fn, key in ((lambda: ltt_threshold(sc, wc, alpha, delta, grid), "ltt"),
                            (lambda: naive_threshold(sc, wc, alpha), "naive")):
            acc = st >= thr_fn()
            risk = wt[acc].mean() if acc.any() else 0.0
            if risk > alpha:
                if key == "ltt":
                    viol_ltt += 1
                else:
                    viol_naive += 1
    assert viol_ltt / reps <= delta + 0.03          # guarantee holds (with MC slack)
    assert viol_naive / reps > viol_ltt / reps      # naive is not safe


def test_mondrian_returns_threshold_per_group():
    rng = np.random.default_rng(0)
    s, w = _simulated(600, rng)
    g = np.where(rng.uniform(size=600) < 0.5, "a", "b")
    thr = mondrian_ltt(s, w, g, 0.1, 0.1)
    assert set(thr) == {"a", "b"}


def test_pipeline_end_to_end():
    facts = synthetic_facts(n_companies=4, years=range(2012, 2024))
    items = build_items(facts, seed=0)
    sig = [signals(it, ExtractiveAgent().sample(it, 6, np.random.default_rng(i))) for i, it in enumerate(items[:40])]
    assert all(0 <= s.feats["consistency"] <= 1 for s in sig)
    T = Table(items, run_agent(items, ExtractiveAgent(), 6, seed=0, log_every=0))
    df = run_regimes(T, reps=3, n_train=60, n_cal=100, n_test=150)
    assert set(df.regime) == {"exchangeable", "shift"}
    assert df.coverage.between(0, 1).all()
