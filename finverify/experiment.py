"""End-to-end experiment: answer -> score -> calibrate -> evaluate, under two regimes.

exchangeable  calibration and test drawn from the same mix of evidence conditions
shift         calibrate on older fiscal years with mostly clean evidence,
              test on newer fiscal years with mostly contradictory / incomplete evidence
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .conformal import apply_thresholds, make_grid, ltt_threshold, mondrian_ltt, naive_threshold
from .evidence import CONDITIONS, Item
from .uncertainty import FEATURES, LogisticScorer, Signals, signals

EXCH_MIX = {"clean": 0.25, "noisy": 0.25, "contradictory": 0.25, "incomplete": 0.25}
CAL_MIX = {"clean": 0.60, "noisy": 0.25, "contradictory": 0.10, "incomplete": 0.05}
TEST_MIX = {"clean": 0.15, "noisy": 0.15, "contradictory": 0.35, "incomplete": 0.35}
UNANSWERED = -1.0  # score for "insufficient evidence" answers; all real scores are in [0, 1]

METHODS = [
    "always answer",
    "self-consistency + naive",
    "self-consistency + LTT",
    "verifier + naive",
    "verifier + LTT",
    "verifier + group-wise LTT",
]


@dataclass
class Table:
    items: list[Item]
    sig: list[Signals]

    def __post_init__(self):
        self.X = np.stack([s.vector() for s in self.sig])
        self.wrong = np.array([not s.correct for s in self.sig])
        self.answered = np.array([s.answer is not None for s in self.sig])
        self.cons = np.where(self.answered, self.X[:, FEATURES.index("consistency")], UNANSWERED)
        self.group = np.array([s.group for s in self.sig])
        self.cond = np.array([it.condition for it in self.items])
        self.year = np.array([it.fact.year for it in self.items])


def run_agent(items: list[Item], agent, k: int, seed: int, log_every: int = 500,
              cache_path: Path | None = None) -> list[Signals]:
    """Sample K answers per item. With `cache_path`, answers are appended to a JSONL file
    as they arrive, so a slow or interrupted LLM run resumes where it stopped."""
    cache: dict[str, list] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cache[rec["uid"]] = rec["samples"]
    fh = None
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fh = cache_path.open("a", encoding="utf-8")
    out = []
    try:
        for i, it in enumerate(items):
            samples = cache.get(it.uid)
            if samples is None or len(samples) < k:
                samples = agent.sample(it, k, np.random.default_rng([seed, i]))
                if fh:
                    fh.write(json.dumps({"uid": it.uid, "samples": samples}) + "\n")
                    fh.flush()
            out.append(signals(it, samples[:k]))
            if log_every and (i + 1) % log_every == 0:
                print(f"  answered {i + 1}/{len(items)}", flush=True)
    finally:
        if fh:
            fh.close()
    return out


def _sample_mix(idx: np.ndarray, cond: np.ndarray, mix: dict, n: int, rng) -> np.ndarray:
    picks = []
    for c, w in mix.items():
        pool = idx[cond[idx] == c]
        m = min(len(pool), int(round(w * n)))
        picks.append(rng.choice(pool, size=m, replace=False))
    return np.concatenate(picks)


def _evaluate(T: Table, tr, ca, te, alpha, delta) -> dict[str, dict]:
    scorer = LogisticScorer().fit(T.X[tr][T.answered[tr]], (~T.wrong[tr][T.answered[tr]]).astype(float))
    ver = np.where(T.answered, scorer.predict(T.X), UNANSWERED)

    # "Insufficient" answers always abstain, so thresholds are calibrated on answered items only.
    ca = ca[T.answered[ca]]
    tra = tr[T.answered[tr]]
    # LTT testing order comes from the training split, never from calibration data.
    g_cons, g_ver = make_grid(T.cons[tra]), make_grid(ver[tra])
    g_group = {g: make_grid(ver[tra][T.group[tra] == g]) for g in set(T.group)}
    thr = {
        "always answer": UNANSWERED,
        "self-consistency + naive": naive_threshold(T.cons[ca], T.wrong[ca], alpha),
        "self-consistency + LTT": ltt_threshold(T.cons[ca], T.wrong[ca], alpha, delta, g_cons),
        "verifier + naive": naive_threshold(ver[ca], T.wrong[ca], alpha),
        "verifier + LTT": ltt_threshold(ver[ca], T.wrong[ca], alpha, delta, g_ver),
        "verifier + group-wise LTT": mondrian_ltt(ver[ca], T.wrong[ca], T.group[ca], alpha, delta, g_group),
    }
    res = {}
    for m in METHODS:
        s = T.cons if m.startswith("self-consistency") else ver
        acc = apply_thresholds(s[te], T.group[te], thr[m]) & T.answered[te]
        n_acc = int(acc.sum())
        risk = float(T.wrong[te][acc].mean()) if n_acc else 0.0
        inc = T.cond[te] == "incomplete"
        res[m] = {
            "coverage": n_acc / len(te),
            "selective_risk": risk,
            "violation": float(risk > alpha),
            "answered_unanswerable": float(acc[inc].mean()) if inc.any() else np.nan,
        }
    return res


def run_regimes(T: Table, alpha=0.1, delta=0.1, reps=200, split_year=2018,
                n_train=300, n_cal=500, n_test=800, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    all_idx = np.arange(len(T.items))
    old, new = all_idx[T.year <= split_year], all_idx[T.year > split_year]
    for r in range(reps):
        # exchangeable: one mixed pool, random disjoint split
        pool = _sample_mix(all_idx, T.cond, EXCH_MIX, n_train + n_cal + n_test, rng)
        rng.shuffle(pool)
        tr, ca, te = np.split(pool, [n_train, n_train + n_cal])
        for m, v in _evaluate(T, tr, ca, te, alpha, delta).items():
            rows.append({"regime": "exchangeable", "rep": r, "method": m, **v})
        # shift: old + clean-heavy for train/cal, new + messy for test
        src = _sample_mix(old, T.cond, CAL_MIX, n_train + n_cal, rng)
        rng.shuffle(src)
        tr, ca = src[:n_train], src[n_train:]
        te = _sample_mix(new, T.cond, TEST_MIX, n_test, rng)
        for m, v in _evaluate(T, tr, ca, te, alpha, delta).items():
            rows.append({"regime": "shift", "rep": r, "method": m, **v})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["regime", "method"], sort=False)
    out = g.agg(coverage=("coverage", "mean"), selective_risk=("selective_risk", "mean"),
                violation_rate=("violation", "mean"),
                answered_unanswerable=("answered_unanswerable", "mean")).reset_index()
    order = {m: i for i, m in enumerate(METHODS)}
    return out.sort_values(["regime", "method"], key=lambda s: s.map(order) if s.name == "method" else s)


def condition_report(T: Table) -> pd.DataFrame:
    rows = []
    for c in CONDITIONS:
        mask = T.cond == c
        rows.append({"condition": c, "n": int(mask.sum()),
                     "agent_accuracy": float((~T.wrong[mask]).mean()),
                     "mean_self_consistency": float(T.X[mask, FEATURES.index("consistency")].mean()),
                     "conflict_detected": float((T.group[mask] == "conflict").mean())})
    return pd.DataFrame(rows)


def risk_coverage_plot(T: Table, path: Path, alpha: float, split_year=2018, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = np.arange(len(T.items))
    old, new = idx[T.year <= split_year], idx[T.year > split_year]
    tr = _sample_mix(old, T.cond, CAL_MIX, 400, rng)
    te = _sample_mix(new, T.cond, TEST_MIX, 1000, rng)
    scorer = LogisticScorer().fit(T.X[tr][T.answered[tr]], (~T.wrong[tr][T.answered[tr]]).astype(float))
    ver = np.where(T.answered, scorer.predict(T.X), UNANSWERED)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, s in [("self-consistency", T.cons), ("evidence verifier", ver)]:
        st, w = s[te], T.wrong[te]
        order = np.argsort(-st)
        ok = st[order] > UNANSWERED
        errs = np.cumsum(w[order][ok])
        n = np.arange(1, ok.sum() + 1)
        ax.plot(n / len(te), errs / n, label=name)
    ax.axhline(alpha, ls="--", c="grey", lw=1, label=f"target risk alpha={alpha}")
    ax.set_xlabel("coverage (fraction of questions answered)")
    ax.set_ylabel("selective risk (error rate when answering)")
    ax.set_title("Risk-coverage on shifted test set (newer years, messy evidence)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out: Path, summary: pd.DataFrame, cond: pd.DataFrame, cfg: dict):
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    cond.to_csv(out / "conditions.csv", index=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    def md(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(lines)

    text = [
        "# Results", "",
        f"Config: `{json.dumps(cfg)}`", "",
        "## Agent behaviour by evidence condition", "", md(cond), "",
        f"## Accept/abstain policies (alpha={cfg['alpha']}, delta={cfg['delta']}, {cfg['reps']} random splits)", "",
        "- **coverage**: fraction of test questions answered",
        "- **selective_risk**: error rate among answered questions (target <= alpha)",
        "- **violation_rate**: fraction of splits where selective risk > alpha (LTT promises <= delta when exchangeable)",
        "- **answered_unanswerable**: fraction of questions with no answer in the evidence that were answered anyway", "",
        md(summary), "",
        "![risk-coverage](risk_coverage.png)",
    ]
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
