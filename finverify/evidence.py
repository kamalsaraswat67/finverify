"""Turn facts into QA items with controlled evidence quality.

Evidence conditions (the "distribution" we shift between):
  clean          gold passage + a few hard distractors
  noisy          gold passage (different number format) + many hard distractors
  contradictory  gold passage + 1-2 passages giving a different number for the same item
  incomplete     gold passage removed; only distractors remain (correct behaviour: abstain)

Hard distractors are other facts about the same company: same concept in other
years, or other concepts in the same year. They look relevant to a retriever.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .data import Fact

CONDITIONS = ("clean", "noisy", "contradictory", "incomplete")

_SOURCES_TRUSTED = ["its annual report on Form 10-K", "its audited financial statements"]
_SOURCES_OTHER = ["a third-party analyst note", "a news summary", "an aggregated data feed"]


def period_phrase(f: Fact) -> str:
    d = date.fromisoformat(f.period_end)
    ds = f"{d:%B} {d.day}, {d.year}"
    return f"for the fiscal year ended {ds}" if f.kind == "duration" else f"as of {ds}"


def money(v: float, style: str = "short") -> str:
    sign = "-" if v < 0 else ""
    a = abs(v)
    if style == "full":
        return f"{sign}${a:,.0f}"
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.2f} billion"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.1f} million"
    return f"{sign}${a:,.0f}"


def question_text(f: Fact) -> str:
    verb = "were" if f.concept.endswith(("assets", "equivalents")) else "was"
    return f"What {verb} {f.company}'s {f.concept} {period_phrase(f)}?"


def passage(f: Fact, value: float | None = None, source: str | None = None, style: str = "short") -> str:
    v = f.value if value is None else value
    src = source or _SOURCES_TRUSTED[0]
    return f"In {src}, {f.company} reported {f.concept} of {money(v, style)} {period_phrase(f)}."


@dataclass
class Item:
    uid: str
    fact: Fact
    condition: str
    question: str
    passages: list[str]
    answerable: bool          # is the true value present in the evidence?
    meta: dict = field(default_factory=dict)


def _distractors(f: Fact, pool: list[Fact], rng: np.random.Generator, k: int) -> list[Fact]:
    same_co = [g for g in pool if g.cik == f.cik and g is not f
               and not (g.concept == f.concept and g.period_end == f.period_end)]
    hard = [g for g in same_co if g.concept == f.concept or g.period_end == f.period_end]
    rest = [g for g in same_co if g not in hard]
    rng.shuffle(hard)
    rng.shuffle(rest)
    return (hard + rest)[:k]


def make_item(f: Fact, condition: str, by_company: dict[int, list[Fact]], rng: np.random.Generator) -> Item:
    pool = by_company[f.cik]
    meta: dict = {}
    if condition == "clean":
        ps = [passage(f, source=rng.choice(_SOURCES_TRUSTED))]
        ps += [passage(g) for g in _distractors(f, pool, rng, 3)]
        answerable = True
    elif condition == "noisy":
        ps = [passage(f, source=rng.choice(_SOURCES_TRUSTED), style=rng.choice(["short", "full"]))]
        ps += [passage(g, style=rng.choice(["short", "full"])) for g in _distractors(f, pool, rng, 8)]
        answerable = True
    elif condition == "contradictory":
        n_wrong = 2 if rng.random() < 0.35 else 1          # sometimes the majority is wrong
        ps = [passage(f, source=rng.choice(_SOURCES_TRUSTED))]
        for _ in range(n_wrong):
            shift = rng.uniform(0.05, 0.30) * rng.choice([-1, 1])
            ps.append(passage(f, value=f.value * (1 + shift), source=rng.choice(_SOURCES_OTHER)))
        ps += [passage(g) for g in _distractors(f, pool, rng, 3)]
        meta["n_contradicting"] = n_wrong
        answerable = True
    elif condition == "incomplete":
        ps = [passage(g) for g in _distractors(f, pool, rng, 4)]
        answerable = False
    else:
        raise ValueError(condition)
    order = rng.permutation(len(ps))
    return Item(uid=f"{f.cik}-{f.concept}-{f.period_end}-{condition}", fact=f, condition=condition,
                question=question_text(f), passages=[ps[i] for i in order], answerable=answerable, meta=meta)


def build_items(facts: list[Fact], seed: int = 0, conditions=CONDITIONS) -> list[Item]:
    rng = np.random.default_rng(seed)
    by_company: dict[int, list[Fact]] = {}
    for f in facts:
        by_company.setdefault(f.cik, []).append(f)
    return [make_item(f, c, by_company, rng) for f in facts for c in conditions]


# ---------------------------------------------------------------- number parsing
_NUM = re.compile(r"(-?)\$\s?([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|[bmk])?\b", re.I)
_SCALE = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


_NUM_NODOLLAR = re.compile(r"(-?)()([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)\b", re.I)


def parse_money(text: str) -> float | None:
    """First dollar amount in `text`, e.g. '$394.33 billion' -> 3.9433e11.

    Falls back to '<number> billion|million|thousand' without a '$' (LLMs often drop it),
    but never parses a bare number, which could be a year.
    """
    m = _NUM.search(text)
    if not m:
        m2 = _NUM_NODOLLAR.search(text)
        if not m2:
            return None
        v = float(m2.group(3).replace(",", "")) * _SCALE[m2.group(4).lower()]
        return -v if m2.group(1) else v
    v = float(m.group(2).replace(",", "")) * _SCALE.get((m.group(3) or "").lower(), 1.0)
    return -v if m.group(1) else v


def is_correct(pred: float | None, truth: float, rel_tol: float = 0.01) -> bool:
    if pred is None:
        return False
    return abs(pred - truth) <= rel_tol * max(abs(truth), 1.0)
