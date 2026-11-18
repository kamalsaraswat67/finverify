"""Questions whose evidence is real 10-K text, retrieved with BM25.

For each fact we open the 10-K whose fiscal period ends on that date, split it into passages,
and find *gold* passages: ones that mention the concept and state the reported value in some
unit (table cells in millions/thousands, or prose like "$394.3 billion").

Evidence conditions (same names as the templated benchmark):
  clean          best gold passage + top BM25 non-gold passages from the same filing
  noisy          best gold passage + more BM25 passages from this AND the prior-year 10-K
                 (the prior filing reports the same line items for different years: real hard distractors)
  contradictory  clean evidence + one gold passage with the value perturbed by 5-30%, presented
                 as a data-feed excerpt (semi-synthetic: real text, injected disagreement)
  incomplete     top BM25 passages with every gold passage removed

Ground truth is the value as originally reported in that 10-K (XBRL), so it matches the text.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .data import Fact
from .evidence import CONDITIONS, Item, question_text
from .filings import filing_passages

KEYWORDS = {
    "total revenue": ["net sales", "revenue", "revenues", "total net revenue", "net revenue"],
    "net income": ["net income", "net earnings", "net (loss) income", "net income (loss)"],
    "operating income": ["operating income", "income from operations", "operating profit", "operating loss"],
    "research and development expense": ["research and development"],
    "total assets": ["total assets"],
    "cash and cash equivalents": ["cash and cash equivalents"],
}

_TOK = re.compile(r"[a-z0-9]+")
_MONEYISH = re.compile(r"\d{1,3}(?:,\d{3})+|\$\s?\d|\d\s*(?:billion|million)", re.I)


def renderings(v: float) -> list[str]:
    """Strings a filing might use for value v (USD)."""
    a = abs(v)
    out = []
    for div in (1e6, 1e3):
        n = a / div
        if n >= 10:
            out.append(f"{round(n):,}")
    if a >= 1e9:
        out += [f"{a / 1e9:.1f} billion", f"{a / 1e9:.2f} billion"]
    elif a >= 1e6:
        out.append(f"{a / 1e6:.1f} million")
    return out


def is_gold(p: str, f: Fact, value: float) -> bool:
    low = p.lower()
    if not any(k in low for k in KEYWORDS[f.concept]):
        return False
    return any(re.search(r"(?<![\d,.])" + re.escape(r) + r"(?![\d,]|\.\d)", p) for r in renderings(value))


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [_TOK.findall(d.lower()) for d in docs]
        self.k1, self.b = k1, b
        self.avg = np.mean([len(t) for t in self.toks]) if self.toks else 1.0
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = _TOK.findall(query.lower())
        out = np.zeros(len(self.toks))
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avg)
            out[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return out


def _perturb(p: str, value: float, rng) -> str:
    """Replace the reported value in passage p by a 5-30% shifted value, in the same format."""
    shift = rng.uniform(0.05, 0.30) * rng.choice([-1, 1])
    new = value * (1 + shift)
    for r_old, r_new in zip(renderings(value), renderings(new)):
        if r_old in p:
            return p.replace(r_old, r_new, 1)
    return p


def build_real_items(facts: list[Fact], cache: str | Path = "data/raw/filings", seed: int = 0,
                     max_passage_chars: int = 600, verbose: bool = True) -> list[Item]:
    cache = Path(cache)
    rng = np.random.default_rng(seed)
    items, skipped = [], Counter()
    for n, f in enumerate(facts):
        ps = filing_passages(f.cik, f.period_end, cache)
        if not ps:
            skipped["no 10-K found"] += 1
            continue
        ps = [p[:max_passage_chars] for p in dict.fromkeys(ps)]   # dedupe, cap length
        truth = f.original_value
        gold_idx = [i for i, p in enumerate(ps) if is_gold(p, f, truth)]
        if not gold_idx:
            skipped["value not found in text"] += 1
            continue
        fq = replace(f, value=truth)
        q = question_text(fq)
        query = q + " " + " ".join(KEYWORDS[f.concept])
        sc = BM25(ps).scores(query)
        order = list(np.argsort(-sc))
        gold_best = max(gold_idx, key=lambda i: sc[i])
        gold_set = set(gold_idx)
        # distractors must carry dollar figures: other line items, other years, other segments
        non_gold = [i for i in order if i not in gold_set and _MONEYISH.search(ps[i])]
        prior = []
        prev_end = date(f.year - 1, int(f.period_end[5:7]), min(int(f.period_end[8:10]), 28))
        prev_ps = None
        for d in range(-7, 8):  # 52/53-week fiscal years end on slightly different dates
            cand = str(prev_end + timedelta(days=d + max(0, int(f.period_end[8:10]) - 28)))
            prev_ps = filing_passages(f.cik, cand, cache) if cand in _index_dates(f.cik, cache) else None
            if prev_ps:
                break
        if prev_ps:
            prev_ps = [p[:max_passage_chars] for p in dict.fromkeys(prev_ps) if _MONEYISH.search(p)]
            if prev_ps:
                psc = BM25(prev_ps).scores(query)
                prior = [prev_ps[i] for i in np.argsort(-psc)[:4]]

        for cond in CONDITIONS:
            meta = {"source": "10k", "gold": ps[gold_best]}
            if cond == "clean":
                ev = [ps[gold_best]] + [ps[i] for i in non_gold[:3]]
            elif cond == "noisy":
                ev = [ps[gold_best]] + [ps[i] for i in non_gold[:5]] + prior
            elif cond == "contradictory":
                fake = "Data feed excerpt: " + _perturb(ps[gold_best], truth, rng)
                ev = [ps[gold_best], fake] + [ps[i] for i in non_gold[:3]]
            else:
                ev = [ps[i] for i in non_gold[:5]]
            ev = [ev[i] for i in rng.permutation(len(ev))]
            items.append(Item(uid=f"10k-{f.cik}-{f.concept}-{f.period_end}-{cond}", fact=fq, condition=cond,
                              question=q, passages=ev, answerable=cond != "incomplete", meta=meta))
        if verbose and (n + 1) % 25 == 0:
            print(f"  built {n + 1}/{len(facts)} facts ({len(items)} questions)", flush=True)
    if verbose:
        print(f"  skipped: {dict(skipped)}")
    return items


_INDEX_CACHE: dict[int, set] = {}


def _index_dates(cik: int, cache: Path) -> set:
    from .filings import ten_k_index

    if cik not in _INDEX_CACHE:
        _INDEX_CACHE[cik] = set(ten_k_index(cik, cache))
    return _INDEX_CACHE[cik]
