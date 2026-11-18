"""Ground-truth financial facts from SEC EDGAR XBRL "companyfacts".

Each fact is one audited number from a 10-K: (company, concept, period end, value).
Because the numbers come from XBRL, every question we generate has an exact,
machine-checkable answer -- no human labelling needed.

An offline synthetic generator with the same schema is provided for tests and
for running without network access.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import numpy as np

SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# SEC asks automated clients to identify themselves. Set SEC_USER_AGENT to
# "<your name or project> <your email>" before fetching real data.
DEFAULT_UA = "finverify-research admin@example.org"

# CIK -> display name. Large, long-history US filers.
COMPANIES = {
    320193: "Apple Inc.",
    789019: "Microsoft Corporation",
    1018724: "Amazon.com, Inc.",
    1652044: "Alphabet Inc.",
    1326801: "Meta Platforms, Inc.",
    1045810: "NVIDIA Corporation",
    19617: "JPMorgan Chase & Co.",
    34088: "Exxon Mobil Corporation",
    104169: "Walmart Inc.",
    200406: "Johnson & Johnson",
    80424: "The Procter & Gamble Company",
    21344: "The Coca-Cola Company",
    78003: "Pfizer Inc.",
    50863: "Intel Corporation",
    858877: "Cisco Systems, Inc.",
    1341439: "Oracle Corporation",
    1065280: "Netflix, Inc.",
    320187: "NIKE, Inc.",
    1318605: "Tesla, Inc.",
    70858: "Bank of America Corporation",
}

# (label used in questions/passages, candidate XBRL tags in priority order, kind)
CONCEPTS = [
    ("total revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], "duration"),
    ("net income", ["NetIncomeLoss"], "duration"),
    ("operating income", ["OperatingIncomeLoss"], "duration"),
    ("research and development expense", ["ResearchAndDevelopmentExpense"], "duration"),
    ("total assets", ["Assets"], "instant"),
    ("cash and cash equivalents", ["CashAndCashEquivalentsAtCarryingValue"], "instant"),
]


@dataclass(frozen=True)
class Fact:
    company: str
    cik: int
    concept: str          # human label, e.g. "net income"
    kind: str             # "duration" (fiscal-year flow) or "instant" (balance-sheet stock)
    period_end: str       # ISO date
    value: float          # truth = most recently filed value for this period
    original_value: float # first value ever reported (differs if restated)
    filed: str            # filing date of the truth value

    @property
    def year(self) -> int:
        return int(self.period_end[:4])

    def to_dict(self) -> dict:
        return asdict(self)


def _fetch_companyfacts(cik: int, cache_dir: Path, user_agent: str) -> dict | None:
    import requests

    path = cache_dir / f"CIK{cik:010d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    resp = requests.get(SEC_URL.format(cik=cik), headers={"User-Agent": user_agent}, timeout=30)
    time.sleep(0.15)  # stay well under SEC's 10 requests/second limit
    if resp.status_code != 200:
        print(f"  [warn] CIK {cik}: HTTP {resp.status_code}")
        return None
    path.write_text(resp.text, encoding="utf-8")
    return resp.json()


def _annual_facts(doc: dict, company: str, cik: int) -> list[Fact]:
    gaap = doc.get("facts", {}).get("us-gaap", {})
    out: list[Fact] = []
    for label, tags, kind in CONCEPTS:
        by_end: dict[str, list[dict]] = {}
        for tag in tags:
            for row in gaap.get(tag, {}).get("units", {}).get("USD", []):
                if row.get("form") not in ("10-K", "10-K/A"):
                    continue
                end = row.get("end")
                if not end:
                    continue
                if kind == "duration":
                    start = row.get("start")
                    if not start:
                        continue
                    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                    if not 350 <= days <= 380:
                        continue
                if end in by_end and by_end[end][0]["_tag"] != tag:
                    continue  # keep the highest-priority tag for this period
                by_end.setdefault(end, []).append({**row, "_tag": tag})
        for end, rows in by_end.items():
            rows.sort(key=lambda r: r["filed"])
            out.append(Fact(company, cik, label, kind, end,
                            float(rows[-1]["val"]), float(rows[0]["val"]), rows[-1]["filed"]))
    return out


def load_edgar_facts(cache_dir: str | Path = "data/raw", min_year: int = 2010,
                     companies: dict[int, str] | None = None) -> list[Fact]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ua = os.environ.get("SEC_USER_AGENT", DEFAULT_UA)
    facts: list[Fact] = []
    for cik, name in (companies or COMPANIES).items():
        doc = _fetch_companyfacts(cik, cache, ua)
        if doc is None:
            continue
        facts.extend(f for f in _annual_facts(doc, name, cik) if f.year >= min_year)
    # Drop zero values (uninformative) and deduplicate.
    uniq = {(f.cik, f.concept, f.period_end): f for f in facts if f.value != 0}
    return sorted(uniq.values(), key=lambda f: (f.company, f.concept, f.period_end))


def synthetic_facts(n_companies: int = 12, years=range(2010, 2025), seed: int = 0) -> list[Fact]:
    """Offline stand-in with realistic magnitudes and growth dynamics."""
    rng = np.random.default_rng(seed)
    facts = []
    for c in range(n_companies):
        name = f"Synthetic Corp {chr(65 + c)}"
        base_rev = 10 ** rng.uniform(9.3, 11.5)
        growth = rng.normal(0.06, 0.08)
        for y in years:
            rev = base_rev * (1 + growth) ** (y - years[0]) * np.exp(rng.normal(0, 0.05))
            vals = {
                "total revenue": rev,
                "net income": rev * rng.normal(0.12, 0.08),
                "operating income": rev * rng.normal(0.18, 0.07),
                "research and development expense": rev * abs(rng.normal(0.08, 0.04)),
                "total assets": rev * rng.uniform(1.0, 3.0),
                "cash and cash equivalents": rev * rng.uniform(0.05, 0.4),
            }
            end = f"{y}-12-31"
            for label, _, kind in CONCEPTS:
                v = float(round(vals[label], -5))
                restated = v * (1 + rng.normal(0, 0.03)) if rng.random() < 0.05 else v
                facts.append(Fact(name, 9_000_000 + c, label, kind, end, v, float(round(restated, -5)), f"{y + 1}-02-15"))
    return facts
