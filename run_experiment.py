"""Run the full pipeline and write results/summary.md.

    python run_experiment.py                      # real SEC EDGAR data, offline extractive agent
    python run_experiment.py --source synthetic   # no network needed
    python run_experiment.py --agent llm --max-facts 200 --k 5   # OpenAI-compatible LLM (see README)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from finverify.agent import ExtractiveAgent, LLMAgent
from finverify.data import load_edgar_facts, synthetic_facts
from finverify.evidence import build_items
from finverify.experiment import (Table, condition_report, risk_coverage_plot, run_agent,
                                  run_regimes, summarize, write_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["edgar", "synthetic"], default="edgar")
    ap.add_argument("--agent", choices=["extractive", "llm"], default="extractive")
    ap.add_argument("--k", type=int, default=10, help="samples per question")
    ap.add_argument("--alpha", type=float, default=0.10, help="target selective risk")
    ap.add_argument("--delta", type=float, default=0.10, help="allowed failure probability")
    ap.add_argument("--reps", type=int, default=200, help="random calibration/test splits")
    ap.add_argument("--max-facts", type=int, default=0, help="subsample facts (useful for paid LLMs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] loading facts from {a.source}")
    facts = load_edgar_facts() if a.source == "edgar" else synthetic_facts(seed=a.seed)
    if a.max_facts and len(facts) > a.max_facts:
        keep = np.random.default_rng(a.seed).choice(len(facts), a.max_facts, replace=False)
        facts = [facts[i] for i in sorted(keep)]
    print(f"      {len(facts)} facts, {len({f.cik for f in facts})} companies, "
          f"years {min(f.year for f in facts)}-{max(f.year for f in facts)}")

    items = build_items(facts, seed=a.seed)
    print(f"[2/4] {len(items)} questions; running {a.agent} agent with k={a.k}")
    out = Path(a.out)
    if a.agent == "extractive":
        agent, cache, every = ExtractiveAgent(), None, 500
    else:
        agent, every = LLMAgent(), 10
        cache = out / f"answers_{agent.model.replace(':', '_').replace('/', '_')}.jsonl"
        print(f"      model={agent.model} endpoint={agent.base_url} cache={cache}")
    T = Table(items, run_agent(items, agent, a.k, a.seed, log_every=every, cache_path=cache))

    print(f"[3/4] calibrating and evaluating over {a.reps} splits")
    n = len(items)
    sizes = dict(n_train=min(300, n // 8), n_cal=min(500, n // 6), n_test=min(800, n // 4))
    df = run_regimes(T, a.alpha, a.delta, a.reps, seed=a.seed, **sizes)

    print(f"[4/4] writing {out}/")
    cfg = {**vars(a), **sizes, "n_facts": len(facts), "n_items": n}
    summary = summarize(df)
    write_report(out, summary, condition_report(T), cfg)
    df.to_csv(out / "all_splits.csv", index=False)
    risk_coverage_plot(T, out / "risk_coverage.png", a.alpha)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
