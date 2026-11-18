"""Answering agents. Every agent returns K sampled answers (float or None = 'insufficient').

ExtractiveAgent  - a lexical retrieve-then-read baseline that runs offline. It samples a
                   passage in proportion to its lexical match with the question, so it is
                   stochastic in exactly the way an LLM at temperature > 0 is.
LLMAgent         - any OpenAI-compatible chat endpoint (OpenAI, Azure OpenAI, Ollama,
                   vLLM, LM Studio ...). Configure with LLM_BASE_URL / LLM_MODEL / LLM_API_KEY.
"""
from __future__ import annotations

import os
import re

import numpy as np

from .evidence import Item, parse_money

_STOP = {"what", "was", "were", "the", "for", "of", "s", "a", "an", "and", "in", "as", "its", "inc", "ended"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


class ExtractiveAgent:
    name = "extractive"

    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def sample(self, item: Item, k: int, rng: np.random.Generator) -> list[float | None]:
        q = _tokens(item.question)
        cands, scores = [], []
        for p in item.passages:
            v = parse_money(p)
            if v is None:
                continue
            cands.append(v)
            scores.append(len(q & _tokens(p)) / max(len(q), 1))
        if not cands:
            return [None] * k
        s = np.array(scores) / self.temperature
        prob = np.exp(s - s.max())
        prob /= prob.sum()
        idx = rng.choice(len(cands), size=k, p=prob)
        return [cands[i] for i in idx]


class LLMAgent:
    name = "llm"
    PROMPT = (
        "Answer the question using ONLY the evidence passages. Sources may disagree or be "
        "irrelevant. Table rows may start with their unit and column years in brackets, e.g. "
        "'[in millions; columns: 2022 | 2021]'. Make sure you use the right company total, "
        "the right year and the right unit. Reply with a single dollar amount such as "
        "'$12.34 billion', or reply "
        "INSUFFICIENT if the evidence does not contain the answer.\n\n"
        "Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:"
    )

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.7):
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.temperature = temperature

    def _ask(self, prompt: str) -> str:
        import requests

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={"model": self.model, "temperature": self.temperature, "max_tokens": 30,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def sample(self, item: Item, k: int, rng: np.random.Generator) -> list[float | None]:
        evidence = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(item.passages))
        prompt = self.PROMPT.format(evidence=evidence, question=item.question)
        out = []
        for _ in range(k):
            text = self._ask(prompt)
            out.append(None if "INSUFFICIENT" in text.upper() else parse_money(text))
        return out
