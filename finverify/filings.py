"""Download real 10-K filings from SEC EDGAR and split them into retrievable passages.

Passages are paragraphs and table rows. A table row on its own ("Total net sales | 394,328 |
365,817") is ambiguous, so each row is prefixed with the context a reasonable chunker would
keep: the table's unit ("in millions") and its year header ("2022 | 2021 | 2020").
"""
from __future__ import annotations

import json
import os
import re
import time
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from .data import DEFAULT_UA

SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


def _get(url: str, path: Path) -> bytes | None:
    import requests

    if path.exists():
        return path.read_bytes()
    r = requests.get(url, headers={"User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_UA)}, timeout=60)
    time.sleep(0.15)
    if r.status_code != 200:
        print(f"  [warn] {url}: HTTP {r.status_code}")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return r.content


def ten_k_index(cik: int, cache: Path) -> dict[str, dict]:
    """period end (reportDate) -> {accession, doc} for every original 10-K this company filed."""
    first = _get(SUBMISSIONS.format(name=f"CIK{cik:010d}.json"), cache / f"sub_{cik}.json")
    if first is None:
        return {}
    d = json.loads(first)
    blocks = [d["filings"]["recent"]]
    for f in d["filings"].get("files", []):
        raw = _get(SUBMISSIONS.format(name=f["name"]), cache / f["name"])
        if raw:
            blocks.append(json.loads(raw))
    out = {}
    for b in blocks:
        for form, rep, acc, doc in zip(b["form"], b["reportDate"], b["accessionNumber"], b["primaryDocument"]):
            if form == "10-K" and rep and doc:
                out.setdefault(rep, {"accession": acc, "doc": doc})
    return out


class _Blocks(HTMLParser):
    """Collect paragraph texts and table rows (cells joined with ' | ')."""
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []   # (kind, text)
        self.buf: list[str] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.skip = 0

    def _flush(self):
        t = " ".join("".join(self.buf).split())
        if t:
            self.blocks.append(("p", t))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "ix:header"):
            self.skip += 1
        elif tag == "tr":
            self._flush()
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag in self.BLOCK and self.row is None:
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style", "ix:header"):
            self.skip = max(0, self.skip - 1)
        elif tag in ("td", "th") and self.row is not None and self.cell is not None:
            c = " ".join("".join(self.cell).replace("​", " ").split())
            if c and not re.search(r"[A-Za-z0-9]", c) and c not in (")", "%", ")%"):
                c = ""  # cells holding only symbols / invisible characters
            if c and c not in ("$", ")", "%", ")%"):
                self.row.append(c)
            elif c in (")", "%", ")%") and self.row:
                self.row[-1] += c
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.blocks.append(("row", " | ".join(self.row)))
            self.row = None
        elif tag in self.BLOCK and self.row is None:
            self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        (self.cell if self.cell is not None else self.buf).append(data)


_YEAR_HDR = re.compile(r"^(?:[^|]*\|\s*)?((?:19|20)\d\d)(?:\s*\|\s*(?:19|20)\d\d)+\s*$")
_UNIT = re.compile(r"in (millions|thousands|billions)", re.I)


def passages_from_html(html: str) -> list[str]:
    html = re.sub(r"<ix:header>.*?</ix:header>", " ", html, flags=re.S | re.I)
    p = _Blocks()
    p.feed(unescape(html) if "&lt;" in html[:2000] else html)
    p._flush()
    out, unit, header = [], None, None
    for kind, text in p.blocks:
        m = _UNIT.search(text)
        if m and len(text) < 400:
            unit = m.group(1).lower()
        if kind == "row":
            years = re.findall(r"(?:19|20)\d\d", text)
            if _YEAR_HDR.match(text) or (len(years) >= 2 and not re.search(r"\d,\d{3}", text)):
                header = " | ".join(years)
                continue
            if not re.search(r"\d", text) or len(text) < 8:
                continue
            ctx = []
            if unit:
                ctx.append(f"in {unit}")
            if header:
                ctx.append(f"columns: {header}")
            out.append((f"[{'; '.join(ctx)}] " if ctx else "") + text)
        else:
            if len(text) >= 60:
                header = None if len(text) > 200 else header
                out.append(text)
    return out


def filing_passages(cik: int, report_date: str, cache: Path) -> list[str] | None:
    idx = ten_k_index(cik, cache)
    meta = idx.get(report_date)
    if not meta:
        return None
    acc = meta["accession"].replace("-", "")
    ppath = cache / "passages" / f"{cik}_{report_date}.json"
    if ppath.exists():
        return json.loads(ppath.read_text(encoding="utf-8"))
    raw = _get(ARCHIVE.format(cik=cik, acc=acc, doc=meta["doc"]), cache / "docs" / f"{cik}_{acc}_{meta['doc']}")
    if raw is None:
        return None
    ps = passages_from_html(raw.decode("utf-8", errors="ignore"))
    ppath.parent.mkdir(parents=True, exist_ok=True)
    ppath.write_text(json.dumps(ps), encoding="utf-8")
    return ps
