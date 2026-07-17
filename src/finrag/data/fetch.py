"""Fetch the FinanceBench open-source sample and the PDFs for our company subset."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx

from finrag.config import DataConfig


@dataclass
class Question:
    qid: str
    company: str
    doc_name: str
    question: str
    answer: str
    question_type: str
    evidence_texts: list[str]

    @classmethod
    def from_raw(cls, r: dict) -> "Question":
        return cls(
            qid=r["financebench_id"],
            company=r["company"],
            doc_name=r["doc_name"],
            question=r["question"],
            answer=str(r.get("answer", "")),
            question_type=r.get("question_type", "unknown"),
            evidence_texts=[e.get("evidence_text", "") for e in r.get("evidence", [])],
        )


def _download(client: httpx.Client, url: str, dest: Path) -> bool:
    resp = client.get(url, follow_redirects=True)
    if resp.status_code != 200:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return True


def fetch_questions(cfg: DataConfig) -> list[dict]:
    """Download (or reuse) the 150-question open-source JSONL; return raw records."""
    if not cfg.questions_file.exists():
        url = f"{cfg.financebench_raw}/data/financebench_open_source.jsonl"
        with httpx.Client(timeout=60) as client:
            if not _download(client, url, cfg.questions_file):
                raise RuntimeError(f"Failed to download FinanceBench questions from {url}")
    return [json.loads(line) for line in cfg.questions_file.read_text().splitlines() if line.strip()]


def select_subset(cfg: DataConfig, raw: list[dict]) -> list[dict]:
    """Pick companies (configured, or top-N by question count) and cap total questions."""
    companies = cfg.companies
    if not companies:
        counts = Counter(r["company"] for r in raw)
        companies = [c for c, _ in counts.most_common(cfg.num_companies)]
    subset = [r for r in raw if r["company"] in companies]
    return subset[: cfg.max_questions]


def fetch_pdfs(cfg: DataConfig, subset: list[dict]) -> tuple[list[dict], list[str]]:
    """Download every PDF the subset references. Drop questions whose PDF 404s."""
    doc_names = sorted({r["doc_name"] for r in subset})
    ok_docs, missing = set(), []
    with httpx.Client(timeout=120) as client:
        for doc in doc_names:
            dest = cfg.pdf_dir / f"{doc}.pdf"
            if dest.exists():
                ok_docs.add(doc)
                continue
            url = f"{cfg.financebench_raw}/pdfs/{doc}.pdf"
            if _download(client, url, dest):
                ok_docs.add(doc)
            else:
                missing.append(doc)
    kept = [r for r in subset if r["doc_name"] in ok_docs]
    return kept, missing


def run_fetch(cfg: DataConfig) -> dict:
    raw = fetch_questions(cfg)
    subset = select_subset(cfg, raw)
    kept, missing = fetch_pdfs(cfg, subset)
    cfg.subset_file.write_text("\n".join(json.dumps(r) for r in kept))
    return {
        "total_questions": len(raw),
        "subset_questions": len(kept),
        "companies": sorted({r["company"] for r in kept}),
        "docs": sorted({r["doc_name"] for r in kept}),
        "missing_pdfs": missing,
    }


def load_subset(cfg: DataConfig) -> list[Question]:
    return [
        Question.from_raw(json.loads(line))
        for line in cfg.subset_file.read_text().splitlines()
        if line.strip()
    ]
