"""Table summaries: what actually gets embedded for a table chunk.

LLM-generated when a summary endpoint is live; otherwise a deterministic
rule-based fallback (section path + caption + column headers + a few row
labels), so ingestion never depends on a running model.
"""

from __future__ import annotations

from finrag.chunk.structured import Summarizer
from finrag.util import truncate_tokens

SUMMARY_SYSTEM = (
    "You summarize tables from SEC filings. In ONE short paragraph, state what the table "
    "shows: the metrics, the fiscal years/periods covered, and the units. Mention key row "
    "labels. No commentary, no markdown."
)


def rule_based_summary(table_md: str, caption: str, section_path: str) -> str:
    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    headers = ""
    row_labels: list[str] = []
    if lines:
        headers = ", ".join(c.strip() for c in lines[0].strip("|").split("|") if c.strip())
    for ln in lines[2:12]:  # skip markdown separator row
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and cells[0]:
            row_labels.append(cells[0])
    parts = [f"Table in section: {section_path}."]
    if caption:
        parts.append(f"Caption: {caption}.")
    if headers:
        parts.append(f"Columns: {headers}.")
    if row_labels:
        parts.append(f"Rows include: {', '.join(row_labels[:8])}.")
    return " ".join(parts)


def make_summarizer(llm_chat=None, model: str | None = None) -> Summarizer:
    """llm_chat: callable(system, user, model) -> str, or None for rule-based only."""

    def summarize(table_md: str, caption: str, section_path: str) -> str:
        fallback = rule_based_summary(table_md, caption, section_path)
        if llm_chat is None or model is None:
            return fallback
        user = (
            f"Section: {section_path}\n"
            f"Caption: {caption or '(none)'}\n\n"
            f"{truncate_tokens(table_md, 3000)}"
        )
        try:
            out = llm_chat(SUMMARY_SYSTEM, user, model).strip()
            return out or fallback
        except Exception:
            return fallback

    return summarize
