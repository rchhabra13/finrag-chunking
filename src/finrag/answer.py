"""Prompt assembly + answer generation over retrieved context."""

from __future__ import annotations

from dataclasses import dataclass

from finrag.llm.client import LLM, ChatResult
from finrag.retrieve import ContextBlock

ANSWER_SYSTEM = (
    "You are a financial analyst answering questions about SEC filings. Answer ONLY from the "
    "provided context blocks. Be concise: give the direct answer with figures, units, and fiscal "
    "period first, then at most two sentences of support. Cite the blocks you used as [N]. "
    "If the context does not contain the answer, reply exactly: Insufficient evidence."
)


@dataclass
class Answer:
    text: str
    context_ids: list[str]
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


def build_user_prompt(question: str, blocks: list[ContextBlock]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] ({b.kind}) {b.text}" for i, b in enumerate(blocks))
    return f"Context blocks:\n\n{ctx}\n\nQuestion: {question}"


def answer_question(llm: LLM, model: str, question: str, blocks: list[ContextBlock]) -> Answer:
    res: ChatResult = llm.chat(ANSWER_SYSTEM, build_user_prompt(question, blocks), model)
    return Answer(
        text=res.text.strip(),
        context_ids=[b.source_id for b in blocks],
        latency_s=res.latency_s,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
    )
