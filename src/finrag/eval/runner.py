"""Benchmark runner: (strategy × model) matrix over the question subset.

Answers append to results/answers/*.jsonl and runs resume — re-running skips
(question, strategy, model) triples that already have an answer. Judging is a
separate pass (see judge.py) so answers are never regenerated just to re-grade.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from finrag.answer import answer_question
from finrag.config import Config
from finrag.data.fetch import Question, load_subset
from finrag.llm.client import LLM
from finrag.retrieve import Retriever


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def answers_path(cfg: Config, strategy: str, endpoint: str, model: str) -> Path:
    d = cfg.eval.results_dir / "answers"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{strategy}__{endpoint}__{sanitize(model)}.jsonl"


def _done_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["qid"] for line in path.read_text().splitlines() if line.strip()
    }


def run_eval(
    cfg: Config,
    strategy_names: list[str],
    model_refs: list[tuple[str, str]],  # (endpoint_name, model)
    limit: int | None = None,
    log=print,
) -> None:
    questions: list[Question] = load_subset(cfg.data)
    if limit:
        questions = questions[:limit]

    llms = {ep: LLM(cfg.llm.endpoint(ep), cfg.llm) for ep, _ in model_refs}

    for strategy in strategy_names:
        retriever = Retriever(cfg, cfg.strategies[strategy])
        # Retrieval is model-independent — do it once per question per strategy.
        log(f"[{strategy}] retrieving context for {len(questions)} questions ...")
        blocks_by_qid = {q.qid: retriever.retrieve(q.question, q.doc_name) for q in questions}

        for ep_name, model in model_refs:
            path = answers_path(cfg, strategy, ep_name, model)
            done = _done_qids(path)
            todo = [q for q in questions if q.qid not in done]
            if not todo:
                log(f"[{strategy}] {ep_name}/{model}: all {len(questions)} done")
                continue
            log(f"[{strategy}] {ep_name}/{model}: {len(todo)} questions")
            with path.open("a") as f:
                for i, q in enumerate(todo, 1):
                    record = {
                        "qid": q.qid,
                        "company": q.company,
                        "doc_name": q.doc_name,
                        "question_type": q.question_type,
                        "question": q.question,
                        "gold": q.answer,
                        "strategy": strategy,
                        "endpoint": ep_name,
                        "model": model,
                    }
                    try:
                        ans = answer_question(llms[ep_name], model, q.question, blocks_by_qid[q.qid])
                        record.update(asdict(ans))
                        record["answer"] = record.pop("text")
                    except Exception as e:  # keep the run alive; judge marks it error
                        record.update(answer="", error=f"{type(e).__name__}: {e}")
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    if i % 5 == 0 or i == len(todo):
                        log(f"  {i}/{len(todo)}")
