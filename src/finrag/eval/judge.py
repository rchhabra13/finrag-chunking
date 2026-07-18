"""LLM-judge: grade stored answers against FinanceBench gold answers.

Separate pass over results/answers/*.jsonl → results/judged/*.jsonl. Verdicts:
correct / incorrect / refusal (+ error for failed generations). Judge model is
configurable — re-run with a stronger model later; already-judged records skip.
"""

from __future__ import annotations

import json
import re

from finrag.config import Config
from finrag.llm.client import LLM, discover

JUDGE_SYSTEM = (
    "You grade answers to financial questions against a gold answer. Verdicts:\n"
    "- correct: materially matches the gold answer. Numbers must match after rounding and "
    "unit normalization ($1,577 million == $1.577 billion == ~$1.6B). Extra correct detail is fine.\n"
    "- incorrect: contradicts the gold answer, wrong figure, wrong period, or answers a "
    "different question.\n"
    "- refusal: declines to answer / says there is insufficient evidence.\n"
    'Respond with ONLY JSON: {"verdict": "correct|incorrect|refusal", "reason": "<one sentence>"}'
)

VERDICTS = ("correct", "incorrect", "refusal")


def parse_verdict(raw: str) -> tuple[str, str]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).lower().strip()
            if v in VERDICTS:
                return v, str(obj.get("reason", ""))
        except json.JSONDecodeError:
            pass
    low = raw.lower()
    if "insufficient evidence" in low:  # judges sometimes invent this as a verdict
        return "refusal", raw.strip()[:200]
    for v in VERDICTS:  # fallback: first verdict word present ("correct" must not
        if re.search(rf"\b{v}\b", low):  # match inside "incorrect", hence \b)
            return v, raw.strip()[:200]
    return "incorrect", f"unparseable judge output: {raw[:150]}"


def judge_prompt(question: str, gold: str, answer: str) -> str:
    return f"Question: {question}\n\nGold answer: {gold}\n\nCandidate answer: {answer}"


def run_judge(cfg: Config, model_override: str | None = None, log=print) -> None:
    ep_name = cfg.judge.endpoint
    model = model_override or cfg.judge.model
    if model is None:
        live = discover(cfg.llm)
        if ep_name not in live or not live[ep_name]:
            raise RuntimeError(f"Judge endpoint '{ep_name}' not reachable and no model configured")
        model = live[ep_name][0]
    llm = LLM(cfg.llm.endpoint(ep_name), cfg.llm)
    log(f"judge: {ep_name}/{model}")

    answers_dir = cfg.eval.results_dir / "answers"
    judged_dir = cfg.eval.results_dir / "judged"
    judged_dir.mkdir(parents=True, exist_ok=True)

    for apath in sorted(answers_dir.glob("*.jsonl")):
        jpath = judged_dir / apath.name
        done = (
            {json.loads(ln)["qid"] for ln in jpath.read_text().splitlines() if ln.strip()}
            if jpath.exists()
            else set()
        )
        records = [json.loads(ln) for ln in apath.read_text().splitlines() if ln.strip()]
        todo = [r for r in records if r["qid"] not in done]
        if not todo:
            continue
        log(f"{apath.name}: judging {len(todo)}")
        with jpath.open("a") as f:
            for r in todo:
                if r.get("error") or not r.get("answer"):
                    verdict, reason = "error", r.get("error", "empty answer")
                else:
                    raw = llm.simple_chat(
                        JUDGE_SYSTEM, judge_prompt(r["question"], r["gold"], r["answer"]), model
                    )
                    verdict, reason = parse_verdict(raw)
                out = {
                    k: r.get(k)
                    for k in (
                        "qid", "company", "doc_name", "question_type",
                        "strategy", "endpoint", "model", "answer", "gold",
                    )
                }
                out.update(verdict=verdict, judge_reason=reason,
                           judge_model=f"{ep_name}/{model}")
                f.write(json.dumps(out) + "\n")
                f.flush()


def load_judged(cfg: Config) -> list[dict]:
    judged_dir = cfg.eval.results_dir / "judged"
    records: list[dict] = []
    for p in sorted(judged_dir.glob("*.jsonl")):
        records.extend(json.loads(ln) for ln in p.read_text().splitlines() if ln.strip())
    return records
