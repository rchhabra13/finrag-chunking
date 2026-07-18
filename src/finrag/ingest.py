"""Ingestion orchestrator: parse PDFs → chunk per strategy → build indexes.

Table summaries are cached per document (keyed by table content hash) so
ablation variants that share chunk content never re-call the summary model.
"""

from __future__ import annotations

import hashlib
import json

from finrag.chunk.models import ChunkSet
from finrag.chunk.naive import chunk_naive
from finrag.chunk.structured import chunk_structured
from finrag.chunk.table_summary import make_summarizer
from finrag.config import Config
from finrag.data.fetch import load_subset
from finrag.index.store import get_embedder, index_chunkset
from finrag.llm.client import LLM, discover
from finrag.parse.docling_parser import parse_or_load


def _cached_summarizer(cfg: Config, doc_name: str, log=print):
    """Summarizer with a per-doc JSON cache; LLM if the summary endpoint is live."""
    cache_path = cfg.data.chunks_dir / doc_name / "table_summaries.json"
    cache: dict[str, str] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    live = discover(cfg.llm)
    llm_chat = None
    model = cfg.summaries.model
    if cfg.summaries.endpoint in live:
        ep_models = live[cfg.summaries.endpoint]
        model = model or (ep_models[0] if ep_models else None)
        if model:
            llm = LLM(cfg.llm.endpoint(cfg.summaries.endpoint), cfg.llm)
            llm_chat = llm.simple_chat
    if llm_chat is None:
        log(f"  (no live summary endpoint — rule-based table summaries for {doc_name})")
    base = make_summarizer(llm_chat, model)

    def summarize(table_md: str, caption: str, path: str) -> str:
        key = hashlib.sha256(table_md.encode()).hexdigest()[:16]
        if key not in cache:
            cache[key] = base(table_md, caption, path)
        return cache[key]

    def save() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))

    return summarize, save


def run_ingest(
    cfg: Config,
    strategy_names: list[str] | None = None,
    force: bool = False,
    log=print,
) -> dict:
    strategies = {
        name: cfg.strategies[name] for name in (strategy_names or list(cfg.strategies))
    }
    # Multiple strategies can share chunk content — dedupe by index_key.
    by_key = {}
    for s in strategies.values():
        by_key.setdefault(s.index_key, s)

    questions = load_subset(cfg.data)
    docs = sorted({(q.doc_name, q.company) for q in questions})
    embedder = get_embedder(cfg.index.embedding_model)
    stats: dict[str, dict[str, int]] = {}

    for doc_name, company in docs:
        pdf = cfg.data.pdf_dir / f"{doc_name}.pdf"
        log(f"parsing {doc_name} ...")
        tree = parse_or_load(pdf, cfg.data.parsed_dir, doc_name, company)
        paras, tables = tree.n_blocks()
        log(f"  {paras} paragraphs, {tables} tables")

        summarize, save_cache = _cached_summarizer(cfg, doc_name, log)
        for index_key, s in by_key.items():
            if ChunkSet.exists(cfg.data.chunks_dir, doc_name, index_key) and not force:
                cs = ChunkSet.load(cfg.data.chunks_dir, doc_name, index_key)
            else:
                if s.chunker == "naive":
                    cs = chunk_naive(tree, cfg.chunking.naive)
                else:
                    cs = chunk_structured(
                        tree,
                        cfg.chunking.structured,
                        atomic_tables=s.atomic_tables,
                        ancestry_headers=s.ancestry_headers,
                        summarize_table=summarize,
                        index_key=index_key,
                    )
                cs.save(cfg.data.chunks_dir)
            n = index_chunkset(cfg.index.persist_dir, embedder, cs)
            stats.setdefault(doc_name, {})[index_key] = n
            log(f"  indexed {n:4d} chunks [{index_key}]")
        save_cache()

    return stats
