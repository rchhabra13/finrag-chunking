"""Hybrid retrieval: dense + BM25 fused with RRF, then parent (small-to-big) expansion."""

from __future__ import annotations

from dataclasses import dataclass

from finrag.chunk.models import Chunk, ChunkSet
from finrag.config import Config, StrategyConfig
from finrag.index.store import DocBM25, get_collection, get_embedder


@dataclass
class ContextBlock:
    source_id: str  # chunk id or parent (section) id
    kind: str  # "text" | "table" | "parent"
    text: str


def rrf_fuse(rankings: list[list[str]], k: int) -> list[str]:
    """Reciprocal rank fusion over id rankings; higher fused score first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class Retriever:
    def __init__(self, cfg: Config, strategy: StrategyConfig):
        self.cfg = cfg
        self.strategy = strategy
        self.index_key = strategy.index_key
        self.collection = get_collection(cfg.index.persist_dir, self.index_key)
        self.embedder = get_embedder(cfg.index.embedding_model)
        self._chunksets: dict[str, ChunkSet] = {}
        self._bm25: dict[str, DocBM25] = {}

    def _chunkset(self, doc_name: str) -> ChunkSet:
        if doc_name not in self._chunksets:
            self._chunksets[doc_name] = ChunkSet.load(
                self.cfg.data.chunks_dir, doc_name, self.index_key
            )
        return self._chunksets[doc_name]

    def _doc_bm25(self, doc_name: str) -> DocBM25:
        if doc_name not in self._bm25:
            self._bm25[doc_name] = DocBM25(self._chunkset(doc_name).chunks)
        return self._bm25[doc_name]

    def retrieve(self, question: str, doc_name: str) -> list[ContextBlock]:
        r = self.cfg.retrieval
        pool = r.top_k * 4

        dense_res = self.collection.query(
            query_embeddings=[self.embedder.query(question)],
            n_results=pool,
            where={"doc_name": doc_name},
        )
        dense_ids: list[str] = dense_res["ids"][0] if dense_res["ids"] else []

        rankings = [dense_ids]
        if self.strategy.hybrid_bm25:
            rankings.append(self._doc_bm25(doc_name).top(question, pool))

        fused = rrf_fuse(rankings, r.rrf_k)[: r.top_k]
        by_id: dict[str, Chunk] = {c.id: c for c in self._chunkset(doc_name).chunks}
        hits = [by_id[cid] for cid in fused if cid in by_id]

        return self._to_context(hits, doc_name)

    def _to_context(self, hits: list[Chunk], doc_name: str) -> list[ContextBlock]:
        from finrag.util import n_tokens  # local import: keep module load light

        blocks: list[ContextBlock] = []
        seen_parents: set[str] = set()
        budget = self.cfg.retrieval.context_max_tokens
        used = 0
        parents = self._chunkset(doc_name).parents

        for c in hits:
            if self.strategy.parent_expansion and c.parent_id and c.parent_id in parents:
                if c.parent_id in seen_parents:
                    continue
                seen_parents.add(c.parent_id)
                block = ContextBlock(source_id=c.parent_id, kind="parent", text=parents[c.parent_id])
                # a table hit still contributes its full table — the parent text
                # may have been truncated past it
                if c.kind == "table" and c.payload_text not in block.text:
                    block = ContextBlock(
                        source_id=c.parent_id,
                        kind="parent",
                        text=block.text + "\n\n" + c.payload_text,
                    )
            else:
                block = ContextBlock(source_id=c.id, kind=c.kind, text=c.payload_text)

            cost = n_tokens(block.text)
            if used + cost > budget and blocks:
                continue  # keep earlier (higher-ranked) blocks; skip what doesn't fit
            blocks.append(block)
            used += cost
        return blocks
