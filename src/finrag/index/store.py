"""Vector index (Chroma + bge embeddings) and BM25 over chunk sets."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from finrag.chunk.models import Chunk, ChunkSet

# bge-family models want this prefix on queries (not on passages)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def bm25_tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def query(self, text: str) -> list[float]:
        return self.model.encode(
            BGE_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


@lru_cache(maxsize=4)
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)


def get_collection(persist_dir: Path, index_key: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=f"finrag_{index_key}", metadata={"hnsw:space": "cosine"}
    )


def index_chunkset(persist_dir: Path, embedder: Embedder, cs: ChunkSet, batch: int = 64) -> int:
    """(Re)index one document's chunks for one index_key. Idempotent per doc."""
    col = get_collection(persist_dir, cs.index_key)
    col.delete(where={"doc_name": cs.doc_name})
    for i in range(0, len(cs.chunks), batch):
        part = cs.chunks[i : i + batch]
        col.add(
            ids=[c.id for c in part],
            embeddings=embedder.passages([c.embed_text for c in part]),
            documents=[c.embed_text for c in part],
            metadatas=[
                {"doc_name": c.doc_name, "kind": c.kind, "parent_id": c.parent_id or ""}
                for c in part
            ],
        )
    return len(cs.chunks)


class DocBM25:
    """BM25 over one document's chunks (built lazily, cached by the retriever)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([bm25_tokenize(c.embed_text) for c in chunks]) if chunks else None

    def top(self, query: str, k: int) -> list[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(bm25_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i].id for i in ranked if scores[i] > 0]
