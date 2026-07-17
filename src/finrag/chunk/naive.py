"""Baseline chunker: flatten the document to one text stream, cut fixed windows.

This is deliberately the tutorial-default pipeline the article argues against —
tables get sliced mid-row, section boundaries are invisible, no metadata.
"""

from __future__ import annotations

from finrag.chunk.models import Chunk, ChunkSet
from finrag.config import NaiveChunkConfig
from finrag.parse.tree import ROOT_TITLE, DocumentTree
from finrag.util import window_tokens


def chunk_naive(tree: DocumentTree, cfg: NaiveChunkConfig) -> ChunkSet:
    parts: list[str] = []
    for sec, _path in tree.walk():
        if sec.title and sec.title != ROOT_TITLE:
            parts.append(sec.title)
        for b in sec.blocks:
            parts.append(b.text)
    full_text = "\n\n".join(parts)

    cs = ChunkSet(doc_name=tree.doc_name, index_key="naive")
    for i, piece in enumerate(window_tokens(full_text, cfg.chunk_tokens, cfg.overlap_tokens)):
        cs.chunks.append(
            Chunk(
                id=f"{tree.doc_name}:naive:{i}",
                doc_name=tree.doc_name,
                kind="text",
                embed_text=piece,
                payload_text=piece,
            )
        )
    return cs
