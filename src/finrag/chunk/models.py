"""Chunk data model + on-disk chunk sets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc_name: str
    kind: str  # "text" | "table"
    embed_text: str  # what gets embedded / BM25-indexed (small)
    payload_text: str  # what the answer LLM receives (full table, provenance line)
    section_path: str = ""
    parent_id: str | None = None  # section-level parent for small-to-big expansion


@dataclass
class ChunkSet:
    doc_name: str
    index_key: str
    chunks: list[Chunk] = field(default_factory=list)
    parents: dict[str, str] = field(default_factory=dict)  # parent_id -> section text

    def save(self, chunks_dir: Path) -> None:
        d = chunks_dir / self.doc_name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self.index_key}.chunks.jsonl").write_text(
            "\n".join(json.dumps(asdict(c)) for c in self.chunks)
        )
        (d / f"{self.index_key}.parents.json").write_text(json.dumps(self.parents))

    @classmethod
    def load(cls, chunks_dir: Path, doc_name: str, index_key: str) -> "ChunkSet":
        d = chunks_dir / doc_name
        chunks = [
            Chunk(**json.loads(line))
            for line in (d / f"{index_key}.chunks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        parents = json.loads((d / f"{index_key}.parents.json").read_text())
        return cls(doc_name=doc_name, index_key=index_key, chunks=chunks, parents=parents)

    @classmethod
    def exists(cls, chunks_dir: Path, doc_name: str, index_key: str) -> bool:
        return (chunks_dir / doc_name / f"{index_key}.chunks.jsonl").exists()
