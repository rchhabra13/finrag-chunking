"""Document tree: the structural skeleton built before any chunking happens.

A filing becomes a tree of sections (Item 1, Item 7, subsections...) whose
leaves are blocks — paragraphs and tables. Tables are separate node types so
chunkers can treat them atomically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

BlockKind = Literal["paragraph", "table"]

ROOT_TITLE = "(document start)"  # pseudo-section for pre-header content


@dataclass
class Block:
    id: str
    kind: BlockKind
    text: str  # markdown for tables, plain text for paragraphs
    caption: str = ""  # table caption/nearby title if detected


@dataclass
class Section:
    id: str
    title: str
    level: int  # 1 = top-level Item, deeper = subsections
    blocks: list[Block] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)


@dataclass
class DocumentTree:
    doc_name: str  # FinanceBench doc_name, e.g. "3M_2018_10K"
    company: str
    sections: list[Section] = field(default_factory=list)

    def walk(self) -> Iterator[tuple[Section, list[str]]]:
        """Yield every section with its ancestry path of titles (outermost first)."""

        def _walk(secs: list[Section], path: list[str]) -> Iterator[tuple[Section, list[str]]]:
            for sec in secs:
                cur = path + [sec.title]
                yield sec, cur
                yield from _walk(sec.children, cur)

        yield from _walk(self.sections, [])

    def n_blocks(self) -> tuple[int, int]:
        paras = tables = 0
        for sec, _ in self.walk():
            for b in sec.blocks:
                if b.kind == "table":
                    tables += 1
                else:
                    paras += 1
        return paras, tables

    # -- JSON round-trip -----------------------------------------------------

    def to_json(self, path: Path) -> None:
        def sec_dict(s: Section) -> dict:
            return {
                "id": s.id,
                "title": s.title,
                "level": s.level,
                "blocks": [vars(b) for b in s.blocks],
                "children": [sec_dict(c) for c in s.children],
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "doc_name": self.doc_name,
                    "company": self.company,
                    "sections": [sec_dict(s) for s in self.sections],
                },
                indent=1,
            )
        )

    @classmethod
    def from_json(cls, path: Path) -> "DocumentTree":
        raw = json.loads(path.read_text())

        def sec_from(d: dict) -> Section:
            return Section(
                id=d["id"],
                title=d["title"],
                level=d["level"],
                blocks=[Block(**b) for b in d["blocks"]],
                children=[sec_from(c) for c in d["children"]],
            )

        return cls(
            doc_name=raw["doc_name"],
            company=raw["company"],
            sections=[sec_from(s) for s in raw["sections"]],
        )
