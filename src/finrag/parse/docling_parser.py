"""PDF → DocumentTree via docling.

Docling gives us a reading-order stream of typed items (section headers with
levels, text, tables). We rebuild a hierarchy from header levels, with one
finance-specific twist: any header matching "Item N." (10-K/10-Q top-level
items) is force-promoted to level 1, because heading levels in filing PDFs are
notoriously inconsistent.
"""

from __future__ import annotations

import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel, TableItem, TextItem

from finrag.parse.tree import Block, DocumentTree, Section

ITEM_RE = re.compile(r"^item\s+\d{1,2}[a-c]?\b", re.IGNORECASE)

_converter: DocumentConverter | None = None


def get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def parse_pdf(pdf_path: Path, doc_name: str, company: str) -> DocumentTree:
    result = get_converter().convert(pdf_path)
    doc = result.document

    tree = DocumentTree(doc_name=doc_name, company=company)
    # Preamble section catches content before the first header.
    stack: list[Section] = [Section(id="s0", title="(document start)", level=0)]
    tree.sections.append(stack[0])
    sec_n = block_n = 0
    last_para = ""  # candidate table caption

    for item, _level in doc.iterate_items():
        if isinstance(item, TextItem) and item.label == DocItemLabel.SECTION_HEADER:
            title = item.text.strip()
            if not title:
                continue
            sec_n += 1
            level = 1 if ITEM_RE.match(title) else min(getattr(item, "level", 1) + 1, 6)
            section = Section(id=f"s{sec_n}", title=title, level=level)
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                stack[-1].children.append(section)
            else:
                tree.sections.append(section)
            stack.append(section)
            last_para = ""
        elif isinstance(item, TableItem):
            md = item.export_to_markdown(doc)
            if not md.strip():
                continue
            block_n += 1
            caption = ""
            try:
                caption = (item.caption_text(doc) or "").strip()
            except Exception:
                pass
            if not caption and len(last_para) < 200:
                caption = last_para
            stack[-1].blocks.append(
                Block(id=f"b{block_n}", kind="table", text=md, caption=caption)
            )
        elif isinstance(item, TextItem):
            text = item.text.strip()
            if not text:
                continue
            block_n += 1
            stack[-1].blocks.append(Block(id=f"b{block_n}", kind="paragraph", text=text))
            last_para = text

    return tree


def parse_or_load(pdf_path: Path, parsed_dir: Path, doc_name: str, company: str) -> DocumentTree:
    cache = parsed_dir / f"{doc_name}.json"
    if cache.exists():
        return DocumentTree.from_json(cache)
    tree = parse_pdf(pdf_path, doc_name, company)
    tree.to_json(cache)
    return tree
