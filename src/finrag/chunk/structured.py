"""Structure-aware chunker — the article's four rules in code:

1. chunks never cross a section boundary (we walk the parsed tree);
2. tables are atomic: one table = one chunk, embed a summary, payload is the
   full markdown table;
3. every chunk's embed text carries an ancestry header
   [Company | Filing | Section path | Type];
4. parent-child: small chunks are embedded, their section is the parent
   returned at answer time (expansion happens in retrieval).

`atomic_tables` / `ancestry_headers` are toggleable for the ablation study.
"""

from __future__ import annotations

from typing import Callable

from finrag.chunk.models import Chunk, ChunkSet
from finrag.config import StructuredChunkConfig
from finrag.parse.tree import ROOT_TITLE, DocumentTree
from finrag.util import n_tokens, truncate_tokens, window_tokens

# (table_markdown, caption, section_path) -> one-paragraph summary
Summarizer = Callable[[str, str, str], str]


def _header(company: str, doc_name: str, path: str, kind: str) -> str:
    return f"[Company: {company} | Filing: {doc_name} | Section: {path} | Type: {kind}]"


def chunk_structured(
    tree: DocumentTree,
    cfg: StructuredChunkConfig,
    *,
    atomic_tables: bool,
    ancestry_headers: bool,
    summarize_table: Summarizer,
    index_key: str,
) -> ChunkSet:
    cs = ChunkSet(doc_name=tree.doc_name, index_key=index_key)
    n = 0

    def add(kind: str, embed: str, payload: str, path: str, parent_id: str) -> None:
        nonlocal n
        head = _header(tree.company, tree.doc_name, path, kind)
        cs.chunks.append(
            Chunk(
                id=f"{tree.doc_name}:{index_key}:{n}",
                doc_name=tree.doc_name,
                kind=kind,
                embed_text=f"{head}\n{embed}" if ancestry_headers else embed,
                # payload always carries provenance — that's standard citation
                # hygiene, not the trick under ablation
                payload_text=f"{head}\n{payload}",
                section_path=path,
                parent_id=parent_id,
            )
        )
        n += 1

    for sec, path_list in tree.walk():
        if not sec.blocks:
            continue
        path = " > ".join(t for t in path_list if t != ROOT_TITLE) or "front matter"
        parent_id = f"{tree.doc_name}:{sec.id}"

        # Parent = the whole section as the LLM would want to read it.
        sec_text_parts = [sec.title] + [b.text for b in sec.blocks]
        cs.parents[parent_id] = truncate_tokens("\n\n".join(sec_text_parts), cfg.parent_tokens)

        prose: list[str] = []

        def flush_prose() -> None:
            if not prose:
                return
            joined = "\n\n".join(prose)
            prose.clear()
            for piece in window_tokens(joined, cfg.child_tokens, 0):
                add("text", piece, piece, path, parent_id)

        for b in sec.blocks:
            if b.kind == "table" and atomic_tables:
                flush_prose()
                summary = summarize_table(b.text, b.caption, path)
                # embed = summary + head of the real table, so dense search gets
                # the semantics and BM25 gets exact row labels and figures
                embed = f"{summary}\n{truncate_tokens(b.text, 150)}"
                payload = f"{b.caption}\n{b.text}" if b.caption else b.text
                add("table", embed, payload, path, parent_id)
            else:
                # table treated as plain prose when atomic_tables is ablated off
                prose.append(b.text)
                if n_tokens("\n\n".join(prose)) >= cfg.child_tokens:
                    flush_prose()
        flush_prose()

    return cs
