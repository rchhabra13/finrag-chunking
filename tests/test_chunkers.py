from finrag.chunk.naive import chunk_naive
from finrag.chunk.structured import chunk_structured
from finrag.chunk.table_summary import rule_based_summary
from finrag.config import NaiveChunkConfig, StructuredChunkConfig
from finrag.util import n_tokens
from tests.conftest import TABLE_MD


def summarizer(md, caption, path):
    return rule_based_summary(md, caption, path)


def test_naive_chunk_sizes(tree):
    cfg = NaiveChunkConfig(chunk_tokens=64, overlap_tokens=8)
    cs = chunk_naive(tree, cfg)
    assert len(cs.chunks) > 1
    assert all(n_tokens(c.embed_text) <= 64 for c in cs.chunks)
    assert not cs.parents  # naive has no parent map


def structured(tree, **flags):
    return chunk_structured(
        tree,
        StructuredChunkConfig(child_tokens=80, parent_tokens=400),
        atomic_tables=flags.get("atomic_tables", True),
        ancestry_headers=flags.get("ancestry_headers", True),
        summarize_table=summarizer,
        index_key="structured_t1_h1",
    )


def test_tables_are_atomic(tree):
    cs = structured(tree)
    tables = [c for c in cs.chunks if c.kind == "table"]
    assert len(tables) == 1
    assert TABLE_MD in tables[0].payload_text  # full table intact in payload
    assert "Adjusted EBITDA" in tables[0].embed_text  # summary mentions row label


def test_chunks_never_cross_sections(tree):
    cs = structured(tree)
    # every chunk belongs to exactly one section path, and Item 8 text never
    # shares a chunk with Item 7 content
    for c in cs.chunks:
        assert c.section_path
        if "Item 8" in c.section_path:
            assert "strong results" not in c.payload_text


def test_ancestry_headers_toggle(tree):
    with_h = structured(tree, ancestry_headers=True)
    without_h = structured(tree, ancestry_headers=False)
    assert all(c.embed_text.startswith("[Company: TestCo") for c in with_h.chunks)
    assert not any(c.embed_text.startswith("[Company:") for c in without_h.chunks)
    # payload keeps provenance either way
    assert all(c.payload_text.startswith("[Company: TestCo") for c in without_h.chunks)


def test_atomic_tables_off_merges_table_into_prose(tree):
    cs = structured(tree, atomic_tables=False)
    assert not [c for c in cs.chunks if c.kind == "table"]
    assert any("Net income" in c.embed_text for c in cs.chunks)


def test_parent_map(tree):
    cs = structured(tree)
    for c in cs.chunks:
        assert c.parent_id in cs.parents
    recon_parent = next(
        cs.parents[c.parent_id] for c in cs.chunks if c.kind == "table"
    )
    assert "Non-GAAP Reconciliation" in recon_parent


def test_rule_based_summary_mentions_columns():
    s = rule_based_summary(TABLE_MD, "Adjusted EBITDA", "Item 7 > Non-GAAP")
    assert "FY2023" in s and "Item 7" in s and "Adjusted EBITDA" in s
