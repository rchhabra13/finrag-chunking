# Architecture

This repo implements and benchmarks the chunking strategy from the article
["RAG for Financial Docs Is Different"](https://medium.com/@rrchhabra) — four
decisions that all point the same direction: **respect the document's structure
instead of fighting it.**

## Why financial filings break naive chunking

- **Size + repetition.** A 10-K runs 150+ pages and "revenue" appears in dozens
  of sections; a dense retriever has too many plausible-looking chunks.
- **Meaning lives in structure.** `$2,018 million` only means something as
  *Adjusted EBITDA, FY2023, Item 7 reconciliation table*. Fixed windows strip
  that lineage.
- **Tables.** ~75% of FinanceBench questions involve a table; a 512-token
  window happily cuts a balance sheet in half, separating headers from rows.

## Pipeline

```
FinanceBench PDFs ──▶ docling ──▶ document tree ──▶ chunkers ──▶ Chroma + BM25
                                                                     │
answer LLM ◀── context blocks ◀── RRF fusion + parent expansion ◀────┘
     │
LLM judge ──▶ results.json / results.md / charts
```

### 1. Parse structure before chunking anything — `src/finrag/parse/`

`docling_parser.py` converts each PDF with [docling](https://github.com/docling-project/docling)
and rebuilds a hierarchy from section-header levels. Filing-specific twist:
any header matching `Item N` is force-promoted to a top-level section, because
heading levels in filing PDFs are inconsistent. Paragraphs and tables are
separate node types (`tree.py`). Chunks are never allowed to cross a section
boundary.

### 2. Tables are atomic — `src/finrag/chunk/structured.py`

A table is never split. Its **embed text** is a one-paragraph summary
(LLM-generated when a summary endpoint is live, deterministic rule-based
fallback otherwise — `table_summary.py`); its **payload** is the full markdown
table. The summary is what the retriever matches; the table is what the answer
model reads.

### 3. Every chunk carries its ancestry

Before embedding, each chunk gets a contextual header built from the tree:

```
[Company: AMCOR | Filing: AMCOR_2023_10K | Section: Item 7 MD&A > Non-GAAP Reconciliation | Type: table]
```

The fiscal year and section path are literally inside the embedded text, so
FY2021 chunks stop matching FY2023 questions.

### 4. Embed small, retrieve big — `src/finrag/retrieve.py`

Small chunks (~350 tokens) are embedded and searched; when one matches, the
pipeline returns its **parent section** (capped ~2000 tokens) with the table
intact and surrounding prose included. Hybrid search fuses dense results
(bge-small-en-v1.5 in Chroma) with BM25 via reciprocal rank fusion — BM25
catches the exact identifiers embeddings fumble (`FY2023`, `Note 14`, tickers,
dollar figures).

## The benchmark — `src/finrag/eval/`

- **Data:** the [FinanceBench](https://github.com/patronus-ai/financebench)
  open-source sample; default subset = 5 companies with the most questions
  (40 questions, 16 filings). `config.yaml` scales it up.
- **Matrix:** every strategy × every live model. Models are anything behind an
  OpenAI-compatible endpoint (`llm/client.py`) — LM Studio, llama.cpp server,
  Ollama locally; OpenAI/Gemini/Anthropic compat endpoints for cloud.
- **Answers are stored raw** (`results/answers/*.jsonl`) and graded in a
  separate judge pass, so you can re-grade with a stronger judge later without
  re-running generation. Runs resume: existing (question, strategy, model)
  triples are skipped.
- **Verdicts:** correct / incorrect / refusal (+ error), accuracy = correct /
  graded.

### Ablations

`config.yaml` defines full-structured minus one trick each:

| Strategy | What's removed |
|---|---|
| `ablation_no_atomic_tables` | tables split like prose |
| `ablation_no_headers` | no ancestry header in embed text |
| `ablation_no_parent_expansion` | answer sees small chunks only |
| `ablation_no_bm25` | dense-only retrieval |

Chunk-content flags map to separate indexes (`StrategyConfig.index_key`);
retrieval-time flags reuse them.

## Known limitations

- Multi-hop questions (numbers from two filings + arithmetic) need
  decomposition and tool use — out of scope here, phase 2.
- Docling heading detection is imperfect on scanned/exotic layouts; the tree
  degrades to flatter structure rather than failing.
- The rule-based table summary is weaker than an LLM summary for very wide,
  unlabeled tables.
