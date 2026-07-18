# Benchmark results

## Naive vs structure-aware chunking (accuracy)

| Model | Naive | Structured | Δ |
|---|---|---|---|
| chat/claude-fable-5 | 40.0% | 35.0% | -5.0% |
| ollama/llama3.2:3b | 12.5% | 17.5% | +5.0% |
| ollama/qwen2.5:3b | 25.0% | 32.5% | +7.5% |

## Ablations (full structured minus one trick)

| Model | structured | ablation_no_atomic_tables | ablation_no_bm25 | ablation_no_headers | ablation_no_parent_expansion |
|---|---|---|---|---|---|
| chat/claude-fable-5 | 35.0% | — | — | — | — |
| ollama/llama3.2:3b | 17.5% | 15.0% | 20.0% | 12.5% | 15.0% |
| ollama/qwen2.5:3b | 32.5% | 37.5% | 20.0% | 25.0% | 22.5% |

## Verdict detail

| Strategy | Model | n | correct | incorrect | refusal | error | accuracy |
|---|---|---|---|---|---|---|---|
| ablation_no_atomic_tables | ollama/llama3.2:3b | 40 | 6 | 26 | 8 | 0 | 15.0% |
| ablation_no_atomic_tables | ollama/qwen2.5:3b | 40 | 15 | 11 | 14 | 0 | 37.5% |
| ablation_no_bm25 | ollama/llama3.2:3b | 40 | 8 | 25 | 7 | 0 | 20.0% |
| ablation_no_bm25 | ollama/qwen2.5:3b | 40 | 8 | 16 | 16 | 0 | 20.0% |
| ablation_no_headers | ollama/llama3.2:3b | 40 | 5 | 23 | 12 | 0 | 12.5% |
| ablation_no_headers | ollama/qwen2.5:3b | 40 | 10 | 15 | 15 | 0 | 25.0% |
| ablation_no_parent_expansion | ollama/llama3.2:3b | 40 | 6 | 30 | 4 | 0 | 15.0% |
| ablation_no_parent_expansion | ollama/qwen2.5:3b | 40 | 9 | 11 | 20 | 0 | 22.5% |
| naive | chat/claude-fable-5 | 40 | 16 | 10 | 14 | 0 | 40.0% |
| naive | ollama/llama3.2:3b | 40 | 5 | 26 | 9 | 0 | 12.5% |
| naive | ollama/qwen2.5:3b | 40 | 10 | 14 | 16 | 0 | 25.0% |
| structured | chat/claude-fable-5 | 40 | 14 | 17 | 9 | 0 | 35.0% |
| structured | ollama/llama3.2:3b | 40 | 7 | 25 | 8 | 0 | 17.5% |
| structured | ollama/qwen2.5:3b | 40 | 13 | 18 | 9 | 0 | 32.5% |
