"""finrag CLI: fetch → ingest → (models/ask) → eval → judge → report."""

from __future__ import annotations

import json

import typer

from finrag.config import load_config

app = typer.Typer(help="Structure-aware RAG benchmark on FinanceBench", no_args_is_help=True)

CONFIG_OPT = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml")


@app.command()
def fetch(config: str = CONFIG_OPT):
    """Download FinanceBench questions + the PDF subset."""
    from finrag.data.fetch import run_fetch

    cfg = load_config(config)
    info = run_fetch(cfg.data)
    typer.echo(json.dumps(info, indent=2))


@app.command()
def ingest(
    config: str = CONFIG_OPT,
    strategy: list[str] = typer.Option(None, "--strategy", "-s", help="Subset of strategies"),
    force: bool = typer.Option(False, help="Re-chunk and re-index even if cached"),
):
    """Parse PDFs, chunk under every strategy, build indexes."""
    from finrag.ingest import run_ingest

    cfg = load_config(config)
    run_ingest(cfg, strategy or None, force=force)


@app.command()
def models(config: str = CONFIG_OPT):
    """List live OpenAI-compatible endpoints and their models."""
    from finrag.llm.client import discover

    cfg = load_config(config)
    live = discover(cfg.llm)
    if not live:
        typer.echo("No live endpoints. Start LM Studio / Ollama / llama.cpp, or add cloud keys.")
        raise typer.Exit(1)
    for ep, ms in live.items():
        typer.echo(f"{ep}:")
        for m in ms:
            typer.echo(f"  {m}")


@app.command()
def ask(
    question: str,
    doc: str = typer.Option(..., "--doc", help="doc_name, e.g. AMCOR_2023_10K"),
    strategy: str = typer.Option("structured", "--strategy", "-s"),
    endpoint: str = typer.Option(None, "--endpoint", "-e"),
    model: str = typer.Option(None, "--model", "-m"),
    config: str = CONFIG_OPT,
    show_context: bool = typer.Option(False, "--show-context"),
):
    """Ask one question against one filing."""
    from finrag.answer import answer_question
    from finrag.llm.client import LLM, discover
    from finrag.retrieve import Retriever

    cfg = load_config(config)
    live = discover(cfg.llm)
    if not live:
        typer.echo("No live endpoints.")
        raise typer.Exit(1)
    ep_name = endpoint or next(iter(live))
    mdl = model or live[ep_name][0]

    retriever = Retriever(cfg, cfg.strategies[strategy])
    blocks = retriever.retrieve(question, doc)
    if show_context:
        for i, b in enumerate(blocks, 1):
            typer.echo(f"--- [{i}] ({b.kind}) {b.source_id}\n{b.text[:500]}\n")
    ans = answer_question(LLM(cfg.llm.endpoint(ep_name), cfg.llm), mdl, question, blocks)
    typer.echo(f"\n[{ep_name}/{mdl} | {strategy} | {ans.latency_s:.1f}s]\n{ans.text}")


@app.command()
def eval(
    config: str = CONFIG_OPT,
    strategy: list[str] = typer.Option(None, "--strategy", "-s",
                                       help="Default: naive + structured"),
    model: list[str] = typer.Option(None, "--model", "-m",
                                    help="endpoint/model; default: every live model"),
    limit: int = typer.Option(None, "--limit", "-n"),
    ablations: bool = typer.Option(False, "--ablations", help="Include ablation strategies"),
):
    """Run the benchmark matrix; answers land in results/answers/."""
    from finrag.eval.runner import run_eval
    from finrag.llm.client import discover

    cfg = load_config(config)
    strategies = list(strategy) if strategy else ["naive", "structured"]
    if ablations:
        strategies += [s for s in cfg.strategies if s.startswith("ablation_")]

    if model:
        refs = [tuple(m.split("/", 1)) for m in model]
    else:
        live = discover(cfg.llm)
        refs = [
            (ep, m)
            for ep, ms in live.items()
            for m in ms
            if "embed" not in m.lower()  # embedding models can't chat
        ]
    if not refs:
        typer.echo("No models to run. Start a local LLM server or pass --model endpoint/model.")
        raise typer.Exit(1)
    typer.echo(f"strategies: {strategies}")
    typer.echo(f"models: {[f'{e}/{m}' for e, m in refs]}")
    run_eval(cfg, strategies, refs, limit=limit)


@app.command()
def judge(
    config: str = CONFIG_OPT,
    model: str = typer.Option(None, "--model", "-m", help="Override judge model id"),
):
    """Grade stored answers (separate pass; re-runnable with a stronger judge)."""
    from finrag.eval.judge import run_judge

    cfg = load_config(config)
    run_judge(cfg, model_override=model)


@app.command()
def report(config: str = CONFIG_OPT):
    """Aggregate judged results into results.json / results.md / charts."""
    from finrag.eval.report import run_report

    cfg = load_config(config)
    agg = run_report(cfg)
    for r in agg["runs"]:
        typer.echo(f"{r['strategy']:32s} {r['endpoint']}/{r['model']:30s} {r['accuracy']:.1%}")


if __name__ == "__main__":
    app()
