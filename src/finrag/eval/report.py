"""Aggregate judged results → results.json, results.md, charts/*.png."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finrag.config import Config
from finrag.eval.judge import load_judged

# chart chrome (validated reference palette)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"  # structured (the treatment under test)
GRAY = "#b0aea8"  # naive baseline
# ordinal blue ramp for ablations (step 250→450)
ABLATION_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5"]

MAIN_STRATEGIES = ("naive", "structured")


def aggregate(records: list[dict]) -> dict:
    """Nested accuracy stats: run level and per question_type."""
    runs: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for r in records:
        key = (r["strategy"], r["endpoint"], r["model"])
        runs[key]["n"] += 1
        runs[key][r["verdict"]] += 1
        runs[key][f"qt:{r['question_type']}:n"] += 1
        if r["verdict"] == "correct":
            runs[key][f"qt:{r['question_type']}:correct"] += 1

    out = []
    for (strategy, endpoint, model), c in sorted(runs.items()):
        graded = c["n"] - c.get("error", 0)
        by_qt = {}
        for k in list(c):
            if k.startswith("qt:") and k.endswith(":n"):
                qt = k[3:-2]
                qn = c[f"qt:{qt}:n"]
                by_qt[qt] = {
                    "n": qn,
                    "accuracy": round(c.get(f"qt:{qt}:correct", 0) / qn, 3) if qn else 0.0,
                }
        out.append(
            {
                "strategy": strategy,
                "endpoint": endpoint,
                "model": model,
                "n": c["n"],
                "correct": c.get("correct", 0),
                "incorrect": c.get("incorrect", 0),
                "refusal": c.get("refusal", 0),
                "error": c.get("error", 0),
                "accuracy": round(c.get("correct", 0) / graded, 3) if graded else 0.0,
                "by_question_type": by_qt,
            }
        )
    return {"runs": out}


def _model_label(run: dict) -> str:
    return f"{run['endpoint']}/{run['model']}"


def write_markdown(agg: dict, path: Path) -> None:
    runs = agg["runs"]
    models = sorted({_model_label(r) for r in runs})
    by = {(_model_label(r), r["strategy"]): r for r in runs}

    lines = ["# Benchmark results", ""]
    lines += ["## Naive vs structure-aware chunking (accuracy)", ""]
    lines += ["| Model | Naive | Structured | Δ |", "|---|---|---|---|"]
    for m in models:
        naive = by.get((m, "naive"))
        struct = by.get((m, "structured"))
        if not (naive and struct):
            continue
        delta = struct["accuracy"] - naive["accuracy"]
        lines.append(
            f"| {m} | {naive['accuracy']:.1%} | {struct['accuracy']:.1%} | {delta:+.1%} |"
        )

    abl = sorted({r["strategy"] for r in runs if r["strategy"].startswith("ablation_")})
    if abl:
        lines += ["", "## Ablations (full structured minus one trick)", ""]
        lines += ["| Model | " + " | ".join(["structured"] + abl) + " |",
                  "|---" * (len(abl) + 2) + "|"]
        for m in models:
            cells = []
            for s in ["structured"] + abl:
                r = by.get((m, s))
                cells.append(f"{r['accuracy']:.1%}" if r else "—")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Verdict detail", ""]
    lines += ["| Strategy | Model | n | correct | incorrect | refusal | error | accuracy |",
              "|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["strategy"], _model_label(r))):
        lines.append(
            f"| {r['strategy']} | {_model_label(r)} | {r['n']} | {r['correct']} | "
            f"{r['incorrect']} | {r['refusal']} | {r['error']} | {r['accuracy']:.1%} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_main(agg: dict, path: Path) -> None:
    """Grouped bars: accuracy per model, naive (gray) vs structured (blue)."""
    runs = agg["runs"]
    by = {(_model_label(r), r["strategy"]): r["accuracy"] for r in runs}
    models = sorted(
        {_model_label(r) for r in runs if (_model_label(r), "structured") in by
         and (_model_label(r), "naive") in by}
    )
    if not models:
        return
    x = range(len(models))
    w = 0.32
    fig, ax = plt.subplots(figsize=(max(6, 1.9 * len(models)), 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    for dx, strat, color in ((-w / 2, "naive", GRAY), (w / 2, "structured", BLUE)):
        vals = [by[(m, strat)] for m in models]
        bars = ax.bar([i + dx for i in x], vals, width=w, color=color, label=strat)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.split("/", 1)[1] for m in models], color=INK)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("FinanceBench accuracy — naive vs structure-aware chunking",
                 color=INK, fontsize=11, pad=12)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_ablation(agg: dict, path: Path) -> None:
    """Horizontal bars per model-averaged strategy accuracy, ablations vs full."""
    runs = agg["runs"]
    strategies = ["naive", "structured"] + sorted(
        {r["strategy"] for r in runs if r["strategy"].startswith("ablation_")}
    )
    means = {}
    for s in strategies:
        vals = [r["accuracy"] for r in runs if r["strategy"] == s]
        if vals:
            means[s] = sum(vals) / len(vals)
    if len(means) <= 2:
        return
    order = sorted(means, key=means.__getitem__)
    colors = {}
    ramp = iter(ABLATION_RAMP)
    for s in order:
        colors[s] = GRAY if s == "naive" else BLUE if s == "structured" else next(ramp, ABLATION_RAMP[-1])
    labels = {s: s.replace("ablation_no_", "− ").replace("_", " ") for s in order}

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(order) + 1.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    bars = ax.barh(range(len(order)), [means[s] for s in order], height=0.55,
                   color=[colors[s] for s in order])
    for i, (b, s) in enumerate(zip(bars, order)):
        ax.text(means[s] + 0.01, i, f"{means[s]:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[s] for s in order], color=INK, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Ablation: which chunking trick carries the accuracy?\n(mean across models)",
                 color=INK, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def run_report(cfg: Config, log=print) -> dict:
    records = load_judged(cfg)
    if not records:
        raise RuntimeError("No judged records — run `finrag eval` then `finrag judge` first")
    agg = aggregate(records)
    rd = cfg.eval.results_dir
    (rd / "results.json").write_text(json.dumps(agg, indent=2))
    write_markdown(agg, rd / "results.md")
    chart_main(agg, rd / "charts" / "accuracy_by_model.png")
    chart_ablation(agg, rd / "charts" / "ablation.png")
    log(f"wrote {rd}/results.json, results.md, charts/")
    return agg
