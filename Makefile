.PHONY: install test lint fetch ingest models eval judge report bench

install:
	uv sync --group dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

fetch:
	uv run finrag fetch

ingest:
	uv run finrag ingest

models:
	uv run finrag models

eval:
	uv run finrag eval --ablations

judge:
	uv run finrag judge

report:
	uv run finrag report

# full benchmark: fetch -> ingest -> eval -> judge -> report
bench: fetch ingest eval judge report
