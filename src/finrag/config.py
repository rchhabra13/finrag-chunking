"""Load and validate config.yaml into typed settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    financebench_raw: str
    dir: Path = Path("data")
    companies: list[str] = Field(default_factory=list)
    num_companies: int = 5
    max_questions: int = 40

    @property
    def pdf_dir(self) -> Path:
        return self.dir / "pdfs"

    @property
    def parsed_dir(self) -> Path:
        return self.dir / "parsed"

    @property
    def chunks_dir(self) -> Path:
        return self.dir / "chunks"

    @property
    def questions_file(self) -> Path:
        return self.dir / "financebench_open_source.jsonl"

    @property
    def subset_file(self) -> Path:
        return self.dir / "subset_questions.jsonl"


class NaiveChunkConfig(BaseModel):
    chunk_tokens: int = 512
    overlap_tokens: int = 64


class StructuredChunkConfig(BaseModel):
    child_tokens: int = 350
    parent_tokens: int = 2000


class ChunkingConfig(BaseModel):
    naive: NaiveChunkConfig = NaiveChunkConfig()
    structured: StructuredChunkConfig = StructuredChunkConfig()


class StrategyConfig(BaseModel):
    chunker: Literal["naive", "structured"]
    atomic_tables: bool = True
    ancestry_headers: bool = True
    parent_expansion: bool = False
    hybrid_bm25: bool = True

    @property
    def index_key(self) -> str:
        """Chunk-content-affecting flags only; strategies sharing a key share an index."""
        if self.chunker == "naive":
            return "naive"
        return f"structured_t{int(self.atomic_tables)}_h{int(self.ancestry_headers)}"


class IndexConfig(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    persist_dir: Path = Path("data/index")


class RetrievalConfig(BaseModel):
    top_k: int = 8
    rrf_k: int = 60
    context_max_tokens: int = 6000


class EndpointConfig(BaseModel):
    name: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    models: list[str] = Field(default_factory=list)

    def resolve_key(self) -> str:
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if not key:
                raise RuntimeError(
                    f"Endpoint '{self.name}' needs env var {self.api_key_env} (see .env.example)"
                )
            return key
        return self.api_key or "none"


class LLMConfig(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 120
    endpoints: list[EndpointConfig] = Field(default_factory=list)

    def endpoint(self, name: str) -> EndpointConfig:
        for ep in self.endpoints:
            if ep.name == name:
                return ep
        raise KeyError(f"No endpoint named '{name}' in config.yaml")


class ModelRef(BaseModel):
    endpoint: str
    model: str | None = None


class EvalConfig(BaseModel):
    results_dir: Path = Path("results")


class Config(BaseModel):
    data: DataConfig
    chunking: ChunkingConfig = ChunkingConfig()
    strategies: dict[str, StrategyConfig]
    index: IndexConfig = IndexConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    summaries: ModelRef
    judge: ModelRef
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def check_strategy_endpoints(self) -> "Config":
        names = {ep.name for ep in self.llm.endpoints}
        for ref in (self.summaries, self.judge):
            if ref.endpoint not in names:
                raise ValueError(f"Unknown endpoint '{ref.endpoint}' referenced in config")
        return self


def load_config(path: str | Path = "config.yaml") -> Config:
    load_dotenv()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
