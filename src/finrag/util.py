"""Shared token utilities (tiktoken cl100k_base — a reasonable proxy for all models)."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


def truncate_tokens(text: str, max_tokens: int) -> str:
    toks = _enc().encode(text, disallowed_special=())
    if len(toks) <= max_tokens:
        return text
    return _enc().decode(toks[:max_tokens])


def window_tokens(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size token windows with overlap — the classic naive splitter."""
    toks = _enc().encode(text, disallowed_special=())
    step = max(size - overlap, 1)
    out = []
    for start in range(0, len(toks), step):
        piece = toks[start : start + size]
        if not piece:
            break
        out.append(_enc().decode(piece))
        if start + size >= len(toks):
            break
    return out
