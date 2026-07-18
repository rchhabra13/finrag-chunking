"""One client for every model: any OpenAI-compatible endpoint.

LM Studio, llama.cpp server, and Ollama all speak the OpenAI chat-completions
API, as do OpenAI, Gemini, and Anthropic via their compat endpoints — so a
single thin wrapper covers local and cloud alike. Endpoints are declared in
config.yaml; live models are discovered via GET /v1/models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
from openai import OpenAI

from finrag.config import EndpointConfig, LLMConfig


@dataclass
class ChatResult:
    text: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


class LLM:
    def __init__(self, ep: EndpointConfig, cfg: LLMConfig):
        self.ep = ep
        self.cfg = cfg
        self.client = OpenAI(
            base_url=ep.base_url, api_key=ep.resolve_key(), timeout=cfg.timeout_s, max_retries=1
        )

    def chat(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        t0 = time.perf_counter()
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if not self.ep.omit_sampling_params:
            kwargs["temperature"] = self.cfg.temperature if temperature is None else temperature
        resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        return ChatResult(
            text=resp.choices[0].message.content or "",
            latency_s=time.perf_counter() - t0,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

    def simple_chat(self, system: str, user: str, model: str) -> str:
        return self.chat(system, user, model).text


def endpoint_alive(ep: EndpointConfig, timeout: float = 3.0) -> bool:
    try:
        r = httpx.get(f"{ep.base_url.rstrip('/')}/models", timeout=timeout,
                      headers={"Authorization": f"Bearer {ep.resolve_key()}"})
        return r.status_code == 200
    except Exception:
        return False


def list_models(ep: EndpointConfig, timeout: float = 5.0) -> list[str]:
    """Models on a live endpoint: config list if set, else GET /v1/models."""
    if ep.models:
        return ep.models
    try:
        r = httpx.get(f"{ep.base_url.rstrip('/')}/models", timeout=timeout,
                      headers={"Authorization": f"Bearer {ep.resolve_key()}"})
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []


def discover(cfg: LLMConfig) -> dict[str, list[str]]:
    """endpoint name -> live model ids (unreachable endpoints excluded)."""
    out: dict[str, list[str]] = {}
    for ep in cfg.endpoints:
        try:
            if endpoint_alive(ep):
                models = list_models(ep)
                if models:
                    out[ep.name] = models
        except RuntimeError:
            continue  # missing cloud key — skip silently, it's optional
    return out
