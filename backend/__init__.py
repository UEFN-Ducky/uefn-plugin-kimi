"""Kimi gateway — LLM provider via Moonshot Open Platform (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any


def _fetch_models(api_key: str, **kw: Any) -> Any:
    from .model_fetch import fetch_models

    return fetch_models(api_key, verify=bool(kw.get("verify", False)))


def _fetch_usage(api_key: str, **kw: Any) -> Any:
    from .usage import fetch_usage

    return fetch_usage(api_key, model=str(kw.get("model") or ""))


def register(api) -> None:
    from .kimi_provider import KimiProvider
    from .model_fetch import clear_model_cache

    api.register_llm_provider(
        "kimi",
        factory=lambda api_key, model, **kw: KimiProvider(api_key, model, **kw),
        fetch_models=_fetch_models,
        fetch_usage=_fetch_usage,
        test_key_model="kimi-k3",
        tool_schema="openai",
        clear_model_cache=clear_model_cache,
        shows_thinking_effort=True,
    )
    from . import graph_nodes

    graph_nodes.register_nodes(api)
    api.log("Kimi gateway contribution active (Providers)")
