"""Kimi / Moonshot model list fetch for this gateway plugin."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from backend.agent.model_fetch import ModelInfo, _cache_put

from .kimi_provider import KIMI_BASE_URL

_log = logging.getLogger(__name__)
_CACHE_MAX = 512
_CACHE_TTL_S = 6 * 3600.0

_KIMI_MODELS_CACHE: dict[str, tuple[float, list[ModelInfo]]] = {}

# Fallback when /v1/models is unavailable (platform.kimi.ai chat API models).
_FALLBACK_MODELS: tuple[tuple[str, bool, int | None], ...] = (
    ("kimi-k3", False, None),
    ("kimi-k2.7", False, None),
    ("kimi-k2.7-code", False, None),
    ("moonshot-v1-8k", False, 8192),
    ("moonshot-v1-32k", False, 32768),
    ("moonshot-v1-128k", False, 131072),
    ("moonshot-v1-8k-vision-preview", True, 8192),
    ("moonshot-v1-32k-vision-preview", True, 32768),
    ("moonshot-v1-128k-vision-preview", True, 131072),
)


def _key_hash(api_key: str) -> str:
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()


def clear_model_cache() -> None:
    _KIMI_MODELS_CACHE.clear()


def fetch_models(api_key: str, **_kw: Any) -> list[ModelInfo]:
    return _fetch_kimi(api_key)


def _fallback_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            id=mid,
            display_name=mid,
            supports_vision=vision,
            supports_tools=True,
            context_limit=ctx,
        )
        for mid, vision, ctx in _FALLBACK_MODELS
    ]


def _info_from_id(model_id: str) -> ModelInfo:
    mid = model_id.strip()
    vision = "vision" in mid.lower()
    ctx: int | None = None
    lower = mid.lower()
    if "128k" in lower:
        ctx = 131072
    elif "32k" in lower:
        ctx = 32768
    elif "8k" in lower:
        ctx = 8192
    return ModelInfo(
        id=mid,
        display_name=mid,
        supports_vision=vision,
        supports_tools=True,
        context_limit=ctx,
    )


def _fetch_kimi(api_key: str) -> list[ModelInfo]:
    cache_key = _key_hash(api_key or "")
    hit = _KIMI_MODELS_CACHE.get(cache_key)
    if hit is not None and (time.time() - hit[0]) < _CACHE_TTL_S:
        return list(hit[1])

    models: list[ModelInfo] = []
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=KIMI_BASE_URL)
        listed = client.models.list()
        for item in listed.data or []:
            mid = str(getattr(item, "id", "") or "").strip()
            if mid:
                models.append(_info_from_id(mid))
        models.sort(key=lambda m: m.id)
    except Exception as exc:
        _log.warning("Kimi /v1/models unavailable: %s", exc)
        models = _fallback_models()

    if not models:
        models = _fallback_models()

    _cache_put(_KIMI_MODELS_CACHE, cache_key, (time.time(), models))
    return list(models)
