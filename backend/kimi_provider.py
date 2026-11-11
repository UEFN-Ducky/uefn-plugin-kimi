"""Kimi (Moonshot) Chat Completions provider — OpenAI-compatible API."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, AsyncIterator

from backend.agent.multimodal_content import build_openai_user_content
from backend.agent.prompt_cache import PromptCachePayload
from backend.agent.providers.base import (
    ProviderMessage,
    StreamEvent,
    StreamEventKind,
    ToolCallRequest,
)
from backend.agent.providers.cache_utils import openai_system_messages, parse_openai_usage
from backend.agent.providers.thinking import ThinkSplitter, reasoning_from_delta
from backend.agent.thinking_effort import normalize_thinking_effort

KIMI_BASE_URL = "https://api.moonshot.ai/v1"
# Kimi issues keys from three places and none works on another's server (Kimi's error
# reference: 401 "Invalid Authentication" is most often a key used on the wrong base URL):
# the API platform, a Kimi Code subscription, and the China platform.
KIMI_BASES = (KIMI_BASE_URL, "https://api.kimi.ai/coding/v1", "https://api.moonshot.cn/v1")
_BASE_BY_KEY: dict[str, str] = {}  # sha256(key) -> the server that accepted it


def _probe(base: str, key: str) -> int:
    """HTTP status of a free models listing with this key (0 = could not reach the server)."""
    try:
        import httpx

        return httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=8.0).status_code
    except Exception:
        return 0


def find_base(api_key: str, base_url: str = "") -> tuple[str, int]:
    """(server this key works on, last HTTP status). ``base_url`` set on the account wins.
    A key no server accepts gets the platform's URL and the status it answered."""
    key = (api_key or "").strip()
    explicit = (base_url or "").strip().rstrip("/")
    bases = (explicit,) if explicit else KIMI_BASES
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    if not explicit and digest in _BASE_BY_KEY:
        return _BASE_BY_KEY[digest], 200
    if not key:
        return bases[0], 401
    status = 0
    for base in bases:
        code = _probe(base, key)
        if code == 200:
            if not explicit:
                _BASE_BY_KEY[digest] = base
            return base, 200
        status = code or status
    return bases[0], status


def resolve_base(api_key: str, base_url: str = "") -> str:
    return find_base(api_key, base_url)[0]


_K3_EFFORT = {"low": "low", "medium": "high", "high": "max"}
# kimi-k3, kimi-k3.1-pro, and Kimi Code's own ids: k3, k3.1-pro, k3-256k.
_K3_RE = re.compile(r"(?:^|[-/])k3(?:$|[.\-])")


def _is_k3(mid: str) -> bool:
    return bool(_K3_RE.search(mid))


def kimi_supports_thinking(model: str) -> bool:
    mid = (model or "").strip().lower()
    if "k2.7" in mid or "k2-7" in mid:
        return False
    return _is_k3(mid) or "k2.5" in mid or "k2-5" in mid or "k2.6" in mid or "k2-6" in mid


def thinking_menu(model: str) -> dict | None:
    if not kimi_supports_thinking(model):
        return None
    mid = (model or "").strip().lower()
    if _is_k3(mid):
        return {
            "lo": "Faster",
            "hi": "Smarter",
            "levels": [
                {"id": "off", "label": "Off", "thinking_tokens": 0, "hint": "lowest available (reasoning_effort=low)"},
                {"id": "low", "label": "Low", "thinking_tokens": None, "hint": "reasoning_effort=low, no token cap"},
                {"id": "medium", "label": "Med", "thinking_tokens": None, "hint": "reasoning_effort=high, no token cap"},
                {"id": "high", "label": "High", "thinking_tokens": None, "hint": "reasoning_effort=max, no token cap"},
            ],
        }
    return {
        "lo": "Faster",
        "hi": "Smarter",
        "levels": [
            {"id": "off", "label": "Off", "thinking_tokens": 0, "hint": "thinking disabled"},
            {"id": "high", "label": "On", "thinking_tokens": None, "hint": "thinking enabled, no token cap"},
        ],
    }


def kimi_effort_body(model: str, thinking_effort: str) -> dict[str, Any]:
    mid = (model or "").strip().lower()
    effort = normalize_thinking_effort(thinking_effort)
    if _is_k3(mid):
        if effort == "off":
            return {"reasoning_effort": "low"}
        return {"reasoning_effort": _K3_EFFORT.get(effort, "high")}
    if "k2.7" in mid or "k2-7" in mid:
        return {}
    if "k2.5" in mid or "k2-5" in mid or "k2.6" in mid or "k2-6" in mid:
        return {"thinking": {"type": "disabled" if effort == "off" else "enabled"}}
    return {}


def kimi_thinks(model: str, thinking_effort: str) -> bool:
    """This request runs with thinking on (K3 always does; K2.5/2.6 unless Off)."""
    body = kimi_effort_body(model, thinking_effort)
    return "reasoning_effort" in body or (body.get("thinking") or {}).get("type") == "enabled"


def _attach_reasoning(msg: dict[str, Any], m: ProviderMessage, *, required: bool = False) -> None:
    """Pass the step's ``reasoning_content`` back, as Kimi's thinking models require
    inside a tool-call loop (and across turns for the models that keep thinking):
    the assistant message goes back as the API returned it.

    ``required``: thinking is on and this is a tool-call message. One that has no thinking
    recorded (made by another model, or an older chat) still needs the field, or Kimi
    answers 400 "thinking is enabled but reasoning_content is missing in assistant tool
    call message" and the chat can never continue."""
    if (m.thinking or "").strip():
        msg["reasoning_content"] = m.thinking
    elif required:
        msg["reasoning_content"] = " "


# Core's default (429 / rate limit / quota) plus Kimi's wordings that can arrive without
# the status code: the platform's "Your account … is suspended due to insufficient balance,
# please recharge your account", and Kimi Code's "You've reached your 5-hour usage limit",
# "Your credit balance is insufficient", "Extra Usage balance has run out".
_QUOTA_RE = re.compile(
    r"\b429\b|rate[_ ]?limit|insufficient_quota|quota|insufficient balance|recharge your account"
    r"|usage limit|credit balance is insufficient|extra usage",
    re.IGNORECASE,
)


def is_quota_error(text: str) -> bool:
    return bool(_QUOTA_RE.search(text or ""))


class KimiProvider:
    def __init__(self, api_key: str, model: str, *, thinking_effort: str = "off", base_url: str = "",
                 **_kw: Any) -> None:
        self._api_key = api_key
        self._model = model
        self._thinking_effort = normalize_thinking_effort(thinking_effort)
        self._base_url = (base_url or "").strip()

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=resolve_base(self._api_key, self._base_url))

    def _to_openai_messages(
        self,
        system: str,
        messages: list[ProviderMessage],
        *,
        cache: PromptCachePayload | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = list(openai_system_messages(cache, fallback_system=system))
        thinks = kimi_thinks(self._model, self._thinking_effort)
        for m in messages:
            if m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.content,
                    }
                )
                continue
            if m.role == "assistant" and m.tool_calls:
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                }
                _attach_reasoning(msg, m, required=thinks)
                out.append(msg)
                continue
            if m.role == "user" and m.attachments:
                out.append({"role": "user", "content": build_openai_user_content(m.content, m.attachments)})
                continue
            msg = {"role": m.role, "content": m.content}
            if m.role == "assistant":
                _attach_reasoning(msg, m)
            out.append(msg)
        return out

    async def stream_turn(
        self,
        *,
        system: str,
        messages: list[ProviderMessage],
        tools: list[dict[str, Any]],
        cancel_event: Any | None = None,
        cache: PromptCachePayload | None = None,
    ) -> AsyncIterator[StreamEvent]:
        client = self._client()
        collected_text = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        cancelled = False
        splitter = ThinkSplitter()

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(system, messages, cache=cache),
            "tools": tools if tools else None,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        cache_key = (cache.prompt_cache_key if cache else "") or ""
        if cache_key:
            create_kwargs["prompt_cache_key"] = cache_key
        extra = kimi_effort_body(self._model, self._thinking_effort)
        if extra:
            create_kwargs["extra_body"] = extra

        stream = client.chat.completions.create(**create_kwargs)
        for chunk in stream:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                cancelled = True
                break
            if chunk.usage:
                usage = parse_openai_usage(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning = reasoning_from_delta(delta)
            if reasoning:
                yield StreamEvent(kind=StreamEventKind.THINKING, text=reasoning)
            if delta.content:
                for kind, seg in splitter.feed(delta.content):
                    if kind == "think":
                        yield StreamEvent(kind=StreamEventKind.THINKING, text=seg)
                    else:
                        collected_text += seg
                        yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=seg)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    acc = tool_calls_acc.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments

        if cancelled:
            return

        for kind, seg in splitter.flush():
            if kind == "think":
                yield StreamEvent(kind=StreamEventKind.THINKING, text=seg)
            else:
                collected_text += seg
                yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=seg)

        rebuilt: list[ToolCallRequest] = []
        for acc in tool_calls_acc.values():
            try:
                args = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            rebuilt.append(
                ToolCallRequest(id=acc["id"] or "call", name=acc["name"], arguments=args)
            )
        if rebuilt:
            yield StreamEvent(kind=StreamEventKind.TOOL_CALLS, tool_calls=rebuilt, usage=usage)
        yield StreamEvent(
            kind=StreamEventKind.DONE,
            text=collected_text,
            stop_reason="tool_calls" if rebuilt else "stop",
            usage=usage,
        )

    async def test_connection(self) -> tuple[bool, str]:
        """Is the key good? One free models listing per Kimi server, never a generated reply:
        that cost tokens, took long on a thinking model, and failed a good Kimi Code or China
        key (wrong server) and a good key whose plan lacks the test model."""
        base, status = find_base(self._api_key, self._base_url)
        if status == 200:
            where = {KIMI_BASES[1]: " (Kimi Code)", KIMI_BASES[2]: " (China platform)"}.get(base, "")
            return True, f"Kimi OK{where}"
        if status == 0:
            return False, "Could not reach Kimi. Check your internet connection and try again."
        if status in (401, 403):
            return False, ("Kimi did not accept this key on its API platform, Kimi Code or the China "
                           "platform. Copy the whole key again from the page that issued it.")
        return False, f"Kimi answered HTTP {status} to this key. Try again in a minute."
