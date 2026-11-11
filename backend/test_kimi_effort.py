from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for k in list(sys.modules):
    if k == "backend" or k.startswith("backend."):
        del sys.modules[k]
for p in _here.parents:
    cand = p / "UEFN-Ducky-Release" / "ducky_app"
    if (cand / "backend" / "agent").is_dir():
        sys.path.insert(0, str(_here))
        sys.path.insert(0, str(cand))
        break

from kimi_provider import kimi_effort_body, kimi_supports_thinking, thinking_menu


def test_kimi_effort_body() -> None:
    assert kimi_supports_thinking("kimi-k3")
    assert kimi_supports_thinking("kimi-k2.5")
    assert not kimi_supports_thinking("kimi-k2.7-code")
    assert kimi_effort_body("kimi-k3", "low") == {"reasoning_effort": "low"}
    assert kimi_effort_body("kimi-k3", "medium") == {"reasoning_effort": "high"}
    assert kimi_effort_body("kimi-k3", "high") == {"reasoning_effort": "max"}
    assert kimi_effort_body("kimi-k2.5", "off") == {"thinking": {"type": "disabled"}}
    assert kimi_effort_body("kimi-k2.6", "high") == {"thinking": {"type": "enabled"}}
    assert kimi_effort_body("kimi-k2.7-code", "high") == {}
    assert thinking_menu("kimi-k2.7-code") is None
    assert thinking_menu("kimi-k3")["levels"][0]["id"] == "off"
    assert thinking_menu("kimi-k2.5")["levels"][0]["thinking_tokens"] == 0


def test_kimi_code_model_ids_think_like_k3() -> None:
    """A Kimi Code key lists k3, k3.1-pro, k3-256k (no 'kimi-' prefix); K3.1 Pro answers
    400 'does not support disabled thinking mode', so Off is the lowest effort, never disabled."""
    for mid in ("k3", "k3.1-pro", "k3-256k", "kimi-k3.1-pro"):
        assert kimi_supports_thinking(mid), mid
        assert kimi_effort_body(mid, "off") == {"reasoning_effort": "low"}, mid
        assert thinking_menu(mid)["levels"][-1]["hint"].startswith("reasoning_effort=max"), mid
    assert not kimi_supports_thinking("kimi-for-coding") and kimi_effort_body("kimi-for-coding", "high") == {}
    assert not kimi_supports_thinking("moonshot-v1-8k-vision-preview")


def test_any_kimi_key_finds_its_server(monkeypatch) -> None:
    """Keys from the API platform, Kimi Code and the China platform each work on one server
    only; the wrong one answers 401 Invalid Authentication."""
    import asyncio

    import kimi_provider as kp

    asked: list[str] = []
    monkeypatch.setattr(kp, "_BASE_BY_KEY", {})
    monkeypatch.setattr(kp, "_probe", lambda base, key: asked.append(base) or (200 if base == kp.KIMI_BASES[1] else 401))
    assert kp.resolve_base("code-key") == "https://api.kimi.ai/coding/v1"
    assert kp.resolve_base("code-key") == "https://api.kimi.ai/coding/v1" and len(asked) == 2  # remembered
    assert kp.resolve_base("code-key", "https://proxy.example/v1/") == "https://proxy.example/v1"  # the account's own URL wins
    ok, detail = asyncio.run(kp.KimiProvider("code-key", "kimi-k3").test_connection())
    assert ok and "Kimi Code" in detail  # a free listing, never a generated reply

    monkeypatch.setattr(kp, "_probe", lambda base, key: 401)
    ok, detail = asyncio.run(kp.KimiProvider("bad-key", "kimi-k3").test_connection())
    assert not ok and "did not accept this key" in detail
    assert kp.resolve_base("bad-key") == kp.KIMI_BASE_URL  # not remembered
    monkeypatch.setattr(kp, "_probe", lambda base, key: 0)
    ok, detail = asyncio.run(kp.KimiProvider("any-key", "kimi-k3").test_connection())
    assert not ok and "Could not reach Kimi" in detail


if __name__ == "__main__":
    test_kimi_effort_body()
    print("ok")
