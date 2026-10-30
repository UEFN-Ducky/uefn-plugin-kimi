"""N Kimi keys: core picks the key account and hands its key to the factory each turn."""

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

from backend.agent.providers import _QUOTA_RE as CORE_QUOTA_RE
from kimi_provider import KimiProvider, is_quota_error

# Moonshot's own error bodies (openai SDK str(): "Error code: N - {...}").
BALANCE = ("Your account org-1 <ak-1> is suspended due to insufficient balance, "
           "please recharge your account or check your plan and billing details")
RATE = ("Error code: 429 - {'error': {'message': 'Your account org-1 request reached organization max RPM: 3', "
        "'type': 'rate_limit_reached_error'}}")


def test_each_turn_uses_the_key_core_picked() -> None:
    # register()'s factory is `lambda api_key, model, **kw: KimiProvider(api_key, model, **kw)`.
    assert KimiProvider("key-1", "kimi-k3")._api_key == "key-1"
    assert KimiProvider("key-2", "kimi-k3")._api_key == "key-2"


def test_insufficient_balance_switches_key() -> None:
    """Core's default misses Moonshot's bare balance message, so the plugin registers its own."""
    assert not CORE_QUOTA_RE.search(BALANCE)
    assert is_quota_error(BALANCE)
    assert is_quota_error(f"Error code: 429 - {{'error': {{'message': '{BALANCE}', "
                          "'type': 'exceeded_current_quota_error'}}")
    assert is_quota_error(RATE) and CORE_QUOTA_RE.search(RATE)
    assert not is_quota_error("Error code: 400 - Invalid request: model not found")
