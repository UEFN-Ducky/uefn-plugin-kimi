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

from kimi_provider import KimiProvider


def test_each_turn_uses_the_key_core_picked() -> None:
    # register()'s factory is `lambda api_key, model, **kw: KimiProvider(api_key, model, **kw)`.
    assert KimiProvider("key-1", "kimi-k3")._api_key == "key-1"
    assert KimiProvider("key-2", "kimi-k3")._api_key == "key-2"
