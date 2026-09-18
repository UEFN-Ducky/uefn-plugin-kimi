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


if __name__ == "__main__":
    test_kimi_effort_body()
    print("ok")
