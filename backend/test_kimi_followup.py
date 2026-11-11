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

from backend.agent.providers.base import ProviderMessage, ToolCallRequest
from kimi_provider import KimiProvider


def test_followup_comes_after_previous_reply():
    msgs = [
        ProviderMessage(role="user", content="first"),
        ProviderMessage(
            role="assistant",
            content="Checking",
            tool_calls=[ToolCallRequest(id="t1", name="ping", arguments={})],
        ),
        ProviderMessage(role="tool", tool_call_id="t1", content='{"ok":true}'),
        ProviderMessage(role="assistant", content="Here is the answer."),
        ProviderMessage(role="user", content="follow up"),
    ]
    out = KimiProvider("k", "kimi-k2")._to_openai_messages("sys", msgs)
    roles = [m["role"] for m in out if m["role"] != "system"]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assert out[-2]["content"] == "Here is the answer."
    assert out[-1]["content"] == "follow up"


def test_a_tool_call_with_no_thinking_recorded_still_carries_the_field_when_thinking_is_on():
    """Kimi: 400 'thinking is enabled but reasoning_content is missing in assistant tool call
    message at index 2'. A chat started on another model could never continue on K3."""
    msgs = [
        ProviderMessage(role="user", content="go"),
        ProviderMessage(role="assistant", content="", tool_calls=[ToolCallRequest(id="t1", name="ping", arguments={})]),
        ProviderMessage(role="tool", tool_call_id="t1", content="{}"),
        ProviderMessage(role="assistant", content="Done."),
    ]
    out = KimiProvider("k", "kimi-k3")._to_openai_messages("sys", msgs)
    assert out[2]["tool_calls"] and out[2]["reasoning_content"].strip() == "" and out[2]["reasoning_content"]
    assert "reasoning_content" not in out[4]  # only tool-call messages need it
    on = KimiProvider("k", "kimi-k2.6", thinking_effort="high")._to_openai_messages("sys", msgs)
    off = KimiProvider("k", "kimi-k2.6", thinking_effort="off")._to_openai_messages("sys", msgs)
    assert "reasoning_content" in on[2] and "reasoning_content" not in off[2]


def test_quota_wordings_from_kimi_code_switch_the_key():
    from kimi_provider import is_quota_error

    for text in ("You've reached your 5-hour usage limit. Your quota will reset when the current 5-hour window ends.",
                 "Your credit balance is insufficient. To continue, purchase extra usage or upgrade your plan",
                 "Your organization's Extra Usage balance has run out."):
        assert is_quota_error(text), text
    assert not is_quota_error("Your current subscription does not have access to k3.1-pro")


def test_reasoning_content_goes_back_inside_a_tool_loop():
    """Kimi's thinking models want the assistant message back as returned, reasoning included."""
    msgs = [
        ProviderMessage(role="user", content="wire the device"),
        ProviderMessage(
            role="assistant",
            content="",
            thinking="Read status first.",
            tool_calls=[ToolCallRequest(id="t1", name="ducky_get_status", arguments={})],
        ),
        ProviderMessage(role="tool", tool_call_id="t1", content='{"ok":true}'),
        ProviderMessage(role="assistant", content="Done.", thinking="Status was fine."),
        ProviderMessage(role="assistant", content="No thoughts here."),
    ]
    out = KimiProvider("k", "kimi-k2-thinking")._to_openai_messages("sys", msgs)
    assert out[2]["tool_calls"] and out[2]["reasoning_content"] == "Read status first."
    assert out[4]["reasoning_content"] == "Status was fine."
    assert "reasoning_content" not in out[5]
    assert "reasoning_content" not in out[1] and "reasoning_content" not in out[3]
