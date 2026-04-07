"""Tests for inference policy guardrails and hard-task shortcut defaults."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference


def _hard_history() -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": (
                '{"action_type":"propose","price":89.0,"payment_days":30,'
                '"use_treds":true,"propose_dynamic_discounting":true,'
                '"dynamic_discount_annual_rate":0.02}'
            ),
        }
    ]


def test_hard_two_step_default_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_HARD_TWO_STEP", raising=False)
    assert inference._hard_two_step_policy_enabled() is False


def test_hard_two_step_can_be_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_HARD_TWO_STEP", "1")
    assert inference._hard_two_step_policy_enabled() is True


def test_no_forced_accept_when_default_disabled(monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_HARD_TWO_STEP", raising=False)
    action = {"action_type": "propose", "price": 88.0, "payment_days": 30}

    out = inference._coerce_hard_accept_after_propose(
        action=action,
        history=_hard_history(),
        task_name="hard",
        round_number=1,
    )

    assert out["action_type"] == "propose"


def test_forced_accept_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_HARD_TWO_STEP", "1")
    action = {"action_type": "propose", "price": 88.0, "payment_days": 30}

    out = inference._coerce_hard_accept_after_propose(
        action=action,
        history=_hard_history(),
        task_name="hard",
        round_number=1,
    )

    assert out["action_type"] == "accept"
    assert out["propose_dynamic_discounting"] is True
