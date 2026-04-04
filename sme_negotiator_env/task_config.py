"""Task definitions for the SME negotiation hackathon gradient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, Optional


@dataclass(frozen=True)
class TaskConfig:
    """Static configuration for one evaluation task."""

    name: str
    description: str
    difficulty: str
    # Negotiation / buyer dynamics (existing sim)
    initial_buyer_price: float
    initial_buyer_days: int
    max_rounds: int
    volume: int
    cost_threshold: float
    liquidity_threshold: int
    concede_low: float
    concede_high: float
    day_floor: int
    day_step_low: int
    day_step_high: int
    # Financial / strategic (exposed in state & observation)
    sme_monthly_revenue: float
    current_payment_terms_days: int
    sme_supplier_payment_days: int
    interest_rate_annual: float
    buyer_power_score: float
    secondary_buyer_power: Optional[float]
    negotiation_round_start: int
    # Which grader key in ``sme_negotiator_env.graders.TASK_GRADERS``
    grader_id: str
    context_note: str = ""


def default_task_config(task_id: str) -> TaskConfig:
    """Factory: realistic defaults per task."""
    if task_id == "payment-terms-easy":
        return TaskConfig(
            name="payment-terms-easy",
            description="Single SME vs cooperative buyer: reduce payment terms from 90d toward 60d or better.",
            difficulty="easy",
            initial_buyer_price=100.0,
            initial_buyer_days=90,
            max_rounds=10,
            volume=1000,
            cost_threshold=80.0,
            liquidity_threshold=60,
            concede_low=0.012,
            concede_high=0.035,
            day_floor=50,
            day_step_low=2,
            day_step_high=5,
            sme_monthly_revenue=500_000.0,
            current_payment_terms_days=90,
            sme_supplier_payment_days=30,
            interest_rate_annual=0.22,
            buyer_power_score=0.3,
            secondary_buyer_power=None,
            negotiation_round_start=0,
            grader_id="payment-terms-easy",
            context_note="Single buyer; target payment terms 90→≤60 days.",
        )
    if task_id == "payment-terms-medium":
        return TaskConfig(
            name="payment-terms-medium",
            description=(
                "SME under working-capital stress: tighten 60 to 45 days or better and agree a late payment "
                "penalty clause with a neutral buyer."
            ),
            difficulty="medium",
            initial_buyer_price=100.0,
            initial_buyer_days=60,
            max_rounds=12,
            volume=1000,
            cost_threshold=80.0,
            liquidity_threshold=45,
            concede_low=0.006,
            concede_high=0.018,
            day_floor=35,
            day_step_low=2,
            day_step_high=5,
            sme_monthly_revenue=500_000.0,
            current_payment_terms_days=750,
            sme_supplier_payment_days=20,
            interest_rate_annual=0.22,
            buyer_power_score=0.6,
            secondary_buyer_power=None,
            negotiation_round_start=0,
            grader_id="payment-terms-medium",
            context_note=(
                "Working-capital gap exceeds 2x monthly revenue (long receivable vs supplier terms). "
                "Negotiate 45 days or better and a late payment penalty clause."
            ),
        )
    if task_id == "payment-terms-hard":
        return TaskConfig(
            name="payment-terms-hard",
            description=(
                "Two-buyer consortium (hostile): negotiate dynamic discounting for faster payment; "
                "reward from NPV of financing vs status quo."
            ),
            difficulty="hard",
            initial_buyer_price=96.0,
            initial_buyer_days=100,
            max_rounds=16,
            volume=5000,
            cost_threshold=78.0,
            liquidity_threshold=55,
            concede_low=0.003,
            concede_high=0.009,
            day_floor=45,
            day_step_low=1,
            day_step_high=3,
            sme_monthly_revenue=600_000.0,
            current_payment_terms_days=120,
            sme_supplier_payment_days=25,
            interest_rate_annual=0.22,
            buyer_power_score=0.85,
            secondary_buyer_power=0.82,
            negotiation_round_start=0,
            grader_id="payment-terms-hard",
            context_note="Consortium of two buyers; leverage is high — use dynamic discounting.",
        )
    raise KeyError(f"Unknown task_id: {task_id}")


# Aliases for older client / notebook code
_LEGACY_ALIASES: Dict[str, str] = {
    "payment-term-negotiation": "payment-terms-medium",
    "early-payment-discount": "payment-terms-easy",
    "treds-enrollment": "payment-terms-hard",
}

TASK_REGISTRY: Final[Dict[str, TaskConfig]] = {
    "payment-terms-easy": default_task_config("payment-terms-easy"),
    "payment-terms-medium": default_task_config("payment-terms-medium"),
    "payment-terms-hard": default_task_config("payment-terms-hard"),
}


def resolve_task_id(requested: Optional[str], *, difficulty: Optional[str] = None) -> str:
    """Map kwargs / legacy names to a canonical task id."""
    if requested:
        rid = str(requested).strip()
        if rid in _LEGACY_ALIASES:
            return _LEGACY_ALIASES[rid]
        if rid in TASK_REGISTRY:
            return rid
    if difficulty:
        d = str(difficulty).lower()
        if d in ("easy", "e"):
            return "payment-terms-easy"
        if d in ("medium", "m"):
            return "payment-terms-medium"
        if d in ("hard", "h"):
            return "payment-terms-hard"
    return "payment-terms-medium"
