"""Typed OpenEnv models for the SME negotiation environment."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import ConfigDict, Field, computed_field
from openenv.core import Action, Observation, State


class NegotiationAction(Action):
    """Action the SME agent can take each round."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action_type: Literal["propose", "accept", "reject"] = "propose"
    price: float = Field(..., description="Proposed price in ₹/unit", ge=0)
    payment_days: int = Field(..., description="Proposed payment days", ge=0)
    use_treds: bool = Field(False, description="Whether to propose TReDS financing")
    reason: Optional[str] = Field(None, description="Agent's reasoning (optional)")
    # Medium / hard task extensions
    propose_late_payment_penalty_clause: bool = Field(
        False,
        description="If true, SME requests a contractual late-payment penalty (medium task).",
    )
    propose_dynamic_discounting: bool = Field(
        False,
        description="If true, SME proposes dynamic discounting for early payment (hard task).",
    )
    dynamic_discount_annual_rate: float = Field(
        0.0,
        ge=0.0,
        le=0.95,
        description="Annualized discount for early payment as a fraction (e.g. 0.08 = 8%).",
    )


class NegotiationObservation(Observation):
    """What the agent sees after each step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    round_number: int
    max_rounds: int
    buyer_price: float
    buyer_days: int
    buyer_accepted: bool
    negotiation_done: bool
    cost_threshold: float
    liquidity_threshold: int
    volume: int
    difficulty: str
    price_score: float
    days_score: float
    treds_bonus: float
    step_reward: float
    message: str
    # Financial / task context
    task_name: str = ""
    sme_monthly_revenue: float = 0.0
    working_capital_gap: float = 0.0
    interest_rate_annual: float = 0.0
    buyer_power_score: float = 0.0
    secondary_buyer_power: Optional[float] = None
    current_payment_terms_days: int = 0
    sme_supplier_payment_days: int = 0


class NegotiationState(State):
    """Full episode state including SME financials."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str
    seed: int
    difficulty: str
    task_name: str = ""
    step_count: int
    max_steps: int
    negotiation_round: int = 0
    max_rounds: int = 0
    deal_reached: bool
    final_price: Optional[float]
    final_days: Optional[int]
    treds_used: bool
    cumulative_reward: float
    buyer_price: float
    buyer_days: int
    initial_buyer_days: int
    cost_threshold: float
    liquidity_threshold: int
    volume: int
    message: str
    # Financial realism
    sme_monthly_revenue: float = Field(..., ge=0.0)
    current_payment_terms_days: int = Field(..., ge=0)
    sme_supplier_payment_days: int = Field(..., ge=0)
    interest_rate_annual: float = Field(0.22, ge=0.0, le=1.0)
    buyer_power_score: float = Field(0.5, ge=0.0, le=1.0)
    secondary_buyer_power: Optional[float] = Field(None, ge=0.0, le=1.0)
    agreed_terms: Optional[int] = Field(None, ge=0)
    late_payment_penalty_agreed: bool = False
    dynamic_discounting_agreed: bool = False
    agreed_dynamic_discount_annual: float = Field(0.0, ge=0.0, le=0.95)

    @computed_field
    @property
    def working_capital_gap(self) -> float:
        """INR tied up between supplier cash-out and buyer cash-in (simplified annualized gap)."""
        return self.sme_monthly_revenue * (
            self.current_payment_terms_days - self.sme_supplier_payment_days
        ) / 365.0


def default_negotiation_state(
    *,
    episode_id: str,
    seed: int,
    difficulty: str,
    task_name: str,
    max_steps: int,
    max_rounds: int,
    buyer_price: float,
    buyer_days: int,
    initial_buyer_days: int,
    cost_threshold: float,
    liquidity_threshold: int,
    volume: int,
    sme_monthly_revenue: float = 500_000.0,
    current_payment_terms_days: int = 90,
    sme_supplier_payment_days: int = 30,
    interest_rate_annual: float = 0.22,
    buyer_power_score: float = 0.4,
    secondary_buyer_power: Optional[float] = None,
    message: str = "",
) -> NegotiationState:
    """Factory with realistic INR / payment-term defaults for hackathon demos."""
    return NegotiationState(
        episode_id=episode_id,
        seed=seed,
        difficulty=difficulty,
        task_name=task_name,
        step_count=0,
        max_steps=max_steps,
        negotiation_round=0,
        max_rounds=max_rounds,
        deal_reached=False,
        final_price=None,
        final_days=None,
        treds_used=False,
        cumulative_reward=0.0,
        buyer_price=buyer_price,
        buyer_days=buyer_days,
        initial_buyer_days=initial_buyer_days,
        cost_threshold=cost_threshold,
        liquidity_threshold=liquidity_threshold,
        volume=volume,
        message=message,
        sme_monthly_revenue=sme_monthly_revenue,
        current_payment_terms_days=current_payment_terms_days,
        sme_supplier_payment_days=sme_supplier_payment_days,
        interest_rate_annual=interest_rate_annual,
        buyer_power_score=buyer_power_score,
        secondary_buyer_power=secondary_buyer_power,
        agreed_terms=None,
        late_payment_penalty_agreed=False,
        dynamic_discounting_agreed=False,
        agreed_dynamic_discount_annual=0.0,
    )
