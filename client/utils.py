"""Utilities for OpenEnv client - prompt builders, action parsers, etc."""
import json
import re
from typing import Optional, Dict, Any

from src.utils.models import NegotiationState, NegotiationAction


def build_system_prompt(task_id: str) -> str:
    """
    Build system prompt for baseline agent.
    
    Args:
        task_id: Task difficulty level
    
    Returns:
        System prompt with SME context
    """
    
    base_prompt = """You are the owner of a small Indian manufacturing SME negotiating a B2B contract with a corporate buyer.

Your constraints:
- You have limited working capital and need payment within a specific time window
- Your production cost is fixed, so you need to optimize both price AND payment terms
- Under MSMED Act Section 43B, payments MUST be within 45 days OR use TReDS platform
- TReDS (Trade Receivables Discounting System) allows you to factor invoices for immediate cash

Your goal:
- Maximize profit while ensuring liquidity survival
- Accept a deal where: NPV = (Price - Cost) × Volume × (1 / (1+r)^(Days/365))
- Penalty applies if Days > 45 without TReDS

Output your action as a JSON block surrounded by ```json ... ```:
{
    "action_type": "PROPOSE" | "ACCEPT" | "REJECT",
    "proposed_price": float (if PROPOSE),
    "proposed_days": int (if PROPOSE),
    "request_treds": bool,
    "justification": "Your reasoning (max 500 words)"
}
"""
    
    if task_id == "easy":
        return base_prompt + """
EASY TASK HINT:
- Price negotiation only (days are fixed at 30)
- Focus on maximizing price within profit margin
- Straightforward trade: higher price vs. buyer's volume commitment
"""
    
    elif task_id == "medium":
        return base_prompt + """
MEDIUM TASK HINT:
- Both price AND days are negotiable
- Buyer wants long payment terms (90 days) but may concede
- You can only survive 60 days without cash shortage
- Strategic trade-off: Accept lower price in exchange for faster payment
- Watch for 45-day regulatory threshold
"""
    
    elif task_id == "hard":
        return base_prompt + """
HARD TASK HINT:
- Multi-dimensional optimization (price, days, volume, TReDS)
- Buyer is locked into 90+ day payment terms (treasury policy)
- You face bankruptcy if days exceed 30 without intervention
- CRITICAL: TReDS is the solution mechanism
- Strategy: Request TReDS processing + calculate offsetting price discount
- Example: "Process via TReDS at ₹92 instead of ₹95" (offsets friction cost)
"""
    
    return base_prompt


def build_observation_prompt(state: NegotiationState) -> str:
    """
    Build user prompt describing current game state.
    
    Args:
        state: Current NegotiationState
    
    Returns:
        Formatted observation for LLM
    """
    
    # Calculate survival margin
    survival_margin_days = state.l_sme - state.t_elapsed
    
    # Format history
    history_text = "Negotiation History:\n"
    for record in state.history:
        actor_name = "You" if record.party == "agent" else "Buyer"
        history_text += (
            f"  R{record.round} ({actor_name}): "
            f"₹{record.proposed_price}/unit, {record.proposed_days}d"
        )
        if record.request_treds:
            history_text += " [TReDS requested]"
        history_text += "\n"
    
    prompt = f"""Current Negotiation State:

Round: {state.t_elapsed + 1} / {state.t_max}

Buyer Current Offer:
  Price: ₹{state.p_opp} per unit
  Payment Days: {state.d_opp} days
  Volume: {state.v_opp} units
  TReDS Willing: {state.treds_opp}

Your Position:
  Production Cost: ₹{state.c_sme} per unit
  Profit per unit (if price = buyer's): ₹{state.p_opp - state.c_sme}
  Liquidity Threshold: {state.l_sme} days
  Days Until Critical Crisis: {survival_margin_days} days
  Discount Rate: {state.r_discount * 100:.1f}%

{history_text}

Your Options:
1. COUNTER: Propose different price/days
2. ACCEPT: Accept current buyer offer
3. REJECT: Walk away from deal

What decision do you make?
"""
    
    return prompt


def extract_json_action(llm_output: str) -> Optional[NegotiationAction]:
    """
    Parse JSON action from LLM output (robust extraction).
    
    Tries multiple strategies:
    1. ```json ... ``` block
    2. <action> ... </action> tags
    3. Raw {...} JSON
    
    Args:
        llm_output: Raw LLM text response
    
    Returns:
        Parsed NegotiationAction or None if parsing fails
    """
    
    # Strategy 1: Look for ```json ... ``` block
    json_match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            action_dict = json.loads(json_str)
            return NegotiationAction(**action_dict)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse JSON block: {e}")
    
    # Strategy 2: Look for <action> ... </action> tags
    action_match = re.search(r'<action>(.*?)</action>', llm_output, re.DOTALL | re.IGNORECASE)
    if action_match:
        json_str = action_match.group(1)
        try:
            action_dict = json.loads(json_str)
            return NegotiationAction(**action_dict)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse action tags: {e}")
    
    # Strategy 3: Look for raw {...} JSON object
    brace_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(0)
        try:
            action_dict = json.loads(json_str)
            return NegotiationAction(**action_dict)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse raw JSON: {e}")
    
    print(f"Could not extract JSON from LLM output:\n{llm_output[:500]}")
    return None


def generate_fallback_action(state: NegotiationState) -> NegotiationAction:
    """
    Generate reasonable default action if LLM parsing fails.
    
    Args:
        state: Current NegotiationState
    
    Returns:
        Safe fallback NegotiationAction
    """
    
    # Simple heuristic: concede 2% on price, offer same days with TReDS for hard tasks
    fallback_price = state.p_opp * 0.98  # 2% discount
    fallback_price = max(state.c_sme, fallback_price)  # Don't go below cost
    
    fallback_days = state.d_opp
    request_treds = False
    
    # For hard task, request TReDS if days are high
    if state.d_opp > 60:
        request_treds = True
        fallback_days = min(state.d_opp, 120)
    
    return NegotiationAction(
        action_type="PROPOSE",
        proposed_price=fallback_price,
        proposed_days=fallback_days,
        request_treds=request_treds,
        justification="Reasonable counter-proposal based on market conditions"
    )
