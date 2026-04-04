#!/usr/bin/env python3
"""Inference runner for the SME negotiation environment."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from sme_negotiator_env.client import SMENegotiatorEnv
from sme_negotiator_env.llm_action_parser import parse_llm_text_to_negotiation_action
from sme_negotiator_env.models import NegotiationAction

logger = logging.getLogger(__name__)

load_dotenv()

# LLM: always Hugging Face OpenAI-compatible router (hackathon rule — no other provider defaults)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# OpenEnv simulation server URL only (set OPENENV_BASE_URL in the environment for your deployment)
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")

NEGOTIATION_SYSTEM_PROMPT = (
    "You are a B2B negotiation assistant. Respond ONLY with valid JSON containing "
    "keys: action_type, price, payment_days, use_treds, reason, and optionally "
    "propose_late_payment_penalty_clause, propose_dynamic_discounting, dynamic_discount_annual_rate. "
    "action_type must be one of: propose, accept, reject."
)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _observation_to_dict(observation: Any) -> Dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return observation
    return dict(observation)


def format_observation(obs: Dict[str, Any]) -> str:
    return (
        f"Round={obs.get('round_number')} | Task={obs.get('task_name')} | "
        f"BuyerPrice={obs.get('buyer_price')} | BuyerDays={obs.get('buyer_days')} | "
        f"LiquidityThreshold={obs.get('liquidity_threshold')} | CostThreshold={obs.get('cost_threshold')} | "
        f"MonthlyRevenueINR={obs.get('sme_monthly_revenue')} | WCGap={obs.get('working_capital_gap')} | "
        f"BuyerPower={obs.get('buyer_power_score')}"
    )


def _safe_fallback_action(observation: Any) -> Dict[str, Any]:
    return {
        "action_type": "propose",
        "price": round(max(float(observation.cost_threshold) + 1.0, float(observation.buyer_price) * 0.99), 2),
        "payment_days": int(max(int(observation.liquidity_threshold), int(observation.buyer_days) - 5)),
        "use_treds": bool(int(observation.buyer_days) > int(observation.liquidity_threshold) + 20),
        "reason": "Fallback action due to model output issue",
    }


def get_agent_action(observation: Dict[str, Any], history: List[dict], task_name: str) -> Dict[str, Any]:
    user_message = (
        f"Task={task_name}\n"
        f"Current observation:\n{format_observation(observation)}\n"
        "Return only JSON action."
    )

    messages = [{"role": "system", "content": NEGOTIATION_SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_message}
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=180,
    )

    content = completion.choices[0].message.content
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        action = json.loads(raw)
        if not isinstance(action, dict):
            raise ValueError("LLM JSON root must be an object")
        return {
            "action_type": str(action.get("action_type", "propose")).lower(),
            "price": float(action.get("price", observation.get("buyer_price", 0.0))),
            "payment_days": int(action.get("payment_days", observation.get("buyer_days", 0))),
            "use_treds": bool(action.get("use_treds", False)),
            "reason": str(action.get("reason", "")),
        }
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("LLM output not valid JSON (%s); using prose/regex parser. Raw snippet: %r", exc, raw[:300])
        parsed = parse_llm_text_to_negotiation_action(raw, observation, allow_json=False)
        return parsed.model_dump()


def _to_model_action(action_payload: Dict[str, Any], observation: Any) -> NegotiationAction:
    action_type = str(action_payload.get("action_type", "propose")).lower()
    if action_type not in {"propose", "accept", "reject"}:
        action_type = "propose"

    price = float(action_payload.get("price", observation.buyer_price))
    payment_days = int(action_payload.get("payment_days", observation.buyer_days))
    use_treds = bool(action_payload.get("use_treds", False))
    reason = str(action_payload.get("reason", "Model-selected action"))

    return NegotiationAction(
        action_type=action_type,
        price=round(price, 2),
        payment_days=payment_days,
        use_treds=use_treds,
        reason=reason,
    )


async def run_episode(env: SMENegotiatorEnv, difficulty: str, seed: int) -> Dict[str, Any]:
    """Run one episode using model-guided actions with strict stdout formatting."""

    task_name = difficulty.lower()
    episode_id = f"{task_name}-{seed}"
    history: List[dict] = []

    all_rewards: List[float] = []
    round_number = 0
    success = False
    result: Any = None
    observation: Any = None
    final_score = 0.0

    try:
        result = await env.reset(seed=seed, difficulty=difficulty, episode_id=episode_id, task_name=task_name)
        observation = result.observation

        print(
            f"[START] task={task_name} env=openenv-sme-negotiator model={MODEL_NAME}",
            flush=True,
        )

        # Termination is driven by the environment (``done``), including max rounds — do not stop early here.
        while not result.done:
            obs_dict = _observation_to_dict(observation)

            llm_error: str | None = None
            try:
                action_payload = get_agent_action(obs_dict, history, task_name)
            except Exception as e:
                llm_error = str(e)
                action_payload = _safe_fallback_action(observation)

            action = _to_model_action(action_payload, observation)
            action_json = json.dumps(action.model_dump(), ensure_ascii=True)

            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            all_rewards.append(reward)

            err_out = "null" if llm_error is None else json.dumps(llm_error)
            print(
                f'[STEP] step={round_number + 1} action={action_json} reward={reward:.2f} '
                f'done={"true" if done else "false"} error={err_out}',
                flush=True,
            )

            history.append({"role": "user", "content": format_observation(obs_dict)})
            history.append({"role": "assistant", "content": json.dumps(action_payload, ensure_ascii=True)})

            round_number += 1

        final_score = float(result.reward or 0.0)
        meta = getattr(result.observation, "metadata", None) or {}
        if isinstance(meta, dict) and "success" in meta:
            success = bool(meta["success"])
        else:
            success = bool(result.done and final_score > 0.0)
    finally:
        total_reward = sum(all_rewards)
        print(
            f'[END] success={"true" if success else "false"} steps={round_number} '
            f'rewards={",".join(f"{r:.2f}" for r in all_rewards)}',
            flush=True,
        )

    return {
        "difficulty": difficulty,
        "seed": seed,
        "final_score": final_score,
        "total_reward": total_reward,
        "steps": round_number,
        "success": success,
        "step_rewards": all_rewards,
        "final_observation": _observation_to_dict(observation) if observation is not None else {},
    }


async def main() -> None:
    """Run three episodes per difficulty and write a compact results file."""

    results: Dict[str, Any] = {
        "metadata": {
            "llm_api_base_url": API_BASE_URL,
            "openenv_base_url": OPENENV_BASE_URL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
        },
        "tasks": {},
    }

    async with SMENegotiatorEnv(base_url=OPENENV_BASE_URL) as env:
        for difficulty in ["EASY", "MEDIUM", "HARD"]:
            episode_results: List[Dict[str, Any]] = []
            for seed in [1000, 1001, 1002]:
                episode_results.append(await run_episode(env, difficulty, seed))

            scores = [episode["final_score"] for episode in episode_results]
            rewards = [episode["total_reward"] for episode in episode_results]
            successes = [episode["success"] for episode in episode_results]

            results["tasks"][difficulty] = {
                "episodes": episode_results,
                "summary": {
                    "mean_final_score": sum(scores) / len(scores),
                    "mean_total_reward": sum(rewards) / len(rewards),
                    "success_rate": sum(1 for success in successes if success) / len(successes),
                },
            }

    overall_scores = [
        episode["final_score"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]
    overall_rewards = [
        episode["total_reward"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]
    overall_successes = [
        episode["success"]
        for task in results["tasks"].values()
        for episode in task["episodes"]
    ]

    results["summary"] = {
        "overall_mean_score": sum(overall_scores) / len(overall_scores),
        "overall_mean_reward": sum(overall_rewards) / len(overall_rewards),
        "overall_success_rate": sum(1 for success in overall_successes if success) / len(overall_successes),
    }

    with open("inference_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
