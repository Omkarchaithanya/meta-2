#!/usr/bin/env python3
"""Quick test of environment functionality."""
from src.env.sme_negotiation import SMENegotiationEnv
from src.utils.models import NegotiationAction

print("Testing SME Negotiation Environment...")
print("\n" + "="*50)

# Test 1: Environment instantiation and reset
print("\n[Test 1] Environment Reset")
env = SMENegotiationEnv()
obs = env.reset(task_id='easy', seed=42)
print(f"[OK] Reset successful")
print(f"  - Price offered: ${obs.p_opp}")
print(f"  - Days offered: {obs.d_opp}")
print(f"  - SME cost: ${obs.c_sme}")
print(f"  - Max rounds: {obs.t_max}")

# Test 2: Simple step
print("\n[Test 2] First Action (PROPOSE)")
action = NegotiationAction(
    action_type="PROPOSE",
    proposed_price=95.0,
    proposed_days=35
)

obs, reward, terminated, info = env.step(action)
print(f"[OK] Step successful")
print(f"  - Reward: {reward:.4f}")
print(f"  - Terminated: {terminated}")
print(f"  - New buyer offer - Price: ${obs.p_opp}, Days: {obs.d_opp}")

# Test 3: Continue negotiation
print("\n[Test 3] Accepting opponent's counter-offer")
if not terminated:
    action = NegotiationAction(
        action_type="ACCEPT",
        proposed_price=obs.p_opp,
        proposed_days=obs.d_opp
    )
    obs, reward, terminated, info = env.step(action)
    print(f"[OK] Accept successful")
    print(f"  - Final reward: {reward:.4f}")
    print(f"  - Terminated: {terminated}")
    if terminated:
        print(f"  - Episode ended after {obs.t_elapsed} rounds")

# Test 4: Determinism - run again with same seed
print("\n[Test 4] Determinism Check (same seed = same trajectory)")
env1 = SMENegotiationEnv()
obs1 = env1.reset(task_id='easy', seed=100)
action1 = NegotiationAction(action_type="PROPOSE", proposed_price=95.0, proposed_days=35)
obs1_after, reward1, _, _ = env1.step(action1)

env2 = SMENegotiationEnv()
obs2 = env2.reset(task_id='easy', seed=100)
action2 = NegotiationAction(action_type="PROPOSE", proposed_price=95.0, proposed_days=35)
obs2_after, reward2, _, _ = env2.step(action2)

assert obs1_after.p_opp == obs2_after.p_opp, "Prices don't match!"
assert obs1_after.d_opp == obs2_after.d_opp, "Days don't match!"
assert reward1 == reward2, "Rewards don't match!"
print(f"[OK] Determinism verified")
print(f"  - Both trajectories produced same buyer counter-offer: ${obs1_after.p_opp}, {obs1_after.d_opp} days")
print(f"  - Both generated same reward: {reward1:.4f}")

print("\n" + "="*50)
print("[SUCCESS] All environment tests passed!")
