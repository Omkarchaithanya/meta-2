#!/usr/bin/env python3
"""Debug script to understand medium/hard task failures."""
import sys
from src.env.sme_negotiation import SMENegotiationEnv
from client.utils import generate_fallback_action

print("\n=== DEBUGGING MEDIUM/HARD TASK FAILURES ===\n")

for task_id in ["easy", "medium", "hard"]:
    print(f"\n[{task_id.upper()}]")
    print("-" * 50)
    
    env = SMENegotiationEnv()
    obs = env.reset(task_id=task_id, seed=123)
    
    print(f"Initial state:")
    print(f"  Price: {obs.p_opp}, Days: {obs.d_opp}")
    print(f"  Cost: {obs.c_sme}, Liquidity threshold: {obs.l_sme}")
    print(f"  Max rounds: {obs.t_max}")
    
    for step_num in range(5):
        try:
            action = generate_fallback_action(obs)
            print(f"\nStep {step_num + 1}:")
            print(f"  Action: {action.action_type} - Price: {action.proposed_price:.2f}, Days: {action.proposed_days}")
            
            obs, reward, terminated, info = env.step(action)
            
            print(f"  Result: Reward={reward:.4f}, Terminated={terminated}")
            if 'failure_reason' in info:
                print(f"  Failure: {info['failure_reason']}")
            
            if terminated:
                print(f"  Episode ended after {step_num + 1} steps")
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
print("\n=== END DEBUG ===\n")
