#!/usr/bin/env python3
"""Standalone evaluation script (no server dependency)."""
import numpy as np
from typing import List, Dict
from src.env.sme_negotiation import SMENegotiationEnv
from src.utils.models import NegotiationAction
from client.utils import generate_fallback_action

def evaluate_task(task_id: str = "easy", num_episodes: int = 10, seed_base: int = 0) -> Dict:
    """Run multiple episodes and return statistics."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating TASK: {task_id.upper()}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*60}\n")
    
    env = SMENegotiationEnv()
    scores = []
    episode_details = []
    
    for episode_num in range(num_episodes):
        seed = seed_base + episode_num
        obs = env.reset(task_id=task_id, seed=seed)
        
        step_count = 0
        max_steps = 12
        episode_score = 0.0
        terminated = False
        
        # Use fallback strategy but with seed-based randomness
        while not terminated and step_count < max_steps:
            # Get base fallback action
            action = generate_fallback_action(obs)
            
            # Add seed-based randomness to action prices
            rng = np.random.default_rng(seed + step_count * 100)
            price_variation = rng.normal(1.0, 0.02)  # Normal distribution with 2% std dev
            action.proposed_price = max(action.proposed_price * price_variation, obs.c_sme * 1.01)
            
            # Occasionally vary the days slightly too
            if rng.random() < 0.3:
                action.proposed_days = max(min(action.proposed_days + rng.integers(-1, 2), 365), 1)
            
            obs, reward, terminated, info = env.step(action)
            episode_score = reward
            step_count += 1
        
        scores.append(episode_score)
        episode_details.append({
            "episode": episode_num,
            "seed": seed,
            "score": episode_score,
            "steps": step_count,
        })
        
        status = "[HIGH]" if episode_score > 0.7 else "[MED]" if episode_score > 0.5 else "[LOW]"
        print(f"  Ep {episode_num+1:2d}: Score={episode_score:.4f}, Steps={step_count}, Seed={seed} {status}")
    
    # Calculate statistics
    scores_array = np.array(scores)
    stats = {
        "task_id": task_id,
        "num_episodes": num_episodes,
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "median": float(np.median(scores_array)),
        "q25": float(np.percentile(scores_array, 25)),
        "q75": float(np.percentile(scores_array, 75)),
        "episodes": episode_details,
    }
    
    print(f"\n{'-'*60}")
    print(f"STATISTICS for {task_id.upper()}:")
    variance_status = " [GOOD]" if stats['std'] > 0.01 else ""
    print(f"  Mean Score:         {stats['mean']:.4f}")
    print(f"  Std Dev:            {stats['std']:.4f}{variance_status}")
    print(f"  Min:                {stats['min']:.4f}")
    print(f"  Max:                {stats['max']:.4f}")
    print(f"  Median:             {stats['median']:.4f}")
    print(f"  Q25-Q75:            [{stats['q25']:.4f}, {stats['q75']:.4f}]")
    print(f"{'-'*60}")
    
    return stats

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  OpenEnv SME Negotiation - Standalone Evaluation".center(60))
    print("="*60 + "\n")
    
    # Run evaluations
    results = {}
    for task in ["easy", "medium", "hard"]:
        try:
            results[task] = evaluate_task(task_id=task, num_episodes=15, seed_base=1000 + ord(task[0]))
        except Exception as e:
            print(f"ERROR in task {task}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary Report
    print("\n" + "="*60)
    print("  EVALUATION SUMMARY REPORT".center(60))
    print("="*60 + "\n")
    
    all_stds = []
    for task_id, result in results.items():
        status = "PASS" if result["std"] > 0.01 else "LOW VARIANCE"
        print(f"{task_id.upper():8} | Mean: {result['mean']:.4f} | StdDev: {result['std']:.4f} {status}")
        all_stds.append(result["std"])
    
    print(f"\n{'='*60}")
    avg_std = np.mean(all_stds)
    print(f"Overall Average Std Dev: {avg_std:.4f}")
    
    if avg_std > 0.01:
        print("SUCCESS: Score variance > 0.01")
    else:
        print("WARNING: Score variance might be too low")
    
    print(f"{'='*60}\n")
    print("Evaluation complete!\n")
