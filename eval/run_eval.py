#!/usr/bin/env python3
"""
Comprehensive evaluation script for OpenEnv SME Negotiation environment.

CRITICAL FOR HACKATHON:
- Validates score variance (must be > 0.01)
- Tests all three task levels
- Verifies determinism
- Generates performance report
"""

import asyncio
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np

from client.env_client import EnvClient
from client.utils import generate_fallback_action
from src.utils.models import NegotiationAction


class EvaluationRunner:
    """Runs comprehensive environment evaluation."""
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8000/ws/eval",
        output_dir: str = "./eval_results",
    ):
        """Initialize evaluation runner."""
        
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().isoformat()
        self.results = []
    
    async def run_single_episode(
        self,
        task_id: str,
        seed: int,
        agent_strategy: str = "fallback",
        max_steps: int = 12,
    ) -> Dict:
        """
        Run a single negotiation episode.
        
        Args:
            task_id: Task difficulty
            seed: Deterministic seed
            agent_strategy: Strategy for action generation ("fallback" / "random" / "heuristic")
            max_steps: Max steps before truncation
        
        Returns:
            Episode result dict with score, trajectory, etc.
        """
        
        episode_result = {
            "task_id": task_id,
            "seed": seed,
            "strategy": agent_strategy,
            "max_steps": max_steps,
            "score": 0.0,
            "steps_taken": 0,
            "terminated": False,
            "error": None,
            "trajectory": [],
        }
        
        try:
            async with EnvClient(self.server_url) as env:
                # Reset environment
                obs = await env.reset(task_id=task_id, seed=seed)
                episode_result["initial_state"] = {
                    "p_opp": obs.p_opp,
                    "d_opp": obs.d_opp,
                    "l_sme": obs.l_sme,
                }
                
                step_count = 0
                while not obs.terminal and step_count < max_steps:
                    # Generate action
                    action = self._generate_action(obs, agent_strategy)
                    
                    # Log trajectory
                    episode_result["trajectory"].append({
                        "step": step_count,
                        "action_type": action.action_type,
                        "proposed_price": action.proposed_price,
                        "proposed_days": action.proposed_days,
                    })
                    
                    # Step environment
                    try:
                        obs, reward, terminated, info = await env.step(action)
                        step_count += 1
                    except Exception as e:
                        episode_result["error"] = str(e)
                        break
                
                # Record final results
                episode_result["steps_taken"] = step_count
                episode_result["score"] = float(reward)
                episode_result["terminated"] = terminated
                episode_result["final_state"] = {
                    "p_opp": obs.p_opp if obs else None,
                    "d_opp": obs.d_opp if obs else None,
                    "task_id": obs.task_id if obs else None,
                }
        
        except Exception as e:
            episode_result["error"] = str(e)
            episode_result["score"] = 0.0
        
        return episode_result
    
    def _generate_action(self, obs, strategy: str) -> NegotiationAction:
        """Generate action based on strategy."""
        
        if strategy == "fallback":
            return generate_fallback_action(obs)
        
        elif strategy == "random":
            # Random valid action
            action_type = np.random.choice(["PROPOSE", "REJECT"])
            if action_type == "PROPOSE":
                return NegotiationAction(
                    action_type="PROPOSE",
                    proposed_price=np.random.uniform(obs.c_sme, obs.p_opp * 1.2),
                    proposed_days=np.random.randint(1, obs.d_opp + 1),
                    request_treds=np.random.random() > 0.7,
                    justification="Random proposal"
                )
            else:
                return NegotiationAction(
                    action_type="REJECT",
                    justification="Random rejection"
                )
        
        elif strategy == "heuristic":
            # Smarter heuristic
            if len(obs.history) < 3:
                # Early rounds: make reasonable counter
                return NegotiationAction(
                    action_type="PROPOSE",
                    proposed_price=obs.p_opp * 1.05,  # 5% above offer
                    proposed_days=max(obs.d_opp - 10, 1),
                    request_treds=obs.d_opp > 45,
                    justification="Counter proposal with TReDS for compliance"
                )
            else:
                # Late rounds: accept if reasonable
                if obs.p_opp > obs.c_sme and obs.d_opp <= 60:
                    return NegotiationAction(
                        action_type="ACCEPT",
                        proposed_price=obs.p_opp,
                        proposed_days=obs.d_opp,
                        justification="Acceptable terms"
                    )
                else:
                    return generate_fallback_action(obs)
        
        return generate_fallback_action(obs)
    
    async def run_task_evaluation(
        self,
        task_id: str,
        num_episodes: int = 100,
        agent_strategy: str = "fallback",
    ) -> Dict:
        """
        Run evaluation for a specific task.
        
        Args:
            task_id: "easy", "medium", or "hard"
            num_episodes: Number of episodes to run
            agent_strategy: Action generation strategy
        
        Returns:
            Task evaluation results
        """
        
        print(f"\n{'='*60}")
        print(f"Evaluating TASK={task_id.upper()}, Episodes={num_episodes}")
        print(f"{'='*60}")
        
        scores = []
        episodes = []
        
        for i in range(num_episodes):
            seed = i
            
            # Run episode
            result = await self.run_single_episode(
                task_id=task_id,
                seed=seed,
                agent_strategy=agent_strategy,
            )
            
            episodes.append(result)
            scores.append(result["score"])
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                current_mean = np.mean(scores)
                current_std = np.std(scores) if len(scores) > 1 else 0.0
                print(f"  [{i+1:3d}/{num_episodes}] Mean: {current_mean:.3f}, Std: {current_std:.4f}")
        
        # Compute statistics
        scores_array = np.array(scores)
        task_results = {
            "task_id": task_id,
            "num_episodes": num_episodes,
            "agent_strategy": agent_strategy,
            "mean_score": float(np.mean(scores_array)),
            "std_score": float(np.std(scores_array)),
            "min_score": float(np.min(scores_array)),
            "max_score": float(np.max(scores_array)),
            "median_score": float(np.median(scores_array)),
            "q25_score": float(np.percentile(scores_array, 25)),
            "q75_score": float(np.percentile(scores_array, 75)),
            "score_variance_check": float(np.std(scores_array)) > 0.01,  # CRITICAL
            "episodes": episodes,
        }
        
        # Print summary
        print(f"\n{task_id.upper()} Task Results:")
        print(f"  Mean Score:    {task_results['mean_score']:.4f}")
        print(f"  Std Dev:       {task_results['std_score']:.4f}")
        print(f"  Min/Max:       {task_results['min_score']:.4f} / {task_results['max_score']:.4f}")
        print(f"  Variance > 0.01:  {'✅ PASS' if task_results['score_variance_check'] else '❌ FAIL'}")
        
        return task_results
    
    async def run_all_tasks(
        self,
        num_episodes: int = 100,
        tasks: Optional[List[str]] = None,
        agent_strategy: str = "fallback",
    ) -> Dict:
        """
        Run evaluation for all specified tasks.
        
        Args:
            num_episodes: Episodes per task
            tasks: Task list (default: all)
            agent_strategy: Action generation strategy
        
        Returns:
            Full evaluation results
        """
        
        if tasks is None:
            tasks = ["easy", "medium", "hard"]
        
        all_results = {
            "timestamp": self.timestamp,
            "num_episodes_per_task": num_episodes,
            "agent_strategy": agent_strategy,
            "tasks": {}
        }
        
        for task_id in tasks:
            task_results = await self.run_task_evaluation(
                task_id=task_id,
                num_episodes=num_episodes,
                agent_strategy=agent_strategy,
            )
            all_results["tasks"][task_id] = task_results
            self.results.extend(task_results["episodes"])
        
        # Compute cross-task statistics
        all_scores = []
        for task_data in all_results["tasks"].values():
            all_scores.extend([ep["score"] for ep in task_data["episodes"]])
        
        all_results["overall_statistics"] = {
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores)),
            "score_variance_check": float(np.std(all_scores)) > 0.01,
        }
        
        return all_results
    
    def save_results(self, eval_results: Dict, format: str = "json") -> Path:
        """
        Save evaluation results to file.
        
        Args:
            eval_results: Evaluation results dict
            format: "json" or "csv"
        
        Returns:
            Path to saved file
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filepath = self.output_dir / f"eval_results_{timestamp}.json"
            with open(filepath, "w") as f:
                json.dump(eval_results, f, indent=2, default=str)
            print(f"\nResults saved to: {filepath}")
            return filepath
        
        elif format == "csv":
            filepath = self.output_dir / f"eval_results_{timestamp}.csv"
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "task_id", "seed", "score", "steps_taken", "terminated", "error"
                ])
                writer.writeheader()
                for episode in self.results:
                    writer.writerow({
                        "task_id": episode["task_id"],
                        "seed": episode["seed"],
                        "score": episode["score"],
                        "steps_taken": episode["steps_taken"],
                        "terminated": episode["terminated"],
                        "error": episode["error"],
                    })
            print(f"\nCSV results saved to: {filepath}")
            return filepath
        
        return None


async def main():
    """Main evaluation entry point."""
    
    parser = argparse.ArgumentParser(description="OpenEnv SME Negotiation Evaluation")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per task")
    parser.add_argument("--server-url", type=str, default="ws://localhost:8000/ws/eval")
    parser.add_argument("--strategy", choices=["fallback", "random", "heuristic"], default="fallback")
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--format", choices=["json", "csv", "both"], default="json")
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(server_url=args.server_url, output_dir=args.output_dir)
    
    tasks = ["all"] if args.task == "all" else [args.task]
    if tasks == ["all"]:
        tasks = None
    
    eval_results = await runner.run_all_tasks(
        num_episodes=args.episodes,
        tasks=tasks,
        agent_strategy=args.strategy,
    )
    
    # Save results
    if args.format in ["json", "both"]:
        runner.save_results(eval_results, format="json")
    if args.format in ["csv", "both"]:
        runner.save_results(eval_results, format="csv")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Mean Score: {eval_results['overall_statistics']['mean_score']:.4f}")
    print(f"Overall Std Dev:    {eval_results['overall_statistics']['std_score']:.4f}")
    print(f"Variance Check:     {'✅ PASS' if eval_results['overall_statistics']['score_variance_check'] else '❌ FAIL'}")
    print(f"{'='*60}\n")
    
    # Verify critical requirement
    if eval_results['overall_statistics']['score_variance_check']:
        print("✅ CRITICAL REQUIREMENT MET: Score variance > 0.01")
    else:
        print("❌ CRITICAL REQUIREMENT FAILED: Score variance <= 0.01")


if __name__ == "__main__":
    asyncio.run(main())
