#!/usr/bin/env python3
"""
Metrics and analytics module for OpenEnv environment.
Tracks performance, fairness, and constraint satisfaction.
"""
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: int
    task_id: str
    seed: int
    final_score: float
    num_steps: int
    terminated: bool
    final_price: Optional[float] = None
    final_days: Optional[int] = None
    npv_raw: Optional[float] = None
    buyer_profit_margin: Optional[float] = None
    sme_profit_margin: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TaskMetrics:
    """Aggregate metrics for a task."""
    task_id: str
    num_episodes: int
    avg_score: float
    std_score: float
    min_score: float
    max_score: float
    median_score: float
    q25_score: float
    q75_score: float
    avg_steps: float
    success_rate: float  # % of non-zero scores
    variance: float     # > 0.01 for hackathon requirement
    meets_requirement: bool


class MetricsCollector:
    """Collates metrics from multiple episodes."""
    
    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
        self.task_metrics: Dict[str, TaskMetrics] = {}
    
    def add_episode(self, metrics: EpisodeMetrics):
        """Add a single episode's metrics."""
        self.episodes.append(metrics)
    
    def compute_task_metrics(self, task_id: str) -> TaskMetrics:
        """Compute aggregate metrics for a task."""
        task_episodes = [e for e in self.episodes if e.task_id == task_id]
        
        if not task_episodes:
            raise ValueError(f"No episodes found for task: {task_id}")
        
        scores = np.array([e.final_score for e in task_episodes])
        steps = np.array([e.num_steps for e in task_episodes])
        
        # Calculate metrics
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        median_score = float(np.median(scores))
        q25_score = float(np.percentile(scores, 25))
        q75_score = float(np.percentile(scores, 75))
        avg_steps = float(np.mean(steps))
        
        # Success rate: % of episodes with score > 0
        success_rate = float(np.sum(scores > 0.0) / len(scores))
        
        # Variance requirement
        meets_requirement = std_score > 0.01
        
        metrics = TaskMetrics(
            task_id=task_id,
            num_episodes=len(task_episodes),
            avg_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            max_score=max_score,
            median_score=median_score,
            q25_score=q25_score,
            q75_score=q75_score,
            avg_steps=avg_steps,
            success_rate=success_rate,
            variance=std_score,
            meets_requirement=meets_requirement
        )
        
        self.task_metrics[task_id] = metrics
        return metrics
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate comprehensive metrics report."""
        # Compute metrics for all tasks
        tasks = set(e.task_id for e in self.episodes)
        for task_id in sorted(tasks):
            self.compute_task_metrics(task_id)
        
        # Overall statistics
        all_scores = np.array([e.final_score for e in self.episodes])
        overall_stats = {
            "total_episodes": len(self.episodes),
            "unique_tasks": len(tasks),
            "overall_mean_score": float(np.mean(all_scores)),
            "overall_std_score": float(np.std(all_scores)),
            "overall_min_score": float(np.min(all_scores)),
            "overall_max_score": float(np.max(all_scores)),
        }
        
        # Build report
        report = {
            "generated_at": datetime.now().isoformat(),
            "overall_statistics": overall_stats,
            "task_metrics": {
                task_id: asdict(metrics)
                for task_id, metrics in self.task_metrics.items()
            },
            "hackathon_requirement": {
                "variance_threshold": 0.01,
                "all_tasks_meet_requirement": all(
                    m.meets_requirement for m in self.task_metrics.values()
                ),
                "task_compliance": {
                    task_id: m.meets_requirement
                    for task_id, m in self.task_metrics.items()
                }
            }
        }
        
        # Save if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Metrics report saved to: {output_file}")
        
        return report
    
    def print_summary(self):
        """Print a human-readable summary."""
        if not self.task_metrics:
            print("No metrics to display. Run evaluation first.")
            return
        
        print("\n" + "="*70)
        print("  OPENENV SME NEGOTIATION - METRICS SUMMARY".center(70))
        print("="*70)
        
        for task_id, metrics in sorted(self.task_metrics.items()):
            status = "PASS" if metrics.meets_requirement else "FAIL"
            print(f"\n{task_id.upper():8} [{status}]")
            print(f"  Episodes:       {metrics.num_episodes}")
            print(f"  Avg Score:      {metrics.avg_score:.4f}")
            print(f"  Std Dev:        {metrics.std_score:.4f} (requirement > 0.01)")
            print(f"  Score Range:    [{metrics.min_score:.4f}, {metrics.max_score:.4f}]")
            print(f"  Median:         {metrics.median_score:.4f}")
            print(f"  Success Rate:   {metrics.success_rate*100:.1f}%")
            print(f"  Avg Steps:      {metrics.avg_steps:.1f}")
        
        print(f"\n{'='*70}")
        overall = self.task_metrics
        all_meet = all(m.meets_requirement for m in overall.values())
        status = "SUCCESS" if all_meet else "NEEDS IMPROVEMENT"
        print(f"Overall Status: {status}".center(70))
        print(f"{'='*70}\n")


def create_performance_report(episodes: List[EpisodeMetrics]) -> str:
    """Generate a text report from episodes."""
    collector = MetricsCollector()
    for ep in episodes:
        collector.add_episode(ep)
    
    # Compute metrics
    tasks = set(e.task_id for e in episodes)
    for task_id in sorted(tasks):
        collector.compute_task_metrics(task_id)
    
    # Generate text report
    lines = [
        "PERFORMANCE ANALYSIS REPORT",
        "=" * 60,
        f"Total Episodes: {len(episodes)}",
        f"Timestamp: {datetime.now().isoformat()}",
        ""
    ]
    
    for task_id, metrics in collector.task_metrics.items():
        lines.append(f"\nTask: {task_id.upper()}")
        lines.append(f"  Episodes: {metrics.num_episodes}")
        lines.append(f"  Mean Score: {metrics.avg_score:.4f}")
        lines.append(f"  Std Dev: {metrics.std_score:.4f}")
        lines.append(f"  Variance Requirement Met: {metrics.meets_requirement}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("Metrics module loaded successfully")
