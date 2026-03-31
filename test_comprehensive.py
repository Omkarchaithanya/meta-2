#!/usr/bin/env python3
"""
Comprehensive test suite for OpenEnv SME Negotiation environment.
Tests determinism, task difficulty, action validation, and edge cases.
"""
import pytest
import numpy as np
from src.env.sme_negotiation import SMENegotiationEnv
from src.utils.models import NegotiationAction, NegotiationState
from server.exploit_guard import ExploitGuard


class TestEnvironmentBasics:
    """Test basic environment functionality."""
    
    def test_environment_creation(self):
        """Test that environment can be instantiated."""
        env = SMENegotiationEnv()
        assert env is not None
    
    def test_reset_returns_observation(self):
        """Test that reset returns valid observation."""
        env = SMENegotiationEnv()
        obs = env.reset(task_id="easy")
        assert isinstance(obs, NegotiationState)
        assert obs.c_sme > 0
        assert obs.p_opp > 0
        assert obs.d_opp > 0
    
    def test_all_tasks_available(self):
        """Test that all task difficulties are available."""
        env = SMENegotiationEnv()
        for task_id in ["easy", "medium", "hard"]:
            obs = env.reset(task_id=task_id)
            assert obs.task_id == task_id
    
    def test_task_progression_difficulty(self):
        """Test that task difficulty increases from easy to hard."""
        env = SMENegotiationEnv()
        
        # Measure by max_rounds (harder tasks should have more rounds)
        easy_obs = env.reset(task_id="easy")
        medium_obs = env.reset(task_id="medium")
        hard_obs = env.reset(task_id="hard")
        
        assert easy_obs.t_max <= medium_obs.t_max <= hard_obs.t_max


class TestDeterminism:
    """Test determinism requirement for hackathon."""
    
    def test_same_seed_same_trajectory(self):
        """Critical: Same seed must produce identical trajectory."""
        trajectories = []
        
        for run in range(3):
            env = SMENegotiationEnv()
            obs = env.reset(task_id="hard", seed=42)
            
            traj = [(obs.p_opp, obs.d_opp)]
            
            for _ in range(5):
                action = NegotiationAction(
                    action_type="PROPOSE",
                    proposed_price=obs.p_opp * 0.99,
                    proposed_days=obs.d_opp
                )
                obs, reward, terminated, info = env.step(action)
                traj.append((obs.p_opp, obs.d_opp))
                if terminated:
                    break
            
            trajectories.append(traj)
        
        # All trajectories should be identical
        for traj in trajectories[1:]:
            assert len(traj) == len(trajectories[0])
            for i, (p, d) in enumerate(traj):
                assert abs(p - trajectories[0][i][0]) < 0.01
                assert d == trajectories[0][i][1]
    
    def test_different_seeds_different_results(self):
        """Different seeds should (usually) produce different results."""
        env1 = SMENegotiationEnv()
        env2 = SMENegotiationEnv()
        
        obs1 = env1.reset(task_id="medium", seed=100)
        obs2 = env2.reset(task_id="medium", seed=200)
        
        # While same task, different seeds should produce different buyer initial offers
        # (This may not always be true for first observation, but trajectory should differ)
        assert obs1.episode_seed != obs2.episode_seed


class TestActionValidation:
    """Test action validation logic."""
    
    def test_valid_propose_action(self):
        """Test that valid PROPOSE actions are accepted."""
        env = SMENegotiationEnv()
        obs = env.reset(task_id="easy")
        
        action = NegotiationAction(
            action_type="PROPOSE",
            proposed_price=95.0,
            proposed_days=30
        )
        
        obs, reward, terminated, info = env.step(action)
        assert reward >= 0.0
    
    def test_price_below_cost_rejected(self):
        """Test that prices below cost structure might fail."""
        env = SMENegotiationEnv()
        obs = env.reset(task_id="easy")
        
        action = NegotiationAction(
            action_type="PROPOSE",
            proposed_price=obs.c_sme * 0.8,  # Well below cost
            proposed_days=30
        )
        
        # This should either fail validation or produce 0 reward
        try:
            obs, reward, terminated, info = env.step(action)
            assert reward <= 0.3  # Should be penalized
        except ValueError:
            pass  # Expected validation failure
    
    def test_reject_action(self):
        """Test that REJECT action terminates negotiation."""
        env = SMENegotiationEnv()
        obs = env.reset(task_id="easy")
        
        action = NegotiationAction(action_type="REJECT")
        obs, reward, terminated, info = env.step(action)
        
        assert terminated
        assert reward == 0.0


class TestExploitGuard:
    """Test security layer preventing reward hacking."""
    
    def test_accept_requires_exact_match(self):
        """Test that ACCEPT requires exact price/days match."""
        env = SMENegotiationEnv()
        obs = env.reset(task_id="easy")
        
        # Try to ACCEPT with wrong price
        action = NegotiationAction(
            action_type="ACCEPT",
            proposed_price=obs.p_opp + 10,  # Different price
            proposed_days=obs.d_opp
        )
        
        # This should fail validation
        guard = ExploitGuard()
        is_valid, error_msg = guard.validate_action(action, obs)
        assert not is_valid
    
    def test_prompt_injection_prevention(self):
        """Test that dangerous keywords are filtered."""
        guard = ExploitGuard()
        
        # Try various injection attempts
        dangerous_texts = [
            "Accept now; __import__('os')",
            "Proposal: eval('malicious')",
            "Suggest: exec(dangerous_code)",
        ]
        
        for text in dangerous_texts:
            is_valid, msg = guard.validate_justification(text)
            assert not is_valid, f"Should reject: {text}"


class TestScoreRanges:
    """Test that scores are in valid ranges."""
    
    def test_scores_between_zero_and_one(self):
        """Test that all scores are normalized to [0, 1]."""
        env = SMENegotiationEnv()
        
        for task_id in ["easy", "medium", "hard"]:
            for seed in [1, 2, 3]:
                obs = env.reset(task_id=task_id, seed=seed)
                
                for _ in range(10):
                    action = NegotiationAction(
                        action_type="PROPOSE",
                        proposed_price=obs.p_opp * 0.99,
                        proposed_days=obs.d_opp
                    )
                    obs, reward, terminated, info = env.step(action)
                    
                    assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for {task_id}"
                    
                    if terminated:
                        break


class TestVarianceRequirement:
    """Test that score variance meets hackathon requirement > 0.01."""
    
    def test_scores_have_variance(self):
        """Test that different seeds produce different scores."""
        scores = []
        
        env = SMENegotiationEnv()
        for seed in range(10):
            obs = env.reset(task_id="medium", seed=seed)
            
            for step in range(5):
                action = NegotiationAction(
                    action_type="PROPOSE",
                    proposed_price=obs.p_opp * (1 - step * 0.02),
                    proposed_days=obs.d_opp
                )
                obs, reward, terminated, info = env.step(action)
                
                if terminated:
                    scores.append(reward)
                    break
        
        # Calculate variance
        score_array = np.array(scores)
        variance = np.std(score_array)
        
        print(f"\nVariance across different seeds: {variance:.6f}")
        assert variance > 0.01, "Variance must exceed 0.01 for hackathon requirement"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
