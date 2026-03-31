"""Unit tests for OpenEnv SME Negotiation environment."""
import pytest
from src.env.sme_negotiator import SMENegotiationEnv
from src.utils.models import NegotiationAction
from src.utils.grader import DeterministicGrader, GraderConfig


class TestEnvironmentInitialization:
    """Test environment setup and reset."""
    
    def test_reset_easy_task(self):
        """Test reset on easy task."""
        env = SMENegotiationEnv()
        state = env.reset(task_id="easy", seed=42)
        
        assert state is not None
        assert state.task_id == "easy"
        assert state.t_elapsed == 0
        assert state.p_opp > 0
        assert state.d_opp == 30
    
    def test_reset_medium_task(self):
        """Test reset on medium task."""
        env = SMENegotiationEnv()
        state = env.reset(task_id="medium", seed=42)
        
        assert state.task_id == "medium"
        assert state.d_opp == 90
        assert state.l_sme == 60
    
    def test_reset_hard_task(self):
        """Test reset on hard task."""
        env = SMENegotiationEnv()
        state = env.reset(task_id="hard", seed=42)
        
        assert state.task_id == "hard"
        assert state.d_opp == 120
        assert state.l_sme == 30
        assert state.v_opp == 5000
    
    def test_deterministic_reset(self):
        """Test that same seed produces same state."""
        env1 = SMENegotiationEnv()
        state1 = env1.reset(task_id="easy", seed=12345)
        
        env2 = SMENegotiationEnv()
        state2 = env2.reset(task_id="easy", seed=12345)
        
        assert state1.p_opp == state2.p_opp
        assert state1.d_opp == state2.d_opp
        assert state1.c_sme == state2.c_sme
    
    def test_invalid_task_id(self):
        """Test error handling for invalid task."""
        env = SMENegotiationEnv()
        
        with pytest.raises(ValueError):
            env.reset(task_id="invalid_task")


class TestActionValidation:
    """Test action validation logic."""
    
    def test_valid_propose_action(self):
        """Test valid PROPOSE action."""
        action = NegotiationAction(
            action_type="PROPOSE",
            proposed_price=90.0,
            proposed_days=45,
            request_treds=False,
            justification="Test justification"
        )
        
        is_valid, msg = action.validate_action()
        assert is_valid
    
    def test_propose_missing_price(self):
        """Test PROPOSE without price."""
        action = NegotiationAction(
            action_type="PROPOSE",
            proposed_days=45,
        )
        
        is_valid, msg = action.validate_action()
        assert not is_valid
    
    def test_valid_accept_action(self):
        """Test valid ACCEPT action."""
        action = NegotiationAction(
            action_type="ACCEPT",
            proposed_price=90.0,
            proposed_days=45
        )
        
        is_valid, msg = action.validate_action()
        assert is_valid
    
    def test_valid_reject_action(self):
        """Test valid REJECT action."""
        action = NegotiationAction(action_type="REJECT")
        
        is_valid, msg = action.validate_action()
        assert is_valid
    
    def test_justification_word_limit(self):
        """Test justification word limit enforcement."""
        long_justification = " ".join(["word"] * 501)  # 501 words
        
        action = NegotiationAction(
            action_type="PROPOSE",
            proposed_price=90.0,
            proposed_days=45,
            justification=long_justification
        )
        
        is_valid, msg = action.validate_action()
        assert not is_valid


class TestStepMechanics:
    """Test single step execution."""
    
    def test_propose_generates_counter_offer(self):
        """Test that PROPOSE generates counter-offer."""
        env = SMENegotiationEnv()
        state = env.reset(task_id="easy", seed=42)
        
        action = NegotiationAction(
            action_type="PROPOSE",
            proposed_price=90.0,
            proposed_days=30,
            justification="Testing counter-offer generation"
        )
        
        observation, reward, terminated, info = env.step(action)
        
        assert not terminated  # Easy task should not end in one round
        assert reward == 0.0  # Intermediate reward is always 0
        assert "counter_price" in info
        assert "counter_days" in info
    
    def test_accept_action(self):
        """Test ACCEPT action terminating episode."""
        env = SMENegotiationEnv()
        state = env.reset(task_id="easy", seed=42)
        
        # Accept current offer
        action = NegotiationAction(
            action_type="ACCEPT",
            proposed_price=state.p_opp,
            proposed_days=state.d_opp
        )
        
        observation, reward, terminated, info = env.step(action)
        
        assert terminated
        assert info.get('success') == True
        assert reward > 0.0 or reward == 0.0  # Terminal reward
    
    def test_reject_action(self):
        """Test REJECT action."""
        env = SMENegotiationEnv()
        state = env.reset(task_id="easy", seed=42)
        
        action = NegotiationAction(action_type="REJECT")
        
        observation, reward, terminated, info = env.step(action)
        
        assert terminated
        assert info.get('success') == False
        assert reward == 0.0


class TestDeterministicGrader:
    """Test deterministic reward grading."""
    
    def setup_method(self):
        """Setup grader for testing."""
        config = GraderConfig(
            sme_cost=80.0,
            sme_liquidity_threshold=60,
            market_discount_rate=0.08,
            internal_cost_of_capital=0.12,
        )
        self.grader = DeterministicGrader(config)
    
    def test_successful_deal_score(self):
        """Test scoring for successful deal."""
        score, details = self.grader.calculate_grader_score(
            final_price=100.0,
            final_days=30,
            final_volume=1000,
            treds_utilized=False,
            success=True,
            u_max=10000.0,
            u_min=0.0
        )
        
        assert 0.0 <= score <= 1.0
        assert details['success'] == True
    
    def test_failed_deal_score(self):
        """Test that failed deals score 0.0."""
        score, details = self.grader.calculate_grader_score(
            final_price=100.0,
            final_days=30,
            final_volume=1000,
            treds_utilized=False,
            success=False,
            u_max=10000.0,
            u_min=0.0
        )
        
        assert score == 0.0
    
    def test_liquidity_penalty(self):
        """Test liquidity penalty calculation."""
        # Deal without TReDS beyond regulatory limit
        score_bad, _ = self.grader.calculate_grader_score(
            final_price=100.0,
            final_days=90,
            final_volume=1000,
            treds_utilized=False,
            success=True,
            u_max=10000.0,
            u_min=0.0
        )
        
        # Same deal with TReDS
        score_good, _ = self.grader.calculate_grader_score(
            final_price=100.0,
            final_days=90,
            final_volume=1000,
            treds_utilized=True,
            success=True,
            u_max=10000.0,
            u_min=0.0
        )
        
        # TReDS version should score better
        assert score_good > score_bad
    
    def test_bankruptcy_scenario(self):
        """Test scenario where payment exceeds survival threshold."""
        score, details = self.grader.calculate_grader_score(
            final_price=100.0,
            final_days=80,  # > 60 threshold without TReDS
            final_volume=1000,
            treds_utilized=False,
            success=True,
            u_max=10000.0,
            u_min=0.0
        )
        
        assert score == 0.0  # Bankruptcy
        assert details['reason'] == "Failed liquidity threshold without TReDS"


class TestFullEpisode:
    """Test complete episodes."""
    
    def test_easy_task_completion(self):
        """Test complete Easy task episode."""
        env = SMENegotiationEnv()
        state = env.reset(task_id="easy", seed=42)
        
        episode_done = False
        step_count = 0
        
        while not episode_done and step_count < 5:
            action = NegotiationAction(
                action_type="PROPOSE",
                proposed_price=95.0,
                proposed_days=30,
                justification="Fair offer"
            )
            
            state, reward, terminated, info = env.step(action)
            step_count += 1
            episode_done = terminated
        
        assert episode_done
        assert step_count > 0
    
    def test_score_normalization(self):
        """Test that scores are properly normalized."""
        env = SMENegotiationEnv()
        
        scores = []
        for seed in range(10):
            state = env.reset(task_id="easy", seed=seed)
            
            # Follow simple heuristic until done
            for _ in range(10):
                action = NegotiationAction(
                    action_type="PROPOSE",
                    proposed_price=state.p_opp * 0.99,
                    proposed_days=state.d_opp,
                    justification="Conceding slightly"
                )
                
                state, reward, terminated, info = env.step(action)
                if terminated:
                    scores.append(info.get('score', 0.0))
                    break
        
        # Check all scores are normalized
        for score in scores:
            assert 0.0 <= score <= 1.0


class TestLiquidityConstraints:
    """Test liquidity constraint handling."""
    
    def test_hard_task_liquidity_crisis(self):
        """Test Hard task liquidity crisis scenario."""
        env = SMENegotiationEnv()
        state = env.reset(task_id="hard", seed=100)
        
        # Hard task: 30-day survival, 120-day buyer demand
        assert state.l_sme == 30
        assert state.d_opp == 120
        
        # Standard accept without TReDS should fail
        action = NegotiationAction(
            action_type="ACCEPT",
            proposed_price=state.p_opp,
            proposed_days=state.d_opp,
            request_treds=False
        )
        
        observation, reward, terminated, info = env.step(action)
        assert terminated
        # Accepting 120 days without TReDS with 30-day threshold = bankruptcy
        assert info.get('score', 1.0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
