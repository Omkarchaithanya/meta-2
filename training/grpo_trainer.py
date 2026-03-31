"""TRL GRPOTrainer integration for OpenEnv SME Negotiation RL training."""
import asyncio
import logging
from typing import List, Dict, Optional, Callable, Any

import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from client.env_client import EnvClient
from client.utils import extract_json_action, generate_fallback_action
from src.utils.models import NegotiationState, NegotiationAction

logger = logging.getLogger(__name__)


class OpenEnvGRPOTrainer:
    """
    GRPO trainer wrapper for OpenEnv SME Negotiation environment.
    
    Integrates:
    - TRL GRPOTrainer for policy optimization
    - Async environment interaction
    - Custom rollout function
    - Reward signal from environment grader
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        server_url: str = "ws://localhost:8000/ws",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./checkpoints",
        per_device_batch_size: int = 4,
        num_generations: int = 8,
        max_completion_length: int = 1024,
        learning_rate: float = 1e-6,
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model_name: HuggingFace model ID (LLaMA 2, Mistral, etc.)
            server_url: WebSocket server URL
            device: Training device (cuda/cpu)
            output_dir: Checkpoint directory
            per_device_batch_size: Batch size for GRPO generation
            num_generations: Number of GRPO generation rounds
            max_completion_length: Max tokens per completion
            learning_rate: Training learning rate
        """
        
        self.model_name = model_name
        self.server_url = server_url
        self.device = device
        self.output_dir = output_dir
        
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        grpo_config = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            learning_rate=learning_rate,
            temperature=0.8,
            topk=50,
            num_return_sequences=1,
        )
        
        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            tokenizer=self.tokenizer,
        )
        
        logger.info("GRPO trainer initialized")
    
    async def rollout_func(
        self,
        prompts: List[str],
        task_id: str = "medium",
        num_episodes: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Async rollout function for environment interaction.
        
        For each prompt:
        1. Run complete negotiation episode against environment
        2. Collect trajectory: (state, action, reward)
        3. Return final episode reward
        
        Args:
            prompts: List of initial prompts (system context)
            task_id: Task difficulty
            num_episodes: Episodes per prompt
        
        Returns:
            List of {query, response, reward, seed} dicts
        """
        
        results = []
        
        for episode_idx in range(num_episodes):
            for prompt_idx, prompt in enumerate(prompts):
                seed = episode_idx * len(prompts) + prompt_idx
                
                try:
                    # Run async negotiation episode
                    reward = await self._run_episode(
                        prompt=prompt,
                        task_id=task_id,
                        seed=seed,
                    )
                    
                    results.append({
                        "query": prompt[:100],  # First 100 chars for logging
                        "response": f"Negotiation episode {seed}",
                        "reward": reward,
                        "seed": seed,
                    })
                    
                except Exception as e:
                    logger.error(f"Episode {seed} failed: {e}")
                    results.append({
                        "query": prompt[:100],
                        "response": "Error",
                        "reward": 0.0,
                        "seed": seed,
                    })
        
        return results
    
    async def _run_episode(
        self,
        prompt: str,
        task_id: str,
        seed: int,
        max_steps: int = 12,
    ) -> float:
        """
        Run single negotiation episode.
        
        Args:
            prompt: System prompt
            task_id: Task difficulty  
            seed: Deterministic seed
            max_steps: Max steps before truncation
        
        Returns:
            Final episode reward (0.0-1.0)
        """
        
        async with EnvClient(self.server_url) as env:
            # Reset episode
            obs = await env.reset(task_id=task_id, seed=seed)
            
            total_steps = 0
            while not obs.terminal and total_steps < max_steps:
                # Generate action via model
                action = await self._generate_action(obs, prompt)
                
                # Step environment
                obs, reward, terminated, info = await env.step(action)
                
                total_steps += 1
            
            # Return final reward (0.0 during episode, final score on terminal)
            return float(reward)
    
    async def _generate_action(
        self,
        state: NegotiationState,
        system_prompt: str,
    ) -> NegotiationAction:
        """
        Generate action via LLM.
        
        Args:
            state: Current NegotiationState
            system_prompt: System context
        
        Returns:
            NegotiationAction
        """
        
        # TODO: Integrate with model.generate() for actual LLM inference
        # For now, return fallback
        return generate_fallback_action(state)
    
    def train(self, num_episodes: int = 100):
        """
        Run GRPO training loop.
        
        Args:
            num_episodes: Total episodes to run
        """
        
        logger.info(f"Starting GRPO training: {num_episodes} episodes")
        
        # Placeholder for actual training logic
        # In practice, this would:
        # 1. Sample prompts
        # 2. Run async rollouts via rollout_func()
        # 3. Compute rewards
        # 4. Call trainer.train()
        
        logger.info("GRPO training complete")
