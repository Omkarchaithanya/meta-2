"""AsyncEnvClient - Async WebSocket client for OpenEnv server."""
import json
import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import websockets
from src.utils.models import NegotiationState, NegotiationAction, OfferRecord

logger = logging.getLogger(__name__)


class EnvClient:
    """
    Async WebSocket client for OpenEnv SME Negotiation environment.
    
    Provides a Gymnasium-compatible interface for agents:
    - async def reset(task_id, seed) -> NegotiationState
    - async def step(action) -> (observation, reward, terminated, info)
    
    Example:
        async with EnvClient(server_url) as env:
            obs = await env.reset(task_id="HARD", seed=42)
            action = NegotiationAction(...)
            obs, reward, done, info = await env.step(action)
    """
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8000/ws/openenv-sme",
        session_id: str = "default",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize async client.
        
        Args:
            server_url: WebSocket URL (e.g., ws://localhost:8000/ws/{session_id})
            session_id: Unique session identifier
            timeout: Timeout per step (seconds)
            max_retries: Max retries on connection failure
            retry_delay: Base delay for exponential backoff (seconds)
        """
        
        self.server_url = f"{server_url}" if "{session_id}" not in server_url else server_url.replace("{session_id}", session_id)
        self.session_id = session_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.websocket = None
        self.connected = False
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Establish WebSocket connection with exponential backoff retry."""
        retries = 0
        while retries < self.max_retries:
            try:
                logger.info(f"Connecting to {self.server_url} (attempt {retries + 1}/{self.max_retries})")
                self.websocket = await websockets.connect(self.server_url, ping_interval=20)
                self.connected = True
                logger.info(f"Connected to OpenEnv server: {self.session_id}")
                return
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    raise ConnectionError(f"Failed to connect after {self.max_retries} attempts: {e}")
                
                wait_time = self.retry_delay * (2 ** (retries - 1))
                logger.warning(f"Connection failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from server")
    
    async def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None
    ) -> NegotiationState:
        """
        Reset environment (Gymnasium-style).
        
        Args:
            task_id: "easy", "medium", or "hard"
            seed: Optional deterministic seed
        
        Returns:
            Initial observation (NegotiationState)
        """
        if not self.connected:
            await self.connect()
        
        message = {
            "type": "reset",
            "task_id": task_id,
            "seed": seed
        }
        
        async with self._lock:
            await self._send_message(message)
            response = await self._receive_message()
        
        if response.get("type") == "error":
            raise RuntimeError(f"Reset failed: {response.get('message')}")
        
        state_dict = response.get("observation", {})
        return NegotiationState(**state_dict)
    
    async def step(self, action: NegotiationAction) -> tuple:
        """
        Execute one step in negotiation (Gymnasium-style).
        
        Args:
            action: NegotiationAction instance
        
        Returns:
            (observation, reward, terminated, info)
        """
        if not self.connected:
            await self.connect()
        
        message = {
            "type": "step",
            "action": action.model_dump()
        }
        
        async with self._lock:
            await self._send_message(message)
            response = await self._receive_message()
        
        if response.get("type") == "error":
            raise RuntimeError(f"Step failed: {response.get('message')}")
        
        result = response.get("result", {})
        
        observation = NegotiationState(**result.get("observation", {}))
        reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("terminated", False))
        info = result.get("info", {})
        
        return observation, reward, terminated, info
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send JSON message to server."""
        try:
            await asyncio.wait_for(
                self.websocket.send(json.dumps(message)),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Send timeout after {self.timeout}s")
        except Exception as e:
            self.connected = False
            raise ConnectionError(f"Send failed: {e}")
    
    async def _receive_message(self) -> Dict[str, Any]:
        """Receive JSON message from server."""
        try:
            data = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.timeout
            )
            return json.loads(data)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Receive timeout after {self.timeout}s")
        except Exception as e:
            self.connected = False
            raise ConnectionError(f"Receive failed: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def close(self) -> None:
        """Alias for disconnect()."""
        await self.disconnect()
