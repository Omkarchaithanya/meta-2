"""Application factory for the OpenEnv SME Negotiation environment."""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.env.sme_negotiation import SMENegotiationEnv
from src.utils.models import NegotiationAction
from src.utils.schemas import build_environment_schema, serialize_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_environment(request: Request) -> SMENegotiationEnv:
    environment = getattr(request.app.state, "environment", None)
    if environment is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    return environment


def _get_active_episodes(request: Request) -> dict:
    active_episodes = getattr(request.app.state, "active_episodes", None)
    if active_episodes is None:
        active_episodes = {}
        request.app.state.active_episodes = active_episodes
    return active_episodes


def create_app(environment: Optional[SMENegotiationEnv] = None) -> FastAPI:
    """Create the ASGI app using a factory pattern."""

    app = FastAPI(title="OpenEnv SME Negotiator")
    app.state.environment = environment
    app.state.active_episodes = {}

    @app.on_event("startup")
    async def startup_event() -> None:
        if app.state.environment is None:
            app.state.environment = SMENegotiationEnv()
        logger.info("OpenEnv SME Negotiation environment initialized")

    @app.get("/health")
    async def health_check() -> dict:
        return {"status": "healthy", "environment": "sme-negotiator"}

    @app.get("/state")
    async def state_snapshot(request: Request) -> dict:
        environment = _get_environment(request)
        current_state = serialize_state(environment.current_state)
        return {
            "environment": "sme-negotiator",
            "task_id": environment.task_config.task_id if environment.task_config else None,
            "episode_seed": environment.episode_seed if environment.current_state else None,
            "state": current_state.model_dump() if current_state else None,
        }

    @app.get("/schema")
    async def environment_schema(request: Request) -> dict:
        environment = _get_environment(request)
        return build_environment_schema(environment)

    @app.post("/reset")
    async def reset_episode(request: Request, task_id: str = "easy", seed: Optional[int] = None):
        environment = _get_environment(request)
        try:
            initial_state = environment.reset(task_id=task_id, seed=seed)
            episode_id = f"{task_id}_{seed or 'random'}"
            _get_active_episodes(request)[episode_id] = {
                "task_id": task_id,
                "seed": seed,
                "state": initial_state,
            }
            return JSONResponse(
                status_code=200,
                content={
                    "episode_id": episode_id,
                    "state": initial_state.model_dump(),
                },
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/step")
    async def step_episode(request: Request, action_dict: dict, episode_id: Optional[str] = None):
        environment = _get_environment(request)

        try:
            action = NegotiationAction(**action_dict)
        except ValidationError as error:
            return JSONResponse(status_code=400, content={"error": str(error), "success": False})

        try:
            observation, reward, terminated, info = environment.step(action)
            return JSONResponse(
                status_code=200,
                content={
                    "observation": observation.model_dump(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "info": info,
                },
            )
        except RuntimeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.websocket("/ws/negotiate")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        environment = getattr(app.state, "environment", None)
        if environment is None:
            await websocket.close(code=1000, reason="Environment not initialized")
            return

        await websocket.accept()
        episode_id = None

        try:
            while True:
                message = await websocket.receive_text()
                try:
                    msg_data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                action_type = msg_data.get("action")

                if action_type == "reset":
                    task_id = msg_data.get("task_id", "easy")
                    seed = msg_data.get("seed", None)
                    try:
                        initial_state = environment.reset(task_id=task_id, seed=seed)
                        episode_id = f"{task_id}_{seed or 'random'}"
                        app.state.active_episodes[episode_id] = {
                            "task_id": task_id,
                            "seed": seed,
                            "state": initial_state,
                        }
                        await websocket.send_json(
                            {
                                "type": "initialized",
                                "episode_id": episode_id,
                                "state": initial_state.model_dump(),
                            }
                        )
                    except ValueError as error:
                        await websocket.send_json({"type": "error", "message": f"Reset failed: {str(error)}"})

                elif action_type == "step":
                    move_data = msg_data.get("move", {})
                    try:
                        action = NegotiationAction(**move_data)
                    except ValidationError as error:
                        await websocket.send_json({"type": "error", "message": f"Invalid action: {str(error)}"})
                        continue

                    try:
                        observation, reward, terminated, info = environment.step(action)
                        await websocket.send_json(
                            {
                                "type": "step_result",
                                "observation": observation.model_dump(),
                                "reward": float(reward),
                                "terminated": bool(terminated),
                                "info": info,
                            }
                        )
                    except Exception as error:
                        await websocket.send_json({"type": "error", "message": f"Step failed: {str(error)}"})

                elif action_type == "status":
                    await websocket.send_json(
                        {
                            "type": "status",
                            "episode_id": episode_id,
                            "current_state": environment.current_state.model_dump() if environment.current_state else None,
                        }
                    )

                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown action: {action_type}"})
        except WebSocketDisconnect:
            logger.info("Client disconnected from episode %s", episode_id)
        except Exception as error:
            logger.error("WebSocket error: %s", str(error))
            await websocket.send_json({"type": "error", "message": "Internal server error"})

    @app.get("/tasks")
    async def list_tasks(request: Request) -> dict:
        environment = _get_environment(request)
        tasks = {}
        for task_id, config in environment.TASKS.items():
            tasks[task_id] = {
                "id": config.task_id,
                "name": config.name,
                "description": config.description,
                "difficulty": task_id,
                "max_rounds": config.max_rounds,
                "negotiation_variables": {
                    "price": True,
                    "days": True if task_id in ["medium", "hard"] else False,
                    "volume": task_id == "hard",
                    "treds": task_id == "hard",
                },
            }

        return {"tasks": tasks}

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "OpenEnv SME Negotiation Environment",
            "version": "0.1.0",
            "description": "Reinforcement Learning environment for B2B contract negotiation",
            "endpoints": {
                "health": "/health",
                "reset": "POST /reset",
                "step": "POST /step",
                "state": "GET /state",
                "schema": "GET /schema",
                "websocket": "WS /ws/negotiate",
                "tasks": "GET /tasks",
            },
            "tasks": ["easy", "medium", "hard"],
        }

    return app


app = create_app()
