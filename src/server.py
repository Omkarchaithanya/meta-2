"""FastAPI server for OpenEnv SME Negotiation environment."""
import json
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.env.sme_negotiation import SMENegotiationEnv
from src.utils.models import NegotiationAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenEnv SME Negotiator")

# Global environment instance
env: Optional[SMENegotiationEnv] = None
active_episodes = {}


@app.on_event("startup")
async def startup_event():
    """Initialize environment on server startup."""
    global env
    env = SMENegotiationEnv()
    logger.info("OpenEnv SME Negotiation environment initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "sme-negotiator"}


@app.post("/reset")
async def reset_episode(task_id: str = "easy", seed: Optional[int] = None):
    """
    HTTP endpoint to reset a new episode.
    
    Args:
        task_id: Task difficulty ("easy", "medium", "hard")
        seed: Optional deterministic seed
    
    Returns:
        Initial observation state
    """
    
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    try:
        initial_state = env.reset(task_id=task_id, seed=seed)
        
        # Store episode metadata
        episode_id = f"{task_id}_{seed or 'random'}"
        active_episodes[episode_id] = {
            "task_id": task_id,
            "seed": seed,
            "state": initial_state,
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "episode_id": episode_id,
                "state": initial_state.model_dump(),
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step_episode(action_dict: dict, episode_id: Optional[str] = None):
    """
    HTTP endpoint for stepping through environment.
    
    Args:
        action_dict: Action parameters as JSON
        episode_id: Optional episode identifier
    
    Returns:
        Next observation, reward, terminated flag, and info
    """
    
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    try:
        action = NegotiationAction(**action_dict)
    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "success": False}
        )
    
    try:
        observation, reward, terminated, info = env.step(action)
        
        return JSONResponse(
            status_code=200,
            content={
                "observation": observation.model_dump(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "info": info,
            }
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/negotiate")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time negotiation.
    
    Protocol:
    1. Client sends: {"action": "reset", "task_id": "easy", "seed": 42}
    2. Server responds: {"type": "state", "data": {...}}
    3. Client sends: {"action": "step", "move": {...}}
    4. Server responds: {"type": "result", "reward": 0.5, "done": false, data": {...}}
    """
    
    if env is None:
        await websocket.close(code=1000, reason="Environment not initialized")
        return
    
    await websocket.accept()
    episode_id = None
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            
            try:
                msg_data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                continue
            
            action_type = msg_data.get("action")
            
            # Handle reset
            if action_type == "reset":
                task_id = msg_data.get("task_id", "easy")
                seed = msg_data.get("seed", None)
                
                try:
                    initial_state = env.reset(task_id=task_id, seed=seed)
                    episode_id = f"{task_id}_{seed or 'random'}"
                    
                    await websocket.send_json({
                        "type": "initialized",
                        "episode_id": episode_id,
                        "state": initial_state.model_dump(),
                    })
                except ValueError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Reset failed: {str(e)}"
                    })
            
            # Handle step
            elif action_type == "step":
                move_data = msg_data.get("move", {})
                
                try:
                    action = NegotiationAction(**move_data)
                except ValidationError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid action: {str(e)}"
                    })
                    continue
                
                try:
                    observation, reward, terminated, info = env.step(action)
                    
                    await websocket.send_json({
                        "type": "step_result",
                        "observation": observation.model_dump(),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "info": info,
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Step failed: {str(e)}"
                    })
            
            # Handle status query
            elif action_type == "status":
                await websocket.send_json({
                    "type": "status",
                    "episode_id": episode_id,
                    "current_state": env.current_state.model_dump() if env.current_state else None,
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from episode {episode_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": "Internal server error"
        })


@app.get("/tasks")
async def list_tasks():
    """Return available tasks and their configurations."""
    
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    tasks = {}
    for task_id, config in env.TASKS.items():
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
async def root():
    """Root endpoint with API information."""
    return {
        "name": "OpenEnv SME Negotiation Environment",
        "version": "0.1.0",
        "description": "Reinforcement Learning environment for B2B contract negotiation",
        "endpoints": {
            "health": "/health",
            "reset": "POST /reset",
            "step": "POST /step",
            "websocket": "WS /ws/negotiate",
            "tasks": "GET /tasks",
        },
        "tasks": ["easy", "medium", "hard"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
