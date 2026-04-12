"""
FastAPI application for the Medication Reconciliation Environment.

Exposes the MedReconciliationEnvironment over HTTP and WebSocket endpoints,
compatible with the OpenEnv EnvClient.

Endpoints:
    POST /reset   — Reset the environment, returns initial observation
    POST /step    — Execute an action, returns observation + reward
    GET  /state   — Get current environment state
    GET  /schema  — Get action/observation schemas
    WS   /ws      — WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os
from urllib.parse import urlparse, parse_qs

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import MedReconciliationAction, MedReconciliationObservation
    from .environment import MedReconciliationEnvironment
except (ImportError, ModuleNotFoundError):
    from models import MedReconciliationAction, MedReconciliationObservation
    from server.environment import MedReconciliationEnvironment


def _env_factory() -> MedReconciliationEnvironment:
    """Create environment — task controlled by MED_RECON_TASK env var."""
    task = os.getenv("MED_RECON_TASK", "easy").lower()
    if task not in ("easy", "medium", "hard", "control"):
        task = "easy"
    return MedReconciliationEnvironment(task=task)


# Override the reset endpoint to support task selection via request body
from fastapi import Request
from fastapi.responses import JSONResponse

app = create_app(
    _env_factory,
    MedReconciliationAction,
    MedReconciliationObservation,
    env_name="med_reconciliation",
    max_concurrent_envs=10,
)


# Add /tasks endpoint so validators can enumerate tasks with graders
@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Easy — Exact Duplicate",
                "description": "Patient on warfarin has it listed twice in discharge. Double dosing a blood thinner = fatal bleeding risk.",
                "difficulty": "easy",
                "has_grader": True,
                "max_score": 1.0,
                "min_score": 0.0,
            },
            {
                "id": "medium",
                "name": "Medium — Brand/Generic Duplicate",
                "description": "Patient takes Ultram (brand) at home, hospital prescribes tramadol (generic). Same drug, double dose with serotonin syndrome risk.",
                "difficulty": "medium",
                "has_grader": True,
                "max_score": 1.0,
                "min_score": 0.0,
            },
            {
                "id": "hard",
                "name": "Hard — Three Hidden Issues",
                "description": "Three dangerous issues: Coumadin+aspirin interaction, digoxin dose doubled, metoprolol missing from discharge.",
                "difficulty": "hard",
                "has_grader": True,
                "max_score": 1.0,
                "min_score": 0.0,
            },
            {
                "id": "control",
                "name": "Control — No Issues",
                "description": "Control task with no medication errors to detect false positives.",
                "difficulty": "control",
                "has_grader": True,
                "max_score": 1.0,
                "min_score": 0.0,
            },
        ]
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    os.environ["MED_RECON_TASK"] = args.task
    main()
