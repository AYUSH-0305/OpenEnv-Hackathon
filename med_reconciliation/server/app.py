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


# Task is configurable via environment variable — defaults to "easy"
# Read dynamically in factory so container restarts pick up changes
def _env_factory() -> MedReconciliationEnvironment:
    task = os.getenv("MED_RECON_TASK", "easy").lower()
    if task not in ("easy", "medium", "hard"):
        task = "easy"
    return MedReconciliationEnvironment(task=task)


app = create_app(
    _env_factory,
    MedReconciliationAction,
    MedReconciliationObservation,
    env_name="med_reconciliation",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, default="easy", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    os.environ["MED_RECON_TASK"] = args.task
    main(port=args.port)
