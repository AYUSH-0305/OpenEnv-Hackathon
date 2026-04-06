"""Medication Reconciliation Environment Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import MedReconciliationAction, MedReconciliationObservation


class MedReconciliationEnv(
    EnvClient[MedReconciliationAction, MedReconciliationObservation, State]
):
    """
    Client for the Medication Reconciliation Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated session.

    Example:
        >>> with MedReconciliationEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(obs.home_medications)
        ...     print(obs.discharge_medications)
        ...
        ...     result = env.step(MedReconciliationAction(
        ...         action_type="flag_duplicate",
        ...         drug_a="metformin",
        ...         drug_b="metformin",
        ...         reasoning="Same drug appears twice in discharge list"
        ...     ))
        ...     print(result.reward)
    """

    def _step_payload(self, action: MedReconciliationAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "drug_a": action.drug_a,
            "drug_b": action.drug_b,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MedReconciliationObservation]:
        obs_data = payload.get("observation", {})
        observation = MedReconciliationObservation(
            task_id=obs_data.get("task_id", ""),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            home_medications=obs_data.get("home_medications", []),
            discharge_medications=obs_data.get("discharge_medications", []),
            flags_submitted=obs_data.get("flags_submitted", []),
            step_feedback=obs_data.get("step_feedback", ""),
            total_issues=obs_data.get("total_issues", 0),
            issues_found=obs_data.get("issues_found", 0),
            false_positives=obs_data.get("false_positives", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
