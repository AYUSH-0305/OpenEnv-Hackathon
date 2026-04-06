"""
Medication Reconciliation Environment Implementation.

The agent receives a patient's home medication list and a hospital discharge
prescription. It must identify all discrepancies: duplicates (exact or brand/generic),
dangerous drug interactions, dose mismatches, and missing medications.

Real-world context:
    Medication reconciliation errors cause ~1.5 million patient injuries per year
    in the US. They most commonly occur during care transitions (hospital → home)
    when no one systematically compares the two medication lists.
"""

import json
import os
from typing import Any, Dict, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..graders.graders import grade_episode
    from ..models import MedReconciliationAction, MedReconciliationObservation
except ImportError:
    from graders.graders import grade_episode
    from models import MedReconciliationAction, MedReconciliationObservation

# Paths to data files
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_TASKS_DIR = os.path.join(_DATA_DIR, "tasks")
_INTERACTIONS_FILE = os.path.join(_DATA_DIR, "drug_interactions.json")

# Task difficulty order
_TASK_FILES = {
    "easy": os.path.join(_TASKS_DIR, "easy.json"),
    "medium": os.path.join(_TASKS_DIR, "medium.json"),
    "hard": os.path.join(_TASKS_DIR, "hard.json"),
}

# Reward constants
_REWARD_CORRECT_FLAG = 0.3       # correct issue identified
_REWARD_PARTIAL_FLAG = 0.1       # correct drug pair, wrong type
_REWARD_FALSE_POSITIVE = -0.15   # incorrect flag
_REWARD_SUBMIT_BONUS = 0.1       # bonus for submitting (completing episode)
_MAX_STEPS = 10                  # max steps before forced submission


def _load_json(path: str) -> Dict:
    with open(os.path.normpath(path), "r") as f:
        return json.load(f)


class MedReconciliationEnvironment(Environment):
    """
    Medication Reconciliation Environment.

    The agent sees two medication lists and must flag all discrepancies
    before calling 'submit'. Graders score performance 0.0–1.0.

    Tasks:
        easy   — one obvious exact duplicate
        medium — brand/generic duplicate (requires drug knowledge)
        hard   — dangerous drug interaction + dose mismatch across doctors

    Supports concurrent WebSocket sessions (each client gets its own instance).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy"):
        """
        Initialize the environment.

        Args:
            task: One of 'easy', 'medium', 'hard'. Defaults to 'easy'.
        """
        if task not in _TASK_FILES:
            raise ValueError(f"task must be one of {list(_TASK_FILES.keys())}, got '{task}'")

        self._task_name = task
        self._task_data: Dict[str, Any] = {}
        self._flags_submitted: List[Dict[str, Any]] = []
        self._issues_found: int = 0
        self._found_issue_indices: set = set()  # track which planted issues already found
        self._false_positives: int = 0
        self._cumulative_reward: float = 0.0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done: bool = False

        # Load drug interaction DB
        interactions_data = _load_json(_INTERACTIONS_FILE)
        self._brand_to_generic: Dict[str, str] = interactions_data.get("brand_to_generic", {})
        self._interactions: List[Dict[str, Any]] = interactions_data.get("interactions", [])

    def _load_task(self) -> Dict[str, Any]:
        return _load_json(_TASK_FILES[self._task_name])

    def _normalize(self, name: str) -> str:
        name = name.strip().lower()
        return self._brand_to_generic.get(name, name)

    def _check_flag_against_issues(
        self, action: MedReconciliationAction
    ) -> tuple[bool, bool, str]:
        """
        Check a flag against planted issues.

        Already-found issues are skipped to prevent double-counting.
        Brand names are resolved to generics before comparison.

        Returns:
            (is_correct, is_partial, feedback_message)
        """
        action_to_issue_type = {
            "flag_duplicate": "duplicate",
            "flag_interaction": "interaction",
            "flag_dose_mismatch": "dose_mismatch",
            "flag_missing": "missing",
        }
        expected_type = action_to_issue_type.get(action.action_type)
        da = self._normalize(action.drug_a)
        db = self._normalize(action.drug_b) if action.drug_b else da

        for idx, issue in enumerate(self._task_data.get("planted_issues", [])):
            if idx in self._found_issue_indices:
                continue  # already found, skip

            ia = self._normalize(issue["drug_a"])
            ib = self._normalize(issue["drug_b"]) if issue.get("drug_b") else ia

            if issue["type"] == "missing":
                drugs_match = da == ia or da == ib
            else:
                drugs_match = (da == ia and db == ib) or (da == ib and db == ia)

            if drugs_match and expected_type == issue["type"]:
                self._found_issue_indices.add(idx)
                return True, False, f"Correct: {issue['description']}"
            elif drugs_match:
                self._found_issue_indices.add(idx)
                return False, True, f"Partially correct — right drug(s), wrong issue type. Hint: {issue['description']}"

        return False, False, f"Incorrect flag — no matching issue found for '{action.drug_a}' / '{action.drug_b}'"

    def reset(self, seed: int = None, episode_id: str = None) -> MedReconciliationObservation:
        """
        Reset the environment and load a fresh task.

        The task can be overridden by passing episode_id as:
        'easy', 'medium', or 'hard'
        """
        # Allow task override via episode_id parameter
        if episode_id and episode_id in ("easy", "medium", "hard"):
            self._task_name = episode_id

        self._task_data = self._load_task()
        self._flags_submitted = []
        self._issues_found = 0
        self._found_issue_indices = set()
        self._false_positives = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return MedReconciliationObservation(
            task_id=self._task_data["task_id"],
            task_difficulty=self._task_data["difficulty"],
            home_medications=self._task_data["home_medications"],
            discharge_medications=self._task_data["discharge_medications"],
            flags_submitted=[],
            step_feedback="Episode started. Review the medication lists and flag discrepancies.",
            total_issues=self._task_data["total_issues"],
            issues_found=0,
            false_positives=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: MedReconciliationAction) -> MedReconciliationObservation:  # type: ignore[override]
        """
        Execute one step — either flag a discrepancy or submit findings.

        Args:
            action: MedReconciliationAction with action_type and drug details

        Returns:
            MedReconciliationObservation with updated state and step reward
        """
        self._state.step_count += 1
        step_reward = 0.0
        feedback = ""

        if self._done:
            return self._build_obs("Episode already complete. Call reset() to start a new episode.", 0.0)

        if action.action_type == "submit" or self._state.step_count >= _MAX_STEPS:
            # Grade the full episode
            score, details = grade_episode(
                self._flags_submitted,
                self._task_data.get("planted_issues", []),
                self._brand_to_generic,
            )
            step_reward = _REWARD_SUBMIT_BONUS + score * 0.5
            self._cumulative_reward += step_reward
            self._done = True
            feedback = (
                f"Episode complete. Final score: {score:.3f}. "
                f"Issues found: {details['issues_found']}/{details['total_issues']}. "
                f"False positives: {details['false_positives']}."
            )
            return self._build_obs(feedback, step_reward, done=True)

        # Process a flag action
        flag_record = {
            "action_type": action.action_type,
            "drug_a": action.drug_a,
            "drug_b": action.drug_b,
            "reasoning": action.reasoning,
        }
        self._flags_submitted.append(flag_record)

        is_correct, is_partial, feedback = self._check_flag_against_issues(action)

        if is_correct:
            step_reward = _REWARD_CORRECT_FLAG
            self._issues_found += 1
        elif is_partial:
            step_reward = _REWARD_PARTIAL_FLAG
            self._issues_found += 1
        else:
            step_reward = _REWARD_FALSE_POSITIVE
            self._false_positives += 1

        self._cumulative_reward += step_reward

        # Hint the agent to submit when all issues are found
        all_found = self._issues_found >= self._task_data.get("total_issues", 1)
        if all_found:
            feedback += " All issues identified — call submit to complete the episode."

        return self._build_obs(feedback, step_reward, done=False)

    def _build_obs(
        self, feedback: str, reward: float, done: bool = False
    ) -> MedReconciliationObservation:
        return MedReconciliationObservation(
            task_id=self._task_data.get("task_id", ""),
            task_difficulty=self._task_data.get("difficulty", self._task_name),
            home_medications=self._task_data.get("home_medications", []),
            discharge_medications=self._task_data.get("discharge_medications", []),
            flags_submitted=list(self._flags_submitted),
            step_feedback=feedback,
            total_issues=self._task_data.get("total_issues", 0),
            issues_found=self._issues_found,
            false_positives=self._false_positives,
            done=done,
            reward=reward,
            metadata={
                "cumulative_reward": round(self._cumulative_reward, 4),
                "step": self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        return self._state
