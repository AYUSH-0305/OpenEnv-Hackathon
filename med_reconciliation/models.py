"""
Data models for the Medication Reconciliation Environment.

The agent receives a patient's home medication list and a discharge prescription,
then must identify discrepancies: duplicates, dose mismatches, dangerous interactions,
and missing medications.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MedReconciliationAction(Action):
    """
    Action for the Medication Reconciliation environment.

    The agent issues one flag per step, identifying a specific discrepancy
    in the medication lists.

    Action types:
        - "flag_duplicate"    : Same drug appears twice (exact or brand/generic)
        - "flag_interaction"  : Two drugs have a dangerous interaction
        - "flag_dose_mismatch": Same drug has different doses across lists
        - "flag_missing"      : A medication is missing from discharge list
        - "submit"            : Agent is done — submit all findings for grading
    """

    action_type: str = Field(
        ...,
        description=(
            "Type of action: 'flag_duplicate', 'flag_interaction', "
            "'flag_dose_mismatch', 'flag_missing', or 'submit'"
        ),
    )
    drug_a: str = Field(
        default="",
        description="Primary drug name involved in the flag",
    )
    drug_b: str = Field(
        default="",
        description="Secondary drug name (required for interaction and duplicate flags)",
    )
    reasoning: str = Field(
        default="",
        description="Agent's explanation for this flag (used for partial credit)",
    )


class MedReconciliationObservation(Observation):
    """
    Observation from the Medication Reconciliation environment.

    The agent sees both medication lists and a running log of its flags.
    """

    task_id: str = Field(default="", description="Current task identifier")
    task_difficulty: str = Field(
        default="easy", description="Task difficulty: easy, medium, or hard"
    )
    home_medications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Patient's home medication list. Each entry has: name, dose, frequency",
    )
    discharge_medications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Discharge prescription list. Each entry has: name, dose, frequency",
    )
    flags_submitted: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Flags the agent has submitted so far this episode",
    )
    step_feedback: str = Field(
        default="",
        description="Feedback on the last action taken",
    )
    total_issues: int = Field(
        default=0,
        description="Total number of issues planted in this scenario (revealed at episode end)",
    )
    issues_found: int = Field(
        default=0,
        description="Number of correct issues found so far",
    )
    false_positives: int = Field(
        default=0,
        description="Number of incorrect flags submitted so far",
    )
