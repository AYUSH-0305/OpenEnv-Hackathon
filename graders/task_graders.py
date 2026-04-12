"""
Task-specific grader classes for the Medication Reconciliation Environment.
Each grader class is referenced in openenv.yaml and instantiated by the validator.
"""

from typing import Any, Dict


class EasyGrader:
    """
    Grader for the Easy task — exact duplicate detection.
    Scores agent performance on identifying warfarin duplicate.
    Returns a score strictly in (0, 1).
    """

    def grade(self, sample: Dict[str, Any], item: Dict[str, Any] = None) -> float:
        """
        Grade the agent's performance.

        Args:
            sample: Agent's submission containing flags and actions
            item: Task item (optional)

        Returns:
            Score strictly between 0 and 1
        """
        flags = sample.get("flags", []) or sample.get("flags_submitted", [])
        if not flags:
            return 0.01

        for flag in flags:
            action = flag.get("action_type", "")
            drug_a = flag.get("drug_a", "").lower().strip()
            drug_b = flag.get("drug_b", "").lower().strip()
            if action == "flag_duplicate" and drug_a == "warfarin" and drug_b == "warfarin":
                return 0.99

        return 0.01

    def __call__(self, sample: Dict[str, Any], item: Dict[str, Any] = None) -> float:
        return self.grade(sample, item)


class MediumGrader:
    """
    Grader for the Medium task — brand/generic duplicate detection.
    Scores agent performance on identifying Ultram/tramadol equivalence.
    Returns a score strictly in (0, 1).
    """

    BRAND_MAP = {"ultram": "tramadol", "tramadol": "tramadol"}

    def _norm(self, name: str) -> str:
        return self.BRAND_MAP.get(name.lower().strip(), name.lower().strip())

    def grade(self, sample: Dict[str, Any], item: Dict[str, Any] = None) -> float:
        flags = sample.get("flags", []) or sample.get("flags_submitted", [])
        if not flags:
            return 0.01

        for flag in flags:
            action = flag.get("action_type", "")
            drug_a = self._norm(flag.get("drug_a", ""))
            drug_b = self._norm(flag.get("drug_b", ""))
            if action == "flag_duplicate" and "tramadol" in (drug_a, drug_b):
                return 0.99

        return 0.01

    def __call__(self, sample: Dict[str, Any], item: Dict[str, Any] = None) -> float:
        return self.grade(sample, item)


class HardGrader:
    """
    Grader for the Hard task — three hidden issues.
    Scores agent performance on identifying all three life-threatening discrepancies.
    Returns a score strictly in (0, 1).
    """

    BRAND_MAP = {"coumadin": "warfarin"}

    def _norm(self, name: str) -> str:
        n = name.lower().strip()
        return self.BRAND_MAP.get(n, n)

    def grade(self, sample: Dict[str, Any], item: Dict[str, Any] = None) -> float:
        flags = sample.get("flags", []) or sample.get("flags_submitted", [])
        if not flags:
            return 0.01

        found_interaction = False
        found_dose_mismatch = False
        found_missing = False
        false_positives = 0

        for flag in flags:
            action = flag.get("action_type", "")
            drug_a = self._norm(flag.get("drug_a", ""))
            drug_b = self._norm(flag.get("drug_b", ""))

            if action == "flag_interaction":
                drugs = {drug_a, drug_b}
                if "warfarin" in drugs and "aspirin" in drugs:
                    found_interaction = True
                else:
                    false_positives += 1
            elif action == "flag_dose_mismatch":
                if "digoxin" in (drug_a, drug_b):
                    found_dose_mismatch = True
                else:
                    false_positives += 1
            elif action == "flag_missing":
                if "metoprolol" in (drug_a, drug_b):
                    found_missing = True
                else:
                    false_positives += 1
            elif action not in ("submit", "flag_duplicate"):
                false_positives += 1

        issues_found = sum([found_interaction, found_dose_mismatch, found_missing])
        raw_score = issues_found / 3.0
        penalty = false_positives * 0.1
        score = max(0.01, min(0.99, raw_score - penalty))
        return score

    def __call__(self, sample: Dict[str, Any], item: Dict[str, Any] = None) -> float:
        return self.grade(sample, item)
