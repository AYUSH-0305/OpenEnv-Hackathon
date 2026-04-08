"""
Deterministic graders for the Medication Reconciliation Environment.

Each grader takes the list of flags submitted by the agent and the
planted issues for the current task, and returns a score in [0.0, 1.0].

Scoring logic:
    - Base score = issues_correctly_identified / total_issues
    - False positive penalty = 0.1 per incorrect flag (min score 0.0)
    - Partial credit for correct drug pair but wrong action type = 0.3 per issue

All graders are deterministic and reproducible.
"""
from typing import Any, Dict, List, Tuple
from openenv.core.env_server import grader

def _normalize(name: str, brand_to_generic: Dict[str, str]) -> str:
    """Lowercase and resolve brand name to generic."""
    name = name.strip().lower()
    return brand_to_generic.get(name, name)


def _drugs_match(
    flag_a: str,
    flag_b: str,
    issue_a: str,
    issue_b: str,
    brand_to_generic: Dict[str, str],
) -> bool:
    """Check if the drug pair in a flag matches the planted issue (order-insensitive)."""
    fa = _normalize(flag_a, brand_to_generic)
    fb = _normalize(flag_b, brand_to_generic)
    ia = _normalize(issue_a, brand_to_generic)
    ib = _normalize(issue_b, brand_to_generic)
    return (fa == ia and fb == ib) or (fa == ib and fb == ia)

@grader("easy")
@grader("medium")
@grader("hard")

def grade_episode(
    flags: List[Dict[str, Any]],
    planted_issues: List[Dict[str, Any]],
    brand_to_generic: Dict[str, str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade a completed episode.

    Args:
        flags: List of flags submitted by the agent. Each flag has:
               action_type, drug_a, drug_b, reasoning
        planted_issues: Ground truth issues for the task. Each issue has:
                        type, drug_a, drug_b, description
        brand_to_generic: Mapping of brand names to generic names

    Returns:
        Tuple of (score: float in [0.0, 1.0], details: dict with breakdown)
    """
    if not planted_issues:
        # Control task — no issues to find. Penalize any false positives.
        false_positives = sum(1 for f in flags if f.get("action_type") not in ("submit", ""))
        penalty = false_positives * 0.1
        score = max(0.0, 1.0 - penalty)
        return round(score, 4), {
            "total_issues": 0,
            "issues_found": 0,
            "full_credits": 0,
            "partial_credits": 0,
            "false_positives": false_positives,
            "raw_score": 1.0,
            "penalty": round(penalty, 4),
            "final_score": round(score, 4),
            "explanations": {
                "false_positives": [f"False positive: flagged {f.get('drug_a')}/{f.get('drug_b')} but this patient has no medication errors" for f in flags if f.get("action_type") not in ("submit", "")],
                "missed_issues": [],
            },
        }

    total_issues = len(planted_issues)
    matched_issues = set()  # indices of planted issues that were correctly found
    false_positives = 0
    partial_credits = 0
    fp_explanations = []  # human-readable false positive explanations

    # Map action_type to issue type
    action_to_issue_type = {
        "flag_duplicate": "duplicate",
        "flag_interaction": "interaction",
        "flag_dose_mismatch": "dose_mismatch",
        "flag_missing": "missing",
    }

    for flag in flags:
        action_type = flag.get("action_type", "")
        drug_a = flag.get("drug_a", "")
        drug_b = flag.get("drug_b", "")

        if action_type == "submit":
            continue

        expected_issue_type = action_to_issue_type.get(action_type)
        found_match = False

        for idx, issue in enumerate(planted_issues):
            if idx in matched_issues:
                continue

            issue_type = issue["type"]
            ia = _normalize(issue["drug_a"], brand_to_generic)
            # For missing issues, drug_b may equal drug_a — normalize carefully
            ib = _normalize(issue.get("drug_b", ""), brand_to_generic) if issue.get("drug_b") else ia

            da = _normalize(drug_a, brand_to_generic)
            # For flag_missing, drug_b may be empty — treat as same as drug_a
            db = _normalize(drug_b, brand_to_generic) if drug_b else da

            # For missing issues, only drug_a needs to match
            if issue_type == "missing":
                drugs_ok = da == ia or da == ib
            else:
                drugs_ok = (da == ia and db == ib) or (da == ib and db == ia)

            if drugs_ok and expected_issue_type == issue_type:
                matched_issues.add(idx)
                found_match = True
                break
            elif drugs_ok and expected_issue_type != issue_type:
                matched_issues.add(idx)
                partial_credits += 1
                found_match = True
                break

        if not found_match:
            false_positives += 1
            # Record what the false positive was for explainability
            fp_explanations.append(
                f"False positive: '{action_type}' on {drug_a}/{drug_b} — no matching planted issue"
            )

    full_credits = len(matched_issues) - partial_credits
    raw_score = (full_credits + partial_credits * 0.3) / total_issues
    penalty = false_positives * 0.1
    final_score = max(0.0, min(1.0, raw_score - penalty))

    # Find missed issues for explainability
    missed_issues = []
    for idx, issue in enumerate(planted_issues):
        if idx not in matched_issues:
            missed_issues.append(
                f"Missed: {issue['type']} — {issue['description']}"
            )

    details = {
        "total_issues": total_issues,
        "issues_found": len(matched_issues),
        "full_credits": full_credits,
        "partial_credits": partial_credits,
        "false_positives": false_positives,
        "raw_score": round(raw_score, 4),
        "penalty": round(penalty, 4),
        "final_score": round(final_score, 4),
        "explanations": {
            "false_positives": fp_explanations,
            "missed_issues": missed_issues,
        },
    }

    return round(final_score, 4), details
