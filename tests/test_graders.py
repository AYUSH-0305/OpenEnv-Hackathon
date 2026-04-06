"""Tests for the Medication Reconciliation graders."""

import pytest
from med_reconciliation.graders.graders import grade_episode

BRAND_MAP = {
    "coumadin": "warfarin",
    "ultram": "tramadol",
    "lopressor": "metoprolol",
    "lanoxin": "digoxin",
    "zoloft": "sertraline",
}


def test_easy_correct_duplicate():
    flags = [{"action_type": "flag_duplicate", "drug_a": "warfarin", "drug_b": "warfarin", "reasoning": "dup"}]
    issues = [{"type": "duplicate", "drug_a": "warfarin", "drug_b": "warfarin", "description": "test"}]
    score, details = grade_episode(flags, issues, BRAND_MAP)
    assert score == 1.0
    assert details["false_positives"] == 0


def test_medium_brand_generic_duplicate():
    flags = [{"action_type": "flag_duplicate", "drug_a": "ultram", "drug_b": "tramadol", "reasoning": "brand"}]
    issues = [{"type": "duplicate", "drug_a": "ultram", "drug_b": "tramadol", "description": "test"}]
    score, details = grade_episode(flags, issues, BRAND_MAP)
    assert score == 1.0


def test_hard_all_three_issues():
    flags = [
        {"action_type": "flag_interaction", "drug_a": "coumadin", "drug_b": "aspirin", "reasoning": "bleed"},
        {"action_type": "flag_dose_mismatch", "drug_a": "digoxin", "drug_b": "digoxin", "reasoning": "dose"},
        {"action_type": "flag_missing", "drug_a": "metoprolol", "drug_b": "", "reasoning": "missing"},
    ]
    issues = [
        {"type": "interaction", "drug_a": "coumadin", "drug_b": "aspirin", "description": "test"},
        {"type": "dose_mismatch", "drug_a": "digoxin", "drug_b": "digoxin", "description": "test"},
        {"type": "missing", "drug_a": "metoprolol", "drug_b": "metoprolol", "description": "test"},
    ]
    score, details = grade_episode(flags, issues, BRAND_MAP)
    assert score == 1.0
    assert details["issues_found"] == 3
    assert details["false_positives"] == 0


def test_false_positive_penalized():
    flags = [{"action_type": "flag_interaction", "drug_a": "aspirin", "drug_b": "metformin", "reasoning": "wrong"}]
    issues = [{"type": "duplicate", "drug_a": "warfarin", "drug_b": "warfarin", "description": "test"}]
    score, details = grade_episode(flags, issues, BRAND_MAP)
    assert score == 0.0
    assert details["false_positives"] == 1


def test_partial_credit_wrong_type():
    flags = [{"action_type": "flag_duplicate", "drug_a": "coumadin", "drug_b": "aspirin", "reasoning": "wrong type"}]
    issues = [{"type": "interaction", "drug_a": "coumadin", "drug_b": "aspirin", "description": "test"}]
    score, details = grade_episode(flags, issues, BRAND_MAP)
    # partial credit = 0.3 per issue (raw_score = 0.3/1 = 0.3, no penalty)
    assert score == pytest.approx(0.3)
    assert details["partial_credits"] == 1
    assert details["false_positives"] == 0


def test_score_clamped_to_zero_on_excess_false_positives():
    flags = [
        {"action_type": "flag_interaction", "drug_a": "x", "drug_b": "y", "reasoning": "wrong"},
        {"action_type": "flag_interaction", "drug_a": "a", "drug_b": "b", "reasoning": "wrong"},
        {"action_type": "flag_interaction", "drug_a": "c", "drug_b": "d", "reasoning": "wrong"},
    ]
    issues = [{"type": "duplicate", "drug_a": "warfarin", "drug_b": "warfarin", "description": "test"}]
    score, _ = grade_episode(flags, issues, BRAND_MAP)
    assert score == 0.0


def test_empty_flags_scores_zero():
    issues = [{"type": "duplicate", "drug_a": "warfarin", "drug_b": "warfarin", "description": "test"}]
    score, details = grade_episode([], issues, BRAND_MAP)
    assert score == 0.0


def test_submit_action_ignored():
    flags = [
        {"action_type": "submit", "drug_a": "", "drug_b": "", "reasoning": "done"},
    ]
    issues = [{"type": "duplicate", "drug_a": "warfarin", "drug_b": "warfarin", "description": "test"}]
    score, details = grade_episode(flags, issues, BRAND_MAP)
    assert score == 0.0
    assert details["false_positives"] == 0
