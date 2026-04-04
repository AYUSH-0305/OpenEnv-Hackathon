"""Tests for the Medication Reconciliation Environment core logic."""

import pytest


def test_environment_reset_easy():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    env = MedReconciliationEnvironment(task="easy")
    obs = env.reset()
    assert obs.task_difficulty == "easy"
    assert len(obs.home_medications) > 0
    assert len(obs.discharge_medications) > 0
    assert obs.done is False
    assert obs.issues_found == 0
    assert obs.false_positives == 0


def test_environment_reset_medium():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    env = MedReconciliationEnvironment(task="medium")
    obs = env.reset()
    assert obs.task_difficulty == "medium"
    assert obs.total_issues == 1


def test_environment_reset_hard():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    env = MedReconciliationEnvironment(task="hard")
    obs = env.reset()
    assert obs.task_difficulty == "hard"
    assert obs.total_issues == 3


def test_correct_flag_gives_positive_reward():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    from med_reconciliation.models import MedReconciliationAction
    env = MedReconciliationEnvironment(task="easy")
    env.reset()
    obs = env.step(MedReconciliationAction(
        action_type="flag_duplicate",
        drug_a="warfarin",
        drug_b="warfarin",
        reasoning="duplicate warfarin"
    ))
    assert obs.reward > 0
    assert obs.issues_found == 1


def test_false_positive_gives_negative_reward():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    from med_reconciliation.models import MedReconciliationAction
    env = MedReconciliationEnvironment(task="easy")
    env.reset()
    obs = env.step(MedReconciliationAction(
        action_type="flag_interaction",
        drug_a="aspirin",
        drug_b="metformin",
        reasoning="wrong"
    ))
    assert obs.reward < 0
    assert obs.false_positives == 1


def test_submit_ends_episode():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    from med_reconciliation.models import MedReconciliationAction
    env = MedReconciliationEnvironment(task="easy")
    env.reset()
    obs = env.step(MedReconciliationAction(action_type="submit"))
    assert obs.done is True


def test_invalid_task_raises():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    with pytest.raises(ValueError):
        MedReconciliationEnvironment(task="impossible")


def test_state_returns_episode_id():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    env = MedReconciliationEnvironment(task="easy")
    env.reset()
    state = env.state
    assert state.episode_id is not None
    assert state.step_count >= 0


def test_reset_clears_previous_state():
    from med_reconciliation.server.environment import MedReconciliationEnvironment
    from med_reconciliation.models import MedReconciliationAction
    env = MedReconciliationEnvironment(task="easy")
    env.reset()
    env.step(MedReconciliationAction(
        action_type="flag_duplicate", drug_a="warfarin", drug_b="warfarin", reasoning="test"
    ))
    obs = env.reset()
    assert obs.issues_found == 0
    assert obs.false_positives == 0
    assert len(obs.flags_submitted) == 0
