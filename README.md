---
title: Medication Reconciliation
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Medication Reconciliation — OpenEnv Environment

> An OpenEnv environment where an AI agent acts as a clinical pharmacist,
> identifying dangerous discrepancies between a patient's home medications
> and their hospital discharge prescription.

## Why This Matters

Medication reconciliation errors cause **~1.5 million patient injuries per year** in the US.
They most commonly occur during care transitions — when a patient leaves the hospital and
nobody systematically compares the two medication lists. This is a real problem that costs
lives and billions in preventable harm annually.

This environment trains and evaluates AI agents on exactly that task — giving the RL/agent
community a medically grounded, high-stakes benchmark that doesn't exist anywhere else.

---

## The Task

The agent receives two medication lists:

- **Home medications** — what the patient was taking before hospitalization
- **Discharge medications** — what the hospital prescribed on discharge

The agent must identify all discrepancies by issuing flag actions, then call `submit` when done.
Rewards are given at every step — the agent gets signal on each flag, not just at the end.

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | string | One of: `flag_duplicate`, `flag_interaction`, `flag_dose_mismatch`, `flag_missing`, `submit` |
| `drug_a` | string | Primary drug name involved |
| `drug_b` | string | Secondary drug name (for duplicate/interaction flags) |
| `reasoning` | string | Agent's explanation — used for partial credit scoring |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `task_difficulty` | string | `easy`, `medium`, or `hard` |
| `home_medications` | list[dict] | Home med list: `{name, dose, frequency}` |
| `discharge_medications` | list[dict] | Discharge list: `{name, dose, frequency}` |
| `flags_submitted` | list[dict] | All flags submitted so far this episode |
| `step_feedback` | string | Feedback on the last action taken |
| `total_issues` | int | Total planted issues in this scenario |
| `issues_found` | int | Correct issues identified so far |
| `false_positives` | int | Incorrect flags submitted so far |
| `done` | bool | Whether the episode is complete |
| `reward` | float | Step reward |

---

## Tasks

### Easy — Exact Duplicate (1 issue)
Patient on warfarin has it listed twice in discharge. Double dosing a blood thinner = fatal bleeding risk.

### Medium — Brand/Generic Duplicate (1 issue)
Patient takes Ultram (brand) at home, hospital prescribes tramadol (generic) — same drug, double dose.
Patient is also on sertraline (SSRI), making the duplicate life-threatening via serotonin syndrome.

### Hard — Three Hidden Issues (3 issues)
1. Coumadin + aspirin interaction (Coumadin = warfarin, major bleeding risk)
2. Digoxin dose doubled 0.125mg → 0.25mg (narrow therapeutic index, fatal arrhythmia risk)
3. Metoprolol missing from discharge (abrupt beta-blocker withdrawal = heart attack risk)

---

## Reward Function

| Event | Reward |
|-------|--------|
| Correct flag | +0.30 |
| Partial flag (right drugs, wrong type) | +0.10 |
| False positive | -0.15 |
| Submit bonus | +0.10 |

---

## Setup

```bash
pip install openenv-core pydantic openai
docker build -t med-recon-env .
docker run -p 7860:7860 -e MED_RECON_TASK=easy med-recon-env
```

## Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token"
export MED_RECON_TASK="all"
python inference.py
```

## Baseline Scores

| Task | Model | Score | Steps |
|------|-------|-------|-------|
| Easy | Qwen/Qwen2.5-72B-Instruct | 0.300 | 1 |
| Medium | Qwen/Qwen2.5-72B-Instruct | 0.300 | 1 |
| Hard | Qwen/Qwen2.5-72B-Instruct | 0.300 | 1 |

Scores normalized to [0.0, 1.0]. Success threshold is 0.5.
The hard task has 3 planted issues — a perfect score requires identifying all 3.
