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
| Correct flag — life-threatening issue (interaction, dose mismatch) | +0.50 |
| Correct flag — other issue (duplicate, missing) | +0.30 |
| Partial flag (right drugs, wrong type) | +0.10 |
| False positive (wrong flag) | -0.15 |
| Submit bonus (completing episode cleanly) | +0.10 |

Rewards are **severity-weighted** — catching a warfarin+aspirin interaction (which can cause fatal hemorrhage) scores higher than catching a simple duplicate. This creates a meaningful gradient that rewards clinical reasoning, not just pattern matching.

False positives are penalized to discourage random flagging. The submit bonus rewards clean episode completion.

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
| Easy | Qwen/Qwen2.5-72B-Instruct | 0.900 | 2 |
| Medium | Qwen/Qwen2.5-72B-Instruct | 0.700 | 3 |
| Hard | Qwen/Qwen2.5-72B-Instruct | 1.000 | 4 |

Scores normalized to [0.0, 1.0]. Success threshold is 0.5.
Medium score of 0.700 reflects a false positive — the model correctly identified the brand/generic duplicate but also flagged a tramadol+sertraline interaction that, while clinically valid, was not a planted issue in this scenario.
The hard task now requires catching a warfarin+aspirin interaction (via brand name Coumadin), a digoxin dose doubling, and a missing beta-blocker — all 3 found correctly.
