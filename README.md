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

Action type meanings:
- `flag_duplicate` — same drug appears twice (exact name or brand/generic equivalent)
- `flag_interaction` — two drugs together cause a dangerous interaction
- `flag_dose_mismatch` — same drug has different doses between the two lists
- `flag_missing` — a home medication is absent from the discharge list
- `submit` — agent is done, submit all findings for final grading

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

Patient on warfarin (blood thinner) has it listed twice in the discharge prescription.
A double dose of warfarin can cause fatal internal bleeding.

Patient context: 72-year-old male with atrial fibrillation admitted for knee replacement surgery.

Planted issue: `duplicate` — warfarin appears twice in discharge list.

Expected baseline score: ~0.90

---

### Medium — Brand/Generic Duplicate (1 issue)

Patient takes Ultram (brand name for tramadol) at home for chronic back pain.
Hospital discharge adds tramadol by generic name — creating a duplicate.
The patient is also on sertraline (SSRI), making a tramadol double-dose life-threatening
via serotonin syndrome.

Patient context: 58-year-old female with chronic back pain and depression admitted for lumbar surgery.
Pain management prescribed tramadol post-operatively without checking the home med list.

Planted issue: `duplicate` — Ultram = tramadol (brand/generic equivalence required).

Expected baseline score: ~0.60

---

### Hard — Three Hidden Issues (3 issues)

Three dangerous discrepancies hidden across a complex medication list, each written by a
different specialist who didn't cross-check the others.

Patient context: 81-year-old female with atrial fibrillation, heart failure, and diabetes
admitted for hip fracture repair. Cardiologist, orthopedic surgeon, and hospitalist each
updated the medication list independently.

Planted issues:

1. **Interaction** — Coumadin (brand name for warfarin) + aspirin prescribed together.
   Warfarin + aspirin = major bleeding interaction. Agent must recognize Coumadin = warfarin.

2. **Dose mismatch** — Digoxin dose silently doubled from 0.125mg to 0.25mg.
   Digoxin has an extremely narrow therapeutic index — doubling the dose causes
   bradycardia, heart block, and potentially fatal arrhythmia.

3. **Missing medication** — Metoprolol (beta-blocker) completely absent from discharge list.
   Abrupt beta-blocker withdrawal causes rebound tachycardia, hypertension, and can
   precipitate a myocardial infarction.

Expected baseline score: ~0.30

---

## Reward Function

| Event | Reward |
|-------|--------|
| Correct flag (right drugs + right type) | +0.30 |
| Partial flag (right drugs, wrong type) | +0.10 |
| False positive (wrong flag) | -0.15 |
| Submit bonus (completing episode) | +0.10 |

Rewards are dense — the agent receives a signal on every step.
False positives are penalized to discourage random flagging.
The submit bonus encourages the agent to complete the episode cleanly.

Maximum steps per episode: 10

---

## Environment API

```python
from med_reconciliation import MedReconciliationEnv, MedReconciliationAction

# Connect to running server
async with MedReconciliationEnv(base_url="http://localhost:8000") as env:
    result = await env.reset()
    obs = result.observation

    # Agent flags a duplicate
    result = await env.step(MedReconciliationAction(
        action_type="flag_duplicate",
        drug_a="warfarin",
        drug_b="warfarin",
        reasoning="Warfarin appears twice in the discharge list"
    ))
    print(result.reward)   # 0.3
    print(result.done)     # False

    # Agent submits
    result = await env.step(MedReconciliationAction(action_type="submit"))
    print(result.done)     # True
```

---

## Setup & Usage

### Prerequisites

```bash
pip install openenv-core pydantic openai
```

### Run locally (without Docker)

```bash
cd med_reconciliation
MED_RECON_TASK=easy uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t med-recon-env ./med_reconciliation/server
docker run -p 8000:8000 -e MED_RECON_TASK=easy med-recon-env
```

Change `MED_RECON_TASK` to `medium` or `hard` to switch tasks.

### Run inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
export MED_RECON_TASK="easy"

python inference.py
```

---

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| Easy | Qwen2.5-72B-Instruct | TBD |
| Medium | Qwen2.5-72B-Instruct | TBD |
| Hard | Qwen2.5-72B-Instruct | TBD |

Baseline scores will be updated after inference runs are complete.

---

## Validation

Run the pre-submission validator before submitting:

```bash
./scripts/validate-submission.sh https://your-space.hf.space
```

All 3 checks must pass:
1. HF Space live — POST `/reset` returns HTTP 200
2. Docker build succeeds
3. `openenv validate` passes

---

## Project Structure

```
OpenEnv-Hackathon/
├── inference.py                      # Mandatory root-level inference script
├── README.md                         # This file
├── CONTRIBUTING.md                   # Team contributor guide
├── requirements.txt
└── med_reconciliation/
    ├── __init__.py
    ├── models.py                     # Pydantic Action + Observation types
    ├── client.py                     # EnvClient wrapper
    ├── openenv.yaml                  # OpenEnv spec config
    ├── pyproject.toml
    ├── data/
    │   ├── drug_interactions.json    # Drug interaction DB + brand/generic map
    │   └── tasks/
    │       ├── easy.json             # Exact duplicate scenario
    │       ├── medium.json           # Brand/generic duplicate scenario
    │       └── hard.json             # 3-issue complex scenario
    ├── graders/
    │   └── graders.py                # Deterministic 0.0–1.0 scoring
    └── server/
        ├── environment.py            # Core reset/step/state logic
        ├── app.py                    # FastAPI server
        ├── Dockerfile
        └── requirements.txt
```
