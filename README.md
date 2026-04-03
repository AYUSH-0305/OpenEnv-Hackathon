# Medication Reconciliation Environment

> An OpenEnv environment where an AI agent acts as a clinical pharmacist,
> identifying dangerous discrepancies between a patient's home medications
> and their hospital discharge prescription.

## Why This Matters

Medication reconciliation errors cause **~1.5 million patient injuries per year** in the US.
They most commonly occur during care transitions — when a patient leaves the hospital and
nobody systematically compares the two medication lists. This environment trains and evaluates
agents on exactly that task.

---

## Environment Description

The agent receives two medication lists:
- **Home medications** — what the patient was taking before hospitalization
- **Discharge medications** — what the hospital prescribed on discharge

The agent must identify all discrepancies by issuing flag actions, then call `submit` when done.

---

## Action Space

| Field         | Type   | Description |
|---------------|--------|-------------|
| `action_type` | string | One of: `flag_duplicate`, `flag_interaction`, `flag_dose_mismatch`, `flag_missing`, `submit` |
| `drug_a`      | string | Primary drug name involved |
| `drug_b`      | string | Secondary drug name (for duplicate/interaction flags) |
| `reasoning`   | string | Agent's explanation (used for partial credit) |

## Observation Space

| Field                  | Type         | Description |
|------------------------|--------------|-------------|
| `task_id`              | string       | Current task identifier |
| `task_difficulty`      | string       | `easy`, `medium`, or `hard` |
| `home_medications`     | list[dict]   | Home med list: `{name, dose, frequency}` |
| `discharge_medications`| list[dict]   | Discharge list: `{name, dose, frequency}` |
| `flags_submitted`      | list[dict]   | Flags submitted so far this episode |
| `step_feedback`        | string       | Feedback on the last action |
| `total_issues`         | int          | Total planted issues (revealed at end) |
| `issues_found`         | int          | Correct issues found so far |
| `false_positives`      | int          | Incorrect flags submitted |
| `done`                 | bool         | Whether the episode is complete |
| `reward`               | float        | Step reward |

---

## Tasks

### Easy — Exact Duplicate
One drug appears twice in the discharge list with the same name and dose.
Expected difficulty: any LLM should catch this.
Baseline score: ~0.9

### Medium — Brand/Generic Duplicate
The same drug appears under its brand name on one list and generic name on the other
(e.g. Zoloft + sertraline). Agent must recognize pharmaceutical equivalence.
Baseline score: ~0.6

### Hard — Drug Interaction + Dose Mismatch
Two drugs prescribed by different doctors have a dangerous interaction (serotonin syndrome risk).
Additionally, a dose was silently doubled. Agent must have pharmacological knowledge.
Baseline score: ~0.3

---

## Reward Function

| Event | Reward |
|-------|--------|
| Correct flag (right drugs + right type) | +0.30 |
| Partial flag (right drugs, wrong type) | +0.10 |
| False positive (wrong flag) | -0.15 |
| Submit bonus | +0.10 |

Rewards are dense — the agent gets signal on every step, not just at the end.

---

## Setup & Usage

### Prerequisites
```bash
pip install openenv-core
```

### Run locally
```bash
cd med_reconciliation
MED_RECON_TASK=easy uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -t med-reconciliation-env ./med_reconciliation/server
docker run -p 8000:8000 -e MED_RECON_TASK=easy med-reconciliation-env
```

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

| Task   | Model                    | Score |
|--------|--------------------------|-------|
| Easy   | Qwen2.5-72B-Instruct     | ~0.90 |
| Medium | Qwen2.5-72B-Instruct     | ~0.60 |
| Hard   | Qwen2.5-72B-Instruct     | ~0.30 |

---

## Validation

Run the pre-submission validator before submitting:
```bash
./scripts/validate-submission.sh https://your-space.hf.space
```

All 3 checks must pass: HF Space live, Docker build, `openenv validate`.
