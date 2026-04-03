# Contributor Guide — Medication Reconciliation OpenEnv

## Branch Strategy

We work on the `development` branch only. Nobody touches `main` until the project is complete and all 3 parts are tested and working.

```
main          ← final submission only, do NOT push here directly
development   ← all active work happens here
```

### Setup (do this once)

```bash
git clone <repo-url>
cd OpenEnv-Hackathon
git checkout development
```

If the `development` branch doesn't exist yet:
```bash
git checkout -b development
git push -u origin development
```

### Daily workflow

```bash
# Before starting any work — always pull first
git pull origin development

# After finishing your work
git add .
git commit -m "person1: brief description of what you did"
git push origin development
```

Commit message format:
- `person1: add hard task scenario`
- `person2: fix grader scoring logic`
- `person3: update inference logging format`

---

## Current Status

| File | Status | Owner |
|------|--------|-------|
| `models.py` | COMPLETE & LOCKED | Person 1 |
| `server/environment.py` | COMPLETE & LOCKED | Person 1 |
| `data/drug_interactions.json` | COMPLETE & LOCKED | Person 1 |
| `data/tasks/easy.json` | COMPLETE & LOCKED | Person 1 |
| `data/tasks/medium.json` | COMPLETE & LOCKED | Person 1 |
| `data/tasks/hard.json` | COMPLETE & LOCKED | Person 1 |
| `graders/graders.py` | COMPLETE & LOCKED | Person 1/2 |
| `server/app.py` | IN PROGRESS | Person 2 |
| `server/Dockerfile` | IN PROGRESS | Person 2 |
| `openenv.yaml` | IN PROGRESS | Person 2 |
| `inference.py` | IN PROGRESS | Person 3 |
| `README.md` | IN PROGRESS | Person 3 |

---

## Repo Structure — Do Not Change

```
OpenEnv-Hackathon/
├── inference.py                 ← Person 3 owns this
├── README.md                    ← Person 3 owns this
├── CONTRIBUTING.md              ← this file, nobody edits
├── requirements.txt             ← Person 3 owns this
└── med_reconciliation/
    ├── __init__.py              ← do not touch
    ├── models.py                ← Person 1 owns this — LOCKED
    ├── client.py                ← do not touch
    ├── openenv.yaml             ← Person 2 owns this
    ├── pyproject.toml           ← Person 2 owns this
    ├── data/
    │   ├── drug_interactions.json   ← Person 1 owns this — LOCKED
    │   └── tasks/
    │       ├── easy.json            ← Person 1 owns this — LOCKED
    │       ├── medium.json          ← Person 1 owns this — LOCKED
    │       └── hard.json            ← Person 1 owns this — LOCKED
    ├── graders/
    │   ├── __init__.py          ← do not touch
    │   └── graders.py           ← LOCKED — tested and verified
    └── server/
        ├── __init__.py          ← do not touch
        ├── environment.py       ← Person 1 owns this — LOCKED
        ├── app.py               ← Person 2 owns this
        ├── Dockerfile           ← Person 2 owns this
        └── requirements.txt     ← Person 2 owns this
```

**Golden rule: only edit files you own. If you need to change someone else's file, message the team first.**

---

## What Has Been Built (Person 1 — COMPLETE)

All core environment logic is done and tested. Here is what exists:

**3 Task Scenarios:**

- `easy.json` — Patient on warfarin has it listed twice in discharge. Double dosing a blood thinner = fatal bleeding risk. One planted issue: `duplicate`.

- `medium.json` — Patient takes Ultram (brand) at home, hospital prescribes tramadol (generic) — same drug, double dose. Patient is also on sertraline (SSRI), making the duplicate life-threatening via serotonin syndrome. One planted issue: `duplicate` (requires brand/generic knowledge).

- `hard.json` — Three issues hidden across a complex list for an 81-year-old with heart failure:
  1. Coumadin (warfarin brand) + aspirin = major bleeding interaction (requires knowing Coumadin = warfarin)
  2. Digoxin dose silently doubled 0.125mg → 0.25mg (narrow therapeutic index drug, can cause fatal arrhythmia)
  3. Metoprolol (beta-blocker) completely missing from discharge (abrupt withdrawal = heart attack risk)

**Grader — verified test results:**
```
Easy   (correct flag)     → score = 1.0
Medium (brand/generic)    → score = 1.0
Hard   (all 3 issues)     → score = 1.0
False positive            → score = 0.0
Partial credit            → score = 0.1
```

**Reward structure per step:**
- Correct flag: +0.30
- Partial flag (right drugs, wrong type): +0.10
- False positive: -0.15
- Submit bonus: +0.10

---

## Person 2 — Graders, Server & Docker

**Status: Ready to start — Person 1 work is complete and locked.**

**You own:**
- `med_reconciliation/server/app.py`
- `med_reconciliation/server/Dockerfile`
- `med_reconciliation/server/requirements.txt`
- `med_reconciliation/openenv.yaml`
- `med_reconciliation/pyproject.toml`

**Your tasks in order:**

1. Review `graders/graders.py` — it is already written and tested. Understand the `grade_episode(flags, planted_issues, brand_to_generic)` function signature. You do not need to change it.

2. Review `server/app.py`. The `MED_RECON_TASK` env var controls which task loads (`easy`, `medium`, `hard`). The factory function creates a new `MedReconciliationEnvironment` per session.

3. Build and test Docker:
```bash
docker build -t med-recon-env ./med_reconciliation/server
docker run -p 8000:8000 -e MED_RECON_TASK=easy med-recon-env
```

4. Test the server responds correctly:
```bash
# Should return HTTP 200 with initial observation
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" -d '{}'

# Should return observation + reward
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "flag_duplicate", "drug_a": "warfarin", "drug_b": "warfarin", "reasoning": "duplicate"}'
```

5. Verify `openenv validate` passes:
```bash
pip install openenv-core
cd med_reconciliation
openenv validate
```

6. The `/reset` endpoint MUST return HTTP 200 — this is the first check in the pre-submission validator.

**Do NOT touch:**
- `models.py`
- `server/environment.py`
- `graders/graders.py`
- `data/` folder
- `inference.py`
- `README.md`

---

## Person 3 — Inference Script, README & Validation

**Status: Can start on README now. Wait for Person 2's server before testing inference.**

**You own:**
- `inference.py` (root level)
- `README.md` (root level)
- `requirements.txt` (root level)

**Your tasks in order:**

1. The stdout format in `inference.py` is strict — any deviation breaks automated scoring:
```
[START] task=easy env=med_reconciliation model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=flag_duplicate(warfarin,warfarin) reward=0.30 done=false error=null
[END] success=true steps=1 score=0.800 rewards=0.30
```
   - `reward` and `rewards` — 2 decimal places
   - `score` — 3 decimal places
   - `done` and `success` — lowercase `true` or `false`
   - `error` — raw error string or `null`

2. Set up env vars and run inference against the local server (once Person 2 has it running):
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token"
export MED_RECON_TASK="easy"
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

3. Run inference for all 3 tasks and record baseline scores in `README.md`.

4. Run the pre-submission validation script:
```bash
./scripts/validate-submission.sh https://your-space.hf.space
```

5. Keep `README.md` updated with environment description, action/observation space tables, task descriptions, setup instructions, and baseline scores.

**Do NOT touch:**
- `models.py`
- `server/environment.py`
- `graders/graders.py`
- `data/` folder
- `server/app.py`
- `server/Dockerfile`

---

## Merging to Main

Only do this when ALL of the following are true:

- [ ] `openenv validate` passes
- [ ] `docker build` succeeds
- [ ] `docker run` starts and `/reset` returns HTTP 200
- [ ] `inference.py` runs without errors and produces correct `[START]`/`[STEP]`/`[END]` logs for all 3 tasks
- [ ] Baseline scores recorded in README
- [ ] All 3 team members have reviewed and approved

Then:
```bash
git checkout main
git merge development
git push origin main
```

---

## Communication Rules

- If you need to change `models.py` — tell everyone immediately, it breaks everything downstream
- If you change the task JSON format — tell Person 2 (grader depends on it)
- If you change the server port or endpoints — tell Person 3 (inference depends on it)
- Never force push to any branch
- Never commit `.env` files or API keys
- Always `git pull origin development` before starting work
