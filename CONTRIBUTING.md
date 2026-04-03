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

**Commit message format:**
- `person1: add hard task scenario`
- `person2: fix grader scoring logic`
- `person3: update inference logging format`

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
    ├── models.py                ← Person 1 owns this
    ├── client.py                ← do not touch
    ├── openenv.yaml             ← Person 2 owns this
    ├── pyproject.toml           ← Person 2 owns this
    ├── data/
    │   ├── drug_interactions.json   ← Person 1 owns this
    │   └── tasks/
    │       ├── easy.json            ← Person 1 owns this
    │       ├── medium.json          ← Person 1 owns this
    │       └── hard.json            ← Person 1 owns this
    ├── graders/
    │   ├── __init__.py          ← do not touch
    │   └── graders.py           ← Person 2 owns this
    └── server/
        ├── __init__.py          ← do not touch
        ├── environment.py       ← Person 1 owns this
        ├── app.py               ← Person 2 owns this
        ├── Dockerfile           ← Person 2 owns this
        └── requirements.txt     ← Person 2 owns this
```

**Golden rule: only edit files you own. If you need to change someone else's file, message the team first.**

---

## Person 1 — Core Environment & Data

**You own:**
- `med_reconciliation/models.py`
- `med_reconciliation/server/environment.py`
- `med_reconciliation/data/drug_interactions.json`
- `med_reconciliation/data/tasks/easy.json`
- `med_reconciliation/data/tasks/medium.json`
- `med_reconciliation/data/tasks/hard.json`

**Your tasks in order:**

1. Review `models.py` — make sure `MedReconciliationAction` and `MedReconciliationObservation` have all the fields you need. If you add or rename a field, immediately tell Person 2 and Person 3 so they can update their files.

2. Review all 3 task JSON files. The `planted_issues` array is the ground truth the grader uses. Every issue must follow this exact format:
```json
{
  "type": "duplicate|interaction|dose_mismatch|missing",
  "drug_a": "<generic or brand name>",
  "drug_b": "<generic or brand name>",
  "description": "<clinical explanation>"
}
```

3. Review `drug_interactions.json`. Make sure every drug pair used in your task files has a corresponding entry in the `interactions` array, and every brand name used has an entry in `brand_to_generic`.

4. Review `environment.py`. Walk through `reset()`, `step()`, and `_check_flag_against_issues()`. Make sure the reward values and logic match what you expect.

5. Test your environment locally:
```bash
cd OpenEnv-Hackathon
pip install openenv-core pydantic
python -c "
from med_reconciliation.server.environment import MedReconciliationEnvironment
env = MedReconciliationEnvironment(task='easy')
obs = env.reset()
print('Home meds:', obs.home_medications)
print('Discharge meds:', obs.discharge_medications)
"
```

**Do NOT touch:**
- `client.py`
- `graders/graders.py`
- `server/app.py`
- `server/Dockerfile`
- `inference.py`
- `README.md`

---

## Person 2 — Graders, Server & Docker

**You own:**
- `med_reconciliation/graders/graders.py`
- `med_reconciliation/server/app.py`
- `med_reconciliation/server/Dockerfile`
- `med_reconciliation/server/requirements.txt`
- `med_reconciliation/openenv.yaml`
- `med_reconciliation/pyproject.toml`

**Your tasks in order:**

1. Wait for Person 1 to confirm `models.py` is stable before starting. The grader imports from it.

2. Review `graders/graders.py`. The `grade_episode()` function takes:
   - `flags` — list of flags the agent submitted
   - `planted_issues` — from the task JSON file
   - `brand_to_generic` — from `drug_interactions.json`
   
   It returns a `(score: float, details: dict)` tuple. Score must always be in `[0.0, 1.0]`.

3. Test the grader in isolation:
```bash
python -c "
from med_reconciliation.graders.graders import grade_episode
flags = [{'action_type': 'flag_duplicate', 'drug_a': 'warfarin', 'drug_b': 'warfarin', 'reasoning': 'test'}]
issues = [{'type': 'duplicate', 'drug_a': 'warfarin', 'drug_b': 'warfarin', 'description': 'test'}]
brand_map = {}
score, details = grade_episode(flags, issues, brand_map)
print(score, details)  # should be close to 1.0
"
```

4. Review `server/app.py`. The `MED_RECON_TASK` env var controls which task loads. Make sure the factory function works.

5. Build and test Docker:
```bash
cd med_reconciliation
docker build -t med-recon-env ./server
docker run -p 8000:8000 -e MED_RECON_TASK=easy med-recon-env
# In another terminal:
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

6. Verify `openenv validate` passes:
```bash
pip install openenv-core
cd med_reconciliation
openenv validate
```

**Do NOT touch:**
- `models.py`
- `server/environment.py`
- `data/` folder
- `inference.py`
- `README.md`

---

## Person 3 — Inference Script, README & Validation

**You own:**
- `inference.py` (root level)
- `README.md` (root level)
- `requirements.txt` (root level)

**Your tasks in order:**

1. Wait for Person 1 to confirm `models.py` is stable and Person 2 to confirm the server runs before testing inference.

2. Review `inference.py`. The stdout format is strict — any deviation breaks automated scoring:
```
[START] task=easy env=med_reconciliation model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=flag_duplicate(warfarin,warfarin) reward=0.30 done=false error=null
[END] success=true steps=1 score=0.800 rewards=0.30
```
   - `reward` and `rewards` — 2 decimal places
   - `score` — 3 decimal places
   - `done` and `success` — lowercase `true` or `false`
   - `error` — raw error string or `null`

3. Set up your environment variables and run inference against the local server:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token"
export MED_RECON_TASK="easy"
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

4. Run inference for all 3 tasks and record the baseline scores in `README.md`.

5. Run the pre-submission validation script:
```bash
./scripts/validate-submission.sh https://your-space.hf.space
```
All 3 checks must pass before we merge to main.

6. Keep `README.md` updated with:
   - Environment description
   - Action/observation space tables
   - Task descriptions with difficulty
   - Setup instructions
   - Baseline scores table

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

- If you change `models.py` — tell everyone immediately
- If you change the task JSON format — tell Person 2 (grader depends on it)
- If you change the server port or endpoints — tell Person 3 (inference depends on it)
- Never force push to any branch
- Never commit `.env` files or API keys
