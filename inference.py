"""
Medication Reconciliation — OpenEnv Inference Script
=====================================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL        The API endpoint for the LLM
    MODEL_NAME          The model identifier to use for inference
    HF_TOKEN            Your Hugging Face / API key
    LOCAL_IMAGE_NAME    Docker image name (if using from_docker_image)
    MED_RECON_TASK      Task: easy | medium | hard (default: runs all 3)

STDOUT FORMAT (strict — only these 3 line types):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MED_RECON_TASK", "").lower().strip()
BENCHMARK = "med_reconciliation"
MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 400
SUCCESS_SCORE_THRESHOLD = 0.5
ALL_TASKS = ["easy", "medium", "hard"]
_MAX_REWARD = 1.0
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://arpann09-med-reconciliation.hf.space")


def _debug(msg: str) -> None:
    """Write debug info to stderr only — never pollutes stdout."""
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


# ── Strict stdout logging (only these 3 types allowed on stdout) ───────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── Prompt helpers ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a senior clinical pharmacist performing medication reconciliation.
Before taking any action, follow this strict reasoning chain:

STEP 1 — NORMALIZE: Convert all brand names to generic names.
  Key mappings: Coumadin=warfarin, Ultram=tramadol, Zoloft=sertraline,
  Lopressor=metoprolol, Lanoxin=digoxin, Lasix=furosemide, Glucophage=metformin

STEP 2 — COMPARE LISTS: Check for duplicates, dose mismatches, missing meds, interactions.
  Dangerous interactions: warfarin+aspirin (bleeding), SSRI+tramadol (serotonin syndrome),
  digoxin+amiodarone (toxicity)

STEP 3 — DECIDE: Only flag TRUE clinical errors. If lists are identical, submit immediately.

For each discrepancy respond with JSON on one line:
{"action_type": "flag_duplicate|flag_interaction|flag_dose_mismatch|flag_missing", "drug_a": "<name>", "drug_b": "<name>", "reasoning": "<explanation>"}

When done respond with:
{"action_type": "submit", "drug_a": "", "drug_b": "", "reasoning": "Reconciliation complete"}
""").strip()


def build_user_prompt(patient_context, home_meds, discharge_meds, flags_so_far, last_feedback, step):
    home_str = "\n".join(f"  - {m['name']} {m['dose']} {m['frequency']}" for m in home_meds)
    discharge_str = "\n".join(f"  - {m['name']} {m['dose']} {m['frequency']}" for m in discharge_meds)
    flags_str = "\n".join(f"  - {f['action_type']}: {f['drug_a']} / {f['drug_b']}" for f in flags_so_far) if flags_so_far else "  None yet"
    context_str = f"\nPATIENT CONTEXT: {patient_context}" if patient_context else ""
    return f"Step {step}{context_str}\n\nHOME MEDICATIONS:\n{home_str}\n\nDISCHARGE MEDICATIONS:\n{discharge_str}\n\nFLAGS ALREADY SUBMITTED (do NOT repeat):\n{flags_str}\n\nLAST FEEDBACK: {last_feedback}\n\nRespond with a single JSON object."


def parse_action_dict(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "submit", "drug_a": "", "drug_b": "", "reasoning": "parse error"}


def _baseline_agent(home_meds, discharge_meds, flags_so_far):
    """Deterministic fallback when LLM unavailable."""
    already_flagged = {(f.get("drug_a", "").lower(), f.get("drug_b", "").lower()) for f in flags_so_far}
    brand_map = {
        "coumadin": "warfarin", "ultram": "tramadol", "zoloft": "sertraline",
        "lopressor": "metoprolol", "lanoxin": "digoxin", "lasix": "furosemide",
        "glucophage": "metformin", "tylenol": "acetaminophen", "advil": "ibuprofen",
        "motrin": "ibuprofen", "prilosec": "omeprazole",
    }

    def norm(name):
        return brand_map.get(name.strip().lower(), name.strip().lower())

    home_generic = {norm(m["name"]): m for m in home_meds}
    discharge_generic = {norm(m["name"]): m for m in discharge_meds}
    discharge_names = [norm(m["name"]) for m in discharge_meds]

    for name in discharge_names:
        if discharge_names.count(name) > 1:
            key = (name, name)
            if key not in already_flagged:
                return {"action_type": "flag_duplicate", "drug_a": name, "drug_b": name, "reasoning": f"{name} appears twice"}

    for d_name in list(discharge_generic.keys()):
        for h_name in list(home_generic.keys()):
            if d_name != h_name and norm(d_name) == norm(h_name):
                key = tuple(sorted([d_name, h_name]))
                if key not in already_flagged:
                    return {"action_type": "flag_duplicate", "drug_a": h_name, "drug_b": d_name, "reasoning": "brand/generic duplicate"}

    for name in home_generic:
        if name in discharge_generic:
            h_dose = home_generic[name].get("dose", "")
            d_dose = discharge_generic[name].get("dose", "")
            if h_dose != d_dose:
                key = (name, name)
                if key not in already_flagged:
                    return {"action_type": "flag_dose_mismatch", "drug_a": name, "drug_b": name, "reasoning": f"dose changed {h_dose} to {d_dose}"}

    for name in home_generic:
        if name not in discharge_generic:
            key = (name, "")
            if key not in already_flagged:
                return {"action_type": "flag_missing", "drug_a": name, "drug_b": "", "reasoning": f"{name} missing from discharge"}

    all_d = [norm(m["name"]) for m in discharge_meds]
    if "warfarin" in all_d and "aspirin" in all_d:
        key = ("warfarin", "aspirin")
        if key not in already_flagged:
            return {"action_type": "flag_interaction", "drug_a": "warfarin", "drug_b": "aspirin", "reasoning": "bleeding risk"}

    return {"action_type": "submit", "drug_a": "", "drug_b": "", "reasoning": "Reconciliation complete"}


def get_model_response(client, patient_context, home_meds, discharge_meds, flags_so_far, last_feedback, step):
    user_prompt = build_user_prompt(patient_context, home_meds, discharge_meds, flags_so_far, last_feedback, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return parse_action_dict((completion.choices[0].message.content or "").strip())
    except Exception as exc:
        _debug(f"Model request failed: {exc}")
        return _baseline_agent(home_meds, discharge_meds, flags_so_far)


# ── Main loop ──────────────────────────────────────────────────────────────────
async def run_task(client, task):
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        from models import MedReconciliationAction
        from client import MedReconciliationEnv
    except ImportError as e:
        _debug(f"Import error: {e}")
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return False, 0, 0.0, []

    env = None
    try:
        if LOCAL_IMAGE_NAME:
            env = await MedReconciliationEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = MedReconciliationEnv(base_url=ENV_BASE_URL)

        try:
            result = await env.reset(episode_id=task)
        except Exception:
            result = await env.reset()

        obs = result.observation
        last_feedback = getattr(obs, 'step_feedback', '')

        for step in range(1, MAX_STEPS + 1):
            if getattr(result, 'done', False):
                break

            data = get_model_response(
                client,
                getattr(obs, 'patient_context', ''),
                getattr(obs, 'home_medications', []),
                getattr(obs, 'discharge_medications', []),
                getattr(obs, 'flags_submitted', []),
                last_feedback,
                step,
            )

            action = MedReconciliationAction(
                action_type=data.get("action_type", "submit"),
                drug_a=data.get("drug_a", ""),
                drug_b=data.get("drug_b", ""),
                reasoning=data.get("reasoning", ""),
            )
            action_str = f"{action.action_type}({action.drug_a},{action.drug_b})"
            error_str = None
            reward = 0.0
            done = False

            try:
                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                last_feedback = getattr(obs, 'step_feedback', '')
            except Exception as exc:
                _debug(f"env.step error: {exc}")
                error_str = str(exc)[:80]
                done = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_str)

            if done:
                break

        total_reward = sum(rewards)
        score = min(max(total_reward / _MAX_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        _debug(f"run_task error: {e}")
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return success, steps_taken, score, rewards


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "placeholder")
    if TASK_NAME in ("easy", "medium", "hard", "control"):
        tasks = [TASK_NAME]
    else:
        tasks = ALL_TASKS

    for task in tasks:
        await run_task(client, task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        _debug(f"Fatal error: {e}")
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(0)
