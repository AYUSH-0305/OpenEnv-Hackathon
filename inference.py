"""
Medication Reconciliation — OpenEnv Inference Script
=====================================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL    The API endpoint for the LLM
    MODEL_NAME      The model identifier to use for inference
    HF_TOKEN        Your Hugging Face / API key
    IMAGE_NAME      Docker image name (if using from_docker_image)
    MED_RECON_TASK  Task difficulty: easy | medium | hard (default: easy)

STDOUT FORMAT (strict — do not modify):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from models import MedReconciliationAction
from client import MedReconciliationEnv

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MED_RECON_TASK", "easy").lower()
BENCHMARK = "med_reconciliation"
MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5
ALL_TASKS = ["easy", "medium", "hard", "control"]
_MAX_REWARD = 1.0

# Single Space URL for all tasks — task is passed via episode_id on reset
_base = os.getenv("ENV_BASE_URL", "https://arpann09-med-reconciliation.hf.space")
TASK_URLS = {
    "easy":   _base,
    "medium": _base,
    "hard":   _base,
}


# ── Logging helpers (strict format — do not change) ────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt helpers ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a senior clinical pharmacist performing medication reconciliation.
Before taking any action, follow this strict reasoning chain:

STEP 1 — NORMALIZE: Convert all brand names to generic names.
  Key mappings: Coumadin=warfarin, Ultram=tramadol, Zoloft=sertraline,
  Lopressor=metoprolol, Lanoxin=digoxin, Lasix=furosemide, Glucophage=metformin,
  Tylenol=acetaminophen, Advil/Motrin=ibuprofen, Prilosec=omeprazole

STEP 2 — COMPARE LISTS: For each drug in the home list, check the discharge list for:
  a) Exact duplicates (same drug listed twice)
  b) Brand/generic duplicates (same drug under different names)
  c) Dose mismatches (especially dangerous for narrow therapeutic index drugs: digoxin, warfarin, lithium)
  d) Missing medications (especially dangerous for beta-blockers, corticosteroids)

STEP 3 — CHECK INTERACTIONS: For every pair of drugs in the discharge list:
  - warfarin/coumadin + aspirin/NSAIDs = major bleeding risk
  - SSRI (sertraline, fluoxetine) + tramadol = serotonin syndrome
  - digoxin + amiodarone = digoxin toxicity
  - ACE inhibitor (lisinopril) + potassium = hyperkalemia

STEP 4 — DECIDE: Only flag TRUE clinical errors. If the lists are identical, submit immediately.

For each discrepancy, respond with a JSON object on a single line:
{"action_type": "flag_duplicate|flag_interaction|flag_dose_mismatch|flag_missing", "drug_a": "<generic_name>", "drug_b": "<generic_name>", "reasoning": "<clinical explanation>"}

When done (or if no discrepancies exist), respond with:
{"action_type": "submit", "drug_a": "", "drug_b": "", "reasoning": "Reconciliation complete"}

False positives are penalized. Only flag what you are clinically certain about.
""").strip()


def build_user_prompt(
    patient_context: str,
    home_meds: List[Dict[str, Any]],
    discharge_meds: List[Dict[str, Any]],
    flags_so_far: List[Dict[str, Any]],
    last_feedback: str,
    step: int,
) -> str:
    home_str = "\n".join(
        f"  - {m['name']} {m['dose']} {m['frequency']}" for m in home_meds
    )
    discharge_str = "\n".join(
        f"  - {m['name']} {m['dose']} {m['frequency']}" for m in discharge_meds
    )
    flags_str = (
        "\n".join(f"  - {f['action_type']}: {f['drug_a']} / {f['drug_b']}" for f in flags_so_far)
        if flags_so_far
        else "  None yet"
    )
    context_str = f"\nPATIENT CONTEXT: {patient_context}" if patient_context else ""
    return textwrap.dedent(f"""
Step {step}{context_str}

HOME MEDICATIONS:
{home_str}

DISCHARGE MEDICATIONS:
{discharge_str}

FLAGS SUBMITTED SO FAR:
{flags_str}

LAST FEEDBACK: {last_feedback}

Identify the next discrepancy or submit if all are found. Respond with a single JSON object.
""").strip()


def parse_action(text: str) -> MedReconciliationAction:
    """Parse model output into a MedReconciliationAction. Falls back to submit on error."""
    text = text.strip()
    # Find JSON object in response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            return MedReconciliationAction(
                action_type=data.get("action_type", "submit"),
                drug_a=data.get("drug_a", ""),
                drug_b=data.get("drug_b", ""),
                reasoning=data.get("reasoning", ""),
            )
        except json.JSONDecodeError:
            pass
    return MedReconciliationAction(action_type="submit", drug_a="", drug_b="", reasoning="parse error")


def get_model_action(
    client: OpenAI,
    patient_context: str,
    home_meds: List[Dict[str, Any]],
    discharge_meds: List[Dict[str, Any]],
    flags_so_far: List[Dict[str, Any]],
    last_feedback: str,
    step: int,
) -> MedReconciliationAction:
    user_prompt = build_user_prompt(patient_context, home_meds, discharge_meds, flags_so_far, last_feedback, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return MedReconciliationAction(action_type="submit", drug_a="", drug_b="", reasoning="model error")


# ── Main loop ──────────────────────────────────────────────────────────────────
async def run_task(client: OpenAI, task: str) -> tuple[bool, int, float, List[float]]:
    """Run one episode for a given task. Returns (success, steps, score, rewards)."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    # Set task env var so local server factory picks it up
    os.environ["MED_RECON_TASK"] = task

    base_url = TASK_URLS.get(task, os.getenv("ENV_BASE_URL", "http://localhost:8000"))

    if IMAGE_NAME:
        env = await MedReconciliationEnv.from_docker_image(IMAGE_NAME)
    else:
        env = MedReconciliationEnv(base_url=base_url)

    try:
        # Pass task name via episode_id so server loads the correct scenario
        try:
            result = await env.reset(episode_id=task)
        except Exception:
            result = await env.reset()
        obs = result.observation
        last_feedback = obs.step_feedback

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(
                client,
                obs.patient_context if hasattr(obs, 'patient_context') else "",
                obs.home_medications,
                obs.discharge_medications,
                obs.flags_submitted,
                last_feedback,
                step,
            )

            action_str = f"{action.action_type}({action.drug_a},{action.drug_b})"

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_feedback = obs.step_feedback

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Normalize score
        total_reward = sum(rewards)
        score = min(max(total_reward / _MAX_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return success, steps_taken, score, rewards


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ALL_TASKS if TASK_NAME == "all" else [TASK_NAME]

    results = []
    for task in tasks:
        success, steps, score, rewards = await run_task(client, task)
        results.append((task, success, steps, score, rewards))

    if len(results) > 1:
        print("\n[SUMMARY] task results:", flush=True)
        for task, success, steps, score, rewards in results:
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[SUMMARY] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
