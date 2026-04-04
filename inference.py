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

from med_reconciliation import MedReconciliationAction, MedReconciliationEnv

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MED_RECON_TASK", "all").lower()
BENCHMARK = "med_reconciliation"
MAX_STEPS = 10
TEMPERATURE = 0.2   # low temp for deterministic medical reasoning
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5
ALL_TASKS = ["easy", "medium", "hard"]

# Max possible reward per episode (used for score normalization)
# correct flag = 0.3, up to 2 issues in hard task + submit bonus
_MAX_REWARD = 1.0


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
You are a clinical pharmacist performing medication reconciliation.
You will be given a patient's home medication list and their hospital discharge prescription.
Your job is to identify ALL discrepancies between the two lists.

Discrepancy types you must check for:
1. DUPLICATE — same drug appears twice (including brand/generic equivalents, e.g. Zoloft = sertraline)
2. INTERACTION — two drugs together cause a dangerous interaction (e.g. serotonin syndrome, bleeding risk)
3. DOSE_MISMATCH — same drug has different doses between home and discharge lists
4. MISSING — a home medication is absent from the discharge list without explanation

For each discrepancy, respond with a JSON object on a single line:
{"action_type": "flag_duplicate|flag_interaction|flag_dose_mismatch|flag_missing", "drug_a": "<name>", "drug_b": "<name>", "reasoning": "<brief explanation>"}

When you have flagged all discrepancies, respond with:
{"action_type": "submit", "drug_a": "", "drug_b": "", "reasoning": "All discrepancies identified"}

Use generic drug names when possible. Be precise — false positives are penalized.
""").strip()


def build_user_prompt(
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
    return textwrap.dedent(f"""
Step {step}

HOME MEDICATIONS:
{home_str}

DISCHARGE MEDICATIONS:
{discharge_str}

FLAGS SUBMITTED SO FAR:
{flags_str}

LAST FEEDBACK: {last_feedback}

Identify the next discrepancy or submit if done. Respond with a single JSON object.
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
    home_meds: List[Dict[str, Any]],
    discharge_meds: List[Dict[str, Any]],
    flags_so_far: List[Dict[str, Any]],
    last_feedback: str,
    step: int,
) -> MedReconciliationAction:
    user_prompt = build_user_prompt(home_meds, discharge_meds, flags_so_far, last_feedback, step)
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

    if IMAGE_NAME:
        env = await MedReconciliationEnv.from_docker_image(IMAGE_NAME)
    else:
        env = MedReconciliationEnv(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))

    try:
        result = await env.reset()
        obs = result.observation
        last_feedback = obs.step_feedback

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(
                client,
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
