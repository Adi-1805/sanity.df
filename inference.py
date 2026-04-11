import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from data_clean_transform.client import DataCleanTransformEnv
from data_clean_transform.models import DataCleanTransformAction

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "EMPTY"

API_BASE_URL = os.getenv("API_BASE_URL") or "https://openrouter.ai/api/v1"
# MODEL_NAME = os.getenv("MODEL_NAME") or "arcee-ai/trinity-large-preview:free"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-oss-120b:free"
# TASK_NAME = os.getenv("TASK_ID", "task1")
TASK_NAME = "task3"
BENCHMARK = "data_clean_transform"
MAX_STEPS = 10
TEMPERATURE = 0.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a data science expert agent evaluating an OpenEnv ML preprocessing challenge.
    Your goal is to parse dataset observations and iteratively execute JSON actions to perfectly clean and preprocess the target dataset according to the given task description.

    You ONLY output a JSON object containing the action. Do not wrap it in markdown block. Just the raw JSON.

    Permitted 'operation' strings: 'drop_duplicates', 'drop_na', 'fill_na', 'astype', 'str_replace', 'to_datetime', 'replace_map', 'impute_from_column', 'impute', 'scale', 'transform', 'feature_eng', 'split_column', 'finish'.

    # Examples

    ## Condition 1: Imputation (Easy)
    {"operation": "impute", "column": "income_usd", "value": null, "kwargs": {"strategy": "median"}}
    {"operation": "impute", "column": "house_price", "value": null, "kwargs": {"strategy": "knn", "cols": ["sqft", "house_price"]}}

    ## Condition 2: Scaling & Transforms (Medium)
    {"operation": "scale", "column": "age", "value": null, "kwargs": {"method": "minmax"}}
    {"operation": "scale", "column": "stock_volume", "value": null, "kwargs": {"method": "robust"}}
    {"operation": "transform", "column": "engagement_time", "value": null, "kwargs": {"method": "log1p"}}

    ## Condition 3: Splitting Columns
    {"operation": "split_column", "column": "product_code", "value": null, "kwargs": {"pat": "([A-Za-z]+)[-]?([0-9]+)", "new_cols": ["product_category", "product_id"]}}

    ## Condition 4: Feature Engineering (Hard)
    {"operation": "feature_eng", "column": "average_order_value", "value": null, "kwargs": {"formula": "total_spent / total_orders"}}

    ## Condition 5: Finishing the task
    {"operation": "finish", "column": null, "value": null, "kwargs": {}}

    IMPORTANT INSTRUCTIONS:
    Observe the 'dataset_head' and 'dataset_info' (including Statistical Summary and non-null counts) carefully to make decisions.
    Do NOT repeat an operation on a column if it has already been successfully applied (e.g., if there are 0 missing values left, do not impute again).
    Once you have resolved ALL criteria described in the 'Task Objective', you MUST output exactly: {"operation": "finish"}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(obs) -> str:
    return textwrap.dedent(
        f"""
        Task Name: {obs.current_task}
        Task Objective: {obs.task_description}
        
        Dataset Info:
        {obs.dataset_info}
        
        Dataset Head (CSV):
        {obs.dataset_head}
        
        Last Action Feedback:
        {obs.last_action_feedback}
        
        Instructions:
        1. Analyze the dataset based on the Task Objective.
        2. Perform ONE operation at a time.
        3. If you have completed the task, call {{"operation": "finish"}}.
        
        What is your next preprocessing operation? Respond with JSON.
        """
    ).strip()


def get_model_action(client: OpenAI, obs) -> DataCleanTransformAction:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        return DataCleanTransformAction(**data)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback to finish if something goes wrong
        return DataCleanTransformAction(operation="finish")


async def run_task(client: OpenAI, env, task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_name)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_obj = get_model_action(client, result.observation)
            action_str = json.dumps(action_obj.model_dump())

            result = await env.step(action_obj)

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step, action=action_str, reward=reward, done=done, error=error
            )

            if done:
                break

        # Final score is the reward from the last step
        score = rewards[-1] if rewards else 0.0
        # Normalization rule: 0.01 - 0.99
        score = max(0.01, min(0.99, score))

        success = score >= 0.8

    except Exception as e:
        print(f"[DEBUG] Main loop error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await DataCleanTransformEnv.from_docker_image(IMAGE_NAME)
    else:
        env = DataCleanTransformEnv(base_url="http://localhost:8000")

    task_id_env = os.getenv("TASK_ID")
    tasks_to_run = [task_id_env] if task_id_env else ["task1", "task2", "task3"]

    try:
        await env.__aenter__()
        for task_name in tasks_to_run:
            await run_task(client, env, task_name)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
