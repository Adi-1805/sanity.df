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

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_ID", "task1")
BENCHMARK = "data_clean_transform"
MAX_STEPS = 25
TEMPERATURE = 0.0 

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a data cleaning expert. Your goal is to clean a messy dataset according to the task description.
    
    Available operations:
    You are an expert Data Engineer AI Agent evaluating an OpenEnv data manipulation challenge.
    Your goal is to parse dataset observations and iteratively execute JSON actions to perfectly clean the target dataset.

    You ONLY output a JSON object containing the action. Do not wrap it in markdown block. Just the raw JSON.

    Permitted 'operation' strings: 'drop_duplicates', 'drop_na', 'fill_na', 'astype', 'str_replace', 'to_datetime', 'replace_map', 'impute_from_column', 'finish'.

    # Examples

    ## Condition 1: Dropping duplicates and NaNs
    {"operation": "drop_duplicates", "column": null, "value": null, "kwargs": {}}
    {"operation": "drop_na", "column": null, "value": null, "kwargs": {"subset": ["critical_column"]}}

    ## Condition 2: Cleaning string currency
    {"operation": "str_replace", "column": "salary", "value": null, "kwargs": {"pat": "\\$", "repl": "", "regex": true}}
    {"operation": "str_replace", "column": "salary", "value": null, "kwargs": {"pat": ",", "repl": "", "regex": false}}
    {"operation": "astype", "column": "salary", "value": "float", "kwargs": {}}

    ## Condition 3: Formatting Dates
    {"operation": "to_datetime", "column": "date_joined", "value": null, "kwargs": {"format": "mixed"}}

    ## Condition 4: Text Normalization (replace_map)
    {"operation": "replace_map", "column": "city", "value": null, "kwargs": {"mapping": {"N.Y.": "New York", "ny": "New York", "L.A.": "Los Angeles"}}}

    ## Condition 5: Contextual Imputation
    {"operation": "impute_from_column", "column": "state", "value": null, "kwargs": {"source_column": "zipcode", "mapping": {10001: "NY", 10002: "NY", 90001: "CA", 90002: "CA"}}}

    ## Condition 6: Finishing the task
    {"operation": "finish", "column": null, "value": null, "kwargs": {}}

    IMPORTANT INSTRUCTIONS:
    Observe the 'dataset_head' and 'dataset_info' carefully. For Task 3, you must normalize ALL variations of city names (e.g., 'Big Apple' to 'New York', 'la' or 'L.A.' to 'Los Angeles') and impute ALL missing states using the zipcodes (10001/10002 -> NY, 90001/90002 -> CA). Since the dataset is now larger (60 rows), you may need several steps to capture all mappings. 
    Stop and 'finish' only when you have resolved all criteria described in 'task_description'.
    """
).strip()


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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


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
        
        What is your next cleaning operation? Respond with JSON.
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
            response_format={"type": "json_object"}
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        return DataCleanTransformAction(**data)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback to finish if something goes wrong
        return DataCleanTransformAction(operation="finish")


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await DataCleanTransformEnv.from_docker_image(IMAGE_NAME)
    else:
        env = DataCleanTransformEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        await env.__aenter__()
        
        result = await env.reset()
        
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

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Final score is the reward from the last step
        score = rewards[-1] if rewards else 0.0
        score = max(0.0001, min(0.9999, score)) if 'score' in locals() else 0.5

        success = score >= 0.8

    except Exception as e:
        print(f"[DEBUG] Main loop error: {e}", flush=True)
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
