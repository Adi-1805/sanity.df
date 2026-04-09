---
title: Data Clean Transform Environment Server
emoji: 🎳
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Data Clean Transform Environment

A realistic OpenEnv environment that simulates a data engineering pipeline. AI Agents are assigned tasks to clean, validate, and standardize messy data using structured Pandas-style operations.

## Supported Tasks

This environment exposes 3 datasets corresponding to different difficulties. Switch between tasks by setting the `TASK_ID` environment variable before running the FastAPI server.

### Task 1: Basic Cleansing (Easy)
- **Status**: Fully completed and tested. `inference.py` scores 1.0.
- **Goal**: Drop duplicates and handle missing critical fields gracefully without deleting valid non-critical rows.
- **Actions Allowed**: `drop_duplicates` and `drop_na`

### Task 2: Formatting & Type Casting (Medium)
- **Status**: Fully completed and tested. `inference.py` is capable of solving this.
- **Goal**: Clean messy currency strings (e.g., `$1,200.50` -> `1200.50` float) and standardize scattered date formats.
- **Actions Allowed**: `str_replace`, `astype`, `to_datetime`

### Task 3: Contextual Imputation (Hard)
- **Status**: Fully completed and tested. `inference.py` achieves 1.0!
- **Goal**: Resolve formatting inconsistencies (e.g., `NY`, `N.Y.` -> `New York`) and impute missing city/state mapping data contextually using standard zipcodes.
- **Actions Allowed**: `replace_map`, `impute_from_column`

## Quick Start

### 1. Environment Setup
Sync the dependencies within both the outer wrapper and the environment configuration.
```bash
uv sync
cd data_clean_transform
uv sync
```

### 2. Launch the Evaluation Server
The server must be running to host the datasets and evaluate the score locally.
```bash
# Optional: Set TASK_ID specifically (defaults to task1)
export TASK_ID="task2"
uv run python server/app.py
```
*(The UI will expose itself locally at http://0.0.0.0:8000)*

### 3. Trigger the AI Agent
Open a separate terminal window to execute the `inference.py` script. The client script evaluates the dataset using the Hugging Face Inference API.
```bash
# Must supply your HF token
export HF_TOKEN="your_hugging_face_token"

# Target the same task as the server
export TASK_ID="task2"
uv run python inference.py
```

## Environment Details

### Action
**DataCleanTransformAction**: Model defining structured JSON actions
- `operation` (str) - The operation string (e.g., `drop_duplicates`, `str_replace`, `finish`)
- `column` (str) - The target column 
- `value` (any) - Values like `target_type` or imputation strategies
- `kwargs` (dict) - Keyword arguments for Pandas methods (e.g., `subset`, `pat`, `repl`)

### Observation
**DataCleanTransformObservation**: Returns environmental progress signals
- `current_task` (str) - Task ID loaded
- `task_description` (str) - Prompt string detailing the AI goal
- `dataset_head` (str) - Text rendering of `df.head(10)`
- `dataset_info` (str) - Text rendering of `df.info()`
- `last_action_feedback` (str) - Logs whether rows were deleted or modified
- `reward` (float) - 0.0 -> 1.0 progress tracking
