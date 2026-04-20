---
title: ML-Ready Preprocessing Environment
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

# AI ML-Ready Preprocessing Environment

A realistic OpenEnv environment that simulates a data science preprocessing pipeline using RL. AI Agents are assigned tasks to clean, scale, impute, and engineer features using structured ML operations.

## Supported Tasks

This environment exposes 3 datasets corresponding to different difficulties. Switch between tasks by setting the `TASK_ID` environment variable before running the FastAPI server.

### Task 1: Intelligent Imputation (Easy)
- **Goal**: Handle missing values and mixed types logically.
- **Challenges**: The agent must look at the statistical summary (e.g., skewness, outliers) to decide whether to use mean, median, mode, or KNN imputation.
- **Actions Allowed**: `impute`, `split_column`

### Task 2: Scaling & Mathematical Transforms (Medium)
- **Goal**: Standardize and normalize various features.
- **Challenges**: Requires detecting distributions (uniform, normal, skewed, sparse) and applying the correct scaler (minmax, standard, robust, maxabs) or mathematical transformation (log1p).
- **Actions Allowed**: `scale`, `transform`

### Task 3: Domain-Driven Feature Construction (Hard)
- **Goal**: Construct new features logically.
- **Challenges**: Given an e-commerce dataset, build semantic features like 'age_at_signup', 'days_since_last_purchase', 'average_order_value', and 'customer_lifetime' while properly handling `NaN`s or division by zero.
- **Actions Allowed**: `feature_eng`

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
- `operation` (str) - The operation string (e.g., `impute`, `scale`, `feature_eng`, `finish`)
- `column` (str) - The target column 
- `value` (any) - Values like `target_type` or imputation strategies
- `kwargs` (dict) - Keyword arguments for Pandas/Sklearn methods (e.g., `strategy`, `method`)

### Observation
**DataCleanTransformObservation**: Returns environmental progress signals
- `current_task` (str) - Task ID loaded
- `task_description` (str) - Prompt string detailing the AI goal
- `dataset_head` (str) - Text rendering of `df.head(10)`
- `dataset_info` (str) - Text rendering of `df.info()` combined with `df.describe()` for statistical context
- `last_action_feedback` (str) - Logs whether rows were modified or scaled correctly
- `reward` (float) - strictly normalized 0.01 -> 0.99 progress tracking based on semantic matching against a hidden "Gold" dataset.
