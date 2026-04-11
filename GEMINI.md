# AI ML-Ready Preprocessing Environment

## Project Overview

This project is a realistic data science preprocessing environment built on the **OpenEnv** specification. It simulates a data engineering/science pipeline where an AI agent must clean, scale, impute, and engineer features using structured ML operations.

The environment challenges the agent with 3 distinct tasks of increasing difficulty:
1.  **Task 1 (Easy) - Intelligent Imputation:** Handling missing values based on statistical distributions using mode, mean, median, or KNN imputation.
2.  **Task 2 (Medium) - Scaling & Mathematical Transforms:** Standardizing and normalizing distributions using scalers (MinMax, StandardScaler, RobustScaler, MaxAbsScaler) and logarithmic transformations.
3.  **Task 3 (Hard) - Domain-Driven Feature Construction:** Creating logical, domain-specific features (e.g., customer lifetime, average order value) from raw e-commerce data.

The core technology stack includes:
*   **Python 3.10+**
*   **FastAPI & Uvicorn** for the environment server.
*   **Pandas & Scikit-Learn** for data manipulation and ML operations.
*   **Pydantic** for strictly typed action and observation models.
*   **OpenAI API Client** (`inference.py`) to run the AI agent against the environment.

## Building and Running

The project uses `uv` for fast dependency management.

### 1. Setup Dependencies
```bash
uv sync
```

### 2. Running the Environment Server
The FastAPI server hosts the dataset state and processes actions.
```bash
# Set the desired task (task1, task2, or task3)
export TASK_ID="task1"
uv run python server/app.py
```
*The server will run locally at `http://0.0.0.0:8000`.*

### 3. Running the AI Agent
Open a separate terminal window to run the inference script.
```bash
# Set your Hugging Face or OpenAI API token
export HF_TOKEN="your_hugging_face_token"

# Match the TASK_ID with the server
export TASK_ID="task1"
uv run python inference.py
```

### Docker Deployment
The environment can be containerized to run as a Hugging Face Space.
```bash
# Build the image
docker build -t data-clean-transform-env .

# Run the container
docker run -p 8000:8000 -e TASK_ID="task2" --name data-env-server --rm data-clean-transform-env
```

## Development Conventions

*   **OpenEnv Spec:** The environment strictly adheres to the OpenEnv protocol. Metadata is defined in `openenv.yaml`. Actions and Observations are strictly typed Pydantic models in `models.py`.
*   **Deterministic Grading:** The environment uses a hidden "Gold Dataset" approach. It performs a programmatic, row-by-row semantic comparison between the agent's modified DataFrame and the hidden target DataFrame to generate a continuous reward signal between `0.01` and `0.99`. 
*   **Safety & Execution:** The environment translates structured JSON actions into Pandas/Scikit-Learn operations. For complex feature engineering, a sandboxed or fallback `eval` approach is utilized over the DataFrame columns.
*   **Dataset Generation:** Synthetic datasets (both "messy" and "gold") are generated deterministically using `scripts/generate_synthetic_data.py`.
