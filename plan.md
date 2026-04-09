# Plan: AI Data Cleaning & Transformation Environment

## 1. Project Overview
The "AI Data Cleaning & Transformation Environment" simulates a realistic data engineering task where an AI agent must clean and standardize messy real-world datasets (e.g., CSV/JSON). The agent will deal with missing values, schema mismatches, duplicates, and inconsistent formats, simulating the daily work of data analysts and engineers.

## 2. Tasks Definition (3 Levels of Difficulty)
We will define 3 specific tasks in `openenv.yaml`, each with a dedicated dataset and grader:

*   **Task 1 (Easy): Basic Cleansing**
    *   **Objective**: Remove exact duplicate rows and drop records where critical fields (e.g., "email" or "id") are missing.
    *   **Grader/Reward Criteria**: 
        *   +0.5 for removing all duplicates.
        *   +0.5 for dropping rows with missing critical fields.
        *   Penalty for dropping valid rows.

*   **Task 2 (Medium): Formatting & Type Casting**
    *   **Objective**: Standardize date formats to `YYYY-MM-DD` and fix schema mismatches (e.g., converting string representations of numbers like "$1,000" to integers/floats).
    *   **Grader/Reward Criteria**:
        *   Partial reward for each correctly formatted date column.
        *   Partial reward for successful type conversions without losing data.
        *   Penalty for introducing `NaN`s during conversion.

*   **Task 3 (Hard): Contextual Imputation & Normalization**
    *   **Objective**: Impute missing categorical values based on context (e.g., inferring "City" from "Zip Code"), normalize inconsistent text fields (e.g., resolving typos in categorical columns like "NY", "New York", "N.Y."), and handle structural anomalies.
    *   **Grader/Reward Criteria**:
        *   Reward based on the percentage of correctly normalized text fields.
        *   Reward for accurate contextual imputation compared to a hidden ground-truth dataset.

## 3. Environment Interface (OpenEnv Spec)

*   **Observation Space (`models.Observation`)**:
    *   `dataset_head`: A string or JSON representation of the first few rows of the dataset.
    *   `dataset_info`: Metadata including column names, data types, null counts, and total row count.
    *   `last_action_feedback`: Output or error message from the previous action.

*   **Action Space (`models.Action`)**:
    *   The agent will generate code or specific structured commands to manipulate the dataset. 
    *   We will provide an action interface that accepts `pandas` code snippets or structured transformation commands (e.g., `{"command": "drop_duplicates", "subset": ["id"]}`).
    *   For security and simplicity, we can restrict the environment to accept structured operations: `operation_type` (e.g., `drop_cols`, `fill_na`, `replace_regex`, `change_type`), `column_name`, and `parameters`.

*   **Reward Function**:
    *   Provides signal over the full trajectory. 
    *   Each step evaluates the current state of the dataset against the ground-truth cleaned dataset.
    *   Returns partial progress (e.g., `current_similarity_score - previous_similarity_score`).
    *   Final completion reward when the dataset perfectly matches the expected schema and data.

## 4. Implementation Steps

1.  **Define Models (`data_clean_transform/models.py`)**:
    *   Update Pydantic models for `Observation`, `Action`, and `Reward` according to the spaces defined above.

2.  **Environment Logic (`data_clean_transform/server/data_clean_transform_environment.py`)**:
    *   Implement `reset()`: Load the messy dataset corresponding to the requested task. Return initial `Observation`.
    *   Implement `step()`: Apply the agent's action (data transformation) to the in-memory dataframe. Calculate the `Reward` by comparing the modified dataframe to the target dataset. Return `Observation`, `Reward`, `done`, `info`.
    *   Implement `state()`: Return the current dataset state.
    *   Implement Grading Logic: Add specific programmatic graders for Easy, Medium, and Hard tasks to yield scores between 0.0 and 1.0.

3.  **Configure Metadata (`data_clean_transform/openenv.yaml`)**:
    *   Define the environment metadata.
    *   List the 3 tasks, pointing to their specific target datasets or grader configurations.

4.  **Baseline Inference Script (`data_clean_transform/inference.py`)**:
    *   Create `inference.py` in the root of the project directory.
    *   Use the standard `openai` client (configured with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`).
    *   Implement the agent loop to query the OpenEnv server, pass observations to the LLM, and execute returned actions.
    *   Strictly implement `[START]`, `[STEP]`, and `[END]` structured logging to standard output.

5.  **Documentation (`data_clean_transform/README.md`)**:
    *   Describe the motivation (real-world data cleaning).
    *   Document Observation/Action spaces.
    *   Detail the 3 tasks and difficulty levels.
    *   Provide setup instructions (`uv sync`, `openenv server`, etc.).
    *   Document the baseline scores achieved by the reference model.

## 5. Deployment & Validation
*   **Dockerfile**: Ensure the provided Dockerfile correctly builds the OpenEnv FastAPI server and installs any required dependencies like `pandas`.
*   **Validation**: Run `openenv validate` locally to ensure spec compliance.
*   **Hugging Face Spaces**: Ensure the container runs correctly so it can be deployed to a HF Space tagged with `openenv`.
*   **Resource Limits**: Verify execution uses <= 2 vCPUs, 8GB RAM, and finishes inference in under 20 minutes.