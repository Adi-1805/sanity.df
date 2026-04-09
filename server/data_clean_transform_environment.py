# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data Clean Transform Environment Implementation.
"""

import os
import io
import pandas as pd
from uuid import uuid4
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataCleanTransformAction, DataCleanTransformObservation
except ImportError:
    from models import DataCleanTransformAction, DataCleanTransformObservation


class DataCleanTransformEnvironment(Environment):
    """
    An environment for simulating data cleaning and transformation tasks.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the data_clean_transform environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.df: Optional[pd.DataFrame] = None
        self.current_task_name: str = ""
        # Use environment variable if set (e.g., in Docker), otherwise use relative path
        self.data_dir = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
        
        # Load tasks configuration
        self.tasks = {
            "task1": {
                "name": "Task 1: Basic Cleansing",
                "dataset": "task1_messy.csv",
                "description": "Clean the dataset by dropping all duplicate rows, and identifying/dropping rows that are entirely invalid (e.g. missing both id and email). Do not drop rows that are just missing non-critical fields like city."
            },
            "task2": {
                "name": "Task 2: Formatting & Type Casting",
                "dataset": "task2_messy.csv",
                "description": "Format date strings in the 'date_joined' column to a standard datetime format, and clean/typecast the messy 'salary' strings to standard floats."
            },
            "task3": {
                "name": "Task 3: Contextual Imputation",
                "dataset": "task3_messy.csv",
                "description": "Perform contextual imputation. Impute missing 'state' values based on the 'zipcode' or 'city' context, and normalize 'city' names logically (e.g., 'N.Y.' or 'ny' to 'New York', 'L.A.' to 'Los Angeles')."
            }
        }
        self.initial_errors: Dict[str, int] = {}

    def reset(self, task_id: Optional[str] = None) -> DataCleanTransformObservation:
        """
        Reset the environment, loading the messy dataset for the current task.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Priority: 1. Passed task_id, 2. Env var (refreshed), 3. Current, 4. task1
        import os
        os_task_id = os.environ.get("TASK_ID")
        self.current_task_name = task_id or os_task_id or self.current_task_name or "task1"
        
        if self.current_task_name not in self.tasks:
            self.current_task_name = "task1"
            
        task_info = self.tasks[self.current_task_name]
        
        dataset_path = os.path.join(self.data_dir, task_info["dataset"])
        self.df = pd.read_csv(dataset_path)
        
        # Calculate baseline errors for continuous grading
        self._calculate_baseline_errors()
        
        return self._make_observation("Environment reset and dataset loaded.")

    def _calculate_baseline_errors(self):
        """Analyze initial dataset to set baselines for continuous grading."""
        df = self.df
        errors = {}
        if self.current_task_name == "task1":
            errors["duplicates"] = df.duplicated().sum()
            errors["missing_critical"] = df[["id", "email"]].isna().any(axis=1).sum()
        elif self.current_task_name == "task2":
            errors["messy_salary"] = len(df) - pd.to_numeric(df["salary"].astype(str).str.replace(r"[\$,\s\€\£\¥]", "", regex=True), errors="coerce").notna().sum()
            errors["messy_date"] = len(df) - pd.to_datetime(df["date_joined"], errors="coerce", format="mixed").notna().sum()
        elif self.current_task_name == "task3":
            valid_cities = {"New York", "Los Angeles"}
            errors["messy_city"] = len(df) - df["city"].isin(valid_cities).sum()
            errors["missing_state"] = df["state"].isna().sum()
            
        self.initial_errors = errors

    def _clamp_score(self, score: float) -> float:
        ranges = {
            "task1": (0.05, 0.98),
            "task2": (0.05, 0.97),
            "task3": (0.05, 0.95)
        }
        low, high = ranges.get(self.current_task_name, (0.05, 0.95))
        clamped = min(1.0, max(0.0, score))
        return low + (clamped * (high - low))

    def step(self, action: DataCleanTransformAction) -> DataCleanTransformObservation:  # type: ignore[override]
        """
        Execute a cleaning operation on the dataset.
        """
        self._state.step_count += 1
        
        if self.df is None:
            return self._make_observation("Error: Environment not reset.", reward=0.0)

        df = self.df

        feedback = ""
        done = False
        
        try:
            if action.operation == "drop_duplicates":
                before = len(df)
                subset = action.kwargs.get("subset")
                df.drop_duplicates(subset=subset, inplace=True)
                after = len(df)
                feedback = f"Dropped {before - after} duplicates."
                
            elif action.operation == "drop_na":
                before = len(df)
                subset = action.kwargs.get("subset")
                df.dropna(subset=subset, inplace=True)
                after = len(df)
                feedback = f"Dropped {before - after} rows with missing values."
                
            elif action.operation == "fill_na":
                col = action.column
                val = action.value
                if col in df.columns:
                    df[col] = df[col].fillna(val)
                    feedback = f"Filled missing values in column '{col}' with '{val}'."
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.operation == "astype":
                col = action.column
                target_type = action.value
                if col in df.columns:
                    df[col] = df[col].astype(target_type)
                    feedback = f"Converted column '{col}' to {target_type}."
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.operation == "str_replace":
                col = action.column
                pat = action.kwargs.get("pat", "")
                repl = action.kwargs.get("repl", "")
                regex = action.kwargs.get("regex", True)
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(pat, repl, regex=regex)
                    feedback = f"Replaced pattern '{pat}' with '{repl}' in column '{col}'."
                else:
                    feedback = f"Error: Column '{col}' not found."
            
            elif action.operation == "to_datetime":
                col = action.column
                kwargs = action.kwargs
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], **kwargs)
                    feedback = f"Converted column '{col}' to datetime."
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.operation == "replace_map":
                col = action.column
                mapping = action.kwargs.get("mapping")
                if col in df.columns and isinstance(mapping, dict):
                    df[col] = df[col].replace(mapping)
                    feedback = f"Replaced values in '{col}' according to mapping."
                else:
                    feedback = f"Error: Column '{col}' not found or mapping is not a dictionary."
                    
            elif action.operation == "impute_from_column":
                col = action.column
                source_col = action.kwargs.get("source_column")
                mapping = action.kwargs.get("mapping")
                if col in df.columns and source_col in df.columns and isinstance(mapping, dict):
                    # fill only missing values in col
                    df[col] = df[col].fillna(df[source_col].map(mapping))
                    feedback = f"Imputed missing values in '{col}' using mapping from '{source_col}'."
                else:
                    feedback = f"Error: Invalid columns or mapping."

            elif action.operation == "finish":
                done = True
                feedback = "Transformation finished. Evaluating results..."
            
            else:
                feedback = f"Error: Unknown operation '{action.operation}'."
                
        except Exception as e:
            feedback = f"Error performing operation '{action.operation}': {str(e)}"

        # Calculate reward
        reward, score_feedback = self._calculate_reward()
        if feedback and not feedback.startswith("Error"):
            feedback += " " + score_feedback
        else:
            feedback = feedback or score_feedback

        return self._make_observation(feedback, reward=reward, done=done)

    def _calculate_reward(self) -> tuple[float, str]:
        """
        Calculate continuous partial reward based on progress toward the goal.
        """
        if self.df is None:
            return self._clamp_score(0.0), ""
            
        df = self.df
        errors = self.initial_errors
            
        if self.current_task_name == "task1":
            # Continuous: (duplicates_fixed / init_duplicates) * 0.4 + (nulls_fixed / init_nulls) * 0.6
            curr_dupes = df.duplicated().sum()
            curr_nulls = df[["id", "email"]].isna().any(axis=1).sum()
            
            init_dupes = max(1, errors.get("duplicates", 0))
            init_nulls = max(1, errors.get("missing_critical", 0))
            
            dupe_reward = max(0, 1 - (curr_dupes / init_dupes)) * 0.4
            null_reward = max(0, 1 - (curr_nulls / init_nulls)) * 0.6
            
            score = dupe_reward + null_reward
            
            clamped_score = self._clamp_score(score)
            return clamped_score, f"Current progress score: {clamped_score:.2f}"
            
        elif self.current_task_name == "task2":
            # Continuous: % of columns correctly typed/formatted
            curr_messy_salary = len(df) - pd.to_numeric(df["salary"].astype(str).str.replace(r"[\$,\s\€\£\¥]", "", regex=True), errors="coerce").notna().sum()
            
            try:
                if pd.api.types.is_datetime64_any_dtype(df["date_joined"]):
                    curr_messy_date = 0
                else:
                    curr_messy_date = len(df) - pd.to_datetime(df["date_joined"], errors="coerce", format="mixed").notna().sum()
            except:
                curr_messy_date = len(df)

            init_salary = max(1, errors.get("messy_salary", 0))
            init_date = max(1, errors.get("messy_date", 0))
            
            salary_reward = max(0, 1 - (curr_messy_salary / init_salary)) * 0.5
            date_reward = max(0, 1 - (curr_messy_date / init_date)) * 0.5
            
            score = salary_reward + date_reward
            clamped_score = self._clamp_score(score)
            return clamped_score, f"Current progress score: {clamped_score:.2f}"
            
        elif self.current_task_name == "task3":
            # Continuous: % of city/state corrected
            valid_cities = {"New York", "Los Angeles"}
            curr_messy_city = len(df) - df["city"].isin(valid_cities).sum()
            curr_missing_state = df["state"].isna().sum()
            
            init_city = max(1, errors.get("messy_city", 0))
            init_state = max(1, errors.get("missing_state", 0))
            
            city_reward = max(0, 1 - (curr_messy_city / init_city)) * 0.5
            state_reward = max(0, 1 - (curr_missing_state / init_state)) * 0.5
            
            score = city_reward + state_reward
            clamped_score = self._clamp_score(score)
            return clamped_score, f"Current progress score: {clamped_score:.2f}"
            
        return self._clamp_score(0.0), ""

    def _make_observation(self, feedback: str, reward: Optional[float] = None, done: bool = False) -> DataCleanTransformObservation:
        """Helper to create an observation from current state."""
        if reward is None:
            reward = self._clamp_score(0.0)
        
        df = self.df
        if df is not None:
            # Get first 10 rows
            dataset_head = df.head(10).to_csv(index=False)
            
            # Get info
            buffer = io.StringIO()
            df.info(buf=buffer)
            dataset_info = buffer.getvalue()
        else:
            dataset_head = ""
            dataset_info = "No dataset loaded."
            
        task_info = self.tasks.get(self.current_task_name, {})
            
        return DataCleanTransformObservation(
            dataset_head=dataset_head,
            dataset_info=dataset_info,
            last_action_feedback=feedback,
            current_task=task_info.get("name", "None"),
            task_description=task_info.get("description", ""),
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.
        """
        return self._state
