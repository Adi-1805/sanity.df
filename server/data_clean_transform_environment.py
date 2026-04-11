# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data Clean Transform Environment Implementation (ML-Ready Preprocessing).
"""

import os
import io
import pandas as pd
import numpy as np
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
    An environment for simulating data cleaning and preprocessing tasks.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the data_clean_transform environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.df: Optional[pd.DataFrame] = None
        self.df_gold: Optional[pd.DataFrame] = None
        self.current_task_name: str = ""
        # Use environment variable if set (e.g., in Docker), otherwise use relative path
        self.data_dir = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
        
        # Load tasks configuration
        self.tasks = {
            "task1": {
                "name": "Task 1: Intelligent Imputation",
                "dataset_messy": "task1_messy.csv",
                "dataset_gold": "task1_gold.csv",
                "description": "Handle missing and mixed values. Split 'product_code' into 'product_category' and 'product_id'. Impute 'color' with Mode, 'weight_kg' with Mean, 'income_usd' with Median, and 'house_price' using KNN or regression on 'sqft'."
            },
            "task2": {
                "name": "Task 2: Scaling & Mathematical Transforms",
                "dataset_messy": "task2_messy.csv",
                "dataset_gold": "task2_gold.csv",
                "description": "Apply standard scalers. 'age' needs MinMax, 'sensor_reading' needs StandardScaler, 'stock_volume' needs RobustScaler, 'sparse_audio_signal' needs MaxAbsScaler, and 'engagement_time' needs Log Transformation (np.log1p)."
            },
            "task3": {
                "name": "Task 3: Domain-Driven Feature Construction",
                "dataset_messy": "task3_messy.csv",
                "dataset_gold": "task3_gold.csv",
                "description": "Construct new features. Create 'age_at_signup' (years), 'days_since_last_purchase' (assume current date is 2025-01-01), 'average_order_value' (total_spent / total_orders), and 'customer_lifetime' (days). Handle missing/zero values carefully."
            }
        }

    def reset(self, task_id: Optional[str] = None) -> DataCleanTransformObservation:
        """
        Reset the environment, loading the messy dataset for the current task.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Priority: 1. Passed task_id, 2. Env var (refreshed), 3. Current, 4. task1
        os_task_id = os.environ.get("TASK_ID")
        self.current_task_name = task_id or os_task_id or self.current_task_name or "task1"
        
        if self.current_task_name not in self.tasks:
            self.current_task_name = "task1"
            
        task_info = self.tasks[self.current_task_name]
        
        messy_path = os.path.join(self.data_dir, task_info["dataset_messy"])
        gold_path = os.path.join(self.data_dir, task_info["dataset_gold"])
        
        self.df = pd.read_csv(messy_path)
        self.df_gold = pd.read_csv(gold_path)
        
        # For dates in Task 3, parse them properly
        if self.current_task_name == "task3":
            for col in ['user_birthdate', 'account_created_date', 'last_purchase_date']:
                self.df[col] = pd.to_datetime(self.df[col])
                self.df_gold[col] = pd.to_datetime(self.df_gold[col])
        
        return self._make_observation("Environment reset and dataset loaded.")

    def step(self, action: DataCleanTransformAction) -> DataCleanTransformObservation:  # type: ignore[override]
        """
        Execute a cleaning operation on the dataset.
        """
        self._state.step_count += 1
        
        if self.df is None:
            return self._make_observation("Error: Environment not reset.", reward=0.01)

        df = self.df

        feedback = ""
        done = False
        
        try:
            if action.operation == "drop_duplicates":
                subset = action.kwargs.get("subset")
                df.drop_duplicates(subset=subset, inplace=True)
                feedback = "Dropped duplicates."
                
            elif action.operation == "drop_na":
                subset = action.kwargs.get("subset")
                df.dropna(subset=subset, inplace=True)
                feedback = "Dropped rows with missing values. (Warning: This may incur penalties!)"
                
            elif action.operation == "fill_na":
                col = action.column
                val = action.value
                if col in df.columns:
                    df[col] = df[col].fillna(val)
                    feedback = f"Filled missing values in '{col}' with '{val}'."
                else:
                    feedback = f"Error: Column '{col}' not found."

            elif action.operation == "impute":
                col = action.column
                strategy = action.kwargs.get("strategy", "mean") # 'mean', 'median', 'mode', 'knn'
                if strategy == "knn":
                    from sklearn.impute import KNNImputer
                    n_neighbors = action.kwargs.get("n_neighbors", 5)
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    cols = action.kwargs.get("cols", df.select_dtypes(include=[np.number]).columns)
                    df[cols] = imputer.fit_transform(df[cols])
                    feedback = f"Applied KNN Imputation to columns: {cols}."
                elif col in df.columns:
                    if strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == "mode":
                        df[col] = df[col].fillna(df[col].mode()[0])
                    feedback = f"Imputed '{col}' using {strategy}."
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.operation == "scale":
                from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
                col = action.column
                method = action.kwargs.get("method", "standard")
                
                scalers = {
                    "minmax": MinMaxScaler,
                    "standard": StandardScaler,
                    "robust": RobustScaler,
                    "maxabs": MaxAbsScaler
                }
                if method in scalers and col in df.columns:
                    scaler = scalers[method]()
                    df[col] = scaler.fit_transform(df[[col]])
                    feedback = f"Scaled '{col}' using {method}."
                else:
                    feedback = f"Error: Invalid column or method '{method}'."

            elif action.operation == "transform":
                col = action.column
                method = action.kwargs.get("method", "log1p")
                if col in df.columns:
                    if method == "log1p":
                        df[col] = np.log1p(df[col])
                    elif method == "log":
                        df[col] = np.log(df[col])
                    feedback = f"Applied {method} transform to '{col}'."
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.operation == "feature_eng":
                # A safe eval alternative for specific feature engineering
                new_col = action.column
                formula = action.kwargs.get("formula")
                
                try:
                    try:
                        # Attempt standard pandas eval
                        df.eval(f"{new_col} = {formula}", inplace=True)
                    except Exception:
                        # Fallback for complex datetime/function operations not supported by df.eval
                        env_globals = {"pd": pd, "np": np, "df": df}
                        env_locals = {col: df[col] for col in df.columns}
                        df[new_col] = eval(formula, env_globals, env_locals)
                    feedback = f"Constructed feature '{new_col}' using formula '{formula}'."
                except Exception as e:
                    feedback = f"Error constructing feature '{new_col}': {e}"
            
            elif action.operation == "split_column":
                col = action.column
                pat = action.kwargs.get("pat", " ")
                new_cols = action.kwargs.get("new_cols", [])
                
                if col in df.columns and len(new_cols) > 0:
                    # Simple regex or str split
                    import re
                    # E.g., separating letters and numbers: pat = r"([A-Za-z]+)[-]?([0-9]+)"
                    extracted = df[col].astype(str).str.extract(pat)
                    for i, new_col in enumerate(new_cols):
                        if i < len(extracted.columns):
                            df[new_col] = extracted[i]
                    df.drop(columns=[col], inplace=True)
                    feedback = f"Split column '{col}' into {new_cols} and dropped original."
                else:
                    feedback = f"Error: Failed to split column '{col}'."

            elif action.operation == "astype":
                col = action.column
                target_type = action.value
                if col in df.columns:
                    # special case for datetime since eval/astype has limits
                    if "datetime" in str(target_type):
                        df[col] = pd.to_datetime(df[col])
                    else:
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

    def _normalize_score(self, raw_score: float) -> float:
        """
        Hackathon strict requirement: Score must be normalized strictly between 0.0 and 1.0
        We map [0.0, 1.0] to [0.01, 0.99].
        """
        clamped = max(0.0, min(1.0, raw_score))
        return 0.01 + (clamped * 0.98)

    def _calculate_reward(self) -> tuple[float, str]:
        """
        Calculate continuous partial reward based on progress toward the goal.
        """
        if self.df is None or self.df_gold is None:
            return self._normalize_score(0.0), ""
            
        df = self.df
        gold = self.df_gold
        score = 0.0
            
        if self.current_task_name == "task1":
            # Max score 1.0. 20% each for: product_code, color, weight_kg, income_usd, house_price
            
            # 1. product_code -> product_category & product_id
            if 'product_category' in df.columns and 'product_id' in df.columns:
                try:
                    df_pid = pd.to_numeric(df['product_id'], errors='coerce')
                    cat_match = (df['product_category'] == gold['product_category']).sum() / len(gold)
                    id_match = (df_pid == gold['product_id']).sum() / len(gold)
                    score += 0.20 * ((cat_match + id_match) / 2)
                except:
                    pass
            
            # 2. color (Mode Imputed)
            if 'color' in df.columns:
                match = (df['color'] == gold['color']).sum() / len(gold)
                score += 0.20 * match
                
            # 3. weight_kg (Mean Imputed)
            if 'weight_kg' in df.columns:
                try:
                    df_w = pd.to_numeric(df['weight_kg'])
                    # close enough due to float precision
                    match = np.isclose(df_w, gold['weight_kg'], rtol=1e-3, atol=1).sum() / len(gold)
                    score += 0.20 * match
                except:
                    pass
                    
            # 4. income_usd (Median Imputed)
            if 'income_usd' in df.columns:
                try:
                    df_i = pd.to_numeric(df['income_usd'])
                    match = np.isclose(df_i, gold['income_usd'], rtol=1e-3, atol=1).sum() / len(gold)
                    score += 0.20 * match
                except:
                    pass
                    
            # 5. house_price (KNN Imputed)
            if 'house_price' in df.columns:
                try:
                    df_h = pd.to_numeric(df['house_price'])
                    match = np.isclose(df_h, gold['house_price'], rtol=1e-3, atol=1).sum() / len(gold)
                    score += 0.20 * match
                except:
                    pass

        elif self.current_task_name == "task2":
            # 5 columns, 20% each
            if 'age' in df.columns:
                try:
                    if np.isclose(df['age'].min(), 0) and np.isclose(df['age'].max(), 1):
                        score += 0.20
                except: pass
                
            if 'sensor_reading' in df.columns:
                try:
                    if np.isclose(df['sensor_reading'].mean(), 0, atol=1e-2) and np.isclose(df['sensor_reading'].std(ddof=0), 1, atol=1e-2):
                        score += 0.20
                except: pass
                
            if 'stock_volume' in df.columns:
                # Robust Scaler checks
                try:
                    q75, q25 = np.percentile(df['stock_volume'], [75, 25])
                    iqr = q75 - q25
                    if np.isclose(df['stock_volume'].median(), 0, atol=1e-2) and np.isclose(iqr, 1.0, atol=1e-2):
                        score += 0.20
                except: pass
                
            if 'sparse_audio_signal' in df.columns:
                # MaxAbs Scaler checks
                try:
                    if np.isclose(np.max(np.abs(df['sparse_audio_signal'])), 1.0) and (df['sparse_audio_signal'] == 0).sum() > 0:
                        score += 0.20
                except: pass
                
            if 'engagement_time' in df.columns:
                # log1p check
                try:
                    if np.isclose(df['engagement_time'], gold['engagement_time'], rtol=1e-2).sum() / len(gold) > 0.8:
                        score += 0.20
                except: pass

        elif self.current_task_name == "task3":
            # 4 target features, 25% each
            
            # age_at_signup
            if 'age_at_signup' in df.columns:
                try:
                    match = np.isclose(pd.to_numeric(df['age_at_signup']), gold['age_at_signup'], rtol=1e-2).sum() / len(gold)
                    score += 0.25 * match
                except: pass
                
            # days_since_last_purchase
            if 'days_since_last_purchase' in df.columns:
                try:
                    match = np.isclose(pd.to_numeric(df['days_since_last_purchase']), gold['days_since_last_purchase']).sum() / len(gold)
                    score += 0.25 * match
                except: pass
                
            # average_order_value
            if 'average_order_value' in df.columns:
                try:
                    match = np.isclose(pd.to_numeric(df['average_order_value']), gold['average_order_value'], rtol=1e-2).sum() / len(gold)
                    score += 0.25 * match
                except: pass
                
            # customer_lifetime
            if 'customer_lifetime' in df.columns:
                try:
                    match = np.isclose(pd.to_numeric(df['customer_lifetime']), gold['customer_lifetime']).sum() / len(gold)
                    score += 0.25 * match
                except: pass

        norm_score = self._normalize_score(score)
        
        # Penalties: Drop NA rows inappropriately
        if len(df) < len(gold):
            norm_score -= 0.1 # Penalty for dropping rows
            norm_score = max(0.01, norm_score)
            
        return norm_score, f"Current progress score: {norm_score:.2f}"

    def _make_observation(self, feedback: str, reward: Optional[float] = None, done: bool = False) -> DataCleanTransformObservation:
        """Helper to create an observation from current state."""
        if reward is None:
            reward = self._normalize_score(0.0)
        
        df = self.df
        if df is not None:
            # Get first 10 rows
            dataset_head = df.head(10).to_csv(index=False)
            
            # Get info including describe
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            
            # Add describe for numerical columns to help the agent decide on scalers/imputation
            try:
                describe_str = df.describe().to_csv()
                dataset_info = f"{info_str}\n\nStatistical Summary:\n{describe_str}"
            except:
                dataset_info = info_str
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
