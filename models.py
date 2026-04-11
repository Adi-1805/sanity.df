# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Data Clean Transform Environment.
"""

from typing import Any, Dict, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DataCleanTransformAction(Action):
    """Action for the Data Clean Transform environment - a structured cleaning operation."""

    operation: str = Field(
        ..., 
        description="The cleaning operation to perform: 'drop_duplicates', 'drop_na', 'fill_na', 'astype', 'str_replace', 'to_datetime', 'replace_map', 'impute_from_column', 'impute', 'scale', 'transform', 'feature_eng', 'split_column', 'finish'"
    )
    column: Optional[str] = Field(None, description="Column to apply the operation to (if applicable)")
    value: Any = Field(None, description="Value for replacement, filling, or the target type (if applicable)")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional arguments for the operation (e.g., {'subset': ['email']})")


class DataCleanTransformObservation(Observation):
    """Observation from the Data Clean Transform environment - dataset state and info."""

    dataset_head: str = Field(..., description="First few rows of the dataset as a CSV string")
    dataset_info: str = Field(..., description="Metadata including column names, types, and null counts")
    last_action_feedback: str = Field(default="", description="Feedback from the previous action (e.g., success or error message)")
    current_task: str = Field(..., description="The name of the current task being performed")
    task_description: str = Field(default="", description="Detailed instructions for the current task")
