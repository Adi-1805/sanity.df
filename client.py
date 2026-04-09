# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data Clean Transform Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import DataCleanTransformAction, DataCleanTransformObservation
except ImportError:
    from models import DataCleanTransformAction, DataCleanTransformObservation


class DataCleanTransformEnv(
    EnvClient[DataCleanTransformAction, DataCleanTransformObservation, State]
):
    """
    Client for the Data Clean Transform Environment.
    """

    def _step_payload(self, action: DataCleanTransformAction) -> Dict[str, Any]:
        """
        Convert DataCleanTransformAction to JSON payload for step message.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DataCleanTransformObservation]:
        """
        Parse server response into StepResult[DataCleanTransformObservation].
        """
        obs_data = payload.get("observation", {})
        observation = DataCleanTransformObservation(
            dataset_head=obs_data.get("dataset_head", ""),
            dataset_info=obs_data.get("dataset_info", ""),
            last_action_feedback=obs_data.get("last_action_feedback", ""),
            current_task=obs_data.get("current_task", ""),
            task_description=obs_data.get("task_description", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
