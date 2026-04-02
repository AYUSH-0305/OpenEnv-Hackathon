# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My First Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyFirstAction, MyFirstObservation


class MyFirstEnv(
    EnvClient[MyFirstAction, MyFirstObservation, State]
):
    """
    Client for the My First Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MyFirstEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(MyFirstAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MyFirstEnv.from_docker_image("my_first_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MyFirstAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MyFirstAction) -> Dict:
        """
        Convert MyFirstAction to JSON payload for step message.

        Args:
            action: MyFirstAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MyFirstObservation]:
        """
        Parse server response into StepResult[MyFirstObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MyFirstObservation
        """
        obs_data = payload.get("observation", {})
        observation = MyFirstObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
