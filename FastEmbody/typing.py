"""Shared type aliases and Protocol definitions for the FastEmbody framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Protocol, Tuple

import torch


TensorDict = MutableMapping[str, torch.Tensor]
ConfigDict = Dict[str, Any]


class SupportsToDict(Protocol):
    def to_dict(self) -> Mapping[str, Any]:
        """Return a dictionary representation."""


class VectorEnv(Protocol):
    """Minimal protocol that training loops expect environments to follow."""

    num_envs: int
    observation_size: int
    action_size: int
    device: torch.device

    def reset(self) -> torch.Tensor:
        """Return the initial observations for every environment instance."""

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Perform a simulation step and return (obs, reward, done, info)."""

    def close(self) -> None:
        """Free any resources owned by the environment."""


@dataclass
class EnvInfo:
    """Metadata describing a constructed environment."""

    name: str
    num_envs: int
    num_observations: int
    num_actions: int
    action_space: Any | None = None
    observation_space: Any | None = None
    extras: Dict[str, Any] | None = None
