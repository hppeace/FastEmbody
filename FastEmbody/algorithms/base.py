"""Algorithm utilities shared across implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from FastEmbody.core.interfaces import AlgorithmBuilder
from FastEmbody.core.registries import Registry


ALGORITHM_REGISTRY: Registry[AlgorithmBuilder] = Registry("algorithm")


@dataclass
class AlgorithmBuildContext:
    env_name: str
    env_metadata: Dict[str, Any]
    experiment_params: Dict[str, Any]
