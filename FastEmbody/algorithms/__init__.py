"""Algorithm integrations."""

from FastEmbody.algorithms.base import ALGORITHM_REGISTRY
from FastEmbody.algorithms import distillation  # noqa: F401 - register distillation
from FastEmbody.algorithms import ppo  # noqa: F401 - ensure registration

__all__ = ["ALGORITHM_REGISTRY"]
