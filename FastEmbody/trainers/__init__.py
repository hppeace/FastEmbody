"""Default trainer implementations."""

from __future__ import annotations

from typing import Any

from FastEmbody.core.interfaces import Trainer
from FastEmbody.trainers.base import TRAINER_REGISTRY, TrainerParams, TrainerWrapper


@TRAINER_REGISTRY.register("on_policy", description="Generic wrapper around algorithms exposing a learn() loop")
def build_on_policy_trainer(**params: Any) -> Trainer:
    config = TrainerParams(**params)
    return TrainerWrapper(config)


__all__ = ["TRAINER_REGISTRY"]
