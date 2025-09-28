"""Trainer registry and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from FastEmbody.core.interfaces import AlgorithmHandle, Trainer
from FastEmbody.core.registries import Registry


TRAINER_REGISTRY: Registry[Trainer] = Registry("trainer")


@dataclass
class TrainerParams:
    num_iterations: int = 1000
    init_at_random_ep_len: bool = True
    progress_bar: bool = True


class TrainerWrapper(Trainer):
    """Wrap an algorithm to expose the :class:`Trainer` protocol."""

    def __init__(self, params: TrainerParams):
        self.params = params

    def run(self, algorithm: AlgorithmHandle, *, num_iterations: int | None = None, checkpoints: dict[str, Any] | None = None) -> None:
        iterations = num_iterations or self.params.num_iterations
        algorithm.learn(
            iterations,
            init_at_random_ep_len=self.params.init_at_random_ep_len,
            progress_bar=self.params.progress_bar,
        )

    def evaluate(self, algorithm: AlgorithmHandle, episodes: int) -> dict[str, Any]:
        return algorithm.evaluate(episodes)
