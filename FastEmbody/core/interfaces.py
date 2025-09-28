"""Protocol definitions for pluggable framework components."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol

from FastEmbody.typing import EnvInfo, VectorEnv


class AlgorithmHandle(Protocol):
    """Minimal interface exposed by training algorithms."""

    def learn(
        self,
        num_iterations: int,
        *,
        init_at_random_ep_len: bool = True,
        progress_bar: bool = True,
    ) -> None:
        """Run the training loop."""

    def evaluate(self, num_episodes: int) -> dict[str, Any]:
        """Optional evaluation phase."""

    def save(self, path: str) -> None:
        """Persist checkpoints to disk."""

    def load(self, path: str) -> None:
        """Load checkpoints from disk."""


class AlgorithmBuilder(Protocol):
    """Factory that turns configs into :class:`AlgorithmHandle` instances."""

    def build(self, env: VectorEnv, env_info: EnvInfo, **kwargs: Any) -> AlgorithmHandle:
        ...


class Trainer(Protocol):
    """Co-ordinates training by gluing envs and algorithms."""

    def run(
        self,
        algorithm: AlgorithmHandle,
        *,
        num_iterations: int,
        checkpoints: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    def evaluate(self, algorithm: AlgorithmHandle, episodes: int) -> dict[str, Any]:
        ...


class Hook(Protocol):
    """Runtime callback hook used by trainers to broadcast events."""

    def on_iteration(self, iteration: int, metrics: dict[str, Any]) -> None:
        ...

    def on_checkpoint(self, path: str) -> None:
        ...

    def on_eval(self, iteration: int, metrics: dict[str, Any]) -> None:
        ...


class Hookable(Protocol):
    def register_hooks(self, hooks: Iterable[Hook]) -> None:
        ...
