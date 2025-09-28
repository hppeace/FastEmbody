"""High-level orchestration utilities for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from FastEmbody.algorithms.base import ALGORITHM_REGISTRY
from FastEmbody.core.config import ExperimentConfig
from FastEmbody.core.interfaces import AlgorithmHandle
from FastEmbody.envs import ENV_REGISTRY
from FastEmbody.trainers import TRAINER_REGISTRY


@dataclass
class ExperimentArtifacts:
    env: Any
    env_info: Any
    algorithm: AlgorithmHandle
    trainer: Any
    raw_cfg: Dict[str, Any]


class ExperimentPipeline:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _build_algorithm_params(self) -> Dict[str, Any]:
        params = dict(self.config.algorithm.params)
        if "device" not in params:
            params["device"] = "cpu"
        return params

    def _build_trainer_params(self, raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(self.config.trainer.params)
        params.setdefault("num_iterations", 1000)
        return params

    def setup(self) -> ExperimentArtifacts:
        env, env_info, raw_cfg = ENV_REGISTRY.build(
            self.config.env.builder,
            name=self.config.env.name,
            params=self.config.env.params,
        )

        algorithm_builder = ALGORITHM_REGISTRY.build(
            self.config.algorithm.builder,
            **self._build_algorithm_params(),
        )
        algorithm = algorithm_builder.build(
            env,
            env_info,
            name=self.config.algorithm.name,
        )

        trainer = TRAINER_REGISTRY.build(
            self.config.trainer.name,
            **self._build_trainer_params(raw_cfg),
        )

        return ExperimentArtifacts(
            env=env,
            env_info=env_info,
            algorithm=algorithm,
            trainer=trainer,
            raw_cfg=raw_cfg,
        )

    def run(self) -> Dict[str, Any]:
        artifacts = self.setup()
        metrics = {}
        try:
            artifacts.trainer.run(
                artifacts.algorithm,
                num_iterations=self._build_trainer_params(artifacts.raw_cfg)["num_iterations"],
            )
        finally:
            artifacts.env.close()
        return metrics

    def evaluate(self, episodes: int = 10) -> Dict[str, Any]:
        artifacts = self.setup()
        try:
            return artifacts.trainer.evaluate(artifacts.algorithm, episodes)
        finally:
            artifacts.env.close()
