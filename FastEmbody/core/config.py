"""Structured configuration objects and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

try:  # optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - dependency optional
    yaml = None
import json

from .registries import Registry


@dataclass
class EnvConfig:
    name: str
    builder: str = "go2"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmConfig:
    name: str
    builder: str = "ppo"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "default_actor_critic"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    name: str = "on_policy"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    log_dir: Optional[str] = None
    interval: int = 10
    checkpoint_interval: int = 100


@dataclass
class ExperimentConfig:
    env: EnvConfig
    algorithm: AlgorithmConfig
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


T = TypeVar("T")


def _coerce_dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    return cls(**data)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML or JSON file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("PyYAML is required to load YAML configuration files")
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported config format: {path.suffix}")

    env = _coerce_dict_to_dataclass(raw["env"], EnvConfig)
    algorithm = _coerce_dict_to_dataclass(raw["algorithm"], AlgorithmConfig)
    trainer = _coerce_dict_to_dataclass(raw.get("trainer", {}), TrainerConfig)
    model = _coerce_dict_to_dataclass(raw.get("model", {}), ModelConfig)
    logging = _coerce_dict_to_dataclass(raw.get("logging", {}), LoggingConfig)

    return ExperimentConfig(
        env=env,
        algorithm=algorithm,
        trainer=trainer,
        model=model,
        logging=logging,
    )


def dump_default_config(
    path: str | Path,
    *,
    env_registry: Registry[Any],
    algorithm_registry: Registry[Any],
) -> None:
    """Create a scaffold configuration file listing registered components.

    This convenience helper is useful when bootstrapping new experiments.
    """

    template = {
        "env": {
            "name": next(iter(env_registry.keys()), "go2"),
            "builder": "go2",
            "params": {},
        },
        "algorithm": {
            "name": next(iter(algorithm_registry.keys()), "ppo"),
            "builder": "ppo",
            "params": {},
        },
        "trainer": {
            "name": "on_policy",
            "params": {"num_iterations": 1000},
        },
        "model": {"name": "default_actor_critic", "params": {}},
        "logging": {"log_dir": "./logs"},
    }

    path = Path(path)
    if path.suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError("Config template path must end with .json or .yaml")

    if path.suffix == ".json":
        with path.open("w", encoding="utf-8") as fp:
            json.dump(template, fp, indent=2)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required to write YAML configuration files")
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(template, fp)
