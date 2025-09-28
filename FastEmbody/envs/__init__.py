"""Environment registry exposing toy and Isaac Gym builders."""

from __future__ import annotations

from typing import Any

from FastEmbody.core.registries import Registry
from FastEmbody.envs.go2 import build_go2_env
from FastEmbody.envs.unitree import build_unitree_env


ENV_REGISTRY: Registry[Any] = Registry("environment")


@ENV_REGISTRY.register("toy_go2", description="Toy Go2 locomotion environment for testing")
def _build_toy_go2(**kwargs: Any):
    return build_go2_env(**kwargs)


@ENV_REGISTRY.register("go2", description="Alias for the toy Go2 environment")
def _build_go2_alias(**kwargs: Any):
    return build_go2_env(**kwargs)


@ENV_REGISTRY.register("unitree", description="Isaac Gym Unitree legged robot tasks")
def _build_unitree(**kwargs: Any):
    return build_unitree_env(**kwargs)


__all__ = ["ENV_REGISTRY"]
