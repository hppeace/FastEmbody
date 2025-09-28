"""Shared environment configuration structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvRuntimeConfig:
    """Lightweight runtime options for vectorised toy environments."""

    num_envs: int = 1024
    seed: int = 1
    device: str = "cpu"
    episode_length: int = 400
