"""Isaac Gym environment builder mirroring unitree_rl_gym."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from isaacgym import gymapi

from unitree_rl_gym.legged_gym.utils import helpers
from unitree_rl_gym.legged_gym.utils.task_registry import task_registry as unitree_registry

from FastEmbody.typing import EnvInfo, VectorEnv


@dataclass
class UnitreeRuntimeConfig:
    headless: bool = True
    physics_engine: str = "physx"
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    num_envs: Optional[int] = None
    seed: Optional[int] = None
    use_gpu_pipeline: bool = True
    use_gpu: bool = True
    subscenes: int = 0
    num_threads: int = 0


def _parse_device(spec: str) -> tuple[str, int]:
    if ":" in spec:
        dev, idx = spec.split(":", maxsplit=1)
        return dev, int(idx)
    return spec, 0


def _physics_engine(name: str) -> int:
    lowered = name.lower()
    if lowered == "physx":
        return gymapi.SIM_PHYSX
    if lowered == "flex":
        return gymapi.SIM_FLEX
    raise ValueError(f"Unsupported physics engine: {name}")


class UnitreeIsaacEnv(VectorEnv):
    """Wrap unitree_rl_gym legged environments to match framework expectations."""

    def __init__(self, env) -> None:
        self._env = env
        self.num_envs = env.num_envs
        self.observation_size = env.num_obs
        self.action_size = env.num_actions
        self.device = torch.device(env.device)

    def reset(self) -> torch.Tensor:
        obs, _ = self._env.reset()
        return obs

    def step(self, actions: torch.Tensor):
        obs, _, reward, dones, info = self._env.step(actions)
        done_tensor = dones.to(reward.dtype)
        return obs, reward, done_tensor, info

    def close(self) -> None:
        viewer = getattr(self._env, "viewer", None)
        if viewer is not None:
            self._env.gym.destroy_viewer(viewer)
        if hasattr(self._env, "gym") and hasattr(self._env, "sim"):
            self._env.gym.destroy_sim(self._env.sim)


def build_unitree_env(*, name: str, params: Dict[str, Any] | None = None):
    params_dict = dict(params or {})
    runtime_cfg_dict = params_dict.pop("runtime", {})
    runtime = UnitreeRuntimeConfig(**runtime_cfg_dict)

    env_cfg_overrides = params_dict.pop("env_cfg_overrides", None)
    train_cfg_overrides = params_dict.pop("train_cfg_overrides", None)

    env_cfg, train_cfg = unitree_registry.get_cfgs(name)
    env_cfg = copy.deepcopy(env_cfg)
    train_cfg = copy.deepcopy(train_cfg)

    if runtime.num_envs is not None:
        env_cfg.env.num_envs = runtime.num_envs
    if runtime.seed is not None:
        train_cfg.seed = runtime.seed
    env_cfg.seed = train_cfg.seed

    if env_cfg_overrides:
        helpers.update_class_from_dict(env_cfg, env_cfg_overrides)
    if train_cfg_overrides:
        helpers.update_class_from_dict(train_cfg, train_cfg_overrides)

    helpers.set_seed(train_cfg.seed)

    sim_type, sim_id = _parse_device(runtime.sim_device)
    args = SimpleNamespace(
        task=name,
        resume=False,
        horovod=False,
        headless=runtime.headless,
        physics_engine=_physics_engine(runtime.physics_engine),
        sim_device=runtime.sim_device,
        sim_device_type=sim_type,
        sim_device_id=sim_id,
        rl_device=runtime.rl_device,
        device=f"{sim_type}:{sim_id}" if sim_type != "cpu" else "cpu",
        use_gpu=runtime.use_gpu,
        use_gpu_pipeline=runtime.use_gpu_pipeline,
        subscenes=runtime.subscenes,
        num_threads=runtime.num_threads,
        num_envs=runtime.num_envs,
        seed=runtime.seed,
        experiment_name=None,
        run_name=None,
        max_iterations=None,
        load_run=None,
        checkpoint=None,
    )

    sim_cfg_dict = {"sim": helpers.class_to_dict(env_cfg.sim)}
    sim_params = helpers.parse_sim_params(args, sim_cfg_dict)

    task_class = unitree_registry.get_task_class(name)
    env = task_class(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )

    wrapped_env = UnitreeIsaacEnv(env)

    env_info = EnvInfo(
        name=name,
        num_envs=env.num_envs,
        num_observations=env.num_obs,
        num_actions=env.num_actions,
        extras={"train_cfg": train_cfg},
    )
    raw_cfg = {"env_cfg": env_cfg, "train_cfg": train_cfg}
    return wrapped_env, env_info, raw_cfg


__all__ = ["build_unitree_env", "UnitreeIsaacEnv", "UnitreeRuntimeConfig"]
