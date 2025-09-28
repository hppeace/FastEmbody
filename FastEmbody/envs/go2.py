"""Toy vectorised Go2 locomotion environment used for PPO training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from FastEmbody.envs.base import EnvRuntimeConfig
from FastEmbody.typing import EnvInfo, VectorEnv


def _to_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


@dataclass
class Go2EnvConfig:
    """Configurable parameters controlling the simplified Go2 dynamics."""

    runtime: EnvRuntimeConfig = EnvRuntimeConfig()
    joint_damping: float = 0.10
    joint_stiffness: float = 0.05
    torque_limit: float = 3.0
    dt: float = 0.02
    command_interval: int = 40
    command_scale: float = 1.0
    noise_scale: float = 0.03
    observation_noise: float = 0.01
    reward_tracking_weight: float = 1.0
    reward_posture_weight: float = 0.05
    reward_action_weight: float = 0.01

    @property
    def num_envs(self) -> int:
        return self.runtime.num_envs

    @property
    def episode_length(self) -> int:
        return self.runtime.episode_length

    @property
    def device(self) -> torch.device:
        return _to_device(self.runtime.device)


class Go2VectorEnv(VectorEnv):
    """A differentiable toy model of a quadruped tracking velocity commands."""

    def __init__(self, cfg: Go2EnvConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.num_envs = cfg.num_envs
        self.num_joints = 12
        self.command_dim = 3  # vx, vy, yaw
        self.observation_size = self.num_joints * 2 + self.command_dim + 1  # + phase indicator
        self.action_size = self.num_joints

        self.state_pos = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.state_vel = torch.zeros_like(self.state_pos)
        self.commands = torch.zeros(self.num_envs, self.command_dim, device=self.device)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(self.num_envs, 1, device=self.device)
        torch.manual_seed(cfg.runtime.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.runtime.seed)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _sample_commands(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        num_to_sample = int(mask.sum().item())
        if num_to_sample == 0:
            return
        commands = (torch.rand(num_to_sample, self.command_dim, device=self.device) * 2 - 1)
        commands *= self.cfg.command_scale
        self.commands[mask] = commands

    def _get_observations(self) -> torch.Tensor:
        noise = self.cfg.observation_noise
        pieces = [self.state_pos, self.state_vel, self.commands, self.phase]
        obs = torch.cat(pieces, dim=-1)
        if noise > 0:
            obs = obs + noise * torch.randn_like(obs)
        return obs

    def _integrate_dynamics(self, actions: torch.Tensor) -> None:
        dt = self.cfg.dt
        torque_limit = self.cfg.torque_limit
        damping = self.cfg.joint_damping
        stiffness = self.cfg.joint_stiffness

        torques = torch.clamp(actions, -1.0, 1.0) * torque_limit
        if self.cfg.noise_scale > 0:
            torques = torques + self.cfg.noise_scale * torch.randn_like(torques)
        vel_acc = torques - damping * self.state_vel - stiffness * self.state_pos
        self.state_vel = self.state_vel + dt * vel_acc
        self.state_pos = self.state_pos + dt * self.state_vel

        # simple phase oscillator to emulate gait progression
        self.phase = (self.phase + dt).fmod(2 * math.pi)

    def _compute_reward(self, actions: torch.Tensor) -> torch.Tensor:
        # Project joint velocities into coarse base velocity estimates
        vel_forward = self.state_vel[:, :4].mean(dim=1)
        vel_lateral = self.state_vel[:, 4:8].mean(dim=1)
        yaw_rate = self.state_vel[:, 8:].mean(dim=1)
        vel_stack = torch.stack([vel_forward, vel_lateral, yaw_rate], dim=-1)

        tracking_error = (vel_stack - self.commands).pow(2).sum(dim=-1)
        tracking_reward = torch.exp(-tracking_error)

        posture_penalty = (self.state_pos.pow(2).mean(dim=1))
        action_penalty = (actions.pow(2).mean(dim=1))

        reward = (
            self.cfg.reward_tracking_weight * tracking_reward
            - self.cfg.reward_posture_weight * posture_penalty
            - self.cfg.reward_action_weight * action_penalty
        )
        return reward

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        self.state_pos.uniform_(-0.1, 0.1)
        self.state_vel.zero_()
        self.episode_step.zero_()
        self.phase.zero_()
        self._sample_commands()
        return self._get_observations()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        self._integrate_dynamics(actions)
        rewards = self._compute_reward(actions)
        self.episode_step += 1

        # resample commands periodically to encourage agility
        if self.cfg.command_interval > 0:
            command_mask = (self.episode_step % self.cfg.command_interval == 0)
            if command_mask.any():
                self._sample_commands(command_mask)

        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_mask |= self.episode_step >= self.cfg.episode_length
        reset_mask |= (self.state_pos.abs().mean(dim=1) > 4.0)

        done = reset_mask.to(torch.float32)
        if reset_mask.any():
            self.state_pos[reset_mask].uniform_(-0.05, 0.05)
            self.state_vel[reset_mask].zero_()
            self.phase[reset_mask] = 0.0
            self.episode_step[reset_mask] = 0
            self._sample_commands(reset_mask)

        obs = self._get_observations()
        info = {
            "commands": self.commands.detach().clone(),
            "reward_tracking": rewards.detach().clone(),
        }
        return obs, rewards, done, info

    def close(self) -> None:
        pass


def build_go2_env(*, name: str, params: Dict[str, object] | None = None) -> tuple[Go2VectorEnv, EnvInfo, Dict[str, object]]:
    params_dict = dict(params or {})
    runtime_cfg_dict = params_dict.pop("runtime", {})
    runtime_cfg = EnvRuntimeConfig(**runtime_cfg_dict)
    env_cfg = Go2EnvConfig(runtime=runtime_cfg, **params_dict)
    env = Go2VectorEnv(env_cfg)
    info = EnvInfo(
        name=name,
        num_envs=env.num_envs,
        num_observations=env.observation_size,
        num_actions=env.action_size,
    )
    return env, info, {"env_cfg": env_cfg}


__all__ = ["Go2VectorEnv", "Go2EnvConfig", "build_go2_env"]
