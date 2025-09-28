"""Student-teacher distillation algorithm implemented in pure PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from FastEmbody.algorithms.base import ALGORITHM_REGISTRY
from FastEmbody.algorithms.utils import evaluate_vector_env, load_model, save_model
from FastEmbody.core.interfaces import AlgorithmBuilder, AlgorithmHandle
from FastEmbody.models import MLP
from FastEmbody.typing import EnvInfo, VectorEnv


@dataclass
class DistillationConfig:
    rollout_length: int = 32
    batch_size: int = 512
    student_epochs: int = 4
    student_hidden_sizes: Iterable[int] = field(default_factory=lambda: (128, 128))
    teacher_hidden_sizes: Iterable[int] = field(default_factory=lambda: (256, 256))
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = 1.0
    loss_type: str = "mse"  # choices: mse, huber
    temperature: float = 1.0
    teacher_checkpoint: Optional[str] = None
    student_checkpoint: Optional[str] = None
    output_activation: str = "tanh"
    device: str = "cpu"


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Iterable[int], output_activation: str) -> None:
        super().__init__()
        self.net = MLP(obs_dim, action_dim, hidden_sizes, activation="tanh", output_activation=output_activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DistillationAgent(AlgorithmHandle):
    def __init__(self, env: VectorEnv, env_info: EnvInfo, cfg: DistillationConfig) -> None:
        self.env = env
        self.env_info = env_info
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        obs_dim = env_info.num_observations
        action_dim = env_info.num_actions

        self.student = PolicyNetwork(obs_dim, action_dim, cfg.student_hidden_sizes, cfg.output_activation).to(self.device)
        self.teacher = PolicyNetwork(obs_dim, action_dim, cfg.teacher_hidden_sizes, cfg.output_activation).to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        if cfg.teacher_checkpoint:
            load_model(self.teacher, cfg.teacher_checkpoint, self.device)
        if cfg.student_checkpoint:
            load_model(self.student, cfg.student_checkpoint, self.device)

        self.optimizer = optim.Adam(
            self.student.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        loss_map = {
            "mse": nn.MSELoss,
            "huber": nn.SmoothL1Loss,
        }
        try:
            self.loss_fn = loss_map[cfg.loss_type]()
        except KeyError:
            raise ValueError(f"Unsupported loss type '{cfg.loss_type}'. Available: {list(loss_map)}")

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _collect_dataset(self, obs: torch.Tensor) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        obs_traces: List[torch.Tensor] = []
        action_traces: List[torch.Tensor] = []
        current_obs = obs
        for _ in range(self.cfg.rollout_length):
            obs_device = current_obs.to(self.device)
            with torch.no_grad():
                teacher_actions = self.teacher(obs_device)
                if self.cfg.temperature != 1.0:
                    teacher_actions = teacher_actions / self.cfg.temperature
            obs_traces.append(obs_device.detach())
            action_traces.append(teacher_actions.detach())

            next_obs, _, _, _ = self.env.step(teacher_actions.to(self.env.device))
            current_obs = next_obs
        stacked_obs = torch.stack(obs_traces, dim=0)
        stacked_actions = torch.stack(action_traces, dim=0)
        dataset_obs = stacked_obs.reshape(-1, stacked_obs.shape[-1])
        dataset_actions = stacked_actions.reshape(-1, stacked_actions.shape[-1])
        data = {"observations": dataset_obs, "actions": dataset_actions}
        return data, current_obs

    def _distill(self, observations: torch.Tensor, targets: torch.Tensor) -> float:
        total = observations.shape[0]
        batch_size = min(self.cfg.batch_size, total)
        indices = torch.arange(total, device=self.device)
        last_loss = 0.0
        for epoch in range(self.cfg.student_epochs):
            perm = indices[torch.randperm(total)]
            for start in range(0, total, batch_size):
                end = start + batch_size
                mb_idx = perm[start:end]
                obs_mb = observations[mb_idx]
                target_mb = targets[mb_idx]
                pred = self.student(obs_mb)
                loss = self.loss_fn(pred, target_mb)

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()
                last_loss = float(loss.detach().cpu())
        return last_loss

    # ------------------------------------------------------------------
    # AlgorithmHandle API
    # ------------------------------------------------------------------
    def learn(
        self,
        num_iterations: int,
        *,
        init_at_random_ep_len: bool = True,
        progress_bar: bool = True,
    ) -> None:
        del init_at_random_ep_len, progress_bar
        obs = self.env.reset()
        for _ in range(num_iterations):
            dataset, obs = self._collect_dataset(obs)
            observations = dataset["observations"].to(self.device)
            targets = dataset["actions"].to(self.device)
            self._distill(observations, targets)

    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        return evaluate_vector_env(
            self.env,
            device=self.device,
            policy_fn=self.student,
            num_episodes=num_episodes,
        )

    def save(self, path: str) -> None:
        save_model(self.student, path)

    def load(self, path: str) -> None:
        load_model(self.student, path, self.device)


class DistillationBuilder(AlgorithmBuilder):
    def __init__(self, cfg: DistillationConfig) -> None:
        self.cfg = cfg

    def build(self, env: VectorEnv, env_info: EnvInfo, **kwargs) -> AlgorithmHandle:
        return DistillationAgent(env, env_info, self.cfg)


@ALGORITHM_REGISTRY.register("distillation", description="Student-teacher distillation")
def register_distillation(**params) -> AlgorithmBuilder:
    cfg = DistillationConfig(**params)
    return DistillationBuilder(cfg)


__all__ = ["DistillationConfig", "DistillationAgent", "register_distillation"]
