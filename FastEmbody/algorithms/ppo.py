"""Self-contained PPO implementation for the toy Go2 environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from FastEmbody.algorithms.base import ALGORITHM_REGISTRY
from FastEmbody.algorithms.utils import evaluate_vector_env, load_model, save_model
from FastEmbody.core.interfaces import AlgorithmBuilder, AlgorithmHandle
from FastEmbody.models import MLP
from FastEmbody.typing import EnvInfo, VectorEnv


@dataclass
class PPOConfig:
    rollout_length: int = 64
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    learning_rate: float = 3e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_sizes: Iterable[int] = field(default_factory=lambda: (128, 128))
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        self.policy_net = MLP(obs_dim, action_dim, hidden_sizes, activation="tanh")
        self.value_net = MLP(obs_dim, 1, hidden_sizes, activation="tanh")
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        mean = self.policy_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        value = self.value_net(obs).squeeze(-1)
        return mean, value, dist

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Normal]:
        mean, value, dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value, dist

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Normal]:
        mean, value, dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy, dist


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.step = 0
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

    def add(self, obs, actions, log_prob, reward, done, value):
        self.observations[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_prob)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.values[self.step].copy_(value)
        self.step += 1

    def reset(self):
        self.step = 0

    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float):
        next_advantage = torch.zeros(self.num_envs, device=self.device)
        for step in reversed(range(self.num_steps)):
            mask = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * last_value * mask - self.values[step]
            next_advantage = delta + gamma * lam * mask * next_advantage
            self.advantages[step] = next_advantage
            self.returns[step] = self.advantages[step] + self.values[step]
            last_value = self.values[step]

    def iter_batches(self, num_mini_batches: int) -> Iterator[Tuple[torch.Tensor, ...]]:
        batch_size = self.num_steps * self.num_envs
        indices = torch.randperm(batch_size, device=self.device)
        mini_batch_size = batch_size // num_mini_batches

        obs = self.observations.reshape(batch_size, -1)
        actions = self.actions.reshape(batch_size, -1)
        log_probs = self.log_probs.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        values = self.values.reshape(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = indices[start:end]
            yield (
                obs[mb_inds],
                actions[mb_inds],
                log_probs[mb_inds],
                returns[mb_inds],
                advantages[mb_inds],
                values[mb_inds],
            )


class PPOAgent(AlgorithmHandle):
    def __init__(self, env: VectorEnv, env_info: EnvInfo, cfg: PPOConfig):
        self.env = env
        self.env_info = env_info
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = ActorCritic(env_info.num_observations, env_info.num_actions, cfg.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.buffer = RolloutBuffer(
            cfg.rollout_length,
            env.num_envs,
            env_info.num_observations,
            env_info.num_actions,
            self.device,
        )

    # --------------------------------------------------------------
    # Core PPO routine
    # --------------------------------------------------------------
    def _collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        self.buffer.reset()
        for _ in range(self.cfg.rollout_length):
            obs_tensor = obs.to(self.device)
            actions, log_prob, value, _ = self.policy.act(obs_tensor)
            next_obs, reward, done, _ = self.env.step(actions.detach().to(self.env.device))

            reward = reward.to(self.device)
            done = done.to(self.device)
            values = value.detach()

            self.buffer.add(obs_tensor, actions.detach(), log_prob.detach(), reward, done, values)
            obs = next_obs
        with torch.no_grad():
            obs_tensor = obs.to(self.device)
            _, last_value, _ = self.policy.forward(obs_tensor)
        self.buffer.compute_returns(last_value.detach(), self.cfg.gamma, self.cfg.lam)
        return obs

    def _update_policy(self):
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages.copy_(advantages)

        for _ in range(self.cfg.num_learning_epochs):
            for batch in self.buffer.iter_batches(self.cfg.num_mini_batches):
                obs_batch, action_batch, old_log_probs, returns_batch, adv_batch, _ = batch
                log_probs, values, entropy, _ = self.policy.evaluate(obs_batch, action_batch)

                ratio = torch.exp(log_probs - old_log_probs)
                surrogate1 = ratio * adv_batch
                surrogate2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * adv_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = 0.5 * (returns_batch - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.value_loss_coef * value_loss
                    - self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

    # --------------------------------------------------------------
    # AlgorithmHandle API
    # --------------------------------------------------------------
    def learn(
        self,
        num_iterations: int,
        *,
        init_at_random_ep_len: bool = True,
        progress_bar: bool = True,
    ) -> None:
        del init_at_random_ep_len, progress_bar  # placeholders to keep interface compatibility
        obs = self.env.reset()
        for _ in range(num_iterations):
            obs = self._collect_rollout(obs)
            self._update_policy()

    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        return evaluate_vector_env(
            self.env,
            device=self.device,
            policy_fn=lambda obs: self.policy.forward(obs)[0],
            num_episodes=num_episodes,
        )

    def save(self, path: str) -> None:
        save_model(self.policy, path)

    def load(self, path: str) -> None:
        load_model(self.policy, path, self.device)


class PPOBuilder(AlgorithmBuilder):
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg

    def build(self, env: VectorEnv, env_info: EnvInfo, **kwargs) -> AlgorithmHandle:
        return PPOAgent(env, env_info, self.cfg)


@ALGORITHM_REGISTRY.register("ppo", description="Self-contained PPO implementation")
def register_ppo(**params) -> AlgorithmBuilder:
    cfg = PPOConfig(**params)
    return PPOBuilder(cfg)


__all__ = ["PPOConfig", "PPOAgent", "register_ppo"]
