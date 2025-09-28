"""Utility helpers shared by multiple algorithms."""

from __future__ import annotations

from typing import Callable, Dict

import torch


def evaluate_vector_env(
    env,
    *,
    device: torch.device,
    policy_fn: Callable[[torch.Tensor], torch.Tensor],
    num_episodes: int,
) -> Dict[str, float]:
    """Run deterministic rollouts and compute mean episodic reward."""

    obs = env.reset()
    returns = torch.zeros(env.num_envs, device=device)
    completed = 0
    episode_returns: list[float] = []

    while completed < num_episodes:
        obs_tensor = obs.to(device)
        actions = policy_fn(obs_tensor)
        next_obs, reward, done, _ = env.step(actions.detach().to(env.device))
        returns += reward.to(device)
        done_mask = done > 0.5
        for idx in torch.nonzero(done_mask, as_tuple=False).flatten().tolist():
            episode_returns.append(float(returns[idx].cpu()))
            returns[idx] = 0.0
            completed += 1
            if completed >= num_episodes:
                break
        obs = next_obs

    mean_reward = sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
    return {"mean_episode_reward": mean_reward}


def save_model(module: torch.nn.Module, path: str) -> None:
    torch.save({"model": module.state_dict()}, path)


def load_model(module: torch.nn.Module, path: str, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    module.load_state_dict(state)


__all__ = ["evaluate_vector_env", "save_model", "load_model"]
