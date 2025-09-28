"""Example training script using the FastEmbody framework."""

from __future__ import annotations

import argparse

from FastEmbody.algorithms import ALGORITHM_REGISTRY  # ensure registrations
from FastEmbody.core import load_experiment_config
from FastEmbody.envs import ENV_REGISTRY
from FastEmbody.pipelines import ExperimentPipeline
from FastEmbody.trainers import TRAINER_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the toy Go2 environment with PPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unitree_go2_ppo.json",
        help="Path to the experiment config (JSON or YAML)",
    )
    parser.add_argument("--num-iterations", type=int, dest="num_iterations", help="Override training iterations")
    parser.add_argument("--log-dir", type=str, dest="log_dir", help="Override logging directory")
    parser.add_argument("--device", type=str, help="Device for the environment and algorithm (e.g. cpu, cuda:0)")
    parser.add_argument("--num-envs", type=int, help="Override number of parallel environments")
    return parser.parse_args()


def _override_runtime(config, args: argparse.Namespace) -> None:
    runtime = config.env.params.setdefault("runtime", {})
    if args.device is not None:
        runtime["device"] = args.device
        config.algorithm.params["device"] = args.device
    if args.num_envs is not None:
        runtime["num_envs"] = args.num_envs

    if args.num_iterations is not None:
        config.trainer.params["num_iterations"] = args.num_iterations
    if args.log_dir is not None:
        config.logging.log_dir = args.log_dir


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    _override_runtime(config, args)

    # lazily ensure registries are populated
    _ = ENV_REGISTRY
    _ = ALGORITHM_REGISTRY
    _ = TRAINER_REGISTRY

    pipeline = ExperimentPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
