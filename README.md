# FastEmbody RL Framework

A compact, fully self-contained reinforcement-learning stack for training and distilling policies on a simplified Unitree Go2 locomotion task. The framework mirrors larger embodied-intelligence codebases (registries, experiment configs, pipelines) without depending on `legged_gym`, `rsl_rl`, or `unitree_rl_gym`.

## Features

- Pure PyTorch PPO implementation with configurable network sizes and optimisation hyper-parameters.
- Student-teacher distillation loop that reuses the same registries and environment abstractions.
- Toy vectorised Go2 environment for quick prototyping plus full Isaac Gym Unitree tasks for high-fidelity simulation.
- Registry-driven architecture for environments, algorithms, and trainers.
- Experiment configuration via YAML/JSON dataclasses and a high-level training pipeline.

## Layout

```
FastEmbody/
├── FastEmbody/                # Framework source code
│   ├── algorithms/          # PPO implementation + registry glue
│   ├── core/                # Config loaders, interfaces, registries
│   ├── envs/                # Go2 toy environment and runtime config
│   ├── pipelines/           # Experiment orchestration utilities
│   └── trainers/            # Trainer wrappers built on AlgorithmHandle
├── configs/                 # Example experiment configs
├── docs/                    # Design notes
└── examples/                # Training entry-points
```

## Quick Start

1. (Optional) create a virtual environment and install requirements:

   ```bash
   pip install torch
   ```

2. Launch PPO training on the toy Go2 task:

   ```bash
   python examples/train_unitree_go2.py --config configs/unitree_go2_ppo.json
   ```

   The script loads the config, instantiates the Go2 environment, builds the PPO agent, and runs the specified number of training iterations.

3. (Optional) Run student-teacher distillation:

   ```bash
   python examples/train_unitree_go2.py --config configs/go2_distillation.json
   ```

   This will let a frozen teacher control the environment while the student learns to imitate its actions.

4. (Isaac Gym) Train on the Unitree Go2 rough-terrain task:

   ```bash
   python examples/train_unitree_go2.py --config configs/unitree_go2_isaac.json
   ```

   The config delegates environment creation to `unitree_rl_gym`, launching an Isaac Gym simulation that mirrors the official Unitree training setup.

## Customisation

- **Environment changes**: implement a new vectorised environment exposing the `VectorEnv` protocol and register a factory with `ENV_REGISTRY`.
- **Algorithm experiments**: create a new class implementing `AlgorithmHandle`, wrap it in a builder, and register it with `ALGORITHM_REGISTRY`.
- **Trainer logic**: extend `TrainerWrapper` or register a bespoke trainer if you need alternative scheduling or evaluation behaviour.

For a deeper architectural overview see [`docs/architecture.md`](docs/architecture.md).
