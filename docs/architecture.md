# FastEmbody Training Framework Architecture

This iteration of the framework is intentionally self-contained: every moving part required to train a simplified Unitree Go2 controller now lives inside the repository. No external legged-robot libraries or third-party reinforcement-learning frameworks are referenced.

## Design Goals

- **Self sufficiency** – algorithms, environments, and training loops are implemented in pure PyTorch so experiments run without special dependencies.
- **Modularity** – registries still back each component family, so swapping in new algorithms or environments is a matter of registering a new factory.
- **Configurability** – experiments are described through dataclass-backed YAML/JSON configs to keep hyper-parameters declarative.
- **Approachability** – the toy Go2 model focuses on clarity (simple dynamics, compact code) so future extensions are straightforward.

## Package Layout

```
FastEmbody/
├── FastEmbody/
│   ├── algorithms/
│   │   ├── __init__.py          # imports register PPO & distillation builders
│   │   ├── base.py              # algorithm registry utilities
│   │   ├── distillation.py      # student-teacher distillation implementation
│   │   └── ppo.py               # pure PyTorch PPO implementation
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # experiment/dataclass helpers
│   │   ├── interfaces.py        # protocols defining common component APIs
│   │   └── registries.py        # generic registry helper
│   ├── envs/
│   │   ├── __init__.py          # registers toy and unitree builders
│   │   ├── base.py              # lightweight runtime config shared by toy envs
│   │   ├── go2.py               # differentiable toy Go2 vector environment
│   │   └── unitree.py           # Isaac Gym adapter mirroring unitree_rl_gym
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── experiment.py        # orchestrates end-to-end training/eval
│   ├── trainers/
│   │   ├── __init__.py          # registers the default on-policy trainer wrapper
│   │   └── base.py              # thin trainer wrapper around AlgorithmHandle
│   └── typing.py                # shared type aliases and `VectorEnv` protocol
├── configs/
│   └── unitree_go2_ppo.json     # example experiment configuration
├── docs/
│   └── architecture.md          # (this file)
├── examples/
│   └── train_unitree_go2.py     # CLI that loads config and launches training
└── README.md                    # user-facing instructions
```

## Components

### Environment
`FastEmbody.envs.go2.Go2VectorEnv` models a batch of Go2 robots with simplified joint dynamics. It tracks joint positions/velocities, samples random velocity commands, and rewards the policy for matching them while remaining stable. Everything is vectorised with PyTorch tensors so rollouts are fast on CPU or GPU.

`FastEmbody.envs.unitree.build_unitree_env` wraps the official `unitree_rl_gym` legged environments, spawning Isaac Gym simulations with the same configs as the upstream project. The wrapper exposes the minimal `VectorEnv` protocol so algorithms can swap between toy and high-fidelity setups without code changes.

### Algorithm
`FastEmbody.algorithms.ppo.PPOAgent` contains a compact PPO learner: an `ActorCritic` network, a rollout buffer with Generalised Advantage Estimation, and the standard clipped objective. The algorithm conforms to the generic `AlgorithmHandle` protocol, so different trainers can drive it.

`FastEmbody.algorithms.distillation.DistillationAgent` implements student-teacher knowledge transfer. A frozen teacher gathers trajectories while the student network minimises an imitation loss over mini-batches drawn from the collected dataset.

### Trainer
`FastEmbody.trainers.base.TrainerWrapper` is the default trainer. It simply forwards `num_iterations` into the algorithm's `learn` method, but keeps the registry-based plug-in point for specialised trainers.

### Pipeline
`FastEmbody.pipelines.experiment.ExperimentPipeline` wires everything together: read config, instantiate environment through the registry, build the PPO agent, wrap it with the trainer, and execute training/evaluation. Since every component advertises a registry key, the pipeline is generic by construction.

## Extensibility

- **Different environment** – implement a new vectorised environment class exposing the `VectorEnv` protocol, register a factory with `ENV_REGISTRY`, and point the config at the new builder.
- **Alternative algorithm** – implement `AlgorithmHandle`, create a builder returning instances of it, and register via `ALGORITHM_REGISTRY.register("name")`.
- **Custom trainer** – if you need bespoke looping logic (e.g., curriculum, evaluation scheduling), create a new trainer class and register it with `TRAINER_REGISTRY`.

These building blocks are intentionally small, making it easy to extend the framework while keeping dependencies minimal.
