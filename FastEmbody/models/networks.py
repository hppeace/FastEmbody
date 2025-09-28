"""Reusable neural network building blocks."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "gelu": nn.GELU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'. Available: {list(mapping)}")
    return mapping[name]()


class MLP(nn.Module):
    """Simple multilayer perceptron reused by algorithms and policies."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Iterable[int],
        *,
        activation: str = "tanh",
        output_activation: str | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation is not None:
            layers.append(_get_activation(output_activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["MLP"]
