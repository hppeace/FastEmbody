from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple, Union

import torch

@dataclass(frozen=True)
class FieldSpec:
    shape: Union[Tuple[int, ...], Mapping[str, Tuple[int, ...]]]
    dtype: torch.dtype = torch.float32
    device: Union[str, torch.device] = "cpu"

FieldSpecTree = Union[FieldSpec, Mapping[str, "FieldSpec"]]

TensorTree = Union[torch.Tensor, Mapping[str, "TensorTree"]]

class BaseStorage:
    def __init__(self, tree: FieldSpecTree) -> None:
        self.tree = tree

    def append(self, value: TensorTree) -> None:
        raise NotImplementedError
    
    def gather(self, indices: torch.Tensor) -> TensorTree:
        raise NotImplementedError
    
    def clear(self) -> None:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError

class TemporalStorage(BaseStorage):
    def __init__(self, horizon: int, num_envs: int, tree: FieldSpecTree) -> None:
        super().__init__(tree=tree)
        self.horizon = horizon
        self.num_envs = num_envs
        self.step = 0
        self.storage = self._temporal_allocater(horizon, num_envs, tree)

    @staticmethod
    def _temporal_allocater(horizon: int, num_env: int, tree: FieldSpecTree) -> TensorTree:
        if isinstance(tree, Mapping):
            return {
                key: TemporalStorage._temporal_allocater(horizon, num_env, sub_spec)
                for key, sub_spec in tree.items()
            }
        return torch.zeros(horizon, num_env, *tree.shape, dtype=tree.dtype, device=tree.device)

    @staticmethod
    def _temporal_append(storage: TensorTree, step: int, value: TensorTree) -> None:
        if isinstance(storage, Mapping):
            for key, sub_storage in storage.items():
                TemporalStorage._temporal_append(sub_storage, step, value[key])
            return
        storage[step].copy_(value)

    @staticmethod
    def _temporal_gather(storage: TensorTree, indices: torch.Tensor, valid_steps: int, num_envs: int) -> TensorTree:
        if isinstance(storage, Mapping):
            return {
                key: TemporalStorage._temporal_gather(sub_storage, indices, valid_steps=valid_steps, num_envs=num_envs)
                for key, sub_storage in storage.items()
            }
        tensor = storage
        flat = tensor[:valid_steps].reshape(valid_steps * num_envs, *tensor.shape[2:])
        indices = indices.to(device=tensor.device, dtype=torch.long)
        return torch.index_select(flat, 0, indices)

    @staticmethod
    def _temporal_clear(tree: TensorTree) -> None:
        if isinstance(tree, Mapping):
            for child in tree.values():
                TemporalStorage._temporal_clear(child)
            return
        tree.zero_()

    def append(self, value: TensorTree) -> None:
        if self.step >= self.horizon:
            raise RuntimeError("TemporalStorage capacity exceeded")
        self._temporal_append(self.storage, self.step, value)
        self.step += 1

    def gather(self, indices: torch.Tensor) -> TensorTree:
        return self._temporal_gather(self.storage, indices, valid_steps=self.step, num_envs=self.num_envs)

    def clear(self) -> None:
        self.step = 0
        self._temporal_clear(self.storage)
    
    def __len__(self) -> int:
        return self.step * self.num_envs

class BatchSampler:
    """Base class for index selection strategies."""
    def __init__(self, batch_size: int = 1) -> None:
        self.batch_size = batch_size

    def sample(self, storage: BaseStorage, **kwargs: Any) -> TensorTree:
        raise NotImplementedError

class RandomSampler(BatchSampler):
    """Uniform random sampling"""
    def __init__(self, batch_size: int = 1) -> None:
        super().__init__(batch_size=batch_size)

    def sample(self, storage: BaseStorage) -> TensorTree:
        total = len(storage)
        if total < self.batch_size:
            raise ValueError("Not enough items in storage to sample the requested batch size")
        indices = torch.randperm(total)[:self.batch_size]
        return storage.gather(indices)

class SampleBuffer:
    def __init__(self, sampler: BatchSampler, storage: BaseStorage) -> None:
        self.sampler = sampler
        self.storage = storage

    def reset(self) -> None:
        self.storage.clear()

    def append(self, value: TensorTree) -> None:
        self.storage.append(value)

    def sample(self) -> TensorTree:
        return self.sampler.sample(self.storage)

__all__ = [
    "FieldSpec",
    "BaseStorage",
    "TemporalStorage",
    "BatchSampler",
    "RandomSampler",
    "SampleBuffer",
]
