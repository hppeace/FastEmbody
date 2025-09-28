
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, Union

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

def _temporal_allocater(horizon: int, num_env: int, tree: FieldSpecTree) -> TensorTree:
    if isinstance(tree, Mapping):
        return {key: _temporal_allocater(horizon, num_env, sub_spec)
                for key, sub_spec in tree.items()}
    return torch.zeros(horizon, num_env, *tree.shape, dtype=tree.dtype, device=tree.device)

def _temporal_append() -> None:
    pass

def _temporal_gather() -> None:
    pass

def _temporal_clear(tree: TensorTree) -> None:
    def _zero(tree: TensorTree) -> None:
        if isinstance(tree, Mapping):
            for child in tree.values():
                _zero(child)
        else:
            tree.zero_()
    _zero(tree)

class TemporalStorage(BaseStorage):
    def __init__(self, horizon: int, num_envs: int, tree: FieldSpecTree) -> None:
        super().__init__(tree=tree)
        self.horizon = horizon
        self.num_envs = num_envs

        self.step = 0
        self.storage = _temporal_allocater(horizon, num_envs, tree)

    def append(self, value: TensorTree) -> None:
        # TODO 检查TensorTree的shape
        if self.step >= self.horizon:
            raise RuntimeError("TemporalStorage capacity exceeded")
        self.step += 1
        _temporal_append(self.storage, self.step, value)

    def gather(self, indices: torch.Tensor) -> TensorTree:
        return _temporal_gather(self.storage, indices)

    def clear(self) -> None:
        self.step = 0
        _temporal_clear(self.storage)
    
    def __len__(self) -> int:
        return self.step * self.num_envs

class BatchSampler:
    """Base class for index selection strategies."""
    def __init__(self, *, batch_size: int = 1) -> None:
        self.batch_size = batch_size

    def sample(self, storage: TensorTree, **kwargs: Any) -> TensorTree:
        raise NotImplementedError

class RandomSampler(BatchSampler):
    """Uniform random sampling"""
    def __init__(self, *, batch_size: int = 1) -> None:
        super().__init__(batch_size=batch_size)

    def sample(self, storage: BaseStorage) -> TensorTree:
        if len(storage) == 0:
            raise ValueError("No storage in the storage dict to sample from!")
        total = len(storage)
        if total < self.batch_size:
            raise ValueError(f"Not enough items {total} in storage to sample the requested batch size {self.batch_size}!")
        indices = torch.randperm(total)[:self.batch_size]
        return storage.gather(indices)

class SampleBuffer:
    def __init__(
        self,
        sampler: BatchSampler,
        storage: BaseStorage,
    ) -> None:
        
        self.sampler = sampler
        self.storage = storage

    def reset(self) -> None:
        self.storage.clear()

    def append(self, value: TensorTree) -> None:
        self.storage.append(value)

    def sample(
        self,
    ) -> Dict[str, TensorTree]:
        return self.sampler.sample(self.storage)

__all__ = [
    "FieldSpec",
    "BaseStorage",
    "TemporalStorage",
    "BatchSampler",
    "RandomSampler",
    "SampleBuffer",
]
