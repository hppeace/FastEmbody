"""Generic registry utilities used across the framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar


T = TypeVar("T")
Factory = Callable[..., T]


@dataclass
class RegistryEntry(Generic[T]):
    factory: Factory[T]
    description: str | None = None
    default_kwargs: Dict[str, Any] | None = None


class Registry(Generic[T]):
    """Simple name -> factory registry.

    Registration is usually performed via the ``@register`` decorator::

        MODELS.register("my_mlp")(MyMLP)

    Each registered factory can expose default kwargs and a short description to
    simplify introspection tooling.
    """

    def __init__(self, name: str):
        self._name = name
        self._items: Dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        key: str,
        *,
        description: str | None = None,
        default_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Factory[T]], Factory[T]]:
        if key in self._items:
            raise KeyError(f"{self._name!r} registry already has key {key!r}")

        def decorator(factory: Factory[T]) -> Factory[T]:
            self._items[key] = RegistryEntry(
                factory=factory, description=description, default_kwargs=default_kwargs
            )
            return factory

        return decorator

    def build(self, key: str, /, **kwargs: Any) -> T:
        try:
            entry = self._items[key]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Unknown {self._name} component: {key!r}") from exc
        final_kwargs = dict(entry.default_kwargs or {})
        final_kwargs.update(kwargs)
        return entry.factory(**final_kwargs)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def describe(self, key: str) -> str | None:
        return self._items[key].description

    def get_entry(self, key: str) -> RegistryEntry[T]:
        return self._items[key]
