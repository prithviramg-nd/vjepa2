"""Minimal name -> builder registry.

Every pluggable component (backbones, heads, later: datasets, transforms,
loggers, ...) registers itself under a string key so YAML configs can select
it by name without the caller importing the concrete class.
"""

from typing import Callable, Dict, Iterable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str = None) -> Callable:
        def deco(fn: Callable) -> Callable:
            key = name or fn.__name__
            if key in self._items:
                raise KeyError(f"{self.kind} '{key}' is already registered")
            self._items[key] = fn
            return fn

        return deco

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def build(self, name: str, **kwargs):
        return self.get(name)(**kwargs)

    def names(self) -> Iterable[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
