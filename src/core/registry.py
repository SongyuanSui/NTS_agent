# src/core/registry.py

from __future__ import annotations

from typing import Any


class Registry:
    """
    Lightweight name -> object/class registry.

    Intended use
    ------------
    - tasks
    - agents
    - pipelines
    - retrievers
    """

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Registry name must be a non-empty string.")
        self.name = name
        self._items: dict[str, Any] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> list[str]:
        return list(self._items.keys())

    def items(self):
        return self._items.items()

    def values(self):
        return self._items.values()

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(
                f"'{key}' is not registered in registry '{self.name}'. "
                f"Available: {sorted(self._items.keys())}"
            )
        return self._items[key]

    def try_get(self, key: str, default: Any = None) -> Any:
        return self._items.get(key, default)

    def register(self, key: str, value: Any, overwrite: bool = False) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Registry key must be a non-empty string.")

        if key in self._items and not overwrite:
            raise ValueError(
                f"'{key}' is already registered in registry '{self.name}'."
            )

        self._items[key] = value

    def unregister(self, key: str) -> None:
        if key not in self._items:
            raise KeyError(
                f"'{key}' is not registered in registry '{self.name}'."
            )
        del self._items[key]

    def decorator(self, key: str, overwrite: bool = False):
        """
        Decorator form of register.

        Example
        -------
        @AGENT_REGISTRY.decorator("channel_selector")
        class ChannelSelectorAgent(...):
            ...
        """
        def _wrapper(obj: Any) -> Any:
            self.register(key=key, value=obj, overwrite=overwrite)
            return obj
        return _wrapper

    def summary(self) -> dict[str, Any]:
        return {
            "registry_name": self.name,
            "num_items": len(self._items),
            "keys": sorted(self._items.keys()),
        }


TASK_REGISTRY = Registry("task")
AGENT_REGISTRY = Registry("agent")
PIPELINE_REGISTRY = Registry("pipeline")
RETRIEVER_REGISTRY = Registry("retriever")