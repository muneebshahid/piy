from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from tile.extensions.hooks import BeforeRunHook, RunHooks
from tile.extensions.run_observers import RunObserver, RunObservers


class Extension(Protocol):
    def register(self, registry: ExtensionRegistry) -> None: ...


class ExtensionRegistry:
    def __init__(self) -> None:
        self._before_run: list[BeforeRunHook] = []
        self._observers: list[RunObserver] = []

    def before_run(self, hook: BeforeRunHook) -> None:
        self._before_run.append(hook)

    def observe(self, observer: RunObserver) -> None:
        self._observers.append(observer)

    def build_run_hooks(self) -> RunHooks:
        return RunHooks(before_run=self._before_run)

    def build_run_observers(self) -> RunObservers:
        return RunObservers(self._observers)


def _register_extensions(extensions: Sequence[Extension]) -> ExtensionRegistry:
    registry = ExtensionRegistry()
    for extension in extensions:
        extension.register(registry)
    return registry
