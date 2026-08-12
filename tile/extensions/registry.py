"""Extension registration and immutable harness assembly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from tile.extensions.hooks import BeforeRunHook, RunHooks
from tile.extensions.run_observers import RunObserver, RunObservers


class Extension(Protocol):
    """Package contributions to one agent harness."""

    def register(self, registry: ExtensionRegistry) -> None:
        """Register this extension's contributions once."""

        ...


class ExtensionRegistry:
    """Collect extension contributions during harness construction."""

    def __init__(self) -> None:
        """Create an empty extension registry."""

        self._before_run: list[BeforeRunHook] = []
        self._observers: list[RunObserver] = []

    def before_run(self, hook: BeforeRunHook) -> None:
        """Register one pre-admission hook in invocation order."""

        self._before_run.append(hook)

    def observe(self, observer: RunObserver) -> None:
        """Register one passive run-event observer in delivery order."""

        self._observers.append(observer)

    def build_run_hooks(self) -> RunHooks:
        """Freeze registered lifecycle hooks for future runs."""

        return RunHooks(before_run=self._before_run)

    def build_run_observers(self) -> RunObservers:
        """Freeze registered observers for future runs."""

        return RunObservers(self._observers)


def _register_extensions(extensions: Sequence[Extension]) -> ExtensionRegistry:
    """Register extensions once and return their assembled contributions."""

    registry = ExtensionRegistry()
    for extension in extensions:
        extension.register(registry)
    return registry
