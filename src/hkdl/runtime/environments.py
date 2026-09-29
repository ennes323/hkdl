"""Content-addressed Variant environments and explicit cache pruning."""

from __future__ import annotations

import os
import shutil
import subprocess  # noqa: F401 - compatibility patch point
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from hkdl.authoring.authoring_records import VariantRecord
from hkdl.storage.storage import RepositoryPaths, publish_directory

from .environment_builder import (
    ENVIRONMENT_SCHEMA_VERSION,  # noqa: F401 - legacy direct import compatibility
    EnvironmentBuilder,
    environment_python,
)
from .environment_pruner import EnvironmentPruner, lock_file, unlock_file
from .environment_types import (
    EnvironmentFailure,
    EnvironmentIdentity,
    PruneEntry,
    PrunePlan,
    PruneResult,
)

for _public_type in (
    EnvironmentFailure,
    EnvironmentIdentity,
    PruneEntry,
    PrunePlan,
    PruneResult,
):
    _public_type.__module__ = __name__


@dataclass
class EnvironmentHandle:
    key: str
    python: Path
    descriptor: int
    on_close: Callable[[int], None] | None = None

    def close(self) -> None:
        if self.descriptor < 0:
            return
        descriptor = self.descriptor
        self.descriptor = -1
        if self.on_close is not None:
            self.on_close(descriptor)
        unlock_file(descriptor)

    def __enter__(self) -> EnvironmentHandle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class EnvironmentStore:
    def __init__(self, repository: RepositoryPaths):
        self.repository = repository
        self._builder = EnvironmentBuilder(repository)
        self._pruner = EnvironmentPruner(repository, self._builder)
        self.root = self._builder.root
        self.store = self._builder.store
        self.locks = self._builder.locks

    def identity(self, variant: VariantRecord) -> EnvironmentIdentity:
        return self._builder.identity(
            variant,
            runtime_identity=self._runtime_identity,
        )

    def acquire(self, variant: VariantRecord) -> EnvironmentHandle:
        identity = self.identity(variant)
        self._builder.ensure_layout()
        lock = self.locks / f"{identity.key}.lock"
        target = self.store / identity.key
        while True:
            descriptor = lock_file(lock, shared=True)
            try:
                if os.path.lexists(target):
                    self._builder.validate(target, identity)
                    self._builder.check(target, variant, identity)
                    return EnvironmentHandle(
                        identity.key,
                        environment_python(target),
                        descriptor,
                    )
            except BaseException:
                unlock_file(descriptor)
                raise
            unlock_file(descriptor)

            # Never queue an exclusive lease behind another caller's ready
            # handle: a competing builder may have published since our read.
            try:
                descriptor = lock_file(lock, shared=False, blocking=False)
            except BlockingIOError:
                continue
            try:
                self._build_if_missing(target, variant, identity)
            finally:
                unlock_file(descriptor)
            # Prune or another builder can run between leases. Recheck under
            # the shared lease that the returned handle will actually retain.

    def _build_if_missing(
        self,
        target: Path,
        variant: VariantRecord,
        identity: EnvironmentIdentity,
    ) -> None:
        if os.path.lexists(target):
            return
        candidate: Path | None = Path(
            tempfile.mkdtemp(prefix=f".{identity.key}.candidate-", dir=self.store)
        )
        try:
            self._builder.synchronize(candidate, variant, identity)
            self._builder.write_manifest(candidate / "environment.json", identity)
            self._builder.validate(candidate, identity)
            try:
                publish_directory(candidate, target)
                candidate = None
            except Exception:
                if os.path.lexists(target):
                    self._builder.validate(target, identity)
                else:
                    raise
        finally:
            if candidate is not None:
                shutil.rmtree(candidate, ignore_errors=True)

    def prepare(self, variant: VariantRecord) -> Path:
        with self.acquire(variant) as environment:
            return environment.python

    def plan_prune(
        self,
        variants: Iterable[VariantRecord],
        *,
        active_variants: set[tuple[str, str]],
        remove_all: bool,
    ) -> PrunePlan:
        return self._pruner.plan(
            variants,
            active_variants=active_variants,
            remove_all=remove_all,
            identity=self.identity,
        )

    def prune(
        self,
        variants: Iterable[VariantRecord],
        *,
        active_variants: set[tuple[str, str]],
        remove_all: bool,
    ) -> PruneResult:
        plan = self.plan_prune(
            variants,
            active_variants=active_variants,
            remove_all=remove_all,
        )
        return self._pruner.prune(plan)

    def _runtime_identity(
        self,
        source: Path,
    ) -> tuple[Path, str, Path, dict[str, str]]:
        return self._builder.runtime_identity(source)


__all__ = [
    "EnvironmentFailure",
    "EnvironmentHandle",
    "EnvironmentIdentity",
    "EnvironmentStore",
    "PruneEntry",
    "PrunePlan",
    "PruneResult",
]
