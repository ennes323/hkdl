"""Private scanning, planning, and deletion service for environment pruning."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
from collections.abc import Callable, Iterable
from pathlib import Path

from hkdl.authoring.authoring_records import VariantRecord
from hkdl.errors import ContractError
from hkdl.storage.filesystem import lock_directory, unlock_directory
from hkdl.storage.storage import RepositoryPaths

from .environment_builder import EnvironmentBuilder, require_real_directory
from .environment_types import (
    EnvironmentFailure,
    EnvironmentIdentity,
    PruneEntry,
    PrunePlan,
    PruneResult,
)

KEY_LENGTH = 64


class EnvironmentPruner:
    def __init__(self, repository: RepositoryPaths, builder: EnvironmentBuilder):
        self.repository = repository
        self.builder = builder
        self.store = builder.store
        self.locks = builder.locks

    def plan(
        self,
        variants: Iterable[VariantRecord],
        *,
        active_variants: set[tuple[str, str]],
        remove_all: bool,
        identity: Callable[[VariantRecord], EnvironmentIdentity],
    ) -> PrunePlan:
        records = tuple(variants)
        store_entries = self._store_entries()
        referenced = (
            {identity(variant).key for variant in records}
            if not remove_all
            and any(kind == "environment" for _, _, kind in store_entries)
            else set()
        )
        entries: list[PruneEntry] = []
        retained = 0
        busy = 0

        for variant in records:
            legacy = variant.path / ".venv"
            if not os.path.lexists(legacy):
                continue
            require_real_directory(legacy, "legacy Variant environment")
            if (variant.experiment, str(variant.document["name"])) in active_variants:
                busy += 1
                continue
            entries.append(PruneEntry("legacy", legacy, None, _logical_bytes(legacy)))

        for path, key, kind in store_entries:
            if kind == "environment" and not remove_all and key in referenced:
                retained += 1
                continue
            if self._is_busy(key):
                busy += 1
                continue
            entries.append(PruneEntry(kind, path, key, _logical_bytes(path)))

        entries.sort(key=lambda entry: os.fsencode(str(entry.path)))
        return PrunePlan(tuple(entries), retained, busy)

    def prune(self, plan: PrunePlan) -> PruneResult:
        removed = 0
        removed_bytes = 0
        busy = plan.busy
        for entry in plan.entries:
            try:
                descriptor, unlock = self._lock_entry(entry)
            except BlockingIOError:
                busy += 1
                continue
            try:
                if self._remove_entry(entry, removed):
                    removed += 1
                    removed_bytes += entry.bytes
            finally:
                unlock(descriptor)
        return PruneResult(removed, removed_bytes, plan.retained, busy)

    def _lock_entry(self, entry: PruneEntry) -> tuple[int, Callable[[int], None]]:
        if entry.kind == "legacy":
            descriptor = self._lock_variant_directory(entry.path.parent)
            return descriptor, unlock_directory
        assert entry.key is not None
        self.builder.ensure_layout()
        descriptor = _lock_file(
            self.locks / f"{entry.key}.lock",
            shared=False,
            blocking=False,
        )
        return descriptor, _unlock_file

    def _remove_entry(self, entry: PruneEntry, removed: int) -> bool:
        try:
            if not os.path.lexists(entry.path):
                return False
            require_real_directory(entry.path, "environment prune target")
            self.builder.ensure_layout()
            detached = self._detach(entry, removed)
            try:
                shutil.rmtree(detached)
            except OSError as error:
                raise EnvironmentFailure(
                    f"could not remove detached environment {detached}: {error}"
                ) from error
            return True
        except EnvironmentFailure:
            raise
        except OSError as error:
            raise EnvironmentFailure(
                f"could not prune environment {entry.path}: {error}"
            ) from error

    def _detach(self, entry: PruneEntry, removed: int) -> Path:
        if entry.kind == "trash":
            return entry.path
        trash_key = (
            entry.key
            or hashlib.sha256(
                entry.path.relative_to(self.repository.root).as_posix().encode("utf-8")
            ).hexdigest()
        )
        detached = self.store / f".{trash_key}.trash-{os.getpid()}-{removed}"
        if os.path.lexists(detached):
            raise EnvironmentFailure(f"prune trash already exists: {detached}")
        os.rename(entry.path, detached)
        return detached

    def _store_entries(self) -> list[tuple[Path, str, str]]:
        if not os.path.lexists(self.store):
            return []
        require_real_directory(self.store, "shared environment store")
        values: list[tuple[Path, str, str]] = []
        with os.scandir(self.store) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    raise ContractError(f"invalid shared environment entry: {path}")
                key, kind = _environment_entry(entry.name)
                if key is None:
                    raise ContractError(f"invalid shared environment entry: {path}")
                values.append((path, key, kind))
        return sorted(values, key=lambda item: os.fsencode(item[0].name))

    def _is_busy(self, key: str) -> bool:
        lock = self.locks / f"{key}.lock"
        if not os.path.lexists(lock):
            return False
        try:
            descriptor = _lock_file(lock, shared=False, blocking=False, create=False)
        except BlockingIOError:
            return True
        _unlock_file(descriptor)
        return False

    @staticmethod
    def _lock_variant_directory(path: Path) -> int:
        require_real_directory(path, "Variant directory")
        return lock_directory(path, blocking=False)


def _environment_entry(name: str) -> tuple[str | None, str]:
    if _environment_key(name):
        return name, "environment"
    if name.startswith(".") and ".candidate-" in name:
        key = name[1 : KEY_LENGTH + 1]
        if _environment_key(key):
            return key, "candidate"
    if name.startswith(".") and ".trash-" in name:
        key = name[1 : KEY_LENGTH + 1]
        if _environment_key(key):
            return key, "trash"
    return None, ""


def _environment_key(value: str) -> bool:
    return len(value) == KEY_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _lock_file(
    path: Path,
    *,
    shared: bool,
    blocking: bool = True,
    create: bool = True,
) -> int:
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise EnvironmentFailure(f"environment lock is unavailable: {path}") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise EnvironmentFailure(f"environment lock is invalid: {path}")
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(descriptor, operation)
    except BlockingIOError:
        os.close(descriptor)
        raise
    return descriptor


def _unlock_file(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _logical_bytes(root: Path) -> int:
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise ContractError(
            f"cannot inspect environment tree {root}: {error}"
        ) from error
    if not stat.S_ISDIR(root_stat.st_mode):
        return root_stat.st_size
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ContractError(
                f"cannot scan environment tree {directory}: {error}"
            ) from error
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ContractError(
                    f"cannot inspect environment entry {entry.path}: {error}"
                ) from error
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(Path(entry.path))
            else:
                total += metadata.st_size
    return total


# The facade reuses these exact lock helpers for environment acquisition.
lock_file = _lock_file
unlock_file = _unlock_file
