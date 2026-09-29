"""Non-mutating workspace admission and journaled-cutover exclusion.

Lock the existing workspace directory: ordinary reads never create lock files.
Workers inherit this descriptor so an orphan worker still excludes migration.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

JOURNAL = ".hkdl/store/authoring-migration.json"
BOOTSTRAP_JOURNAL = ".hkdl/store/bootstrap.json"
PROMOTION_JOURNAL = ".hkdl/store/variant-promotion.json"
_ACCESS: ContextVar[dict[str, tuple[int, bool]]] = ContextVar(
    "hkdl_workspace_access", default={}
)
_LEGACY_ROOTS: ContextVar[frozenset[str]] = ContextVar(
    "hkdl_legacy_migration_roots", default=frozenset()
)


def legacy_access_active(repository) -> bool:
    return str(repository.root) in _LEGACY_ROOTS.get()


class WorkspaceBusy(RuntimeError):
    """A workspace is in maintenance or needs explicit migration recovery."""


def workspace_descriptor(root: Path) -> int | None:
    value = _ACCESS.get().get(str(root))
    return value[0] if value else None


@contextmanager
def borrowed_workspace_access(repository, descriptor: int):
    """Borrow the installer's inherited exclusive admission for a read probe.

    The parent keeps the lock until activation finishes. Never downgrade or
    unlock its shared open-file description from the child process.
    """

    root = repository.root
    held, expected = os.fstat(descriptor), root.stat()
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
        raise WorkspaceBusy("inherited workspace admission belongs to another root")
    for journal in (JOURNAL, BOOTSTRAP_JOURNAL, PROMOTION_JOURNAL):
        if os.path.lexists(root / journal):
            raise WorkspaceBusy(
                f"workspace needs recovery before installation: {journal}"
            )
    token = _ACCESS.set({**_ACCESS.get(), str(root): (descriptor, True)})
    try:
        yield descriptor
    finally:
        _ACCESS.reset(token)


@contextmanager
def workspace_access(
    repository,
    *,
    exclusive=False,
    recovery=False,
    bootstrap=False,
    promotion=False,
    legacy=False,
):
    root = repository.root
    key = str(root)
    current = _ACCESS.get()
    if key in current:
        descriptor, held_exclusive = current[key]
        if exclusive and not held_exclusive:
            raise WorkspaceBusy("workspace operation cannot upgrade to maintenance")
        if legacy and not legacy_access_active(repository):
            raise WorkspaceBusy("ordinary operation cannot enter legacy migration")
        yield descriptor
        return
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    token = None
    legacy_token = None
    try:
        try:
            fcntl.flock(
                descriptor,
                (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise WorkspaceBusy(
                "workspace is busy or in authoring maintenance; retry later"
            ) from error
        if not recovery and os.path.lexists(root / JOURNAL):
            raise WorkspaceBusy(
                "authoring migration needs recovery; run hkdl migrate --authoring --yes"
            )
        if not bootstrap and os.path.lexists(root / BOOTSTRAP_JOURNAL):
            raise WorkspaceBusy(
                "workspace initialization needs recovery; repeat the original "
                "hkdl experiment create command"
            )
        if not promotion and os.path.lexists(root / PROMOTION_JOURNAL):
            raise WorkspaceBusy(
                "Variant promotion needs recovery; repeat the original "
                "hkdl variant promote command"
            )
        if not legacy and not (bootstrap and os.path.lexists(root / BOOTSTRAP_JOURNAL)):
            from hkdl.storage.workspace_modes import require_current_workspace

            require_current_workspace(repository)
        if legacy:
            legacy_token = _LEGACY_ROOTS.set(_LEGACY_ROOTS.get() | {key})
        token = _ACCESS.set({**current, key: (descriptor, exclusive)})
        yield descriptor
    finally:
        if token is not None:
            _ACCESS.reset(token)
        if legacy_token is not None:
            _LEGACY_ROOTS.reset(legacy_token)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def workspace_operation(function=None, *, legacy=False):
    if function is None:
        return lambda function: workspace_operation(function, legacy=legacy)

    @wraps(function)
    def operation(owner, *args, **kwargs):
        with workspace_access(owner.repository, legacy=legacy):
            return function(owner, *args, **kwargs)

    return operation
