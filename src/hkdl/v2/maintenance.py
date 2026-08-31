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
_ACCESS: ContextVar[dict[str, tuple[int, bool]]] = ContextVar(
    "hkdl_workspace_access", default={}
)


class WorkspaceBusy(RuntimeError):
    """A workspace is in maintenance or needs explicit migration recovery."""


def workspace_descriptor(root: Path) -> int | None:
    value = _ACCESS.get().get(str(root))
    return value[0] if value else None


@contextmanager
def workspace_access(repository, *, exclusive=False, recovery=False, bootstrap=False):
    root = repository.root
    key = str(root)
    current = _ACCESS.get()
    if key in current:
        descriptor, held_exclusive = current[key]
        if exclusive and not held_exclusive:
            raise WorkspaceBusy("workspace operation cannot upgrade to maintenance")
        yield descriptor
        return
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    token = None
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
        token = _ACCESS.set({**current, key: (descriptor, exclusive)})
        yield descriptor
    finally:
        if token is not None:
            _ACCESS.reset(token)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def workspace_operation(function):
    @wraps(function)
    def operation(owner, *args, **kwargs):
        repository = getattr(owner, "repository", None)
        if repository is None and hasattr(owner, "authoring"):
            repository = owner.authoring.repository
        if repository is None and hasattr(owner, "root"):
            repository = owner
        if repository is None:
            repository = owner.store.repository
        with workspace_access(repository):
            return function(owner, *args, **kwargs)

    return operation
