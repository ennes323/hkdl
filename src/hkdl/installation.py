"""Installation admission shared by CLI, web servers, and inherited workers."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from hkdl.installer.common import InstallBusy, InstallError, directory_lease
from hkdl.installer.paths import INSTALL_MARKER, LEASE_ENV, InstallPaths

_LEASE: ContextVar[int | None] = ContextVar("hkdl_installation_lease", default=None)


def installed_paths() -> InstallPaths | None:
    """Find managed ownership from this interpreter's release environment.

    Source checkouts and unrelated environments have no installation admission.
    An identified managed root must validate; invalid metadata is not a fallback.
    """
    release = Path(sys.prefix).resolve().parent
    if release.parent.name != "releases":
        return None
    root = release.parent.parent
    if not os.path.lexists(root / INSTALL_MARKER):
        return None
    paths = InstallPaths(root)
    paths.validate()
    return paths


def installation_descriptor() -> int | None:
    return _LEASE.get()


def installation_descriptors() -> tuple[int, ...]:
    descriptor = installation_descriptor()
    return () if descriptor is None else (descriptor,)


def installation_operation(function):
    @wraps(function)
    def operation(*args, **kwargs):
        with installation_access():
            return function(*args, **kwargs)

    return operation


def _validate_current(paths: InstallPaths) -> None:
    current = paths.current()
    if current is None or current.path != Path(sys.prefix).resolve().parent:
        raise InstallBusy("this HKDL release is inactive; use the managed hkdl command")


def _inherited_descriptor(paths: InstallPaths) -> int | None:
    raw = os.environ.get(LEASE_ENV)
    if raw is None:
        return None
    try:
        descriptor = int(raw)
        held, expected = os.fstat(descriptor), paths.root.stat()
    except (ValueError, OSError) as error:
        raise InstallError("invalid inherited HKDL installation lease") from error
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
        raise InstallError("inherited lease belongs to another installation")
    return descriptor


@contextmanager
def installation_lease():
    """Own a lease independently of thread-local admission or a caller's FD."""

    paths = installed_paths()
    if paths is None:
        yield None
        return
    with directory_lease(paths.root, exclusive=False) as descriptor:
        _validate_current(paths)
        yield descriptor


@contextmanager
def installation_access():
    """Borrow inherited admission or hold a shared lease for this operation.

    Nested calls reuse the context descriptor. Only the context that acquires a
    descriptor closes it; a caller or worker keeps ownership of a borrowed one.
    """
    held = _LEASE.get()
    if held is not None:
        yield held
        return
    paths = installed_paths()
    if paths is None:
        yield None
        return
    inherited = _inherited_descriptor(paths)
    if inherited is not None:
        _validate_current(paths)
        token = _LEASE.set(inherited)
        try:
            yield inherited
        finally:
            _LEASE.reset(token)
        return
    with directory_lease(paths.root, exclusive=False) as descriptor:
        _validate_current(paths)
        token = _LEASE.set(descriptor)
        try:
            yield descriptor
        finally:
            _LEASE.reset(token)
