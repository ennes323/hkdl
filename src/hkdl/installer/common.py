"""Strict metadata and filesystem primitives for owned installation state."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class InstallError(ValueError):
    exit_code = 3


class InstallBusy(InstallError):
    exit_code = 5


class InstallFailure(InstallError):
    exit_code = 6


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InstallError(f"duplicate installation metadata key: {key}")
        result[key] = value
    return result


def read_json(payload: bytes) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise InstallError(f"invalid JSON constant: {value}")

    try:
        value = json.loads(payload, object_pairs_hook=_pairs, parse_constant=reject)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InstallError(f"invalid installation metadata: {error}") from error
    if not isinstance(value, dict):
        raise InstallError("installation metadata must be a JSON object")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def real_directory(path: Path, *, create: bool = False) -> Path:
    path = path.absolute()
    if path.resolve() != path or path.is_symlink():
        raise InstallError(f"installation path must not contain symlinks: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise InstallError(f"directory is unavailable: {path}")
    return path


def regular_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise InstallError(f"regular file is unavailable: {path}")
    return path.read_bytes()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_file(path: Path, payload: bytes, *, replace: bool = False) -> None:
    real_directory(path.parent)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise InstallError(f"cannot replace installation metadata: {path}")
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as error:
                raise InstallBusy(
                    f"installation file already exists: {path}"
                ) from error
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def directory_lease(path: Path, *, exclusive: bool):
    real_directory(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(
                descriptor,
                (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise InstallBusy(
                f"HKDL is in use or being updated: {path}; finish the affected "
                "commands, workers or web server and retry"
            ) from error
        yield descriptor
    finally:
        # Closing our reference preserves a lease inherited by a live worker.
        # Explicit LOCK_UN would release that worker's shared open description.
        os.close(descriptor)
