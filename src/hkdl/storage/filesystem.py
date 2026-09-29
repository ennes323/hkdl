"""Raw filesystem locking primitives shared by domain-specific wrappers."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


def lock_directory(path: Path, *, blocking: bool = True) -> int:
    """Open and exclusively lock a directory without domain validation."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(descriptor, operation)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def unlock_directory(descriptor: int) -> None:
    """Release and close a descriptor returned by :func:`lock_directory`."""

    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
