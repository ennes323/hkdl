"""Shared value types for private environment services and their public facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EnvironmentFailure(RuntimeError):
    """Environment discovery, build, validation, or deletion failed."""


@dataclass(frozen=True)
class EnvironmentIdentity:
    key: str
    document: dict[str, Any]
    uv: Path
    python: Path
    extras: tuple[str, ...]


@dataclass(frozen=True)
class PruneEntry:
    kind: str
    path: Path
    key: str | None
    bytes: int


@dataclass(frozen=True)
class PrunePlan:
    entries: tuple[PruneEntry, ...]
    retained: int
    busy: int

    @property
    def bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries)


@dataclass(frozen=True)
class PruneResult:
    removed: int
    bytes: int
    retained: int
    busy: int
