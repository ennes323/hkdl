"""Shared Run and Model values, independent of storage and execution services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    path: Path
    address: str
    snapshot: dict[str, Any]
    request: dict[str, Any]
    state: dict[str, Any]
    graph_identity: dict[str, Any] | None = None
    event_hash: str | None = None


@dataclass(frozen=True)
class ModelRecord:
    path: Path
    address: str
    document: dict[str, Any]
    graph_hash: str | None = None


# Keep existing public type locations, including serialized Python references.
RunRecord.__module__ = "hkdl.storage.runs"
ModelRecord.__module__ = "hkdl.storage.runs"


def run_sort_key(record: RunRecord) -> tuple[Any, ...]:
    return (
        record.request["experiment"].encode("utf-8"),
        record.request["variant"].encode("utf-8"),
        int(record.request["run_id"].removeprefix("run-")),
        record.request["run_id"].encode("utf-8"),
    )
