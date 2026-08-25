"""Transient action journals for one execution Run."""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ContractError
from .run_contracts import TRACKER_ID_PATTERN
from .storage import atomic_replace, atomic_write_new

ATTEMPT_ACTIONS = frozenset({"train", "eval", "export"})
ATTEMPT_PHASES = frozenset({"running", "worker_done", "ready"})


def new_attempt(
    *,
    action: str,
    tracker_run_id: str | None = None,
    candidate: str | None = None,
) -> dict[str, Any]:
    document = {
        "schema_version": 1,
        "action": action,
        "phase": "running",
        "tracker_run_id": tracker_run_id,
        "checkpoint": {"best": None, "last": None},
        "candidate": candidate,
        "result": None,
    }
    validate_attempt(document)
    return document


def validate_attempt(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "action",
        "phase",
        "tracker_run_id",
        "checkpoint",
        "candidate",
        "result",
    }:
        raise ContractError("attempt journal fields are invalid")
    if (
        isinstance(document["schema_version"], bool)
        or not isinstance(document["schema_version"], int)
        or document["schema_version"] != 1
    ):
        raise ContractError("attempt journal schema version is invalid")
    if document["action"] not in ATTEMPT_ACTIONS:
        raise ContractError("attempt journal action is invalid")
    if document["phase"] not in ATTEMPT_PHASES:
        raise ContractError("attempt journal phase is invalid")
    tracker_run_id = document["tracker_run_id"]
    if tracker_run_id is not None and (
        not isinstance(tracker_run_id, str)
        or not TRACKER_ID_PATTERN.fullmatch(tracker_run_id)
    ):
        raise ContractError("attempt journal tracker identity is invalid")
    checkpoint = document["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"best", "last"}:
        raise ContractError("attempt journal checkpoint fields are invalid")
    for value in checkpoint.values():
        if value is not None:
            _owned_path(value, "attempt journal checkpoint")
    candidate = document["candidate"]
    if candidate is not None:
        _owned_path(candidate, "attempt journal candidate")
    result = document["result"]
    if document["phase"] == "running" and result is not None:
        raise ContractError("running attempt journal must not contain a result")
    if document["phase"] != "running" and not isinstance(result, dict):
        raise ContractError("completed attempt journal requires a result")
    _json_compatible(document)
    return document


def write_attempt(
    path: Path,
    document: dict[str, Any],
    *,
    directory_descriptor: int | None = None,
) -> None:
    validate_attempt(document)
    content = _json_text(document)
    if os.path.lexists(path):
        _require_regular_file(path)
        atomic_replace(path, content)
    else:
        atomic_write_new(
            path,
            content,
            directory_descriptor=directory_descriptor,
        )


def load_attempt(path: Path) -> dict[str, Any] | None:
    if not os.path.lexists(path):
        return None
    _require_regular_file(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("attempt journal is invalid JSON") from error
    return validate_attempt(document)


def remove_attempt(path: Path) -> None:
    if not os.path.lexists(path):
        return
    _require_regular_file(path)
    path.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _owned_path(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location} path is invalid")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise ContractError(f"{location} path is invalid")


def _json_compatible(value: Any) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("attempt journal contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _json_compatible(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("attempt journal mapping keys must be strings")
            _json_compatible(item)
        return
    raise ContractError("attempt journal contains a non-JSON value")


def _require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError("attempt journal is unavailable") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ContractError("attempt journal must be a regular non-symlink file")


def _json_text(document: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
