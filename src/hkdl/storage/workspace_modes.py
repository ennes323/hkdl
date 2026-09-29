"""Admission policy for ordinary graph/JSON workspaces.

An empty workspace may be bootstrapped. Older formats are inputs to explicit
migration, not alternative ordinary execution modes.
"""

from __future__ import annotations

import os
from pathlib import Path

from hkdl.errors import ContractError


def require_current_workspace(repository, *, allow_empty: bool = True) -> None:
    root = repository.root
    for directory in (root / ".hkdl", root / ".hkdl/store"):
        if os.path.lexists(directory) and (
            directory.is_symlink() or not directory.is_dir()
        ):
            raise ContractError(
                f"workspace metadata must be a real directory: {directory}"
            )
    storage = root / ".hkdl/store/CURRENT"
    authoring = root / ".hkdl/store/AUTHORING_CURRENT"
    has_storage = _marker(storage)
    has_authoring = _marker(authoring)
    if has_storage and has_authoring:
        from hkdl.storage.graph.graph import V2Graph

        V2Graph(repository).is_active()
        return
    if has_authoring:
        raise ContractError(
            "existing JSON authoring requires active v2 storage; inspect migration state"
        )
    if has_storage:
        raise ContractError(
            "ordinary operations require v2 storage and JSON authoring; "
            "preview hkdl migrate --authoring --dry-run --output json"
        )
    if all(
        _empty(path)
        for path in (root / "experiments", root / "outputs", root / ".hkdl/store")
    ):
        if allow_empty:
            return
        raise ContractError(
            "initialize graph/JSON authoring with hkdl experiment create before execution"
        )
    raise ContractError(
        "ordinary operations require v2 storage and JSON authoring; "
        "preview hkdl migrate --all --dry-run --output json, then migrate "
        "--authoring after the approved storage migration"
    )


def _marker(path: Path) -> bool:
    if not os.path.lexists(path):
        return False
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"invalid workspace format marker: {path}")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ContractError(f"cannot read workspace format marker: {path}") from error
    if content != b"v2\n":
        raise ContractError(f"unsupported workspace format marker: {path}")
    return True


def _empty(path: Path) -> bool:
    if not os.path.lexists(path):
        return True
    try:
        return not path.is_symlink() and path.is_dir() and not any(path.iterdir())
    except OSError as error:
        raise ContractError(f"cannot inspect workspace format: {path}") from error
