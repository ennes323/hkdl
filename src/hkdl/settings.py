"""Validated workspace settings for the v2 authoring surface.

Workspace settings are intentionally a very small, human-readable document.  The
document contains no identity, credential, or connection information; those
values belong to an execution environment and are not part of the research
record.  The only setting currently supported is the workspace default tracker.

The public helpers in this module keep the filesystem concerns in one place so
that CLI and runtime callers can share the same validation and publication
semantics.  A missing settings file is equivalent to the default ``local``
tracker, but reading it is side-effect free.  ``set_tracker`` is the operation
that creates or atomically replaces the file.
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .config import ContractError
from .storage import AlreadyExistsError, atomic_replace, atomic_write_new
from .v2.maintenance import workspace_operation


SETTINGS_SCHEMA_VERSION: Final = 2
SETTINGS_FILENAME: Final = ".hkdl/settings.json"
DEFAULT_TRACKER_BACKENDS: Final[tuple[str, ...]] = ("local",)
SUPPORTED_TRACKER_BACKENDS: Final[frozenset[str]] = frozenset({"local", "mlflow"})


class _DuplicateKey(ValueError):
    """Raised by the strict JSON decoder when a mapping repeats a key."""


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateKey(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _ensure_json_compatible(value: Any, location: str = "settings") -> None:
    """Reject values that JSON cannot represent deterministically.

    ``json.loads`` normally returns only JSON-native values, but retaining this
    check makes the validation boundary explicit and protects the serializer if
    callers construct a document programmatically.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{location} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{location} mapping keys must be strings")
            _ensure_json_compatible(item, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_compatible(item, f"{location}[{index}]")
        return
    raise ContractError(f"{location} contains a non-JSON value")


def _require_real_directory(path: Path, *, missing_ok: bool = False) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return
        raise ContractError(f"settings directory is unavailable: {path}")
    except OSError as error:
        raise ContractError(
            f"cannot inspect settings directory {path}: {error}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"settings directory must be a real directory: {path}")


def _require_real_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"settings file is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"settings file must be a real regular file: {path}")


def _workspace_root(root: Path | str | None) -> Path:
    path = Path.cwd() if root is None else Path(root)
    path = path.absolute()
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"workspace root is unavailable: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise ContractError(f"workspace root must be a real directory: {path}")
    return path


def settings_path(
    root: Path | str | None = None, *, create_parent: bool = False
) -> Path:
    """Return the settings path after validating the workspace boundary.

    ``load_settings`` calls this with ``create_parent=False`` so a read never
    mutates a workspace.  ``set_tracker`` opts into creating a missing real
    ``.hkdl`` directory before the atomic file publication.
    """

    workspace = _workspace_root(root)
    hkdl = workspace / ".hkdl"
    if create_parent:
        try:
            hkdl.mkdir(mode=0o755)
        except FileExistsError:
            pass
        except OSError as error:
            raise ContractError(
                f"cannot create settings directory {hkdl}: {error}"
            ) from error
    _require_real_directory(hkdl, missing_ok=not create_parent)
    return workspace / SETTINGS_FILENAME


def normalize_tracker(value: Any) -> tuple[str, ...]:
    """Normalize a tracker selector to deterministic backend order.

    Accepted selectors are ``none``, ``local``, ``mlflow``, and the combined
    ``local+mlflow`` form.  For CLI and compatibility callers, a combined
    selector may also use a comma (with optional whitespace), and a JSON-style
    list/tuple of the two backend names is accepted.  Mapping input must use
    exactly the existing tracker contract's ``backend`` field.

    The returned tuple is ordered ``local``, then ``mlflow`` and is immutable.
    """

    if isinstance(value, Mapping):
        if set(value) != {"backend"}:
            raise ContractError("tracker must contain only backend")
        value = value["backend"]

    if isinstance(value, str):
        selector = value.strip()
        if selector == "none":
            return ()
        if selector in {"local", "mlflow"}:
            return (selector,)
        # ``local+mlflow`` is the user-facing spelling; comma is retained for
        # compatibility with the current CLI's tracker selector.
        pieces = selector.replace("+", ",").split(",")
        pieces = [piece.strip() for piece in pieces]
        if (
            len(pieces) == 2
            and len(set(pieces)) == 2
            and set(pieces) == set(SUPPORTED_TRACKER_BACKENDS)
        ):
            return ("local", "mlflow")
        raise ContractError("tracker must be none, local, mlflow, or local+mlflow")

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        pieces = list(value)
        if (
            len(pieces) == 2
            and all(isinstance(item, str) for item in pieces)
            and set(pieces) == set(SUPPORTED_TRACKER_BACKENDS)
        ):
            return ("local", "mlflow")
        if (
            len(pieces) == 1
            and isinstance(pieces[0], str)
            and pieces[0] in SUPPORTED_TRACKER_BACKENDS
        ):
            return (pieces[0],)

    raise ContractError("tracker must be none, local, mlflow, or local+mlflow")


def _tracker_document_value(backends: tuple[str, ...]) -> str | list[str]:
    if not backends:
        return "none"
    if len(backends) == 1:
        return backends[0]
    return list(backends)


def _canonical_document(backends: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "tracker": {"backend": _tracker_document_value(backends)},
    }


def validate_settings(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate a persisted settings document and return normalized trackers."""

    if not isinstance(document, Mapping):
        raise ContractError("settings must be a JSON object")
    if set(document) != {"schema_version", "tracker"}:
        raise ContractError("settings must contain exactly schema_version and tracker")
    schema_version = document["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SETTINGS_SCHEMA_VERSION
    ):
        raise ContractError(
            f"settings.schema_version must be {SETTINGS_SCHEMA_VERSION}"
        )
    tracker = document["tracker"]
    if not isinstance(tracker, Mapping):
        raise ContractError("settings.tracker must be an object")
    if set(tracker) != {"backend"}:
        raise ContractError("settings.tracker must contain only backend")
    backends = normalize_tracker(tracker)
    _ensure_json_compatible(dict(document))
    return backends


def settings_document(value: Any = DEFAULT_TRACKER_BACKENDS) -> dict[str, Any]:
    """Build and validate a canonical settings document from a selector."""

    backends = normalize_tracker(value)
    document = _canonical_document(backends)
    validate_settings(document)
    return document


def settings_json(document: Mapping[str, Any]) -> str:
    """Serialize settings deterministically for human-authored storage."""

    # Rebuild the canonical shape so valid compatibility inputs such as a
    # tuple/list in ``tracker.backend`` cannot affect the on-disk ordering or
    # JSON-native representation.
    if not isinstance(document, Mapping):
        raise ContractError("settings must be a JSON object")
    if set(document) != {"schema_version", "tracker"}:
        raise ContractError("settings must contain exactly schema_version and tracker")
    schema_version = document["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SETTINGS_SCHEMA_VERSION
    ):
        raise ContractError(
            f"settings.schema_version must be {SETTINGS_SCHEMA_VERSION}"
        )
    canonical = _canonical_document(normalize_tracker(document["tracker"]))
    validate_settings(canonical)
    return (
        json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _load_document(path: Path) -> dict[str, Any]:
    _require_real_file(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractError(f"cannot read settings file {path}: {error}") from error
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_object_pairs_hook,
            parse_constant=_reject_json_constant,
        )
    except (_DuplicateKey, json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ContractError(f"invalid settings JSON in {path}: {error}") from error
    if not isinstance(document, dict):
        raise ContractError(f"settings must be a JSON object: {path}")
    validate_settings(document)
    return document


@dataclass(frozen=True)
class WorkspaceSettings:
    """Immutable validated view of the workspace settings document."""

    tracker_backends: tuple[str, ...] = DEFAULT_TRACKER_BACKENDS

    def __post_init__(self) -> None:
        # The normalized internal representation uses an empty tuple for
        # ``none``.  Keep that representation constructible while still
        # rejecting an empty persisted JSON list in ``normalize_tracker``.
        normalized = (
            ()
            if self.tracker_backends == ()
            else normalize_tracker(self.tracker_backends)
        )
        object.__setattr__(self, "tracker_backends", normalized)

    @property
    def tracker(self) -> str | list[str]:
        """Return the normalized tracker in the persisted contract shape."""

        return _tracker_document_value(self.tracker_backends)

    @property
    def backend(self) -> str | list[str]:
        """Alias for callers that use the existing ``tracker.backend`` term."""

        return self.tracker

    def as_dict(self) -> dict[str, Any]:
        return _canonical_document(self.tracker_backends)

    def to_document(self) -> dict[str, Any]:
        return self.as_dict()

    def json(self) -> str:
        return settings_json(self.as_dict())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> WorkspaceSettings:
        return cls(validate_settings(document))


class WorkspaceSettingsStore:
    """Filesystem-backed settings operations for one workspace root."""

    def __init__(self, root: Path | str | None = None):
        self.root = _workspace_root(root)

    @property
    def path(self) -> Path:
        return settings_path(self.root)

    @workspace_operation
    def load(self) -> WorkspaceSettings:
        path = self.path
        if not os.path.lexists(path):
            return WorkspaceSettings()
        return WorkspaceSettings.from_document(_load_document(path))

    def show(self) -> dict[str, Any]:
        """Return the canonical document without changing the workspace."""

        return self.load().as_dict()

    @workspace_operation
    def set_tracker(self, value: Any) -> WorkspaceSettings:
        """Validate and atomically publish a new workspace tracker setting."""

        document = settings_document(value)
        serialized = settings_json(document)
        path = settings_path(self.root, create_parent=True)

        # Refuse to overwrite malformed or symlinked state.  This is both a
        # fail-closed guard and a useful protection against accidentally hiding
        # a damaged settings file behind a valid replacement.
        exists = os.path.lexists(path)
        if exists:
            _load_document(path)
            try:
                atomic_replace(path, serialized)
            except OSError as error:
                raise ContractError(
                    f"cannot publish settings {path}: {error}"
                ) from error
        else:
            try:
                atomic_write_new(path, serialized)
            except AlreadyExistsError:
                # A concurrent writer won creation.  Validate its complete
                # document and then perform the requested atomic replacement.
                _load_document(path)
                try:
                    atomic_replace(path, serialized)
                except OSError as error:
                    raise ContractError(
                        f"cannot publish settings {path}: {error}"
                    ) from error
            except OSError as error:
                raise ContractError(
                    f"cannot publish settings {path}: {error}"
                ) from error

        return WorkspaceSettings.from_document(document)


def load_settings(root: Path | str | None = None) -> WorkspaceSettings:
    """Load settings, defaulting to local tracking when absent."""

    return WorkspaceSettingsStore(root).load()


def show_settings(root: Path | str | None = None) -> dict[str, Any]:
    """Return canonical settings suitable for CLI JSON output."""

    return WorkspaceSettingsStore(root).show()


def set_tracker(root: Path | str | None, value: Any) -> WorkspaceSettings:
    """Set and atomically publish the workspace's default tracker."""

    return WorkspaceSettingsStore(root).set_tracker(value)


__all__ = [
    "DEFAULT_TRACKER_BACKENDS",
    "SETTINGS_FILENAME",
    "SETTINGS_SCHEMA_VERSION",
    "SUPPORTED_TRACKER_BACKENDS",
    "WorkspaceSettings",
    "WorkspaceSettingsStore",
    "load_settings",
    "normalize_tracker",
    "set_tracker",
    "settings_document",
    "settings_json",
    "settings_path",
    "show_settings",
    "validate_settings",
]
