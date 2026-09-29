"""Versioned, user-owned installation layout and explicit workspace registry."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import validate_manifest
from .common import (
    InstallError,
    digest,
    directory_lease,
    json_bytes,
    read_json,
    real_directory,
    regular_bytes,
    write_file,
)

INSTALL_MARKER = "installation.json"
INSTALL_DOCUMENT = {"schema_version": 1, "kind": "hkdl-installation"}
LEASE_ENV = "HKDL_INSTALLATION_FD"
ROOT_ENV = "HKDL_INSTALL_ROOT"


def default_root() -> Path:
    return Path(os.environ.get(ROOT_ENV, Path.home() / ".local/share/hkdl")).absolute()


@dataclass(frozen=True)
class InstalledRelease:
    path: Path
    manifest: dict[str, Any]
    identity: str

    @property
    def python(self) -> Path:
        return self.path / ".venv/bin/python"


@dataclass(frozen=True)
class InstallPaths:
    root: Path

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def workspaces(self) -> Path:
        return self.root / "workspaces"

    def initialize(self) -> None:
        real_directory(self.root, create=True)
        with directory_lease(self.root, exclusive=True):
            marker = self.root / INSTALL_MARKER
            if not os.path.lexists(marker):
                if any(self.root.iterdir()):
                    raise InstallError(f"installation root is not empty: {self.root}")
                write_file(marker, json_bytes(INSTALL_DOCUMENT))
            self.validate()
            for name in ("manager", "releases", "staging", "workspaces"):
                real_directory(self.root / name, create=True)

    def validate(self) -> None:
        real_directory(self.root)
        marker = read_json(regular_bytes(self.root / INSTALL_MARKER))
        if marker != INSTALL_DOCUMENT or type(marker.get("schema_version")) is not int:
            raise InstallError(f"unrecognized installation root: {self.root}")

    def current(self) -> InstalledRelease | None:
        self.validate()
        pointer = self.root / "current"
        if not os.path.lexists(pointer):
            return None
        if not pointer.is_symlink():
            raise InstallError(f"active release pointer is not a symlink: {pointer}")
        target = os.readlink(pointer)
        if not re.fullmatch(r"releases/[a-zA-Z0-9._-]+", target):
            raise InstallError("active release pointer leaves the release directory")
        return read_installed_release(self.root / target)

    def registered_workspaces(self) -> list[Path]:
        real_directory(self.workspaces)
        roots = []
        for entry in sorted(self.workspaces.iterdir()):
            if entry.name.startswith("."):
                continue
            value = read_json(regular_bytes(entry))
            if (
                set(value) != {"schema_version", "path"}
                or type(value["schema_version"]) is not int
                or value["schema_version"] != 1
                or not isinstance(value["path"], str)
            ):
                raise InstallError(f"invalid workspace registration: {entry}")
            root = Path(value["path"])
            if not root.is_absolute() or entry.name != registration_name(root):
                raise InstallError(f"workspace registration identity mismatch: {entry}")
            roots.append(real_directory(root))
        return roots

    def register(self, root: Path) -> None:
        root = real_directory(root)
        payload = json_bytes({"schema_version": 1, "path": str(root)})
        with directory_lease(self.workspaces, exclusive=True):
            target = self.workspaces / registration_name(root)
            if os.path.lexists(target):
                if regular_bytes(target) != payload:
                    raise InstallError(f"conflicting workspace registration: {target}")
                return
            write_file(target, payload)

    def is_registered(self, root: Path) -> bool:
        target = self.workspaces / registration_name(root.absolute())
        if not os.path.lexists(target):
            return False
        return read_json(regular_bytes(target)) == {
            "schema_version": 1,
            "path": str(root.absolute()),
        }


def registration_name(root: Path) -> str:
    return digest(str(root).encode()) + ".json"


def read_installed_release(path: Path) -> InstalledRelease:
    real_directory(path)
    raw = regular_bytes(path / "release.json")
    manifest = validate_manifest(read_json(raw))
    state = read_json(regular_bytes(path / "installed.json"))
    identity = digest(raw)
    if (
        state != {"schema_version": 1, "manifest_sha256": identity}
        or type(state.get("schema_version")) is not int
    ):
        raise InstallError(f"release installation is incomplete: {path}")
    return InstalledRelease(path, manifest, identity)
