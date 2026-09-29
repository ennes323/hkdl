"""Read a bounded release bundle without extracting arbitrary archive paths."""

from __future__ import annotations

import io
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import InstallError, digest, read_json, regular_bytes, write_file

PROTOCOL = 1
INSTALLER_VERSION = "1.0.0"
MAX_BUNDLE_BYTES = 128 * 1024 * 1024
VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
SHA256 = re.compile(r"[0-9a-f]{64}")
WORKSPACE_MODES = frozenset({"legacy-yaml", "v2-yaml", "v2-json"})


def version_tuple(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise InstallError(f"invalid HKDL version: {value!r}")
    return tuple(map(int, value.split(".")))


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "bundle_schema",
        "version",
        "python",
        "kind",
        "files",
        "wheel",
        "direct_from",
        "source_routes",
        "workspace_modes",
        "templates",
    }
    if set(value) != expected or type(value["bundle_schema"]) is not int:
        raise InstallError("invalid release manifest fields")
    if value["bundle_schema"] != PROTOCOL:
        raise InstallError("unsupported bundle protocol; obtain its matching installer")
    version_tuple(value["version"])
    if (
        value["python"] != "3.12"
        or not isinstance(value["kind"], str)
        or value["kind"] not in {"candidate", "release"}
    ):
        raise InstallError("unsupported release runtime or kind")
    for field in ("direct_from", "workspace_modes", "source_routes", "templates"):
        if not isinstance(value[field], list):
            raise InstallError(f"release {field} must be a list")
    for version in value["direct_from"]:
        version_tuple(version)
    if not value["workspace_modes"] or any(
        not isinstance(mode, str) or mode not in WORKSPACE_MODES
        for mode in value["workspace_modes"]
    ):
        raise InstallError("unsupported workspace mode in release manifest")
    _validate_routes(value["source_routes"])
    _validate_templates(value["templates"])
    _validate_files(value)
    return value


def _validate_routes(routes: list[Any]) -> None:
    for route in routes:
        if not isinstance(route, list) or len(route) != 2:
            raise InstallError("source transition must contain two versions")
        source, target = map(version_tuple, route)
        if source >= target:
            raise InstallError("source transition must advance to a newer version")


def _validate_templates(templates: list[Any]) -> None:
    seen = set()
    for template in templates:
        if not isinstance(template, dict) or set(template) != {
            "name",
            "version",
            "digest",
        }:
            raise InstallError("invalid bundled Template identity")
        name = template["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise InstallError("invalid bundled Template name")
        version_tuple(template["version"])
        key = (name, template["version"])
        if (
            key in seen
            or not isinstance(template["digest"], str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", template["digest"])
        ):
            raise InstallError("invalid or repeated bundled Template identity")
        seen.add(key)
    if not seen:
        raise InstallError("release must include its Template catalog")


def _validate_files(value: dict[str, Any]) -> None:
    wheel = value["wheel"]
    if (
        not isinstance(wheel, str)
        or wheel != f"hkdl-{value['version']}-py3-none-any.whl"
    ):
        raise InstallError("release wheel name does not match its version")
    files = value["files"]
    if not isinstance(files, dict) or set(files) != {wheel, "requirements.txt"}:
        raise InstallError("release must contain exactly its wheel and dependency lock")
    for item in files.values():
        if (
            not isinstance(item, dict)
            or set(item) != {"sha256", "size"}
            or not isinstance(item["sha256"], str)
            or not SHA256.fullmatch(item["sha256"])
            or type(item["size"]) is not int
            or not 0 < item["size"] <= MAX_BUNDLE_BYTES
        ):
            raise InstallError("invalid release file checksum or size")


@dataclass(frozen=True)
class Bundle:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    payloads: dict[str, bytes]
    archive_digest: str

    @property
    def identity(self) -> str:
        return digest(self.manifest_bytes)

    def unpack(self, root: Path) -> None:
        write_file(root / "release.json", self.manifest_bytes)
        for name, payload in self.payloads.items():
            write_file(root / name, payload)


def load_bundle(path: Path, *, expected_sha256: str | None = None) -> Bundle:
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise InstallError("release bundle exceeds the supported size")
    payload = regular_bytes(path)
    archive_digest = digest(payload)
    if expected_sha256 is not None and (
        not SHA256.fullmatch(expected_sha256) or archive_digest != expected_sha256
    ):
        raise InstallError("release bundle checksum does not match")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return _read_archive(archive, archive_digest)
    except (zipfile.BadZipFile, KeyError, RuntimeError) as error:
        raise InstallError(f"invalid release bundle: {error}") from error


def _read_archive(archive: zipfile.ZipFile, archive_digest: str) -> Bundle:
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(entries) != 3 or len(set(names)) != 3 or "release.json" not in names:
        raise InstallError("release bundle must contain exactly three distinct files")
    for entry in entries:
        mode = entry.external_attr >> 16
        if entry.is_dir() or stat.S_ISLNK(mode) or entry.file_size > MAX_BUNDLE_BYTES:
            raise InstallError("invalid release archive entry")
    if sum(entry.file_size for entry in entries) > MAX_BUNDLE_BYTES:
        raise InstallError("expanded release bundle exceeds the supported size")
    if archive.getinfo("release.json").file_size > 1024 * 1024:
        raise InstallError("release manifest exceeds the supported size")
    raw = archive.read("release.json")
    manifest = validate_manifest(read_json(raw))
    if set(names) != {"release.json", *manifest["files"]}:
        raise InstallError("release archive contains an unexpected path")
    payloads = {}
    for name, expected in manifest["files"].items():
        payload = archive.read(name)
        if len(payload) != expected["size"] or digest(payload) != expected["sha256"]:
            raise InstallError(f"release file checksum does not match: {name}")
        payloads[name] = payload
    return Bundle(manifest, raw, payloads, archive_digest)


def transition_path(source: str, manifest: dict[str, Any]) -> list[str] | None:
    version_tuple(source)
    queue = [[source]]
    visited = {source}
    while queue:
        path = queue.pop(0)
        if path[-1] in manifest["direct_from"]:
            return [*path, manifest["version"]]
        for start, end in manifest["source_routes"]:
            if start == path[-1] and end not in visited:
                visited.add(end)
                queue.append([*path, end])
    return None


def require_transition(source: str, manifest: dict[str, Any]) -> None:
    current, target = version_tuple(source), version_tuple(manifest["version"])
    if target < current or current[0] != target[0]:
        raise InstallError("downgrade or major-version transition is not supported")
    if source not in manifest["direct_from"]:
        route = transition_path(source, manifest)
        detail = " -> ".join(route) if route else "no verified transition is recorded"
        raise InstallError(
            f"direct transition from {source} is unverified: {detail}. "
            "Follow the intermediate source-release instructions before adopting "
            "this bundle; no source update or migration was performed."
        )
