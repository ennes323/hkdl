"""Canonical hashing and immutable publication for HKDL v2 objects."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hkdl.authoring.config import DIGEST_PATTERN
from hkdl.errors import ContractError

SCHEMA_VERSION = 2
OBJECT_KINDS = frozenset(
    {
        "experiment",
        "experiment_revision",
        "variant",
        "variant_revision",
        "source_tree",
        "evaluation_case",
        "export_profile",
        "option_set",
        "run_spec",
        "attempt",
        "attempt_event",
        "model",
        "eval_result",
        "export_result",
        "blob",
        "binding_transaction",
    }
)
_FRAME = b"hkdl-object-v2\0"
_BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ObjectRecord:
    digest: str
    kind: str
    payload: dict[str, Any]
    path: Path | None = None

    @property
    def envelope(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "payload": self.payload,
        }


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict canonical UTF-8 JSON after rejecting non-JSON values."""

    _validate_json(value, "object")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def object_digest(kind: str, payload: Mapping[str, Any]) -> str:
    """Hash an object envelope with explicit type and length framing."""

    envelope = _envelope(kind, payload)
    encoded = canonical_json_bytes(envelope)
    kind_bytes = kind.encode("utf-8")
    framed = (
        _FRAME
        + len(kind_bytes).to_bytes(4, "big")
        + kind_bytes
        + len(encoded).to_bytes(8, "big")
        + encoded
    )
    return f"sha256:{hashlib.sha256(framed).hexdigest()}"


class ObjectStore:
    """Repository-local immutable object and blob store."""

    def __init__(self, repository_root: Path):
        self.repository_root = Path(repository_root).absolute()
        self.root = self.repository_root / ".hkdl/store/v2"
        self.objects = self.root / "objects/sha256"
        self.blobs = self.root / "blobs/sha256"
        self.candidates = self.root / ".candidates"

    def preview(self, kind: str, payload: Mapping[str, Any]) -> ObjectRecord:
        normalized = dict(payload)
        digest = object_digest(kind, normalized)
        return ObjectRecord(digest, kind, normalized)

    def put(self, kind: str, payload: Mapping[str, Any]) -> ObjectRecord:
        record = self.preview(kind, payload)
        self._ensure_layout()
        destination = self.object_path(record.digest)
        content = canonical_json_bytes(record.envelope) + b"\n"
        self._publish_bytes(content, destination, expected=content)
        loaded = self.load(record.digest)
        if loaded.kind != kind or loaded.payload != record.payload:
            raise ContractError("published v2 object disagrees with candidate")
        return loaded

    def load(self, digest: str) -> ObjectRecord:
        path = self.object_path(digest)
        content = _read_regular(path, "v2 object")
        try:
            envelope = json.loads(content.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ContractError(f"v2 object is invalid JSON: {digest}") from error
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema_version",
            "kind",
            "payload",
        }:
            raise ContractError(f"v2 object envelope is invalid: {digest}")
        if envelope["schema_version"] != SCHEMA_VERSION:
            raise ContractError(f"v2 object schema is unsupported: {digest}")
        kind = envelope["kind"]
        payload = envelope["payload"]
        if not isinstance(kind, str) or not isinstance(payload, dict):
            raise ContractError(f"v2 object envelope is invalid: {digest}")
        actual = object_digest(kind, payload)
        if actual != digest:
            raise ContractError(f"v2 object hash mismatch: {digest}")
        canonical = canonical_json_bytes(envelope) + b"\n"
        if canonical != content:
            raise ContractError(f"v2 object is not canonical: {digest}")
        return ObjectRecord(digest, kind, payload, path)

    def put_blob(
        self, path: Path, *, media_type: str = "application/octet-stream"
    ) -> ObjectRecord:
        path = Path(path).absolute()
        digest, size = hash_file(path)
        self._ensure_layout()
        destination = self.blob_path(digest)
        self._publish_file(path, destination, digest)
        return self.put(
            "blob",
            {
                "content_hash": digest,
                "media_type": media_type,
                "size": size,
            },
        )

    def put_blob_bytes(
        self, content: bytes, *, media_type: str = "application/octet-stream"
    ) -> ObjectRecord:
        """Publish already-frozen evidence without an intermediate source file."""
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
        self._ensure_layout()
        self._publish_bytes(content, self.blob_path(digest), expected=content)
        record = self.put(
            "blob",
            {"content_hash": digest, "size": len(content), "media_type": media_type},
        )
        self.verify_blob(record)
        return record

    def verify_blob(self, record_or_digest: ObjectRecord | str) -> Path:
        record = (
            record_or_digest
            if isinstance(record_or_digest, ObjectRecord)
            else self.load(record_or_digest)
        )
        if record.kind != "blob":
            raise ContractError("v2 object is not a blob")
        content_hash = record.payload.get("content_hash")
        size = record.payload.get("size")
        if (
            not isinstance(content_hash, str)
            or not DIGEST_PATTERN.fullmatch(content_hash)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ContractError("v2 blob metadata is invalid")
        path = self.blob_path(content_hash)
        actual_hash, actual_size = hash_file(path)
        if actual_hash != content_hash or actual_size != size:
            raise ContractError(f"v2 blob hash mismatch: {content_hash}")
        return path

    def object_path(self, digest: str) -> Path:
        hexadecimal = _digest_hex(digest)
        return self.objects / hexadecimal[:2] / hexadecimal

    def blob_path(self, digest: str) -> Path:
        hexadecimal = _digest_hex(digest)
        return self.blobs / hexadecimal[:2] / hexadecimal

    def iter_records(self) -> list[ObjectRecord]:
        if not os.path.lexists(self.objects):
            return []
        if self.objects.is_symlink() or not self.objects.is_dir():
            raise ContractError("v2 object catalog must be a real directory")
        records: list[ObjectRecord] = []
        for prefix in sorted(self.objects.iterdir(), key=lambda path: path.name):
            if prefix.is_symlink() or not prefix.is_dir() or len(prefix.name) != 2:
                raise ContractError("v2 object catalog contains an invalid prefix")
            for entry in sorted(prefix.iterdir(), key=lambda path: path.name):
                digest = f"sha256:{entry.name}"
                if entry.name[:2] != prefix.name:
                    raise ContractError("v2 object catalog prefix mismatch")
                records.append(self.load(digest))
        return records

    def _ensure_layout(self) -> None:
        _ensure_real_directory(self.repository_root / ".hkdl")
        _ensure_real_directory(self.repository_root / ".hkdl/store")
        _ensure_real_directory(self.root)
        _ensure_real_directory(self.objects.parent)
        _ensure_real_directory(self.objects)
        _ensure_real_directory(self.blobs.parent)
        _ensure_real_directory(self.blobs)
        _ensure_real_directory(self.candidates)

    def _publish_bytes(
        self, content: bytes, destination: Path, *, expected: bytes
    ) -> None:
        _ensure_real_directory(destination.parent)
        candidate = self.candidates / f"object-{secrets.token_hex(16)}"
        descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            _link_no_replace(candidate, destination)
        finally:
            candidate.unlink(missing_ok=True)
        if _read_regular(destination, "v2 object") != expected:
            raise ContractError("existing v2 object disagrees with content hash")

    def _publish_file(self, source: Path, destination: Path, digest: str) -> None:
        _ensure_real_directory(destination.parent)
        if os.path.lexists(destination):
            actual, _ = hash_file(destination)
            if actual != digest:
                raise ContractError("existing v2 blob disagrees with content hash")
            return
        candidate = self.candidates / f"blob-{secrets.token_hex(16)}"
        source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        candidate_descriptor = os.open(
            candidate,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        try:
            with (
                os.fdopen(source_descriptor, "rb") as reader,
                os.fdopen(candidate_descriptor, "wb") as writer,
            ):
                while chunk := reader.read(_BUFFER_SIZE):
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            actual, _ = hash_file(candidate)
            if actual != digest:
                raise ContractError("v2 blob source changed during publication")
            _link_no_replace(candidate, destination)
        finally:
            candidate.unlink(missing_ok=True)
        actual, _ = hash_file(destination)
        if actual != digest:
            raise ContractError("published v2 blob hash mismatch")


def hash_file(path: Path) -> tuple[str, int]:
    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"cannot read blob source: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"blob source must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        while chunk := handle.read(_BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _envelope(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if kind not in OBJECT_KINDS:
        raise ContractError(f"unsupported v2 object kind: {kind}")
    if not isinstance(payload, Mapping):
        raise ContractError("v2 object payload must be a mapping")
    normalized = dict(payload)
    _validate_json(normalized, "payload")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "payload": normalized,
    }


def _validate_json(value: Any, location: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{location} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item, location)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{location} mapping keys must be strings")
            _validate_json(item, location)
        return
    raise ContractError(f"{location} contains a non-JSON value")


def _digest_hex(digest: str) -> str:
    if not isinstance(digest, str) or not DIGEST_PATTERN.fullmatch(digest):
        raise ContractError("invalid SHA-256 digest")
    return digest.removeprefix("sha256:")


def _read_regular(path: Path, location: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"{location} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{location} must be a regular non-symlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ContractError(f"{location} is unavailable: {path}") from error


def _ensure_real_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o755)
    except FileExistsError:
        pass
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"v2 store directory is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"v2 store path must be a real directory: {path}")


def _link_no_replace(candidate: Path, destination: Path) -> None:
    try:
        os.link(candidate, destination, follow_symlinks=False)
    except FileExistsError:
        pass
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
