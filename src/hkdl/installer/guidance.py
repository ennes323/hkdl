"""Owned research-root guidance, shared by Core and the standalone installer."""

from __future__ import annotations

import io
import os
import stat
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from .bundle import SHA256, version_tuple
from .common import (
    InstallError,
    digest,
    json_bytes,
    read_json,
    real_directory,
    write_file,
)

RESOURCE = "research-agents.md"
WHEEL_RESOURCE = f"hkdl/installer/{RESOURCE}"
RECEIPT = ".hkdl/agent-guidance.json"
HEADER = b'<!-- hkdl:research-guidance version="1" -->\n'
MAX_BYTES = 1024 * 1024
Choice = Callable[[Path], str]
Report = Callable[[str], None]


def reference(version: str) -> str:
    version_tuple(version)
    return f"https://github.com/hukuhaka/hkdl/blob/v{version}/src/{WHEEL_RESOURCE}"


def packaged_guidance() -> bytes:
    return _validate_payload(files(__package__).joinpath(RESOURCE).read_bytes())


def bundle_guidance(wheel: bytes) -> bytes | None:
    """Read the selected candidate's resource, never the old installer's copy."""
    try:
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            entries = [i for i in archive.infolist() if i.filename == WHEEL_RESOURCE]
            if not entries:
                return None  # Older candidates did not distribute guidance.
            if len(entries) != 1 or entries[0].file_size > MAX_BYTES:
                raise InstallError("invalid bundled agent guidance resource")
            return _validate_payload(archive.read(entries[0]))
    except zipfile.BadZipFile as error:
        raise InstallError(
            "cannot read agent guidance from the candidate wheel"
        ) from error


def _validate_payload(payload: bytes) -> bytes:
    if not payload.startswith(HEADER) or len(payload) > MAX_BYTES:
        raise InstallError("invalid HKDL research guidance")
    try:
        payload.decode("utf-8")
    except UnicodeError as error:
        raise InstallError("HKDL research guidance must be UTF-8") from error
    return payload


def _read(path: Path) -> bytes | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        # A dangling symlink is not an absent user file.
        if os.path.lexists(path):
            raise InstallError(f"refusing non-regular guidance path: {path}")
        return None
    except OSError as error:
        raise InstallError(
            f"cannot safely read guidance path: {path}: {error}"
        ) from error
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise InstallError(f"refusing non-regular guidance path: {path}")
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise InstallError(f"guidance file exceeds 1 MiB: {path}")
    return data


@dataclass(frozen=True)
class GuidanceState:
    content: bytes | None
    receipt: bytes | None
    owned: bool


def inspect(root: Path) -> GuidanceState:
    real_directory(root)
    real_directory(root / ".hkdl")
    content = _read(root / "AGENTS.md")
    receipt = _read(root / RECEIPT)
    owned = False
    if content is not None and receipt is not None:
        try:
            record = read_json(receipt)
            owned = (
                set(record) == {"schema_version", "sha256", "version"}
                and type(record["schema_version"]) is int
                and record["schema_version"] == 1
                and isinstance(record["sha256"], str)
                and SHA256.fullmatch(record["sha256"]) is not None
                and content.startswith(HEADER)
                and digest(content) == record["sha256"]
            )
            version_tuple(record["version"])
        except (InstallError, KeyError):
            owned = False
    return GuidanceState(content, receipt, owned)


def _unchanged(root: Path, expected: GuidanceState) -> None:
    if inspect(root) != expected:
        raise InstallError(
            f"guidance changed after review; repeat workspace init: {root}"
        )


def _publish(root: Path, state: GuidanceState, payload: bytes, version: str) -> None:
    """Publish ownership last; interrupted writes never authorize future overwrites."""
    _unchanged(root, state)
    write_file(root / "AGENTS.md", payload, replace=state.content is not None)
    write_file(
        root / RECEIPT,
        json_bytes(
            {"schema_version": 1, "sha256": digest(payload), "version": version}
        ),
        replace=state.receipt is not None,
    )


def initialize(
    root: Path,
    *,
    version: str,
    choose: Choice | None = None,
    report: Report = lambda message: None,
) -> None:
    """Called under exclusive workspace admission after root validation."""
    payload = packaged_guidance()
    state = inspect(root)
    if state.content is None:
        _publish(root, state, payload, version)
        report(f"Installed HKDL guidance: {root / 'AGENTS.md'}")
        return
    if state.owned:
        if state.content != payload:
            _publish(root, state, payload, version)
            report(f"Updated HKDL guidance: {root / 'AGENTS.md'}")
        return
    choice = choose(root) if choose else "keep"
    if choice == "keep":
        report(
            f"Preserved user guidance: {root / 'AGENTS.md'}; HKDL guide: {reference(version)}"
        )
        return
    if choice not in {"replace", "user"}:
        raise InstallError("guidance choice must be keep, replace, or user")
    destination = (
        root / "AGENTS.user.md"
        if choice == "user"
        else root / f"AGENTS.md.backup-{uuid.uuid4().hex}"
    )
    if os.path.lexists(destination):
        raise InstallError(
            f"preserve existing {destination}; choose keep or backup-replace instead"
        )
    _unchanged(root, state)
    write_file(destination, state.content)
    report(f"Preserved previous guidance: {destination}")
    _publish(root, state, payload, version)
    report(f"Installed HKDL guidance: {root / 'AGENTS.md'}")


@dataclass(frozen=True)
class RefreshPlan:
    root: Path
    state: GuidanceState | None
    action: str


def plan_refresh(root: Path, payload: bytes | None) -> RefreshPlan:
    if payload is None:
        return RefreshPlan(root, None, "preserve (candidate has no guidance)")
    try:
        state = inspect(root)
    except (OSError, InstallError) as error:
        return RefreshPlan(root, None, f"preserve ({error})")
    if not state.owned:
        return RefreshPlan(root, state, "preserve (user-owned, modified, or missing)")
    return RefreshPlan(
        root, state, "unchanged" if state.content == payload else "refresh"
    )


def refresh(plan: RefreshPlan, payload: bytes, version: str) -> str:
    """Called after activation under the existing exclusive workspace lease."""
    if plan.action != "refresh" or plan.state is None:
        return f"Guidance {plan.action}: {plan.root}"
    _publish(plan.root, plan.state, payload, version)
    return f"Guidance refreshed: {plan.root / 'AGENTS.md'}"
