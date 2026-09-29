"""Durable, byte-checked recovery of the authoring cutover boundary."""

from __future__ import annotations

import base64
import os
from pathlib import PurePosixPath

from hkdl.authoring.config import DIGEST_PATTERN
from hkdl.authoring.research_json import dump_json, load_json_file
from hkdl.errors import ContractError
from hkdl.storage.storage import atomic_replace, atomic_write_new

from .bindings import _operation
from .graph import V2Graph
from .maintenance import JOURNAL, WorkspaceBusy

MARKER = ".hkdl/store/AUTHORING_CURRENT"


def _read(path):
    if not os.path.lexists(path):
        return None
    if path.is_symlink() or not path.is_file():
        raise ContractError("cutover path must be a regular file")
    return path.read_bytes()


def _encode(content):
    return base64.b64encode(content).decode("ascii") if content is not None else None


def _decode(content):
    try:
        return base64.b64decode(content, validate=True) if content is not None else None
    except (ValueError, TypeError) as error:
        raise ContractError("cutover journal bytes are invalid") from error


def _fsync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_document(document):
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema_version",
            "plan_digest",
            "before_head",
            "operations",
            "graph_only",
            "files",
        }
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["plan_digest"], str)
        or not DIGEST_PATTERN.fullmatch(document["plan_digest"])
        or type(document["graph_only"]) is not bool
    ):
        raise ContractError("cutover journal identity is invalid")
    before = document["before_head"]
    if before is not None and (
        not isinstance(before, str) or not DIGEST_PATTERN.fullmatch(before)
    ):
        raise ContractError("cutover journal HEAD is invalid")
    if not isinstance(document["operations"], list):
        raise ContractError("cutover journal operations are invalid")
    for operation in document["operations"]:
        if not isinstance(operation, dict) or any(
            not isinstance(value, str) for value in operation.values()
        ):
            raise ContractError("cutover journal operation is invalid")
        _operation(operation)
    if not isinstance(document["files"], list):
        raise ContractError("cutover journal files are invalid")
    paths = set()
    for entry in document["files"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "before", "after"}
            or not isinstance(entry["path"], str)
            or entry["path"] in paths
        ):
            raise ContractError("cutover journal file entry is invalid")
        paths.add(entry["path"])
        _decode(entry["before"])
        _decode(entry["after"])
    marker = next((item for item in document["files"] if item["path"] == MARKER), None)
    if marker is None or _decode(marker["after"]) != b"v2\n":
        raise ContractError("cutover journal marker is invalid")
    if document["graph_only"] != (_decode(marker["before"]) == b"v2\n"):
        raise ContractError("cutover journal marker does not match cutover mode")


class CutoverJournal:
    def __init__(self, repository):
        self.repository = repository
        self.path = repository.root / JOURNAL
        self.graph = V2Graph(repository)

    def _target(self, relative):
        parsed = PurePosixPath(relative)
        if (
            parsed.as_posix() != relative
            or parsed.is_absolute()
            or ".." in parsed.parts
        ):
            raise ContractError("cutover journal path escapes workspace")
        special = relative in {".hkdl/settings.json", MARKER}
        authored = (
            len(parsed.parts) in {3, 4}
            and parsed.parts[0] == "experiments"
            and parsed.name
            in {
                "experiment.yaml",
                "variant.yaml",
                "experiment.json",
                "code.json",
                "options.json",
            }
        )
        if not (special or authored):
            raise ContractError(
                "cutover journal target is outside the authored contract"
            )
        path = self.repository.root
        for part in parsed.parts:
            path = path / part
            if path.is_symlink():
                raise ContractError("cutover journal target contains a symlink")
        return path

    def prepare(self, plan, settings_after):
        if os.path.lexists(self.path):
            raise WorkspaceBusy("unfinished authoring cutover requires recovery")
        changes = {}
        for item in plan.files:
            changes[item.source] = None
            changes[item.target] = item.content
        if settings_after is not None:
            changes[".hkdl/settings.json"] = settings_after
        changes[MARKER] = b"v2\n"
        document = {
            "schema_version": 1,
            "plan_digest": plan.plan_digest,
            "before_head": self.graph.bindings.head(),
            "operations": [item.as_dict() for item in plan.binding_operations],
            "graph_only": _read(self.repository.root / MARKER) == b"v2\n",
            "files": [
                {
                    "path": path,
                    "before": _encode(_read(self._target(path))),
                    "after": _encode(content),
                }
                for path, content in sorted(changes.items())
            ],
        }
        atomic_write_new(self.path, dump_json(document))

    def recover(self):
        if not os.path.lexists(self.path):
            return None
        document = load_json_file(self.path)
        _validate_document(document)
        before = document["before_head"]
        current = self.graph.bindings.head()
        changed = current != before
        if changed:
            if current is None:
                raise WorkspaceBusy(
                    "binding HEAD disappeared during unfinished cutover"
                )
            transaction = self.graph.store.load(current)
            if (
                transaction.payload["previous"] != before
                or transaction.payload["operations"] != document["operations"]
            ):
                raise WorkspaceBusy(
                    "binding HEAD diverged from unfinished cutover; preserved for inspection"
                )
        committed = (
            changed
            if document["graph_only"]
            else _read(self.repository.root / MARKER) == b"v2\n"
        )
        if committed and document["operations"] and not changed:
            raise WorkspaceBusy(
                "authoring marker advanced without the planned binding HEAD"
            )
        changes = []
        for entry in document["files"]:
            path = self._target(entry["path"])
            old, new = _decode(entry["before"]), _decode(entry["after"])
            current_bytes = _read(path)
            if current_bytes not in (old, new):
                raise WorkspaceBusy(
                    f"authored file changed during cutover; preserved: {entry['path']}"
                )
            desired = new if committed else old
            if current_bytes != desired:
                changes.append((path, desired))
        if changed and not committed:
            if before is None:
                self.graph.bindings.head_path.unlink()
                _fsync(self.graph.bindings.head_path.parent)
            else:
                atomic_replace(self.graph.bindings.head_path, f"{before}\n")
        for path, content in changes:
            if content is None:
                path.unlink()
                _fsync(path.parent)
            elif os.path.lexists(path):
                atomic_replace(path, content)
            else:
                atomic_write_new(path, content)
        # The disposable index may describe either HEAD. Readers never rely on
        # that index; the explicit rebuild occurs after recovery completes.
        self.path.unlink()
        _fsync(self.path.parent)
        return {
            "plan_digest": document["plan_digest"],
            "outcome": "completed" if committed else "rolled_back",
        }
