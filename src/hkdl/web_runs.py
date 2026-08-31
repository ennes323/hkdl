"""Read-only captured inputs and bounded worker-log views for one Run."""

from __future__ import annotations

import os
import stat
from copy import deepcopy
from typing import Any

from .config import ContractError
from .research_json import (
    split_legacy_variant,
    validate_code_json,
    validate_options_json,
)
from .run_contracts import TERMINAL_STATUSES, validate_tracker
from .runs import RunRecord, RunStore
from .storage import NotFoundError
from .v2.execution import GraphRecorder


MAX_LOG_BYTES = 64 * 1024
MAX_LOG_LINES = 200


def captured_run(
    record: RunRecord, graph: GraphRecorder
) -> tuple[RunRecord, dict[str, Any]]:
    """Prefer committed graph inputs; never consult the current authored draft."""
    split = split_legacy_variant(record.snapshot["variant"])
    code, options = split.code, split.options
    source = "legacy_snapshot"
    if graph.active():
        record = graph.authoritative_record(record)
        identity = graph.identity(record)
        objects = graph.graph.store
        revision = objects.load(identity.variant_revision_hash)
        if revision.kind != "variant_revision":
            raise ContractError("Run Code reference is not a Variant revision")
        provenance = revision.payload["template"]
        code = {
            "schema_version": 2,
            "template": {key: provenance[key] for key in ("name", "version")},
            "components": deepcopy(revision.payload["components"]),
        }
        if identity.option_set_hash is not None:
            option_set = objects.load(identity.option_set_hash)
            if (
                option_set.kind != "option_set"
                or option_set.payload.get("scope") != "full"
            ):
                raise ContractError("Run Options reference is not a full OptionSet")
            options = deepcopy(option_set.payload["document"])
            source = "content_addressed"
        attempt = objects.load(identity.attempt_hash)
        spec = objects.load(identity.run_spec_hash)
        if (
            spec.kind != "run_spec"
            or spec.payload["action"] != record.request["action"]
        ):
            raise ContractError("Run action disagrees with its RunSpec")
        snapshot = deepcopy(record.snapshot)
        snapshot["variant"].update(
            {key: value for key, value in options.items() if key != "schema_version"}
        )
        snapshot["variant"]["components"] = deepcopy(code["components"])
        snapshot["variant"]["template"] = deepcopy(provenance)
        if "tracker_backends" in attempt.payload:
            backends = attempt.payload["tracker_backends"]
            tracker = {"backend": backends if backends else "none"}
            validate_tracker(tracker)
            snapshot["variant"]["tracker"] = tracker
        request = deepcopy(record.request)
        request["exec"]["device"] = spec.payload["device"]
        request["created_at"] = attempt.payload["created_at"]
        if request["action"] == "train":
            request["target"]["seed"] = spec.payload["seed"]
        record = RunRecord(
            record.path,
            record.address,
            snapshot,
            request,
            record.state,
            record.graph_identity,
            record.event_hash,
        )
    validate_code_json(code)
    validate_options_json(options)
    return record, {"source": source, "code": code, "options": options}


def worker_log_view(
    store: RunStore, graph: GraphRecorder, record: RunRecord
) -> dict[str, Any]:
    """Missing optional logs are distinct from corrupt references or files."""
    base = {
        "availability": "not_recorded",
        "text": "",
        "bytes_total": 0,
        "lines_returned": 0,
        "truncated": False,
        "decoding_replaced": False,
    }
    try:
        path = (
            graph.artifact_path(record, "worker.log")
            if graph.active() and record.state["status"] in TERMINAL_STATUSES
            else store.resolve_worker_log(record)
        )
    except NotFoundError:
        return base
    try:
        # no-follow also protects against replacing the validated final file with a link.
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ContractError("Run worker log is not a regular file")
            start = max(0, info.st_size - MAX_LOG_BYTES)
            stream.seek(start)
            tail = stream.read(min(info.st_size, MAX_LOG_BYTES))
    except OSError as error:
        raise ContractError("Run worker log is unavailable") from error
    # A byte-limited tail may start within a UTF-8 character or a line. Skip the
    # incomplete first line when possible; otherwise disclose replacement decoding.
    if start and b"\n" in tail:
        tail = tail.split(b"\n", 1)[1]
    try:
        decoded = tail.decode("utf-8")
        replaced = False
    except UnicodeError:
        decoded = tail.decode("utf-8", errors="replace")
        replaced = True
    lines = decoded.splitlines(keepends=True)
    return {
        **base,
        "availability": "available",
        "text": "".join(lines[-MAX_LOG_LINES:]),
        "bytes_total": info.st_size,
        "lines_returned": min(len(lines), MAX_LOG_LINES),
        "truncated": start > 0 or len(lines) > MAX_LOG_LINES,
        "decoding_replaced": replaced,
    }
