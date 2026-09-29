"""Read-only captured inputs and bounded worker-log views for one Run."""

from __future__ import annotations

import os
import stat
from typing import Any

from hkdl.authoring.research_json import (
    split_legacy_variant,
)
from hkdl.errors import ContractError
from hkdl.execution.run_records import RunRecord
from hkdl.storage.graph.reader import GraphReader
from hkdl.storage.runs import RunStore
from hkdl.storage.storage import NotFoundError

MAX_LOG_BYTES = 64 * 1024
MAX_LOG_LINES = 200


def captured_run(
    record: RunRecord, reader: GraphReader | None
) -> tuple[RunRecord, dict[str, Any]]:
    """Present captured inputs through their graph or legacy owner."""
    if reader is not None:
        record = reader.authoritative_record(record)
        return record, reader.captured_inputs(record)
    split = split_legacy_variant(record.snapshot["variant"])
    return record, {
        "source": "legacy_snapshot",
        "code": split.code,
        "options": split.options,
    }


def worker_log_view(store: RunStore, record: RunRecord) -> dict[str, Any]:
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
        path = store.resolve_worker_log(record)
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
