"""Shared terminal formatting and confirmation for command handlers.

Handlers choose report fields, output streams and when confirmation is required.
These helpers render that choice; services retain mutation and validation policy.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from typing import Any


def table(
    headers: tuple[str, ...],
    rows: Iterable[Sequence[object]],
    *,
    file: Any = None,
) -> None:
    """Materialize rows for aligned columns, defaulting to standard output.

    A mismatched row width is a caller error. This buffers the full table and is
    not the streaming path used for live metric following.
    """
    if file is None:
        file = sys.stdout
    rendered_rows = [tuple(str(value) for value in row) for row in rows]
    if any(len(row) != len(headers) for row in rendered_rows):
        raise AssertionError("table row width does not match headers")

    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) if index < len(row) - 1 else value
            for index, value in enumerate(row)
        )

    print(render(headers), file=file)
    for row in rendered_rows:
        print(render(row), file=file)


def confirm(question: str) -> bool:
    """Prompt on stderr, accepting only y/yes; empty input and EOF decline."""
    print(f"{question} [y/N] ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in {"y", "yes"}


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if value < 1024 or unit == "EiB":
            return f"{size} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable size unit")


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def public_payload(value: Any) -> Any:
    """Omit graph bookkeeping from selected user-facing command reports.

    This is a presentation projection, not a general sensitive-data filter.
    It leaves the original service result intact for internal use.
    """
    if isinstance(value, dict):
        return {
            key: public_payload(item)
            for key, item in value.items()
            if not key.endswith("_hash")
            and key
            not in {
                "before_head",
                "binding_transaction",
                "checkpoint_blob",
                "dependencies",
                "event_chain",
                "attempt_hashes",
                "run_spec_hashes",
                "event_hashes",
                "result_hashes",
                "blob_hashes",
                "operations",
                "plan_digest",
                "producing_attempt",
                "retry_parent",
                "unbound",
                "variant_hashes",
                "revision_hashes",
                "source_draft_fingerprint",
                "target_draft_fingerprint",
            }
        }
    if isinstance(value, list):
        return [public_payload(item) for item in value]
    return value


def short_hash(digest: str) -> str:
    return digest.removeprefix("sha256:")[:12]
