"""Best-effort failure diagnostics shared with the standalone stdlib worker."""

from __future__ import annotations

import os
import re
import sys
import traceback
from urllib.parse import quote

_LIMIT = 16 * 1024
_SECRET_NAME = re.compile(
    r"(?:^|_)(?:TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIALS?|API_?KEY|"
    r"ACCESS_KEY|PRIVATE_KEY)(?:_|$)",
    re.IGNORECASE,
)
_AUTHORITY = re.compile(r"([a-z][a-z0-9+.-]*://)[^\s/@]+@", re.IGNORECASE)
_AUTHORIZATION = re.compile(r"\b(Bearer|Basic)\s+[^\s,;\"']+", re.IGNORECASE)
_ASSIGNMENT = re.compile(
    r"(\b(?:token|access_token|refresh_token|password|passwd|secret|"
    r"client_secret|api[_-]?key|credential|signature|authorization)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;&}\]\"']+)",
    re.IGNORECASE,
)


def redact(text: str) -> str:
    """Remove known external auth values and common credential spellings."""
    values = {
        value
        for name, value in os.environ.items()
        if value
        and (
            _SECRET_NAME.search(name)
            or name in {"MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME"}
        )
    }
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, "<redacted>")
        text = text.replace(quote(value, safe=""), "<redacted>")
    text = _AUTHORITY.sub(r"\1<redacted>@", text)
    text = _AUTHORIZATION.sub(r"\1 <redacted>", text)
    return _ASSIGNMENT.sub(r"\1<redacted>", text)


def _bounded(text: str) -> str:
    text = redact(text)
    if len(text) > _LIMIT:
        return f"[truncated; last {_LIMIT} characters]\n{text[-_LIMIT:]}"
    return text


def report_message(phase: str, message: str) -> None:
    """Diagnostics must not replace the original failure if stderr is broken."""
    try:
        print(f"[hkdl {phase}] {_bounded(message)}", file=sys.stderr, flush=True)
    except Exception:
        pass


def report_exception(phase: str, error: BaseException) -> None:
    """Report exception chains and frame locations, never locals or source lines."""
    try:
        chain: list[BaseException] = []
        current: BaseException | None = error
        while current is not None and all(current is not item for item in chain):
            chain.append(current)
            if len(chain) == 8:
                break
            current = current.__cause__ or (
                None if current.__suppress_context__ else current.__context__
            )
        parts: list[str] = []
        for item in reversed(chain):
            if parts:
                parts.append("The following exception occurred:")
            frames = list(traceback.walk_tb(item.__traceback__))
            if frames:
                parts.append("Traceback (most recent call last):")
            for frame, line in frames[-20:]:
                parts.append(
                    f'  File "{frame.f_code.co_filename}", line {line}, '
                    f"in {frame.f_code.co_name}"
                )
            parts.append(f"{type(item).__name__}: {item}")
        report_message(phase, "\n".join(parts))
    except Exception:
        report_message(phase, type(error).__name__)


def report_process_failure(
    phase: str,
    returncode: int,
    *,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
) -> None:
    report_message(phase, f"process exited with code {returncode}")
    for label, output in (("stdout", stdout), ("stderr", stderr)):
        if output:
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            report_message(phase, f"{label}:\n{output}")
