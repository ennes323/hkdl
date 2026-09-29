"""Bridge root-independent CLI input to workspace initialization services.

Initialization owns the directory lease, marker recovery, and installation
registration. This module supplies the interactive policy for an existing
guidance file and reports whether the workspace was newly created.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hkdl.errors import ContractError
from hkdl.installation import installed_paths
from hkdl.storage.workspace import initialize_managed_workspace, initialize_workspace


def run(args: argparse.Namespace) -> int:
    """Initialize or adopt a workspace through the active installation mode."""

    if args.noun == "workspace":
        installation = installed_paths()
        if installation is not None:
            repository, created = initialize_managed_workspace(
                Path(args.path),
                installation,
                choose_guidance=_choose_workspace_guidance,
                report_guidance=print,
            )
        else:
            repository, created = initialize_workspace(
                Path(args.path),
                choose_guidance=_choose_workspace_guidance,
                report_guidance=print,
            )
        state = "initialized" if created else "already initialized"
        print(f"HKDL workspace {state}: {repository.root}")
        return 0

    raise AssertionError("unreachable command")


def _choose_workspace_guidance(root: Path) -> str:
    """Return the user's policy when initialization finds modified guidance."""
    print(
        f"Existing user or modified guidance: {root / 'AGENTS.md'}\n"
        "  keep: preserve AGENTS.md (default)\n"
        "  replace: back up AGENTS.md and install HKDL guidance\n"
        "  user: preserve it as AGENTS.user.md and install HKDL guidance\n"
        "Choose [keep/replace/user]: ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    choice = sys.stdin.readline().strip().lower() or "keep"
    if choice not in {"keep", "replace", "user"}:
        raise ContractError("choose keep, replace, or user; no guidance was changed")
    return choice
