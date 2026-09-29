"""Connect parsed commands to command families within workspace access.

main owns installation access and error-to-exit translation. This module selects
workspace admission and shared read observations; command modules own procedures
and presentation, while their services enforce operation-specific contracts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hkdl.errors import ContractError
from hkdl.installation import installed_paths
from hkdl.interfaces import completion, web
from hkdl.storage.graph.maintenance import workspace_access
from hkdl.storage.graph.reader import graph_observation
from hkdl.storage.storage import RepositoryPaths, validate_repository_root

from .commands import (
    experiment,
    maintenance,
    migration,
    model,
    status,
    template,
    variant,
    workspace,
)
from .commands import run as run_commands


# Flow: check registration -> select admission -> select observation -> route.
def run(args: argparse.Namespace) -> int:
    """Route parsed arguments under the command's required workspace scope.

    Called within main's installation access. Ordinary commands retain workspace
    admission through the command handler; nested service access reuses it.
    Exceptions propagate to main after this scope unwinds.
    """
    installation = installed_paths()
    if installation is not None and args.noun not in {"workspace", "completion"}:
        if not installation.is_registered(Path.cwd()):
            raise ContractError(
                "workspace is not registered with this installation; "
                "run hkdl workspace init in the intended workspace first"
            )
    # Completion and initialization need no admitted current workspace.
    # Migration and experiment bootstrap own maintenance/recovery admission;
    # web admits individual requests instead of holding a server-lifetime lease.
    if args.noun in {"migrate", "web", "completion", "workspace"} or (
        args.noun == "experiment" and args.verb == "create"
    ):
        return _dispatch(args)
    # Applying promotion may resume its journal, so enter exclusive admission
    # here: a nested service cannot upgrade an already-held shared lease.
    promotion_recovery = (
        args.noun == "variant" and args.verb == "promote" and not args.dry_run
    )
    repository = validate_repository_root()
    with workspace_access(
        repository,
        exclusive=promotion_recovery,
        promotion=promotion_recovery,
        legacy=args.noun == "update",
    ):
        # These direct reads share one graph HEAD through rendering. Following
        # metrics must observe later changes; other read services own snapshots.
        if (args.noun, args.verb) in {
            ("identity", "show"),
            ("run", "logs"),
            ("run", "metrics"),
        } and not getattr(args, "follow", False):
            return _dispatch_observed(repository, args)
        return _dispatch(args)


@graph_observation
def _dispatch_observed(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Keep snapshot-style CLI reads on one HEAD; follow retains live polling."""
    return _dispatch(args)


def _dispatch(args: argparse.Namespace) -> int:
    """Route a command without changing its admission or observation scope."""
    if args.noun == "completion":
        sys.stdout.write(completion.shellcode(args.shell))
        return 0
    if args.noun == "workspace":
        return workspace.run(args)
    repository = validate_repository_root()
    if args.noun == "migrate":
        return migration.run(repository, args)
    if args.noun in {
        "update",
        "storage",
        "index",
        "settings",
        "identity",
        "environment",
    }:
        return maintenance.run(repository, args)
    if args.noun == "web":
        web.serve(repository, args.experiment, port=args.port)
        return 0
    if args.noun == "template":
        return template.run(repository, args)
    if args.noun == "experiment":
        return experiment.run(repository, args)
    if args.noun == "variant":
        return variant.run(repository, args)
    if args.noun == "run":
        return run_commands.run(repository, args)
    if args.noun == "model":
        return model.run(repository, args)
    if args.noun == "status":
        return status.run(repository, args)
    raise AssertionError("unreachable command")
