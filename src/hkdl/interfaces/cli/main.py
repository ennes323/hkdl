"""Translate CLI input and service failures at the process boundary.

The parser owns command syntax; this module checks option combinations and
enters installation access before dispatch owns workspace admission and routing.
Command families and their services own execution and recovery procedures.
"""
# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import sys
from collections.abc import Sequence

from hkdl.authoring.config import DIGEST_PATTERN
from hkdl.errors import ContractError
from hkdl.execution.evaluation import (
    EvaluationFailure,
    EvaluationInterrupted,
    LifecycleConflict,
)
from hkdl.execution.export import ExportFailure, ExportInterrupted
from hkdl.execution.recovery import RecoveryFailure, RecoveryInterrupted
from hkdl.execution.training import TrainingFailure, TrainingInterrupted
from hkdl.installation import installation_access, installed_paths
from hkdl.installer.common import InstallError
from hkdl.interfaces import completion, web
from hkdl.runtime.environments import EnvironmentFailure
from hkdl.storage.graph.authoring_migration import (
    AuthoringMigrationConflict,
)
from hkdl.storage.graph.deletion import (
    DeletionConflict,
    DeletionFailure,
)
from hkdl.storage.graph.graph import (
    DirtyDraftError,
)
from hkdl.storage.graph.maintenance import WorkspaceBusy
from hkdl.storage.graph.migration import MigrationConflict
from hkdl.storage.graph.promotion import (
    PromotionConflict,
    PromotionFailure,
)
from hkdl.storage.status_index import IndexFailure
from hkdl.storage.storage import (
    AlreadyExistsError,
    NotFoundError,
    OwnershipError,
)
from hkdl.update import UpdateConflict, UpdateFailure

from . import dispatch
from .commands import maintenance
from .parser import build_parser


# Flow: parse and validate -> managed update or admitted dispatch -> exit mapping.
def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI invocation and translate handled failures into exit codes.

    Syntax and option-combination errors retain argparse's SystemExit behavior.
    Service failures reach this boundary after acquired access contexts unwind;
    their owners remain responsible for recovery and artifact cleanup.
    """
    parser = build_parser()
    completion.activate(parser)
    args = parser.parse_args(argv)
    if getattr(args, "follow", False) and args.output == "json":
        parser.error("--follow requires --output text")
    if getattr(args, "table", False) and args.output == "json":
        parser.error("--table requires --output text")
    if getattr(args, "noun", None) == "migrate":
        migrate_all = getattr(args, "all", False)
        migrate_authoring = getattr(args, "authoring", False)
        migrate_dry_run = getattr(args, "dry_run", False)
        migrate_yes = getattr(args, "yes", False)
        migrate_output = getattr(args, "output", "text")
        if sum(map(bool, (migrate_all, migrate_authoring, args.path))) != 1:
            parser.error("migrate requires exactly one of PATH, --all, or --authoring")
        if args.path is not None and (migrate_dry_run or migrate_yes):
            parser.error("--dry-run and --yes require --all or --authoring")
        if args.path is not None and migrate_output != "text":
            parser.error("--output requires --all or --authoring")
        if migrate_authoring and bool(migrate_dry_run) == bool(migrate_yes):
            parser.error("--authoring requires exactly one of --dry-run or --yes")
        if getattr(args, "tracker_default", None) is not None and not migrate_authoring:
            parser.error("--tracker-default requires --authoring")
        expected_plan = getattr(args, "expect_plan", None)
        if expected_plan is not None and (
            not migrate_authoring
            or not migrate_yes
            or not DIGEST_PATTERN.fullmatch(expected_plan)
        ):
            parser.error(
                "--expect-plan requires --authoring --yes and a full sha256 digest"
            )
    if (getattr(args, "noun", None), getattr(args, "verb", None)) == (
        "run",
        "delete",
    ) and bool(args.dry_run) == bool(args.yes):
        parser.error("run delete requires exactly one of --dry-run or --yes")
    try:
        # The installer owns update locking; an outer shared installation lease
        # would obstruct its exclusive activation boundary.
        if args.noun == "update" and installed_paths() is not None:
            return maintenance.managed_update(args)
        # Keep installation access around the complete workspace command.
        # installation_access owns whether this lease is acquired or borrowed.
        with installation_access():
            return dispatch.run(args)
    except InstallError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code
    except WorkspaceBusy as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except EvaluationInterrupted as error:
        print(f"interrupted {error.address}", file=sys.stderr)
        return 130
    except ExportInterrupted as error:
        print(f"interrupted {error.address}", file=sys.stderr)
        return 130
    except TrainingInterrupted as error:
        print(f"interrupted {error.address}", file=sys.stderr)
        return 130
    except RecoveryInterrupted as error:
        print(f"interrupted {error.address}", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except TrainingFailure as error:
        address = f" for {error.address}" if error.address else ""
        print(f"error: training failed{address}: {error}", file=sys.stderr)
        return 6
    except EvaluationFailure as error:
        address = f" for {error.address}" if error.address else ""
        print(
            f"error: {error.action} failed{address}: {error}",
            file=sys.stderr,
        )
        return 6
    except ExportFailure as error:
        address = f" for {error.address}" if error.address else ""
        print(f"error: export failed{address}: {error}", file=sys.stderr)
        return 6
    except RecoveryFailure as error:
        address = f" for {error.address}" if error.address else ""
        print(
            f"error: {error.action} recovery failed{address}: {error}", file=sys.stderr
        )
        return 6
    except LifecycleConflict as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except DirtyDraftError as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except DeletionConflict as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except DeletionFailure as error:
        print(f"error: {error}", file=sys.stderr)
        if error.journal is not None:
            print(f"recovery journal: {error.journal}", file=sys.stderr)
        return 6
    except PromotionConflict as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except PromotionFailure as error:
        print(f"error: {error}", file=sys.stderr)
        if error.journal is not None:
            print(f"recovery journal: {error.journal}", file=sys.stderr)
        return 6
    except MigrationConflict as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except AuthoringMigrationConflict as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except UpdateConflict as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except (AlreadyExistsError, OwnershipError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    except NotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except UpdateFailure as error:
        print(f"error: update failed: {error}", file=sys.stderr)
        return 6
    except EnvironmentFailure as error:
        print(f"error: environment operation failed: {error}", file=sys.stderr)
        return 6
    except IndexFailure as error:
        print(f"error: index operation failed: {error}", file=sys.stderr)
        return 6
    except web.WebFailure as error:
        print(f"error: web server failed: {error}", file=sys.stderr)
        return 6
