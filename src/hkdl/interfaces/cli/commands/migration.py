"""Coordinate migration previews, approval checks, and result presentation.

Workspace and authoring cutover services own source validation, exclusive
workspace admission, and atomic cutover or journal recovery. This module
preserves the CLI sequence from recovery and planning through approval to apply.
"""

from __future__ import annotations

import argparse
from typing import Any

from hkdl.authoring.migration import Migration
from hkdl.errors import ContractError
from hkdl.interfaces.cli import io
from hkdl.storage.graph.authoring_migration import (
    AuthoringMigration,
    AuthoringMigrationConflict,
)
from hkdl.storage.graph.migration import MigrationConflict, WorkspaceMigration
from hkdl.storage.storage import RepositoryPaths


# Flow: select migration -> recover and plan -> preview or approve -> apply.
def run(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Run the selected path, authoring, or workspace migration workflow.

    Path migration delegates directly to the authoring service. Cutovers use a
    service-produced plan; this layer renders previews, checks the approved
    authoring digest, and obtains interactive workspace confirmation. The
    cutover services revalidate authority and leases under exclusive admission.
    """

    if args.noun == "migrate":
        if args.path is not None:
            result = Migration(repository).migrate(args.path)
            print(f"already current {result.path} schema={result.schema_version}")
            return 0
        if getattr(args, "authoring", False):
            migration = AuthoringMigration(
                repository, tracker_default=getattr(args, "tracker_default", None)
            )
            expected_plan = getattr(args, "expect_plan", None)
            # Resume an approved journal before deriving the current plan. A
            # completed recovery cannot silently approve additional changes.
            recovered = (
                migration.recover(expected_plan=expected_plan)
                if getattr(args, "yes", False)
                else None
            )
            plan = migration.plan()
            if (
                recovered
                and recovered["outcome"] == "completed"
                and not plan.report.already_current
            ):
                raise AuthoringMigrationConflict(
                    "approved cutover recovered; additional changes require a new dry-run"
                )
            if (
                expected_plan is not None
                and not (recovered and recovered["outcome"] == "completed")
                and expected_plan != plan.plan_digest
            ):
                raise AuthoringMigrationConflict(
                    "migration plan differs from the approved digest; run dry-run again"
                )
            report = plan.report.as_dict()
            dry_run = getattr(args, "dry_run", False)
            output = getattr(args, "output", "text")
            if dry_run:
                if output == "json":
                    io.print_json(
                        {
                            "authoring_migration": report,
                            "dry_run": True,
                            "applied": False,
                        }
                    )
                else:
                    _authoring_migration_report(report, dry_run=True)
                return 0 if plan.report.cutover_ready else 5
            migration.apply(plan)
            if output == "json":
                io.print_json(
                    {
                        "authoring_migration": report,
                        "dry_run": False,
                        "applied": True,
                        "recovery": recovered,
                    }
                )
            else:
                _authoring_migration_report(report, dry_run=False)
                print(f"activated authoring schema 2 plan={plan.report.plan_digest}")
            return 0
        migration = WorkspaceMigration(repository)
        # Plan without writes, present blockers, then let apply revalidate the
        # same authority and Run leases under exclusive workspace admission.
        plan = migration.plan()
        output = getattr(args, "output", "text")
        dry_run = getattr(args, "dry_run", False)
        if dry_run:
            if output == "json":
                io.print_json(
                    {
                        "migration": plan.report.as_dict(),
                        "dry_run": True,
                        "applied": False,
                    }
                )
            else:
                _migration_report(plan.report.as_dict(), dry_run=True)
            return 0
        if output != "json":
            _migration_report(plan.report.as_dict(), dry_run=False)
        if plan.report.active_leases:
            raise MigrationConflict("active Run lease blocks migration")
        if plan.report.malformed_records:
            raise ContractError("malformed v1 records block migration")
        if not getattr(args, "yes", False) and not io.confirm(
            "Cut over this workspace to HKDL v2?"
        ):
            if output == "json":
                io.print_json(
                    {
                        "migration": plan.report.as_dict(),
                        "dry_run": False,
                        "applied": False,
                    }
                )
            else:
                print("Migration cancelled. v1 remains authoritative.")
            return 0
        migration.apply(plan)
        if output == "json":
            io.print_json(
                {
                    "migration": plan.report.as_dict(),
                    "dry_run": False,
                    "applied": True,
                }
            )
        else:
            print(f"activated v2 plan={plan.report.plan_digest}")
        return 0

    raise AssertionError("unreachable command")


def _migration_report(report: dict[str, object], *, dry_run: bool) -> None:
    prefix = "dry-run" if dry_run else "migration plan"
    print(
        f"{prefix} variants={report['variants']} runs={report['runs']} "
        f"models={report['models']} artifacts={report['artifacts']}"
    )
    print(
        f"bytes input={report['input_bytes']} output={report['output_bytes']} "
        f"additional={report['estimated_additional_bytes']}"
    )
    print(f"plan_digest={report['plan_digest']}")
    print(f"cutover_ready={str(report['cutover_ready']).lower()}")
    for address in report["active_leases"]:
        print(f"active_lease {address}")
    for address in report["stale_nonterminal"]:
        print(f"stale_nonterminal {address}")
    for item in report["malformed_records"]:
        print(f"malformed {item['path']}: {item['error']}")
    for item in report["mlflow_identities"]:
        print(f"mlflow {item['attempt']} {item['tracker_run_id']}")
    for digest in report["object_hashes"]:
        print(f"object {digest}")


def _authoring_migration_report(report: dict[str, Any], *, dry_run: bool) -> None:
    prefix = "dry-run" if dry_run else "authoring migration"
    print(
        f"{prefix} experiments={report['experiments']} "
        f"variants={report['variants']} runs={report['runs']} "
        f"models={report['models']} results={report['results']}"
    )
    print(
        f"bytes input={report['input_bytes']} output={report['output_bytes']} "
        f"additional={report['estimated_additional_bytes']}"
    )
    print(f"tracker {report['tracker_before']} -> {report['tracker_after']}")
    if report.get("tracker_default") is not None:
        print(
            f"tracker_default={report['tracker_default']} (explicit; history preserved)"
        )
    print(f"plan_digest={report['plan_digest']}")
    print(f"cutover_ready={str(report['cutover_ready']).lower()}")
    for item in report["transformations"]:
        source = item.get("source", item.get("kind", "record"))
        print(f"transform {source} {item.get('status', 'planned')}")
    for address in report["active_leases"]:
        print(f"active_lease {address}")
    for address in report["stale_nonterminal"]:
        print(f"stale_nonterminal {address}")
    for item in report["malformed_records"]:
        print(f"malformed {item['path']}: {item['error']}")
    for digest in report["object_hashes"]:
        print(f"object {digest}")
