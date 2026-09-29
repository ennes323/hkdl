"""Coordinate Experiment authoring commands and present their results.

Dispatch supplies ordinary workspace admission; creation bypasses it so
Authoring can own bootstrap and recovery admission. Authoring and V2Authoring
own draft validation, identity commits, rename guards, and rename recovery;
ExperimentDeletionService owns deletion planning, locks, journals, and recovery.
This module sequences those services with user confirmation and CLI output.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from hkdl.authoring.authoring import Authoring
from hkdl.authoring.authoring_records import ExperimentRecord
from hkdl.interfaces.cli import io
from hkdl.storage.graph.authoring import V2Authoring
from hkdl.storage.graph.deletion import (
    DeletionConflict,
    ExperimentDeletionPlan,
    ExperimentDeletionService,
)
from hkdl.storage.graph.graph import V2Graph
from hkdl.storage.storage import NotFoundError, RepositoryPaths


# Flow: select the authoring operation; destructive deletion recovers, plans,
# confirms, and executes before the result is rendered.
def run(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Execute one parsed Experiment command under its owning access scope.

    Dispatch admits ordinary commands; creation lets Authoring acquire bootstrap
    and recovery admission. Service conflicts are rendered here when their
    current plan is useful; other failures propagate to main.
    """
    assert args.noun == "experiment"
    authoring = Authoring(repository)

    if args.verb == "create":
        record = authoring.create_experiment(args.experiment, args.template)
        print(f"created experiment {record.document['name']}")
        return 0

    if args.verb == "commit":
        identity = V2Authoring(authoring).commit_experiment(args.experiment)
        payload = {
            "experiment": args.experiment,
            "changed": identity.changed,
        }
        if args.output == "json":
            io.print_json(payload)
        else:
            state = "committed" if identity.changed else "unchanged"
            print(f"{state} experiment {args.experiment}")
        return 0

    if args.verb == "rename":
        plan = V2Authoring(authoring).rename_experiment(
            args.old_name,
            args.new_name,
            dry_run=args.dry_run,
        )
        payload = {
            "old_name": plan.old_name,
            "new_name": plan.new_name,
            "changed": plan.changed,
            "dry_run": args.dry_run,
        }
        if args.output == "json":
            io.print_json(payload)
        else:
            prefix = "would rename" if args.dry_run else "renamed"
            print(f"{prefix} experiment {args.old_name} -> {args.new_name}")
        return 0

    if args.verb == "delete":
        service = ExperimentDeletionService(authoring.repository)
        recovered: tuple[dict[str, Any], ...] = ()
        # Recovery may finish an already committed deletion or restore a
        # pre-commit move. Dry runs remain read-only and skip this mutation.
        if not args.dry_run:
            recovered = service.recover()
            for item in recovered:
                print(
                    f"deletion recovery {item['action']} "
                    f"transaction={item['transaction']}",
                    file=sys.stderr,
                )
        try:
            plan = service.plan(args.experiment)
        except NotFoundError:
            matching = [
                item
                for item in recovered
                if item.get("experiment") == args.experiment
                and item.get("action") == "completed"
            ]
            if not matching:
                raise
            payload = {
                "experiment": args.experiment,
                "recovered": True,
                "transaction": matching[-1]["transaction"],
            }
            if args.output == "json":
                io.print_json({"deletion": payload, "dry_run": False})
            else:
                print(
                    f"recovered deleted experiment {args.experiment} "
                    f"transaction={matching[-1]['transaction']}"
                )
            return 0
        document = io.public_payload(plan.as_dict())
        if args.dry_run:
            if args.output == "json":
                io.print_json({"deletion": document, "dry_run": True})
            else:
                _experiment_deletion_report(document)
            return 0 if plan.ready else 5
        if not plan.ready:
            if args.output == "json":
                print(
                    json.dumps(
                        {"deletion": document, "dry_run": False},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                )
            else:
                _experiment_deletion_report(document, file=sys.stderr)
                print("Nothing was changed.", file=sys.stderr)
            return 5
        _experiment_deletion_report(document, file=sys.stderr)
        if not io.confirm(
            f"Delete Experiment {args.experiment} and all listed contents?"
        ):
            if args.output == "json":
                io.print_json(
                    {
                        "deletion": {
                            "experiment": args.experiment,
                            "cancelled": True,
                        },
                        "dry_run": False,
                    }
                )
            else:
                print("Experiment deletion cancelled. Nothing was changed.")
            return 0
        try:
            # execute rechecks the confirmed plan under entity and Run locks;
            # the service retains uncertain journal state for later recovery.
            result = service.execute(plan)
        except DeletionConflict as error:
            print(f"error: {error}", file=sys.stderr)
            if isinstance(error.plan, ExperimentDeletionPlan):
                changed = io.public_payload(error.plan.as_dict())
                if args.output == "json":
                    print(
                        json.dumps(
                            {"deletion": changed, "dry_run": False},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        file=sys.stderr,
                    )
                else:
                    _experiment_deletion_report(changed, file=sys.stderr)
            print("Nothing was changed.", file=sys.stderr)
            return 5
        result_document = io.public_payload(result.as_dict())
        if args.output == "json":
            io.print_json(
                {
                    "deletion": {
                        "experiment": args.experiment,
                        **result_document,
                    },
                    "dry_run": False,
                }
            )
        else:
            print(
                f"deleted experiment {args.experiment} "
                f"transaction={result_document['transaction']}"
            )
        return 0

    if args.verb == "list":
        experiments = authoring.list_experiments()
        if args.output == "json":
            io.print_json(
                {
                    "experiments": [
                        _experiment_payload(item, None) for item in experiments
                    ]
                }
            )
        else:
            io.table(
                ("NAME", "TYPE", "TEMPLATE"),
                (
                    (
                        item.document["name"],
                        item.document["type"],
                        item.document["template"]["name"],
                    )
                    for item in experiments
                ),
            )
        return 0

    raise AssertionError("unreachable command")


def _experiment_payload(
    item: ExperimentRecord,
    graph: V2Graph | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": item.document["name"],
        "type": item.document["type"],
        "template": {"name": item.document["template"]["name"]},
    }
    if graph is not None:
        experiment_hash = graph.experiment_hash(str(item.document["name"]))
        payload.update(
            experiment_hash=experiment_hash,
            experiment_revision_hash=graph.current_revision(experiment_hash),
        )
    return payload


def _experiment_deletion_report(document: dict[str, Any], *, file: Any = None) -> None:
    if file is None:
        file = sys.stdout
    print(f"Experiment deletion plan: {document['experiment']}", file=file)
    print(f"  Variants: {len(document['variants'])}", file=file)
    print(f"  Runs: {len(document['runs'])}", file=file)
    print(f"  Models: {len(document['models'])}", file=file)
    print(f"  Authored tree: {document['authored_path']}", file=file)
    for path in document["output_paths"]:
        print(f"  Output tree: {path}", file=file)
    print(f"  Ready: {str(document['ready']).lower()}", file=file)
    if document["variants"]:
        print("VARIANTS", file=file)
        for variant in document["variants"]:
            print(f"- {variant}", file=file)
    blockers = document["blockers"]
    if blockers:
        print("BLOCKERS", file=file)
    for blocker in blockers:
        print(f"- {blocker['address']}", file=file)
        print(f"  code: {blocker['code']}", file=file)
        if blocker["status"] is not None:
            print(f"  status: {blocker['status']}", file=file)
        if blocker["lease"] is not None:
            print(f"  lease: {blocker['lease']}", file=file)
        print(f"  reason: {blocker['reason']}", file=file)
        print(f"  next: {blocker['next_action']}", file=file)
