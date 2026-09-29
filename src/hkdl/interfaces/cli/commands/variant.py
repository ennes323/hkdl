"""Coordinate Variant authoring and lifecycle commands for the CLI.

Dispatch supplies ordinary workspace admission and exclusive admission for an
applied promotion. Authoring and V2Authoring own draft publication, commits,
guards, and rename recovery. Deletion and promotion services own plan
revalidation, entity and Run locks, journal recovery, and graph mutations; this
module sequences confirmation and presents their public results.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from hkdl.authoring.authoring import Authoring
from hkdl.authoring.authoring_records import VariantRecord
from hkdl.interfaces.cli import io
from hkdl.storage.graph.authoring import V2Authoring
from hkdl.storage.graph.deletion import (
    DeletionConflict,
    VariantDeletionPlan,
    VariantDeletionService,
)
from hkdl.storage.graph.graph import V2Graph
from hkdl.storage.graph.promotion import (
    PromotionConflict,
    VariantPromotionPlan,
    VariantPromotionService,
)
from hkdl.storage.storage import NotFoundError, RepositoryPaths


# Flow: route ordinary authoring operations; deletion and promotion recover,
# plan, confirm, execute, and then render service-owned results.
def run(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Execute one parsed Variant command within dispatch's access scope.

    Current deletion or promotion plans are rendered for handled conflicts;
    other failures propagate to the process boundary in main.
    """
    assert args.noun == "variant"
    authoring = Authoring(repository)

    if args.verb == "delete":
        service = VariantDeletionService(authoring.repository)
        recovered: tuple[dict[str, Any], ...] = ()
        # Recovery completes committed work or restores pre-commit moves before
        # a fresh plan is built. Dry runs leave pending journals untouched.
        if not args.dry_run:
            recovered = service.recover()
            for item in recovered:
                print(
                    f"deletion recovery {item['action']} "
                    f"transaction={item['transaction']}",
                    file=sys.stderr,
                )
        try:
            plan = service.plan(args.experiment, args.variant)
        except NotFoundError:
            matching = [
                item
                for item in recovered
                if item.get("experiment") == args.experiment
                and item.get("variant") == args.variant
                and item.get("action") == "completed"
            ]
            if not matching:
                raise
            payload = {
                "experiment": args.experiment,
                "variant": args.variant,
                "recovered": True,
                "transaction": matching[-1]["transaction"],
            }
            if args.output == "json":
                io.print_json({"deletion": payload, "dry_run": False})
            else:
                print(
                    f"recovered deleted variant {args.experiment}/{args.variant} "
                    f"transaction={matching[-1]['transaction']}"
                )
            return 0
        document = io.public_payload(plan.as_dict())
        if args.dry_run:
            if args.output == "json":
                io.print_json({"deletion": document, "dry_run": True})
            else:
                _variant_deletion_report(document)
            return 0 if plan.ready and not plan.requires_lineage_confirmation else 5
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
                _variant_deletion_report(document, file=sys.stderr)
                print("Nothing was changed.", file=sys.stderr)
            return 5
        _variant_deletion_report(document, file=sys.stderr)
        lineage_confirmed = False
        if plan.requires_lineage_confirmation:
            if not io.confirm(
                "Continue with automatic active-lineage reconnection while "
                "preserving immutable derivation history?"
            ):
                print(
                    "error: Variant deletion is blocked by active child Variants; "
                    "delete or reorganize them first. Nothing was changed.",
                    file=sys.stderr,
                )
                return 5
            lineage_confirmed = True
        if not io.confirm(
            f"Delete Variant {args.experiment}/{args.variant} and all listed "
            "owned contents?"
        ):
            if args.output == "json":
                io.print_json(
                    {
                        "deletion": {
                            "experiment": args.experiment,
                            "variant": args.variant,
                            "cancelled": True,
                        },
                        "dry_run": False,
                    }
                )
            else:
                print("Variant deletion cancelled. Nothing was changed.")
            return 0
        try:
            # The service revalidates the confirmed plan under its entity and
            # Run locks. Active child lineage may continue through preserved
            # derivation history only after the separate confirmation above.
            result = service.execute(
                plan,
                allow_lineage_reconnection=lineage_confirmed,
            )
        except DeletionConflict as error:
            print(f"error: {error}", file=sys.stderr)
            if isinstance(error.plan, VariantDeletionPlan):
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
                    _variant_deletion_report(changed, file=sys.stderr)
            print("Nothing was changed.", file=sys.stderr)
            return 5
        result_document = io.public_payload(result.as_dict())
        reconnections = [item.as_dict() for item in plan.direct_children]
        if args.output == "json":
            io.print_json(
                {
                    "deletion": {
                        "experiment": args.experiment,
                        "variant": args.variant,
                        "lineage_reconnections": reconnections,
                        **result_document,
                    },
                    "dry_run": False,
                }
            )
        else:
            print(
                f"deleted variant {args.experiment}/{args.variant} "
                f"transaction={result_document['transaction']}"
            )
            for item in plan.direct_children:
                after = item.effective_parent_after or "<active root>"
                print(
                    f"reconnected active lineage {item.address}: "
                    f"{item.current_parent} -> {after}"
                )
        return 0

    if args.verb == "create":
        record = authoring.create_variant(
            args.experiment,
            args.variant,
            template_version=args.template_version,
        )
        print(f"created variant {record.experiment}/{record.document['name']}")
        return 0

    if args.verb == "clone":
        record = authoring.clone_variant(
            args.experiment,
            args.variant,
            source_variant=args.source,
            source_experiment=args.source_experiment,
        )
        print(f"created variant {record.experiment}/{record.document['name']}")
        return 0

    if args.verb == "promote":
        service = VariantPromotionService(authoring.repository)
        recovered: tuple[dict[str, Any], ...] = ()
        # Applied promotion enters exclusive admission in dispatch. Recovery
        # resolves the stable journal before a replacement plan is considered.
        if not args.dry_run:
            recovered = service.recover(
                args.experiment,
                args.source,
                args.target,
            )
            for item in recovered:
                print(
                    f"Variant promotion recovery {item['action']} "
                    f"transaction={item['transaction']}",
                    file=sys.stderr,
                )
            completed = [
                item
                for item in recovered
                if item["action"] == "completed"
                and item["experiment"] == args.experiment
                and item["source"] == args.source
                and item["target"] == args.target
            ]
            if completed:
                payload = {
                    "experiment": args.experiment,
                    "source": args.source,
                    "target": args.target,
                    "changed": True,
                    "recovered": True,
                    "transaction": completed[-1]["transaction"],
                    "target_options": "preserved",
                    "source_variant": "deleted",
                }
                if args.output == "json":
                    io.print_json({"promotion": payload, "dry_run": False})
                else:
                    print(
                        f"recovered promoted variant Code "
                        f"{args.experiment}/{args.source} -> "
                        f"{args.experiment}/{args.target} "
                        f"transaction={completed[-1]['transaction']}"
                    )
                return 0
        plan = service.plan(args.experiment, args.source, args.target)
        document = io.public_payload(plan.as_dict())
        if args.dry_run:
            if args.output == "json":
                io.print_json({"promotion": document, "dry_run": True})
            else:
                _variant_promotion_report(document)
            return 0 if plan.ready else 5
        if not plan.ready:
            if args.output == "json":
                print(
                    json.dumps(
                        {"promotion": document, "dry_run": False},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                )
            else:
                _variant_promotion_report(document, file=sys.stderr)
                print("Nothing was changed.", file=sys.stderr)
            return 5
        _variant_promotion_report(document, file=sys.stderr)
        if not io.confirm(
            f"Promote committed Code from {args.experiment}/{args.source} into "
            f"{args.experiment}/{args.target}? Target Options remain unchanged "
            "and the source Variant plus its active owned closure will be deleted."
        ):
            if args.output == "json":
                io.print_json(
                    {
                        "promotion": {
                            "experiment": args.experiment,
                            "source": args.source,
                            "target": args.target,
                            "changed": False,
                            "cancelled": True,
                        },
                        "dry_run": False,
                    }
                )
            else:
                print("Variant promotion cancelled. Nothing was changed.")
            return 0
        try:
            # execute guards both Variants and rechecks the plan; its journal
            # defines whether failure can roll back or must be recovered.
            result = service.execute(plan)
        except PromotionConflict as error:
            print(f"error: {error}", file=sys.stderr)
            if isinstance(error.plan, VariantPromotionPlan):
                changed = io.public_payload(error.plan.as_dict())
                if args.output == "json":
                    print(
                        json.dumps(
                            {"promotion": changed, "dry_run": False},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        file=sys.stderr,
                    )
                else:
                    _variant_promotion_report(changed, file=sys.stderr)
            print("Nothing was changed.", file=sys.stderr)
            return 5
        internal_result = result.as_dict()
        internal_result.pop("journal", None)
        result_document = io.public_payload(internal_result)
        if args.output == "json":
            io.print_json({"promotion": result_document, "dry_run": False})
        else:
            print(
                f"promoted variant Code {args.experiment}/{args.source} -> "
                f"{args.experiment}/{args.target} "
                f"and deleted source Variant transaction="
                f"{result_document['transaction']}"
            )
        return 0

    if args.verb == "list":
        variants = authoring.list_variants(args.experiment)
        if args.output == "json":
            io.print_json(
                {
                    "experiment": args.experiment,
                    "variants": [
                        _variant_payload(item, None, None) for item in variants
                    ],
                }
            )
        else:
            io.table(
                ("NAME",),
                ((item.document["name"],) for item in variants),
            )
        return 0

    if args.verb == "check":
        record = authoring.check_variant(args.experiment, args.variant)
        if args.output == "json":
            payload = {
                "experiment": record.experiment,
                "variant": record.document["name"],
                "valid": True,
            }
            io.print_json(payload)
        else:
            print(f"valid {record.experiment}/{record.document['name']}")
        return 0

    if args.verb == "commit":
        identity = V2Authoring(authoring).commit_variant(
            args.experiment,
            args.variant,
        )
        payload = {
            "experiment": args.experiment,
            "variant": args.variant,
            "changed": identity.changed,
        }
        if args.output == "json":
            io.print_json(payload)
        else:
            state = "committed" if identity.changed else "unchanged"
            print(f"{state} variant {args.experiment}/{args.variant}")
        return 0

    if args.verb == "rename":
        plan = V2Authoring(authoring).rename_variant(
            args.experiment,
            args.old_name,
            args.new_name,
            dry_run=args.dry_run,
        )
        if args.output == "json":
            io.print_json(
                {
                    "rename": {
                        "experiment": plan.experiment,
                        "old_name": plan.old_name,
                        "new_name": plan.new_name,
                        "changed": plan.changed,
                    },
                    "dry_run": args.dry_run,
                }
            )
        else:
            prefix = "would rename" if args.dry_run else "renamed"
            print(
                f"{prefix} variant {args.experiment}/{args.old_name} to {args.new_name}"
            )
        return 0

    raise AssertionError("unreachable command")


def _variant_payload(
    item: VariantRecord,
    graph: V2Graph | None,
    experiment_hash: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {"name": item.document["name"]}
    if graph is not None and experiment_hash is not None:
        variant_hash = graph.variant_hash(
            experiment_hash,
            str(item.document["name"]),
        )
        payload.update(
            variant_hash=variant_hash,
            variant_revision_hash=graph.current_revision(variant_hash),
        )
    return payload


def _variant_deletion_report(document: dict[str, Any], *, file: Any = None) -> None:
    if file is None:
        file = sys.stdout
    address = f"{document['experiment']}/{document['variant']}"
    print(f"Variant deletion plan: {address}", file=file)
    print(f"  Runs: {len(document['runs'])}", file=file)
    print(f"  Models: {len(document['models'])}", file=file)
    print(f"  Authored tree: {document['authored_path']}", file=file)
    for path in document["output_paths"]:
        print(f"  Output tree: {path}", file=file)
    print(f"  Ready: {str(document['ready']).lower()}", file=file)
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
    derived = document["derived_variants"]
    if not derived:
        return
    print("CHILD VARIANT WARNING", file=file)
    print(
        "Deleting this Variant directly changes the active lineage. HKDL can "
        "continue the lineage while preserving immutable derivation history, "
        "but deleting or reorganizing child Variants first is preferred.",
        file=file,
    )
    for child in derived:
        after = child["effective_parent_after"] or "<active root>"
        print(f"- {child['address']} relation={child['relation']}", file=file)
        if child["reconnected"]:
            print(
                f"  active lineage: {child['current_parent']} -> {after}",
                file=file,
            )
        else:
            print(f"  active parent remains: {after}", file=file)


def _variant_promotion_report(document: dict[str, Any], *, file: Any = None) -> None:
    if file is None:
        file = sys.stdout
    source = f"{document['experiment']}/{document['source']}"
    target = f"{document['experiment']}/{document['target']}"
    print(f"Variant Code promotion plan: {source} -> {target}", file=file)
    print(f"  Ready: {str(document['ready']).lower()}", file=file)
    print(f"  Changed: {str(document['changed']).lower()}", file=file)
    print(f"  Target Options: {document['target_options']}", file=file)
    print(f"  Source Variant: {document['source_variant']}", file=file)
    if document["template_changed"]:
        print("  Code change: Template provenance", file=file)
    if document["components_changed"]:
        print("  Code change: components", file=file)
    for path in document["changed_paths"]:
        print(f"  Source path: {path}", file=file)
    blockers = document["blockers"]
    if blockers:
        print("BLOCKERS", file=file)
    for blocker in blockers:
        print(f"- {blocker['code']}", file=file)
        print(f"  reason: {blocker['reason']}", file=file)
        print(f"  next: {blocker['next_action']}", file=file)
