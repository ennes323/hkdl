"""HKDL command-line interface."""
# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .authoring import Authoring, ExperimentRecord, VariantRecord
from .config import ContractError
from .completion import (
    SHELLS,
    activate as activate_completion,
    complete_devices,
    complete_eval_seeds,
    complete_evaluation_cases,
    complete_experiments,
    complete_models,
    complete_runs,
    complete_source_variants,
    complete_template_references,
    complete_template_versions,
    complete_templates,
    complete_training_groups,
    complete_variants,
    file_completer,
    shellcode as completion_shellcode,
)
from .evaluation import (
    Evaluation,
    EvaluationFailure,
    EvaluationInterrupted,
    LifecycleConflict,
)
from .environments import EnvironmentFailure, EnvironmentStore, PrunePlan
from .export import Export, ExportFailure, ExportInterrupted
from .migration import Migration
from .recovery import Recovery, RecoveryFailure, RecoveryInterrupted
from .run_contracts import MAX_SEED, TERMINAL_STATUSES
from .runs import RunRecord, RunStore
from .settings import WorkspaceSettingsStore
from .status import Status, render_status_tree
from .status_index import IndexFailure, IndexReport
from .storage import (
    AlreadyExistsError,
    NotFoundError,
    OwnershipError,
    storage_usage,
    validate_repository_root,
)
from .training import Training, TrainingFailure, TrainingInterrupted
from .update import UpdateConflict, UpdateFailure, update
from .web import DEFAULT_WEB_PORT, WebFailure, serve as serve_web
from .v2.authoring import V2Authoring
from .v2.authoring_migration import (
    AuthoringMigration,
    AuthoringMigrationConflict,
)
from .v2.deletion import DeletionConflict, RunDeletionService
from .v2.graph import DirtyDraftError
from .v2.graph import V2Graph
from .v2.execution import GraphRecorder
from .v2.migration import MigrationConflict, WorkspaceMigration
from .v2.maintenance import WorkspaceBusy, workspace_access
from .config import DIGEST_PATTERN


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    activate_completion(parser)
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
        if args.noun in {"migrate", "web", "completion"} or (
            args.noun == "experiment" and args.verb == "create"
        ):
            return _dispatch(args)
        with workspace_access(validate_repository_root()):
            return _dispatch(args)
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
    except WebFailure as error:
        print(f"error: web server failed: {error}", file=sys.stderr)
        return 6


def _dispatch(args: argparse.Namespace) -> int:
    if args.noun == "completion":
        sys.stdout.write(completion_shellcode(args.shell))
        return 0

    repository = validate_repository_root()
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
                    _json(
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
                _json(
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
        plan = migration.plan()
        output = getattr(args, "output", "text")
        dry_run = getattr(args, "dry_run", False)
        if dry_run:
            if output == "json":
                _json(
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
        if not getattr(args, "yes", False) and not _confirm(
            "Cut over this workspace to HKDL v2?"
        ):
            if output == "json":
                _json(
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
            _json(
                {
                    "migration": plan.report.as_dict(),
                    "dry_run": False,
                    "applied": True,
                }
            )
        else:
            print(f"activated v2 plan={plan.report.plan_digest}")
        return 0
    if args.noun == "update":
        update(repository, assume_yes=args.yes)
        return 0
    if args.noun == "storage":
        usage = storage_usage(repository)
        if args.output == "json":
            _json({"storage": usage})
        else:
            _table(
                ("CATEGORY", "BYTES", "SIZE"),
                (
                    (category, size, _format_bytes(size))
                    for category, size in usage.items()
                ),
            )
        return 0
    if args.noun == "index":
        status = Status(repository)
        if args.verb == "status":
            _index_report(status.index_status())
        else:
            _index_report(status.rebuild_index())
        return 0
    if args.noun == "web":
        serve_web(repository, args.experiment, port=args.port)
        return 0
    if args.noun == "settings":
        settings = WorkspaceSettingsStore(repository.root)
        if args.verb == "show":
            document = settings.show()
            if args.output == "json":
                _json({"settings": document})
            else:
                backend = document["tracker"]["backend"]
                value = ",".join(backend) if isinstance(backend, list) else backend
                print(f"tracker: {value}")
            return 0
        result = settings.set_tracker(args.backend)
        if args.output == "json":
            _json({"settings": result.as_dict()})
        else:
            backend = result.backend
            value = ",".join(backend) if isinstance(backend, list) else backend
            print(f"tracker: {value}")
        return 0

    authoring = Authoring(repository)

    if (args.noun, args.verb) == ("identity", "show"):
        graph = V2Graph(repository)
        if not graph.is_active():
            raise ContractError("HKDL v2 is not active")
        addresses = list(args.address)
        expected = {"experiment": 1, "variant": 2, "run": 3, "model": 3}
        if len(addresses) != expected[args.kind]:
            raise ContractError(
                f"identity show {args.kind} requires {expected[args.kind]} address values"
            )
        if args.kind == "experiment":
            experiment_hash = graph.experiment_hash(addresses[0])
            payload = {
                "experiment": addresses[0],
                "experiment_hash": experiment_hash,
                "experiment_revision_hash": graph.current_revision(experiment_hash),
            }
        elif args.kind == "variant":
            experiment_hash = graph.experiment_hash(addresses[0])
            variant_hash = graph.variant_hash(experiment_hash, addresses[1])
            revision_hash = graph.current_revision(variant_hash)
            revision = graph.store.load(revision_hash)
            payload = {
                "experiment": addresses[0],
                "variant": addresses[1],
                "experiment_hash": experiment_hash,
                "experiment_revision_hash": graph.current_revision(experiment_hash),
                "variant_hash": variant_hash,
                "variant_revision_hash": revision_hash,
                "source_tree_hash": revision.payload["source_tree"],
            }
        elif args.kind == "run":
            record = RunStore(repository).load(*addresses)
            identity = GraphRecorder(repository).identity(record)
            payload = {
                "experiment": addresses[0],
                "variant": addresses[1],
                "run_id": addresses[2],
                **identity.as_dict(),
            }
        else:
            model_hash = GraphRecorder(repository).model_hash(*addresses)
            model = graph.store.load(model_hash)
            payload = {
                "experiment": addresses[0],
                "variant": addresses[1],
                "model_id": addresses[2],
                "model_hash": model_hash,
                **model.payload,
            }
        if args.output == "json":
            _json({"identity": payload})
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if (args.noun, args.verb) == ("environment", "prune"):
        variants = [
            variant
            for experiment in authoring.list_experiments()
            for variant in authoring.list_variants(str(experiment.document["name"]))
        ]
        runs = RunStore(repository).scan()
        active_variants = {
            (str(run.request["experiment"]), str(run.request["variant"]))
            for run in runs
            if run.state["status"] not in TERMINAL_STATUSES
        }
        environments = EnvironmentStore(repository)
        plan = environments.plan_prune(
            variants,
            active_variants=active_variants,
            remove_all=args.all,
        )
        if args.dry_run:
            _prune_table(plan, repository.root)
            return 0
        if not plan.entries:
            print(f"nothing to prune retained={plan.retained} busy={plan.busy}")
            return 0
        _prune_table(plan, repository.root, file=sys.stderr)
        if not args.yes and not _confirm("Continue with environment prune?"):
            print("Environment prune cancelled. Nothing was removed.")
            return 0
        result = environments.prune(
            variants,
            active_variants=active_variants,
            remove_all=args.all,
        )
        print(
            f"pruned environments={result.removed} "
            f"bytes={result.bytes} size={_format_bytes(result.bytes)} "
            f"retained={result.retained} busy={result.busy}"
        )
        return 0

    if (args.noun, args.verb) == ("template", "list"):
        templates = authoring.list_templates()
        if args.output == "json":
            _json(
                {
                    "templates": [
                        {
                            "name": item.manifest["name"],
                            "version": item.manifest["version"],
                            "type": item.manifest["type"],
                        }
                        for item in templates
                    ]
                }
            )
        else:
            _table(
                ("NAME", "VERSION", "TYPE"),
                (
                    (
                        item.manifest["name"],
                        item.manifest["version"],
                        item.manifest["type"],
                    )
                    for item in templates
                ),
            )
        return 0

    if (args.noun, args.verb) == ("template", "show"):
        template = authoring.show_template(args.reference)
        payload = {
            "name": template.manifest["name"],
            "version": template.manifest["version"],
            "type": template.manifest["type"],
            "digest": template.bundle_digest,
        }
        if args.output == "json":
            _json({"template": payload})
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if (args.noun, args.verb) == ("experiment", "create"):
        record = authoring.create_experiment(args.experiment, args.template)
        print(f"created experiment {record.document['name']}")
        return 0

    if (args.noun, args.verb) == ("experiment", "commit"):
        identity = V2Authoring(authoring).commit_experiment(args.experiment)
        payload = {
            "experiment": args.experiment,
            "changed": identity.changed,
        }
        if args.output == "json":
            _json(payload)
        else:
            state = "committed" if identity.changed else "unchanged"
            print(f"{state} experiment {args.experiment}")
        return 0

    if (args.noun, args.verb) == ("experiment", "rename"):
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
            _json(payload)
        else:
            prefix = "would rename" if args.dry_run else "renamed"
            print(f"{prefix} experiment {args.old_name} -> {args.new_name}")
        return 0

    if (args.noun, args.verb) == ("experiment", "list"):
        experiments = authoring.list_experiments()
        if args.output == "json":
            _json(
                {
                    "experiments": [
                        _experiment_payload(item, None) for item in experiments
                    ]
                }
            )
        else:
            _table(
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

    if (args.noun, args.verb) == ("variant", "create"):
        record = authoring.create_variant(
            args.experiment,
            args.variant,
            template_version=args.template_version,
        )
        print(f"created variant {record.experiment}/{record.document['name']}")
        return 0

    if (args.noun, args.verb) == ("variant", "clone"):
        record = authoring.clone_variant(
            args.experiment,
            args.variant,
            source_variant=args.source,
            source_experiment=args.source_experiment,
        )
        print(f"created variant {record.experiment}/{record.document['name']}")
        return 0

    if (args.noun, args.verb) == ("variant", "list"):
        variants = authoring.list_variants(args.experiment)
        if args.output == "json":
            _json(
                {
                    "experiment": args.experiment,
                    "variants": [
                        _variant_payload(item, None, None) for item in variants
                    ],
                }
            )
        else:
            _table(
                ("NAME",),
                ((item.document["name"],) for item in variants),
            )
        return 0

    if (args.noun, args.verb) == ("variant", "check"):
        record = authoring.check_variant(args.experiment, args.variant)
        if args.output == "json":
            payload = {
                "experiment": record.experiment,
                "variant": record.document["name"],
                "valid": True,
            }
            _json(payload)
        else:
            print(f"valid {record.experiment}/{record.document['name']}")
        return 0

    if (args.noun, args.verb) == ("variant", "commit"):
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
            _json(payload)
        else:
            state = "committed" if identity.changed else "unchanged"
            print(f"{state} variant {args.experiment}/{args.variant}")
        return 0

    if (args.noun, args.verb) == ("variant", "rename"):
        plan = V2Authoring(authoring).rename_variant(
            args.experiment,
            args.old_name,
            args.new_name,
            dry_run=args.dry_run,
        )
        if args.output == "json":
            _json(
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

    if (args.noun, args.verb) == ("run", "train"):
        failed = False
        training = Training(authoring.repository)
        for seed in args.seeds:
            try:
                record = training.train(
                    args.experiment,
                    args.variant,
                    args.training_group,
                    seed=seed,
                    device=args.device,
                    tracker=args.tracker,
                )
            except TrainingFailure as error:
                address = f" for {error.address}" if error.address else ""
                print(
                    f"error: training failed{address}: {error}",
                    file=sys.stderr,
                )
                failed = True
                continue
            print(
                f"trained {record.address} "
                f"model={record.state['result']['model_id']} "
                f"group={record.request['target']['training_group']} "
                f"seed={record.request['target']['seed']}"
            )
        return 6 if failed else 0

    if (args.noun, args.verb) == ("run", "eval"):
        records = Evaluation(authoring.repository).evaluate(
            args.experiment,
            args.variant,
            args.training_group,
            args.evaluation_case,
            seed=args.seed,
            device=args.device,
        )
        if not records:
            print(
                f"no evaluations needed "
                f"{args.experiment}/{args.variant}/{args.training_group}/"
                f"{args.evaluation_case}"
            )
        for record in records:
            print(
                f"evaluated {record.address} "
                f"model={record.request['target']['model_id']} "
                f"case={record.request['target']['evaluation_case']}"
            )
        return 0

    if (args.noun, args.verb) == ("run", "export"):
        record = Export(authoring.repository).export(
            args.experiment,
            args.variant,
            args.model_id,
            device=args.device,
        )
        print(f"exported {record.address} model={record.request['target']['model_id']}")
        return 0

    if (args.noun, args.verb) == ("run", "retry"):
        record = Recovery(authoring.repository).retry(
            args.experiment,
            args.variant,
            args.run_id,
            tracker=args.tracker,
        )
        action = record.request["action"]
        if action == "train":
            print(
                f"trained {record.address} "
                f"model={record.state['result']['model_id']} "
                f"group={record.request['target']['training_group']} "
                f"seed={record.request['target']['seed']}"
            )
        elif action == "eval":
            print(
                f"evaluated {record.address}"
                f" model={record.request['target']['model_id']}"
                f" case={record.request['target']['evaluation_case']}"
            )
        else:
            print(
                f"exported {record.address} "
                f"model={record.request['target']['model_id']}"
            )
        return 0

    if (args.noun, args.verb) == ("run", "delete"):
        service = RunDeletionService(authoring.repository)
        result = service.delete(
            args.experiment,
            args.variant,
            args.run_id,
            cascade=args.cascade,
            force_stale=args.force_stale,
            yes=args.yes,
            dry_run=args.dry_run,
        )
        document = _public_payload(result.as_dict())
        if args.output == "json":
            _json({"deletion": document, "dry_run": args.dry_run})
        elif args.dry_run:
            readiness = "ready" if document["ready"] else "blocked"
            print(
                f"deletion {readiness} {args.experiment}/{args.variant}/{args.run_id} "
                f"runs={len(document['runs'])} models={len(document['models'])}"
            )
            for conflict in document["conflicts"]:
                print(f"conflict: {conflict}")
        else:
            print(
                f"deleted run {args.experiment}/{args.variant}/{args.run_id} "
                f"transaction={document['transaction']}"
            )
        return 5 if args.dry_run and not document["ready"] else 0

    if (args.noun, args.verb) == ("run", "metrics"):
        store = RunStore(authoring.repository)
        record = store.load(args.experiment, args.variant, args.run_id)
        if args.follow:
            _follow_training_metrics(store, record)
            return 0
        recorder = GraphRecorder(authoring.repository)
        metrics = (
            recorder.load_training_metrics(record)
            if recorder.active() and record.state["status"] in TERMINAL_STATUSES
            else store.load_training_metrics(record)
        )
        if args.output == "json":
            _json(
                {
                    "experiment": args.experiment,
                    "variant": args.variant,
                    "run_id": args.run_id,
                    "status": record.state["status"],
                    "partial": metrics["partial"],
                    "metrics": metrics["events"],
                }
            )
        else:
            _table(
                ("STEP", "METRIC", "VALUE"),
                (
                    (event["step"], event["name"], event["value"])
                    for event in metrics["events"]
                ),
            )
        return 0

    if (args.noun, args.verb) == ("run", "logs"):
        store = RunStore(authoring.repository)
        record = store.load(args.experiment, args.variant, args.run_id)
        recorder = GraphRecorder(authoring.repository)
        log_path = (
            recorder.artifact_path(record, "worker.log")
            if recorder.active() and record.state["status"] in TERMINAL_STATUSES
            else store.resolve_worker_log(record)
        )
        with log_path.open("rb") as source:
            while chunk := source.read(64 * 1024):
                sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        return 0

    if (args.noun, args.verb) == ("model", "list"):
        models = RunStore(authoring.repository).scan_model_manifests(
            experiment=args.experiment,
            variant=args.variant,
        )
        payload = [
            {
                "model_id": item.document["model_id"],
                "training_group": item.document["training_group"],
                "seed": item.document["seed"],
                "producer_run": item.document["producer_run"],
                "created_at": item.document["created_at"],
            }
            for item in models
        ]
        if args.output == "json":
            _json(
                {
                    "experiment": args.experiment,
                    "variant": args.variant,
                    "models": payload,
                }
            )
        else:
            _table(
                ("MODEL", "GROUP", "SEED", "RUN", "CREATED"),
                (
                    (
                        item["model_id"],
                        item["training_group"],
                        item["seed"],
                        item["producer_run"],
                        item["created_at"],
                    )
                    for item in payload
                ),
            )
        return 0

    if (args.noun, args.verb) == ("model", "show"):
        model = RunStore(authoring.repository).load_model_manifest(
            args.experiment,
            args.variant,
            args.model_id,
        )
        if args.output == "json":
            _json({"model": dict(model.document)})
        else:
            for key in (
                "model_id",
                "training_group",
                "seed",
                "device",
                "producer_run",
                "created_at",
            ):
                print(f"{key}: {model.document[key]}")
        return 0

    if args.noun == "status":
        document = Status(authoring.repository).query(
            experiment=args.experiment,
            variant=args.variant,
            run_id=args.run_id,
        )
        if args.output == "json":
            _json(document)
        elif getattr(args, "table", False):
            _status_table(document)
        else:
            sys.stdout.write(render_status_tree(document, full=args.full))
        return 0

    raise AssertionError("unreachable command")


def _follow_training_metrics(
    store: RunStore,
    record: RunRecord,
    *,
    wait: Callable[[float], None] = time.sleep,
) -> None:
    experiment = record.request["experiment"]
    variant = record.request["variant"]
    run_id = record.request["run_id"]
    offset = 0
    events: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    header_written = False
    while True:
        current = store.load(experiment, variant, run_id)
        chunk = store.load_training_metric_chunk(current, offset=offset)
        if not header_written:
            print("STEP  METRIC  VALUE", flush=True)
            header_written = True
        for event in chunk["events"]:
            key = (event["name"], event["step"])
            if key in seen:
                raise ContractError("Train metric history contains a duplicate step")
            seen.add(key)
            events.append(event)
            print(
                f"{event['step']}  {event['name']}  {event['value']}",
                flush=True,
            )
        offset = chunk["offset"]
        if current.state["status"] in TERMINAL_STATUSES:
            final = store.load_training_metrics(current)
            if final["events"] != events:
                raise ContractError("Train metric history changed while following")
            partial = str(final["partial"]).lower()
            print(
                f"run {current.address} status={current.state['status']} "
                f"partial={partial}",
                file=sys.stderr,
                flush=True,
            )
            return
        wait(1.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hkdl",
        description="Author and run reproducible ML experiments.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('hkdl')}",
    )
    nouns = parser.add_subparsers(title="commands", dest="noun", required=True)

    template = nouns.add_parser(
        "template",
        help="Inspect available Templates",
        description="Inspect Template bundles available in this checkout.",
    )
    template_verbs = template.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    template_list = template_verbs.add_parser(
        "list",
        help="List available Template versions",
        description="List available Template versions.",
        epilog=_examples("hkdl template list", "hkdl template list -o json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _output_argument(template_list)
    template_show = template_verbs.add_parser(
        "show",
        help="Show one Template version",
        description="Show one Template version and its bundle digest.",
        epilog=_examples("hkdl template show resnet18@1.0.0"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    template_reference = template_show.add_argument(
        "reference",
        metavar="TEMPLATE@VERSION",
        help="Template reference",
    )
    template_reference.completer = complete_template_references
    _output_argument(template_show)

    experiment = nouns.add_parser(
        "experiment",
        help="Create and inspect Experiments",
        description="Create and inspect authored Experiments.",
    )
    experiment_verbs = experiment.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    experiment_create = experiment_verbs.add_parser(
        "create",
        help="Create an Experiment",
        description="Create an Experiment for a Template family.",
        epilog=_examples("hkdl experiment create smoke -t resnet18"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    experiment_create.add_argument(
        "experiment",
        metavar="EXPERIMENT",
        help="Experiment name",
    )
    template_name = experiment_create.add_argument(
        "-t",
        "--template",
        required=True,
        metavar="TEMPLATE",
        help="Template family name",
    )
    template_name.completer = complete_templates
    experiment_list = experiment_verbs.add_parser(
        "list",
        help="List authored Experiments",
        description="List authored Experiments.",
        epilog=_examples("hkdl experiment list", "hkdl experiment list -o json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _output_argument(experiment_list)
    experiment_commit = experiment_verbs.add_parser(
        "commit",
        help="Commit an Experiment draft revision",
        description="Publish the current Experiment draft as an immutable v2 revision.",
        epilog=_examples("hkdl experiment commit smoke"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(experiment_commit)
    _output_argument(experiment_commit)
    experiment_rename = experiment_verbs.add_parser(
        "rename",
        help="Rename one Experiment binding and authored directory",
    )
    old_experiment = experiment_rename.add_argument(
        "old_name", metavar="OLD", help="Existing Experiment name"
    )
    old_experiment.completer = complete_experiments
    experiment_rename.add_argument("new_name", metavar="NEW", help="New name")
    experiment_rename.add_argument(
        "--dry-run", action="store_true", help="Preview without changing authority"
    )
    _output_argument(experiment_rename)

    variant = nouns.add_parser(
        "variant",
        help="Create, clone, and validate Variants",
        description="Create, clone, inspect, and validate authored Variants.",
    )
    variant_verbs = variant.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    variant_create = variant_verbs.add_parser(
        "create",
        help="Create a Variant",
        description="Create a Variant from a Template version.",
        epilog=_examples("hkdl variant create smoke baseline"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(variant_create)
    _variant_argument(variant_create, help_text="Variant name", existing=False)
    template_version = variant_create.add_argument(
        "--template-version",
        metavar="VERSION",
        help="Exact Template version; defaults to the latest numeric version",
    )
    template_version.completer = complete_template_versions
    variant_clone = variant_verbs.add_parser(
        "clone",
        help="Clone an existing Variant",
        description="Clone an existing Variant and its source.",
        epilog=_examples(
            "hkdl variant clone smoke tuned -f baseline",
            ("hkdl variant clone target tuned -f baseline --from-experiment source"),
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(variant_clone)
    _variant_argument(variant_clone, help_text="Target Variant name", existing=False)
    source_variant = variant_clone.add_argument(
        "-f",
        "--from",
        dest="source",
        required=True,
        metavar="VARIANT",
        help="Source Variant name",
    )
    source_variant.completer = complete_source_variants
    source_experiment = variant_clone.add_argument(
        "--from-experiment",
        dest="source_experiment",
        metavar="EXPERIMENT",
        help="Source Experiment; defaults to the target Experiment",
    )
    source_experiment.completer = complete_experiments
    variant_list = variant_verbs.add_parser(
        "list",
        help="List Variants",
        description="List Variants owned by an Experiment.",
        epilog=_examples("hkdl variant list smoke", "hkdl variant list smoke -o json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(variant_list)
    _output_argument(variant_list)
    variant_check = variant_verbs.add_parser(
        "check",
        help="Validate a Variant",
        description="Validate one authored Variant.",
        epilog=_examples("hkdl variant check smoke baseline"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(variant_check)
    _variant_argument(variant_check, help_text="Variant name")
    _output_argument(variant_check)
    variant_commit = variant_verbs.add_parser(
        "commit",
        help="Commit a Variant draft revision",
        description="Publish the current Variant recipe and source as a v2 revision.",
        epilog=_examples("hkdl variant commit smoke baseline"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(variant_commit)
    _variant_argument(variant_commit, help_text="Variant draft to commit")
    _output_argument(variant_commit)
    variant_rename = variant_verbs.add_parser(
        "rename",
        help="Rename one Variant binding",
        description="Atomically move one active Variant name binding without an alias.",
        epilog=_examples(
            "hkdl variant rename smoke old-name new-name --dry-run",
            "hkdl variant rename smoke old-name new-name",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(variant_rename)
    old_name = variant_rename.add_argument("old_name", metavar="OLD")
    old_name.completer = complete_variants
    variant_rename.add_argument("new_name", metavar="NEW")
    variant_rename.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the binding change without writing it",
    )
    _output_argument(variant_rename)

    model = nouns.add_parser(
        "model",
        help="Inspect trained Models",
        description="Inspect immutable Models produced by Train Runs.",
    )
    model_verbs = model.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    model_list = model_verbs.add_parser(
        "list",
        help="List Models",
        description="List Models owned by one Variant.",
        epilog=_examples("hkdl model list smoke baseline"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(model_list)
    _variant_argument(model_list, help_text="Variant owning the Models")
    _output_argument(model_list)
    model_show = model_verbs.add_parser(
        "show",
        help="Show one Model",
        description="Show one immutable Model manifest.",
        epilog=_examples(
            "hkdl model show smoke baseline model-0123456789abcdef0123456789abcdef"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(model_show)
    _variant_argument(model_show, help_text="Variant owning the Model")
    _model_argument(model_show)
    _output_argument(model_show)

    run = nouns.add_parser(
        "run",
        help="Execute Variants",
        description="Execute authored Variants.",
    )
    run_verbs = run.add_subparsers(title="commands", dest="verb", required=True)
    run_train = run_verbs.add_parser(
        "train",
        help="Train a Variant",
        description="Train a Variant and create a persistent Run.",
        epilog=_examples(
            "hkdl run train smoke baseline stability",
            "hkdl run train smoke baseline stability -s 0,1,2 -d cpu",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(run_train)
    _variant_argument(run_train, help_text="Variant to train")
    training_group = run_train.add_argument(
        "training_group",
        metavar="GROUP",
        help="Training Group name",
    )
    training_group.completer = complete_training_groups
    _train_seed_argument(run_train)
    _device_argument(run_train)
    _tracker_argument(run_train)

    run_eval = run_verbs.add_parser(
        "eval",
        help="Evaluate Models in a Training Group",
        description="Create Eval Runs for selected Models and one Evaluation Case.",
        epilog=_examples(
            "hkdl run eval smoke baseline stability clean",
            "hkdl run eval smoke baseline stability clean -s all",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(run_eval)
    _variant_argument(run_eval, help_text="Variant owning the Models")
    evaluation_group = run_eval.add_argument("training_group", metavar="GROUP")
    evaluation_group.completer = complete_training_groups
    evaluation_case = run_eval.add_argument("evaluation_case", metavar="CASE")
    evaluation_case.completer = complete_evaluation_cases
    _eval_seed_argument(run_eval)
    _device_argument(run_eval)

    run_export = run_verbs.add_parser(
        "export",
        help="Export one Model",
        description="Create one Export Run for an immutable Model.",
        epilog=_examples(
            "hkdl run export smoke baseline model-0123456789abcdef0123456789abcdef"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(run_export)
    _variant_argument(run_export, help_text="Variant owning the Model")
    _model_argument(run_export)
    _device_argument(run_export)

    run_retry = run_verbs.add_parser(
        "retry",
        help="Retry a stopped action as a new Run",
        description="Create one new Run from a failed, interrupted, or abandoned Run.",
        epilog=_examples("hkdl run retry smoke baseline run-001"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(run_retry)
    _variant_argument(run_retry, help_text="Variant owning the Run")
    _run_argument(run_retry)
    _tracker_argument(run_retry)

    run_delete = run_verbs.add_parser(
        "delete",
        help="Delete an active Run lineage from the v2 graph",
        description="Plan or apply a recoverable graph-aware Run deletion.",
    )
    _experiment_argument(run_delete)
    _variant_argument(run_delete, help_text="Variant owning the Run")
    _run_argument(run_delete)
    run_delete.add_argument(
        "--cascade",
        action="store_true",
        help="Include retry children, Models, Eval/Export Runs, and results",
    )
    run_delete.add_argument(
        "--force-stale",
        action="store_true",
        help="Permit deletion of unlocked nonterminal Runs",
    )
    run_delete.add_argument(
        "--dry-run", action="store_true", help="Show the exact deletion closure"
    )
    run_delete.add_argument(
        "--yes", action="store_true", help="Confirm and apply the deletion"
    )
    _output_argument(run_delete)

    run_metrics = run_verbs.add_parser(
        "metrics",
        help="Inspect local Train metric history",
        description="Show local scalar history for one Train Run.",
        epilog=_examples(
            "hkdl run metrics smoke baseline run-001",
            "hkdl run metrics smoke baseline run-001 -o json",
            "hkdl run metrics smoke baseline run-001 --follow",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(run_metrics)
    _variant_argument(run_metrics, help_text="Variant owning the Train Run")
    _run_argument(run_metrics)
    _output_argument(run_metrics)
    run_metrics.add_argument(
        "--follow",
        action="store_true",
        help="Follow new local metrics until the Run becomes terminal",
    )

    run_logs = run_verbs.add_parser(
        "logs",
        help="Inspect one Run's action-worker output",
        description="Show merged stdout and stderr captured from one action worker.",
        epilog=_examples("hkdl run logs smoke baseline run-001"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _experiment_argument(run_logs)
    _variant_argument(run_logs, help_text="Variant owning the Run")
    _run_argument(run_logs)

    status = nouns.add_parser(
        "status",
        help="Inspect authoritative Run state",
        description="Show authoritative Runs grouped by Variant, Training Group, and seed.",
        epilog=_examples(
            "hkdl status",
            "hkdl status smoke baseline",
            "hkdl status smoke --table",
            "hkdl status smoke baseline run-001 -o json",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    status.set_defaults(verb=None)
    _experiment_argument(
        status,
        required=False,
        help_text="Experiment filter",
    )
    _variant_argument(
        status,
        required=False,
        help_text="Variant filter",
    )
    _run_argument(status, required=False)
    _output_argument(status)
    status_views = status.add_mutually_exclusive_group()
    status_views.add_argument(
        "--full",
        action="store_true",
        help="Show expanded Run details in text output",
    )
    status_views.add_argument(
        "--table",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show aggregate evaluation results as a table",
    )

    web = nouns.add_parser(
        "web",
        help="Open a local read-only Experiment view",
        description="Serve one Experiment through a loopback-only read-only web view.",
        epilog=_examples("hkdl web smoke", "hkdl web smoke --port 8765"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    web.set_defaults(verb=None)
    _experiment_argument(web, help_text="Experiment to inspect")
    web.add_argument(
        "--port",
        type=_web_port,
        default=DEFAULT_WEB_PORT,
        metavar="PORT",
        help=f"Loopback TCP port (default: {DEFAULT_WEB_PORT})",
    )

    storage = nouns.add_parser(
        "storage",
        help="Inspect repository-owned storage usage",
        description="Show logical bytes used by authored and generated HKDL state.",
        epilog=_examples("hkdl storage", "hkdl storage -o json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    storage.set_defaults(verb=None)
    _output_argument(storage)

    settings = nouns.add_parser(
        "settings",
        help="Inspect and change workspace execution settings",
        description="Manage operational defaults outside research authoring files.",
    )
    settings_verbs = settings.add_subparsers(
        title="commands", dest="verb", required=True
    )
    settings_show = settings_verbs.add_parser("show", help="Show workspace settings")
    _output_argument(settings_show)
    tracker = settings_verbs.add_parser("tracker", help="Manage tracker defaults")
    tracker_verbs = tracker.add_subparsers(
        title="commands", dest="settings_tracker_verb", required=True
    )
    tracker_set = tracker_verbs.add_parser(
        "set", help="Set the workspace default tracker"
    )
    tracker_set.add_argument(
        "backend",
        metavar="BACKEND",
        help="none, local, mlflow, or local+mlflow",
    )
    _output_argument(tracker_set)

    identity = nouns.add_parser(
        "identity",
        help="Inspect content-addressed identity details",
        description="Show full v2 hashes only when explicitly requested.",
    )
    identity_verbs = identity.add_subparsers(
        title="commands", dest="verb", required=True
    )
    identity_show = identity_verbs.add_parser("show", help="Show one identity")
    identity_show.add_argument(
        "kind", choices=("experiment", "variant", "run", "model")
    )
    identity_show.add_argument(
        "address",
        nargs="+",
        metavar="ADDRESS",
        help="Experiment [Variant [Run or Model]]",
    )
    _output_argument(identity_show)

    index = nouns.add_parser(
        "index",
        help="Inspect and rebuild the status projection",
        description="Inspect and rebuild the disposable SQLite status projection.",
    )
    index_verbs = index.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    index_status = index_verbs.add_parser(
        "status",
        help="Inspect projection health without changing it",
        description="Inspect the disposable status projection without changing it.",
        epilog=_examples("hkdl index status"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    index_status.set_defaults(verb="status")
    index_rebuild = index_verbs.add_parser(
        "rebuild",
        help="Rebuild the projection from authoritative files",
        description="Validate authoritative files and atomically rebuild the projection.",
        epilog=_examples("hkdl index rebuild"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    index_rebuild.set_defaults(verb="rebuild")

    environment = nouns.add_parser(
        "environment",
        help="Manage generated Variant environments",
        description="Manage repository-local generated Variant environments.",
    )
    environment_verbs = environment.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    environment_prune = environment_verbs.add_parser(
        "prune",
        help="Remove unused generated environments",
        description="Preview and remove safely reproducible Variant environments.",
        epilog=_examples(
            "hkdl environment prune --dry-run",
            "hkdl environment prune",
            "hkdl environment prune --all --yes",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    environment_prune.add_argument(
        "--all",
        action="store_true",
        help="Also remove inactive environments referenced by current Variants",
    )
    environment_confirmation = environment_prune.add_mutually_exclusive_group()
    environment_confirmation.add_argument(
        "--dry-run",
        action="store_true",
        help="Show removable environments without prompting or modifying state",
    )
    environment_confirmation.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm the shown prune without prompting",
    )

    completion = nouns.add_parser(
        "completion",
        help="Generate shell completion registration",
        description="Generate shell code that registers HKDL completion.",
        epilog=_examples(
            'eval "$(hkdl completion zsh)"',
            'eval "$(hkdl completion bash)"',
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    completion.set_defaults(verb=None)
    completion.add_argument(
        "shell",
        choices=SHELLS,
        metavar="SHELL",
        help="Shell to register: bash or zsh",
    )

    migrate = nouns.add_parser(
        "migrate",
        help="Inspect one schema or migrate the full workspace",
        description=(
            "Inspect or migrate one authored Experiment or Variant file. "
            "Use --all for the complete v1 workspace."
        ),
        epilog=_examples(
            "hkdl migrate experiments/smoke/experiment.yaml",
            "hkdl migrate experiments/smoke/baseline/variant.yaml",
            "hkdl migrate --all --dry-run -o json",
            "hkdl migrate --all --yes",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    migrate.set_defaults(verb=None)
    migrate.add_argument(
        "--expect-plan",
        default=argparse.SUPPRESS,
        metavar="SHA256",
        help="Require the approved authoring migration plan digest",
    )
    migrate.add_argument(
        "--tracker-default",
        choices=("none", "local", "mlflow", "local,mlflow"),
        default=argparse.SUPPRESS,
        help="Explicit future workspace tracker for --authoring; preserve Run history",
    )
    migrate_path = migrate.add_argument(
        "path",
        nargs="?",
        metavar="PATH",
        help="Repository-owned experiment.yaml or variant.yaml",
    )
    migrate_path.completer = file_completer
    migrate.add_argument(
        "--all",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Plan or execute a complete v1-to-v2 workspace import",
    )
    migrate.add_argument(
        "--authoring",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Migrate active-v2 authored YAML and execution identity to schema 2",
    )
    migrate.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Compute and report the full import without writing authority",
    )
    migrate.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Confirm v2 cutover without prompting",
    )
    migrate.add_argument(
        "-o",
        "--output",
        choices=("text", "json"),
        default=argparse.SUPPRESS,
        help="Output format for --all (default: text)",
    )

    update_parser = nouns.add_parser(
        "update",
        help="Update this public HKDL source checkout",
        description="Inspect and update a clean public HKDL source checkout.",
        epilog=_examples("hkdl update", "hkdl update --yes"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    update_parser.set_defaults(verb=None)
    update_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm the shown update without prompting",
    )

    return parser


def _examples(*commands: str) -> str:
    return "examples:\n" + "\n".join(f"  {command}" for command in commands)


def _web_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _index_report(report: IndexReport) -> None:
    print(f"state: {report.state}")
    print(f"path: {report.path}")
    print(
        "schema: "
        + ("-" if report.schema_version is None else str(report.schema_version))
    )
    print(f"variants: {report.variants}")
    print(f"runs: {report.runs}")
    print(f"models: {report.models}")
    if report.detail is not None:
        print(f"detail: {report.detail}")


def _experiment_argument(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
    help_text: str = "Experiment containing the Variant",
) -> None:
    action = parser.add_argument(
        "experiment",
        nargs=None if required else "?",
        metavar="EXPERIMENT",
        help=help_text,
    )
    action.completer = complete_experiments


def _variant_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
    required: bool = True,
    existing: bool = True,
) -> None:
    action = parser.add_argument(
        "variant",
        nargs=None if required else "?",
        metavar="VARIANT",
        help=help_text,
    )
    if existing:
        action.completer = complete_variants


def _run_argument(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
) -> None:
    action = parser.add_argument(
        "run_id",
        nargs=None if required else "?",
        metavar="RUN",
        help="Run ID",
    )
    action.completer = complete_runs


def _train_seed_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-s",
        "--seed",
        dest="seeds",
        type=_parse_seed_list,
        default=(0,),
        metavar="SEED[,SEED...]",
        help="One or more random seeds (default: 0)",
    )


def _eval_seed_argument(parser: argparse.ArgumentParser) -> None:
    action = parser.add_argument(
        "-s",
        "--seed",
        type=_parse_eval_seed,
        default=None,
        metavar="SEED|all",
        help="One Model seed or all; inferred when the group has one Model",
    )
    action.completer = complete_eval_seeds


def _model_argument(parser: argparse.ArgumentParser) -> None:
    action = parser.add_argument(
        "model_id",
        metavar="MODEL",
        help="Model ID",
    )
    action.completer = complete_models


def _parse_seed_list(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not parts or any(
        not part or not part.isascii() or not part.isdigit() for part in parts
    ):
        raise argparse.ArgumentTypeError("seed list must contain decimal integers")
    seeds = tuple(int(part) for part in parts)
    if any(seed > MAX_SEED for seed in seeds):
        raise argparse.ArgumentTypeError(f"seed must not exceed {MAX_SEED}")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seed list must not contain duplicates")
    return seeds


def _parse_eval_seed(value: str) -> int | str:
    if value == "all":
        return value
    return _parse_seed_list(value)[0] if "," not in value else _invalid_eval_seed()


def _invalid_eval_seed():
    raise argparse.ArgumentTypeError("evaluation seed must be one integer or all")


def _device_argument(parser: argparse.ArgumentParser) -> None:
    action = parser.add_argument(
        "-d",
        "--device",
        default="auto",
        metavar="DEVICE",
        help="auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    action.completer = complete_devices


def _tracker_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracker",
        metavar="BACKEND",
        help="Override workspace tracker: none, local, mlflow, or local+mlflow",
    )


def _output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )


def _table(
    headers: tuple[str, ...],
    rows: Iterable[Sequence[object]],
    *,
    file: Any = None,
) -> None:
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


def _prune_table(
    plan: PrunePlan,
    root: Path,
    *,
    file: Any = None,
) -> None:
    if file is None:
        file = sys.stdout
    _table(
        ("KIND", "PATH", "BYTES", "SIZE"),
        (
            (
                entry.kind,
                entry.path.relative_to(root),
                entry.bytes,
                _format_bytes(entry.bytes),
            )
            for entry in plan.entries
        ),
        file=file,
    )
    print(
        f"total environments={len(plan.entries)} bytes={plan.bytes} "
        f"size={_format_bytes(plan.bytes)} retained={plan.retained} busy={plan.busy}",
        file=file,
    )


def _confirm(question: str) -> bool:
    print(f"{question} [y/N] ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in {"y", "yes"}


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if value < 1024 or unit == "EiB":
            return f"{size} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable size unit")


def _status_table(document: dict[str, Any]) -> None:
    _table(
        (
            "EXPERIMENT",
            "VARIANT",
            "GROUP",
            "CASE",
            "METRIC",
            "COUNT",
            "ELIGIBLE",
            "MEAN",
            "SAMPLE_STD",
        ),
        (
            (
                experiment["name"],
                variant["name"],
                group["name"],
                aggregate["evaluation_case"],
                aggregate["metric"],
                aggregate["count"],
                aggregate["eligible"],
                aggregate["mean"],
                aggregate["sample_std"] if aggregate["sample_std"] is not None else "-",
            )
            for experiment in document["experiments"]
            for variant in experiment["variants"]
            for group in variant["training_groups"]
            for aggregate in group["aggregates"]
        ),
    )


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


def _json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _public_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_payload(item)
            for key, item in value.items()
            if not key.endswith("_hash")
            and key
            not in {
                "before_head",
                "binding_transaction",
                "attempt_hashes",
                "run_spec_hashes",
                "event_hashes",
                "result_hashes",
                "blob_hashes",
            }
        }
    if isinstance(value, list):
        return [_public_payload(item) for item in value]
    return value


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


def _short_hash(digest: str) -> str:
    return digest.removeprefix("sha256:")[:12]


if __name__ == "__main__":
    raise SystemExit(main())
