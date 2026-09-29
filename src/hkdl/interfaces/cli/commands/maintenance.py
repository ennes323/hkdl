"""Present maintenance operations owned by installation and storage services.

Managed release activation stays with the standalone installer. Workspace
updates, derived indexes, settings, identity reads, and environment pruning are
delegated to their owners while this module handles CLI selection and output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hkdl.authoring.authoring import Authoring
from hkdl.errors import ContractError
from hkdl.execution.run_contracts import TERMINAL_STATUSES
from hkdl.installation import installed_paths
from hkdl.interfaces.cli import io
from hkdl.runtime.environments import EnvironmentStore, PrunePlan
from hkdl.storage.graph.graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    entity_revision_scope,
    experiment_variant_scope,
    workspace_experiment_scope,
)
from hkdl.storage.graph.reader import current_reader
from hkdl.storage.runs import RunStore
from hkdl.storage.settings import WorkspaceSettingsStore
from hkdl.storage.status import Status
from hkdl.storage.status_index import IndexReport
from hkdl.storage.storage import RepositoryPaths, storage_usage
from hkdl.update import update


def managed_update(args: argparse.Namespace) -> int:
    """Translate managed update options into a standalone installer invocation.

    ``main`` calls this outside its shared installation lease so the installer
    can acquire the exclusive activation boundary it owns.
    """

    # Load the standalone installer only for managed-update dispatch.
    from hkdl.installer.cli import main as installer_main

    installation = installed_paths()
    if installation is None or args.bundle is None:
        raise ContractError(
            "select a release bundle explicitly: hkdl update BUNDLE.zip"
        )
    arguments = ["--root", str(installation.root), "install", args.bundle]
    if args.yes:
        arguments.append("--yes")
    if args.sha256:
        arguments.extend(["--sha256", args.sha256])
    if args.repair:
        arguments.append("--repair")
    return installer_main(arguments)


# Flow: select the service owner -> perform or inspect work -> render its report.
def run(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Delegate the selected workspace maintenance or diagnostic operation.

    Service layers retain validation, locking, and mutation responsibilities;
    this handler shapes their reports for the requested CLI output.
    """

    if args.noun == "update":
        if args.bundle or args.sha256 or args.repair:
            raise ContractError("release bundle updates require a managed installation")
        update(repository, assume_yes=args.yes)
        return 0
    if args.noun == "storage":
        usage = storage_usage(repository)
        if args.output == "json":
            io.print_json({"storage": usage})
        else:
            io.table(
                ("CATEGORY", "BYTES", "SIZE"),
                (
                    (category, size, io.format_bytes(size))
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
    if args.noun == "settings":
        settings = WorkspaceSettingsStore(repository.root)
        if args.verb == "show":
            document = settings.show()
            if args.output == "json":
                io.print_json({"settings": document})
            else:
                backend = document["tracker"]["backend"]
                value = ",".join(backend) if isinstance(backend, list) else backend
                print(f"tracker: {value}")
            return 0
        result = settings.set_tracker(args.backend)
        if args.output == "json":
            io.print_json({"settings": result.as_dict()})
        else:
            backend = result.backend
            value = ",".join(backend) if isinstance(backend, list) else backend
            print(f"tracker: {value}")
        return 0

    authoring = Authoring(repository)

    if (args.noun, args.verb) == ("identity", "show"):
        # Resolve related bindings through dispatch's single graph observation
        # so every reported identity belongs to the same HEAD.
        graph = V2Graph(repository)
        if not graph.is_active():
            raise ContractError("HKDL v2 is not active")
        addresses = list(args.address)
        expected = {"experiment": 1, "variant": 2, "run": 3, "model": 3}
        if len(addresses) != expected[args.kind]:
            raise ContractError(
                f"identity show {args.kind} requires {expected[args.kind]} address values"
            )
        reader = current_reader(repository)
        if args.kind == "experiment":
            experiment_hash = reader.resolve(workspace_experiment_scope(), addresses[0])
            payload = {
                "experiment": addresses[0],
                "experiment_hash": experiment_hash,
                "experiment_revision_hash": reader.resolve(
                    entity_revision_scope(experiment_hash), CURRENT_REVISION_NAME
                ),
            }
        elif args.kind == "variant":
            experiment_hash = reader.resolve(workspace_experiment_scope(), addresses[0])
            variant_hash = reader.resolve(
                experiment_variant_scope(experiment_hash), addresses[1]
            )
            revision_hash = reader.resolve(
                entity_revision_scope(variant_hash), CURRENT_REVISION_NAME
            )
            revision = reader.graph.store.load(revision_hash)
            payload = {
                "experiment": addresses[0],
                "variant": addresses[1],
                "experiment_hash": experiment_hash,
                "experiment_revision_hash": reader.resolve(
                    entity_revision_scope(experiment_hash), CURRENT_REVISION_NAME
                ),
                "variant_hash": variant_hash,
                "variant_revision_hash": revision_hash,
                "source_tree_hash": revision.payload["source_tree"],
            }
        elif args.kind == "run":
            record = RunStore(repository).load(*addresses)
            identity = reader.identity(record)
            payload = {
                "experiment": addresses[0],
                "variant": addresses[1],
                "run_id": addresses[2],
                **identity.as_dict(),
            }
        else:
            model_hash = reader.model_hash(*addresses)
            model = reader.graph.store.load(model_hash)
            payload = {
                "experiment": addresses[0],
                "variant": addresses[1],
                "model_id": addresses[2],
                "model_hash": model_hash,
                **model.payload,
            }
        if args.output == "json":
            io.print_json({"identity": payload})
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if (args.noun, args.verb) == ("environment", "prune"):
        # Snapshot active Variants for retention policy. EnvironmentStore
        # re-plans candidates and uses per-environment locks during deletion.
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
        if not args.yes and not io.confirm("Continue with environment prune?"):
            print("Environment prune cancelled. Nothing was removed.")
            return 0
        result = environments.prune(
            variants,
            active_variants=active_variants,
            remove_all=args.all,
        )
        print(
            f"pruned environments={result.removed} "
            f"bytes={result.bytes} size={io.format_bytes(result.bytes)} "
            f"retained={result.retained} busy={result.busy}"
        )
        return 0

    raise AssertionError("unreachable command")


def _index_report(report: IndexReport) -> None:
    """Render the common report returned by legacy and graph index owners."""
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


def _prune_table(
    plan: PrunePlan,
    root: Path,
    *,
    file: Any = None,
) -> None:
    """Render an environment prune plan before confirmation or as a dry run."""
    if file is None:
        file = sys.stdout
    io.table(
        ("KIND", "PATH", "BYTES", "SIZE"),
        (
            (
                entry.kind,
                entry.path.relative_to(root),
                entry.bytes,
                io.format_bytes(entry.bytes),
            )
            for entry in plan.entries
        ),
        file=file,
    )
    print(
        f"total environments={len(plan.entries)} bytes={plan.bytes} "
        f"size={io.format_bytes(plan.bytes)} retained={plan.retained} busy={plan.busy}",
        file=file,
    )
