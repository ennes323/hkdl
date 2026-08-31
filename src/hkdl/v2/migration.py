"""Verified v1-to-v2 workspace import planning and atomic cutover."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from ..authoring import (
    EXPERIMENT_AUXILIARY_DIRECTORIES,
    Authoring,
    ExperimentRecord,
    VariantRecord,
)
from ..config import ContractError
from ..run_contracts import TERMINAL_STATUSES, validate_tracker
from ..research_json import split_legacy_variant
from ..runs import ModelRecord, RunRecord, RunStore
from ..storage import (
    LockUnavailableError,
    RepositoryPaths,
    atomic_write_new,
    compute_source_digest,
    try_directory_lock,
)
from .bindings import BindingOperation
from .graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    attempt_event_scope,
    comparison_digest,
    comparison_group_scope,
    entity_revision_scope,
    evaluation_case_scope,
    experiment_variant_scope,
    export_profile_scope,
    variant_model_scope,
    variant_run_scope,
    workspace_experiment_scope,
)
from .objects import ObjectRecord, ObjectStore, canonical_json_bytes, hash_file
from .projection import GraphProjection
from .maintenance import workspace_access, workspace_operation

IMPORT_TIMESTAMP = "1970-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class PlannedBlob:
    source: Path | None
    content_hash: str
    size: int
    object_hash: str
    content: bytes | None = None


@dataclass(frozen=True)
class WorkspaceMigrationReport:
    variants: int
    runs: int
    models: int
    artifacts: int
    input_bytes: int
    output_bytes: int
    estimated_additional_bytes: int
    object_counts: dict[str, int]
    object_hashes: list[str]
    active_leases: list[str]
    stale_nonterminal: list[str]
    malformed_records: list[dict[str, str]]
    mlflow_identities: list[dict[str, str]]
    plan_digest: str
    cutover_ready: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "variants": self.variants,
            "runs": self.runs,
            "models": self.models,
            "artifacts": self.artifacts,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "estimated_additional_bytes": self.estimated_additional_bytes,
            "object_counts": self.object_counts,
            "object_hashes": self.object_hashes,
            "active_leases": self.active_leases,
            "stale_nonterminal": self.stale_nonterminal,
            "malformed_records": self.malformed_records,
            "mlflow_identities": self.mlflow_identities,
            "plan_digest": self.plan_digest,
            "cutover_ready": self.cutover_ready,
        }


@dataclass
class WorkspaceMigrationPlan:
    repository: RepositoryPaths
    objects: dict[str, ObjectRecord]
    blobs: dict[str, PlannedBlob]
    transaction_batches: list[list[BindingOperation]]
    transaction_hash: str | None
    report: WorkspaceMigrationReport
    authoritative_digests: dict[str, str]


class _Builder:
    def __init__(self, store: ObjectStore):
        self.store = store
        self.objects: dict[str, ObjectRecord] = {}
        self.blobs: dict[str, PlannedBlob] = {}

    def add(self, kind: str, payload: dict[str, Any]) -> str:
        record = self.store.preview(kind, payload)
        existing = self.objects.get(record.digest)
        if existing is not None and (
            existing.kind != record.kind or existing.payload != record.payload
        ):
            raise ContractError("v2 migration object hash collision")
        self.objects[record.digest] = record
        return record.digest

    def blob(self, path: Path) -> str:
        content_hash, size = hash_file(path)
        payload = {
            "content_hash": content_hash,
            "media_type": "application/octet-stream",
            "size": size,
        }
        object_hash = self.add("blob", payload)
        existing = self.blobs.get(content_hash)
        if existing is None:
            self.blobs[content_hash] = PlannedBlob(
                path.absolute(), content_hash, size, object_hash
            )
        elif existing.size != size:
            raise ContractError("v2 migration blob digest size collision")
        return object_hash

    def blob_bytes(self, content: bytes) -> str:
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        object_hash = self.add(
            "blob",
            {
                "content_hash": digest,
                "size": len(content),
                "media_type": "application/json",
            },
        )
        self.blobs.setdefault(
            digest, PlannedBlob(None, digest, len(content), object_hash, content)
        )
        return object_hash

    def source_tree(self, path: Path | None, legacy_digest: str) -> str:
        if path is None:
            return self.add(
                "source_tree",
                {"availability": "unavailable", "legacy_digest": legacy_digest},
            )
        files: list[dict[str, Any]] = []
        for entry in sorted(
            path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()
        ):
            relative = PurePosixPath(*entry.relative_to(path).parts).as_posix()
            metadata = entry.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ContractError(f"migration source contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(
                    f"migration source contains a non-regular entry: {relative}"
                )
            blob_hash = self.blob(entry)
            blob = self.objects[blob_hash]
            files.append(
                {
                    "path": relative,
                    "blob": blob_hash,
                    "content_hash": blob.payload["content_hash"],
                    "size": blob.payload["size"],
                }
            )
        return self.add("source_tree", {"availability": "available", "files": files})


class WorkspaceMigration:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        failure_hook: Callable[[str], None] | None = None,
    ):
        self.repository = repository
        self.authoring = Authoring(repository)
        self.runs = RunStore(repository)
        self.graph = V2Graph(repository)
        self._failure_hook = failure_hook

    @workspace_operation
    def plan(self) -> WorkspaceMigrationPlan:
        if self.graph.is_active():
            raise ContractError("HKDL v2 is already active")
        staged_head = self.graph.bindings.head()
        malformed: list[dict[str, str]] = []
        experiments, variants = self._authored(malformed)
        runs = self._runs(malformed)
        models = self._models(malformed)
        active_leases, stale = self._leases(runs)
        authoritative = _authority_digests(self.repository, malformed)
        input_bytes = sum(
            path.lstat().st_size
            for root in (self.repository.experiments, self.repository.outputs)
            if os.path.lexists(root)
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not path.name.startswith(".hkdl-index.sqlite3")
        )

        builder = _Builder(self.graph.store)
        operations: list[BindingOperation] = []
        run_by_variant: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
        for run in runs:
            run_by_variant[
                (str(run.request["experiment"]), str(run.request["variant"]))
            ].append(run)
        models_by_variant: dict[tuple[str, str], list[ModelRecord]] = defaultdict(list)
        for model in models:
            models_by_variant[
                (str(model.document["experiment"]), str(model.document["variant"]))
            ].append(model)

        experiment_documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            experiment_documents[str(run.request["experiment"])].append(
                run.snapshot["experiment"]
            )
        for name, record in experiments.items():
            experiment_documents[name].append(record.document)

        experiment_hashes: dict[str, str] = {}
        experiment_snapshot_revisions = {}
        experiment_revisions: dict[str, str] = {}
        for name in sorted(experiment_documents, key=lambda item: item.encode("utf-8")):
            documents = _unique_documents(experiment_documents[name], ignored={"name"})
            created_at = str(documents[-1].get("created_at", IMPORT_TIMESTAMP))
            entity = builder.add(
                "experiment",
                {
                    "created_at": created_at,
                    "creation_nonce": _legacy_nonce(f"experiment:{name}"),
                },
            )
            experiment_hashes[name] = entity
            parent = None
            for document in documents:
                parent = builder.add(
                    "experiment_revision",
                    {
                        "experiment": entity,
                        "parent": parent,
                        "type": document["type"],
                        "question": document["question"],
                        "template": deepcopy(document["template"]),
                    },
                )
                key = canonical_json_bytes(
                    {
                        field: document[field]
                        for field in ("type", "question", "template")
                    }
                )
                experiment_snapshot_revisions.setdefault((name, key), parent)
            if parent is None:
                raise ContractError("migration Experiment has no revision")
            experiment_revisions[name] = parent
            operations.extend(
                [
                    BindingOperation(
                        "bind", workspace_experiment_scope(), name, entity
                    ),
                    BindingOperation(
                        "bind",
                        entity_revision_scope(entity),
                        CURRENT_REVISION_NAME,
                        parent,
                    ),
                ]
            )

        variant_keys = set(variants) | set(run_by_variant) | set(models_by_variant)
        variant_hashes: dict[tuple[str, str], str] = {}
        current_revisions: dict[tuple[str, str], str] = {}
        run_revisions: dict[str, str] = {}
        for key in sorted(
            variant_keys, key=lambda item: (item[0].encode(), item[1].encode())
        ):
            experiment_name, variant_name = key
            experiment_hash = experiment_hashes.get(experiment_name)
            if experiment_hash is None:
                raise ContractError(f"migration Variant has no Experiment: {key}")
            entity = builder.add(
                "variant",
                {
                    "experiment": experiment_hash,
                    "created_at": _variant_created_at(key, run_by_variant, experiments),
                    "creation_nonce": _legacy_nonce(
                        f"variant:{experiment_name}/{variant_name}"
                    ),
                },
            )
            variant_hashes[key] = entity
            revision_inputs: list[
                tuple[dict[str, Any], str, Path | None, str | None]
            ] = []
            for run in sorted(run_by_variant.get(key, []), key=_run_order):
                digest = str(run.request["source_digest"])
                source_path = _matching_authored_source(variants.get(key), digest)
                revision_inputs.append(
                    (run.snapshot["variant"], digest, source_path, run.address)
                )
            authored = variants.get(key)
            if authored is not None:
                digest = compute_source_digest(authored.path / "src")
                revision_inputs.append(
                    (authored.document, digest, authored.path / "src", None)
                )
            if not revision_inputs:
                raise ContractError(
                    f"migration Variant has no revision evidence: {key}"
                )
            parent = None
            last_semantic: dict[str, Any] | None = None
            current_cases: dict[str, str] = {}
            current_profiles: dict[str, str] = {}
            for document, digest, source_path, run_address in revision_inputs:
                source_tree = builder.source_tree(source_path, digest)
                cases = _case_objects(builder, document)
                profiles = _profile_objects(builder, document)
                semantic = {
                    "variant": entity,
                    "template": deepcopy(document["template"]),
                    "source_tree": source_tree,
                    "components": deepcopy(document["components"]),
                }
                if semantic != last_semantic:
                    parent = builder.add(
                        "variant_revision",
                        {
                            **semantic,
                            "parent": parent,
                            "derivation_parent": None,
                        },
                    )
                    last_semantic = semantic
                if run_address is not None:
                    run_revisions[run_address] = str(parent)
                current_cases = cases
                current_profiles = profiles
            if parent is None:
                raise ContractError("migration Variant has no revision")
            current_revisions[key] = parent
            operations.extend(
                [
                    BindingOperation(
                        "bind",
                        experiment_variant_scope(experiment_hash),
                        variant_name,
                        entity,
                    ),
                    BindingOperation(
                        "bind",
                        entity_revision_scope(entity),
                        CURRENT_REVISION_NAME,
                        parent,
                    ),
                ]
            )
            for case_name, case_hash in sorted(current_cases.items()):
                operations.append(
                    BindingOperation(
                        "bind", evaluation_case_scope(parent), case_name, case_hash
                    )
                )
            for profile_name, profile_hash in sorted(current_profiles.items()):
                operations.append(
                    BindingOperation(
                        "bind", export_profile_scope(parent), profile_name, profile_hash
                    )
                )

        run_specs: dict[str, str] = {}
        run_experiment_revisions = {
            run.address: experiment_snapshot_revisions[
                (
                    run.request["experiment"],
                    canonical_json_bytes(
                        {
                            field: run.snapshot["experiment"][field]
                            for field in ("type", "question", "template")
                        }
                    ),
                )
            ]
            for run in runs
        }
        attempt_hashes: dict[str, str] = {}
        comparison_hashes: dict[str, str | None] = {}
        train_runs = [run for run in runs if run.request["action"] == "train"]
        for run in sorted(train_runs, key=_run_order):
            spec, comparison = _train_spec(
                builder,
                run,
                run_revisions[run.address],
                run_experiment_revisions[run.address],
            )
            run_specs[run.address] = spec
            comparison_hashes[run.address] = comparison
            attempt_hashes[run.address] = _attempt(builder, run, spec, attempt_hashes)

        model_hashes: dict[str, str] = {}
        for model in sorted(models, key=lambda item: item.address.encode("utf-8")):
            producer_address = (
                f"{model.document['experiment']}/{model.document['variant']}/"
                f"{model.document['producer_run']}"
            )
            attempt_hash = attempt_hashes.get(producer_address)
            if attempt_hash is None:
                malformed.append(
                    {"path": str(model.path), "error": "producer Run is unavailable"}
                )
                continue
            checkpoint = (
                self.repository.outputs
                / str(model.document["experiment"])
                / str(model.document["variant"])
                / str(model.document["checkpoint"]["path"])
            )
            blob = builder.blob(checkpoint)
            revision = run_revisions[producer_address]
            model_hash = builder.add(
                "model",
                {
                    "producing_attempt": attempt_hash,
                    "experiment_revision": run_experiment_revisions[producer_address],
                    "train_options": builder.objects[
                        run_specs[producer_address]
                    ].payload["train_options"],
                    "variant_revision": revision,
                    "comparison_hash": comparison_hashes[producer_address],
                    "seed": model.document["seed"],
                    "checkpoint_blob": blob,
                    "checkpoint_relative_path": model.document["checkpoint"][
                        "path"
                    ].split("/", 2)[2],
                    "checkpoint_content_hash": builder.objects[blob].payload[
                        "content_hash"
                    ],
                    "created_at": model.document["created_at"],
                },
            )
            model_hashes[model.address] = model_hash

        for run in sorted(
            [run for run in runs if run.request["action"] != "train"],
            key=_run_order,
        ):
            key = (str(run.request["experiment"]), str(run.request["variant"]))
            revision = run_revisions[run.address]
            model_address = f"{key[0]}/{key[1]}/{run.request['target']['model_id']}"
            model_hash = model_hashes.get(model_address)
            if model_hash is None:
                malformed.append(
                    {"path": str(run.path), "error": "referenced Model is unavailable"}
                )
                continue
            if run.request["action"] == "eval":
                cases = _case_objects(builder, run.snapshot["variant"])
                case_name = str(run.request["target"]["evaluation_case"])
                spec = builder.add(
                    "run_spec",
                    {
                        "action": "eval",
                        "experiment_revision": run_experiment_revisions[run.address],
                        "model": model_hash,
                        "evaluation_case": cases[case_name],
                        "evaluator_revision": revision,
                        "device": run.request["exec"]["device"],
                    },
                )
            else:
                profiles = _profile_objects(builder, run.snapshot["variant"])
                spec = builder.add(
                    "run_spec",
                    {
                        "action": "export",
                        "experiment_revision": run_experiment_revisions[run.address],
                        "model": model_hash,
                        "export_profile": profiles["default"],
                        "exporter_revision": revision,
                        "device": run.request["exec"]["device"],
                    },
                )
            run_specs[run.address] = spec
            comparison_hashes[run.address] = None
            attempt_hashes[run.address] = _attempt(builder, run, spec, attempt_hashes)

        artifact_count = 0
        current_events: dict[str, str] = {}
        for run in sorted(runs, key=_run_order):
            attempt_hash = attempt_hashes.get(run.address)
            if attempt_hash is None:
                continue
            allocated = builder.add(
                "attempt_event",
                _event_payload(run, attempt_hash, None, [], None, status="allocated"),
            )
            current_event = allocated
            if run.state["status"] != "allocated":
                artifacts = _artifact_objects(builder, run)
                artifact_count += len(artifacts)
                result_object = _result_object(
                    builder,
                    run,
                    attempt_hash,
                    run_specs[run.address],
                    artifacts,
                    model_hashes,
                )
                current_event = builder.add(
                    "attempt_event",
                    _event_payload(
                        run,
                        attempt_hash,
                        allocated,
                        artifacts,
                        result_object,
                        status=str(run.state["status"]),
                    ),
                )
            current_events[run.address] = current_event

        comparison_transactions = _comparison_binding_transactions(
            train_runs,
            run_revisions,
            comparison_hashes,
        )
        for key, entity in variant_hashes.items():
            for run in sorted(run_by_variant.get(key, []), key=_run_order):
                attempt_hash = attempt_hashes.get(run.address)
                if attempt_hash is None:
                    continue
                operations.extend(
                    [
                        BindingOperation(
                            "bind",
                            variant_run_scope(entity),
                            str(run.request["run_id"]),
                            attempt_hash,
                        ),
                        BindingOperation(
                            "bind",
                            attempt_event_scope(attempt_hash),
                            CURRENT_REVISION_NAME,
                            current_events[run.address],
                        ),
                    ]
                )
            for model in sorted(
                models_by_variant.get(key, []), key=lambda item: item.address
            ):
                model_hash = model_hashes.get(model.address)
                if model_hash is not None:
                    operations.append(
                        BindingOperation(
                            "bind",
                            variant_model_scope(entity),
                            str(model.document["model_id"]),
                            model_hash,
                        )
                    )

        transaction_batches: list[list[BindingOperation]] = []
        if operations:
            transaction_batches.append(operations)
        transaction_batches.extend(comparison_transactions)
        transaction_hash = None
        for index, batch in enumerate(transaction_batches):
            transaction_hash = builder.add(
                "binding_transaction",
                {
                    "previous": transaction_hash,
                    "operations": [operation.as_dict() for operation in batch],
                    "created_at": IMPORT_TIMESTAMP,
                    "nonce": f"v1-full-workspace-import:{index}",
                },
            )
        if staged_head is not None and staged_head != transaction_hash:
            raise ContractError("staged v2 binding state disagrees with migration plan")
        object_counts = dict(
            sorted(Counter(item.kind for item in builder.objects.values()).items())
        )
        object_hashes = sorted(builder.objects)
        object_bytes = sum(
            len(canonical_json_bytes(record.envelope)) + 1
            for record in builder.objects.values()
        )
        blob_bytes = sum(item.size for item in builder.blobs.values())
        output_bytes = object_bytes + blob_bytes
        additional = _additional_bytes(self.graph.store, builder)
        mlflow = [
            {
                "attempt": attempt_hashes[run.address],
                "tracker_run_id": str(run.state["tracker_run_id"]),
            }
            for run in runs
            if run.address in attempt_hashes
            and isinstance(run.state["tracker_run_id"], str)
            and str(run.state["tracker_run_id"]).startswith("mlflow:")
        ]
        digest_payload = {
            "objects": object_hashes,
            "transactions": [
                [operation.as_dict() for operation in batch]
                for batch in transaction_batches
            ],
            "authority": authoritative,
        }
        plan_digest = (
            f"sha256:{hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()}"
        )
        report = WorkspaceMigrationReport(
            variants=len(variant_keys),
            runs=len(runs),
            models=len(model_hashes),
            artifacts=artifact_count,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            estimated_additional_bytes=additional,
            object_counts=object_counts,
            object_hashes=object_hashes,
            active_leases=active_leases,
            stale_nonterminal=stale,
            malformed_records=malformed,
            mlflow_identities=mlflow,
            plan_digest=plan_digest,
            cutover_ready=not active_leases and not malformed,
        )
        return WorkspaceMigrationPlan(
            self.repository,
            builder.objects,
            builder.blobs,
            transaction_batches,
            transaction_hash,
            report,
            authoritative,
        )

    def apply(self, plan: WorkspaceMigrationPlan) -> WorkspaceMigrationReport:
        with workspace_access(self.repository, exclusive=True):
            return self._apply(plan)

    def _apply(self, plan: WorkspaceMigrationPlan) -> WorkspaceMigrationReport:
        if plan.repository != self.repository:
            raise ContractError("migration plan belongs to another repository")
        if not plan.report.cutover_ready:
            if plan.report.active_leases:
                raise MigrationConflict("active Run lease blocks migration")
            raise ContractError("malformed v1 records block migration")
        if _authority_digests(self.repository, []) != plan.authoritative_digests:
            raise MigrationConflict("v1 authority changed after migration planning")
        refreshed_leases, _ = self._leases(self._runs([]))
        if refreshed_leases:
            raise MigrationConflict("active Run lease blocks migration")
        store = self.graph.store
        for blob in plan.blobs.values():
            published = (
                store.put_blob_bytes(
                    blob.content,
                    media_type=plan.objects[blob.object_hash].payload["media_type"],
                )
                if blob.content is not None
                else store.put_blob(blob.source)
            )
            if published.digest != blob.object_hash:
                raise ContractError("migration blob publication disagrees with plan")
        self._phase("after_blobs")
        for digest in sorted(plan.objects):
            record = plan.objects[digest]
            published = store.put(record.kind, record.payload)
            if published.digest != digest:
                raise ContractError("migration object publication disagrees with plan")
        self._phase("after_objects")
        if plan.transaction_hash is not None:
            self.graph.bindings.root.mkdir(mode=0o755, parents=True, exist_ok=True)
            existing_head = self.graph.bindings.head()
            if existing_head is None:
                atomic_write_new(
                    self.graph.bindings.head_path,
                    f"{plan.transaction_hash}\n",
                )
            elif existing_head != plan.transaction_hash:
                raise MigrationConflict("v2 binding HEAD appeared during migration")
        self._phase("after_head")
        if self.graph.bindings.bindings() != _operations_result(
            plan.transaction_batches
        ):
            raise ContractError("migration binding verification failed")
        GraphProjection(self.graph).rebuild()
        self._phase("after_projection")
        if _authority_digests(self.repository, []) != plan.authoritative_digests:
            raise MigrationConflict("v1 authority changed during migration")
        self._phase("before_current")
        self.graph.activate()
        self._phase("after_current")
        return plan.report

    def _phase(self, name: str) -> None:
        if self._failure_hook is not None:
            self._failure_hook(name)

    def _authored(
        self, malformed: list[dict[str, str]]
    ) -> tuple[
        dict[str, ExperimentRecord],
        dict[tuple[str, str], VariantRecord],
    ]:
        experiments: dict[str, ExperimentRecord] = {}
        variants: dict[tuple[str, str], VariantRecord] = {}
        root = self.repository.experiments
        if not os.path.lexists(root):
            return experiments, variants
        for entry in sorted(root.iterdir(), key=lambda item: item.name.encode("utf-8")):
            if entry.name.startswith("."):
                continue
            try:
                experiment = self.authoring.load_experiment(entry.name)
            except (ContractError, OSError) as error:
                malformed.append({"path": str(entry), "error": str(error)})
                continue
            experiments[entry.name] = experiment
            for child in sorted(
                entry.iterdir(), key=lambda item: item.name.encode("utf-8")
            ):
                if child.name.startswith(".") or child.name in {
                    "experiment.yaml",
                    "notes",
                    *EXPERIMENT_AUXILIARY_DIRECTORIES,
                }:
                    continue
                try:
                    variant = self.authoring.load_variant(entry.name, child.name)
                except (ContractError, OSError) as error:
                    malformed.append({"path": str(child), "error": str(error)})
                    continue
                variants[(entry.name, child.name)] = variant
        return experiments, variants

    def _runs(self, malformed: list[dict[str, str]]) -> list[RunRecord]:
        records: list[RunRecord] = []
        for experiment, variant, run_id, path in _generated_entries(
            self.repository.outputs, "runs"
        ):
            try:
                records.append(self.runs.load(experiment, variant, run_id))
            except (ContractError, OSError) as error:
                malformed.append({"path": str(path), "error": str(error)})
        return records

    def _models(self, malformed: list[dict[str, str]]) -> list[ModelRecord]:
        records: list[ModelRecord] = []
        for experiment, variant, model_id, path in _generated_entries(
            self.repository.outputs, "models"
        ):
            try:
                records.append(self.runs.load_model(experiment, variant, model_id))
            except (ContractError, OSError) as error:
                malformed.append({"path": str(path), "error": str(error)})
        return records

    @staticmethod
    def _leases(records: list[RunRecord]) -> tuple[list[str], list[str]]:
        active: list[str] = []
        stale: list[str] = []
        for record in records:
            if record.state["status"] in TERMINAL_STATUSES:
                continue
            try:
                with try_directory_lock(record.path):
                    stale.append(record.address)
            except LockUnavailableError:
                active.append(record.address)
        return active, stale


class MigrationConflict(RuntimeError):
    """Workspace migration conflicts with live or changed v1 authority."""


def _case_objects(builder: _Builder, document: dict[str, Any]) -> dict[str, str]:
    evaluation = document["eval"]
    cases = evaluation.get("cases")
    if isinstance(cases, dict):
        definitions = {name: deepcopy(value) for name, value in cases.items()}
    else:
        definitions = {"default": deepcopy(evaluation)}
    result: dict[str, str] = {}
    for name, definition in sorted(definitions.items()):
        metrics = deepcopy(definition.get("metrics", document["metrics"]))
        result[name] = builder.add(
            "evaluation_case",
            {"definition": definition, "metrics": metrics},
        )
    return result


def _profile_objects(builder: _Builder, document: dict[str, Any]) -> dict[str, str]:
    return {
        "default": builder.add(
            "export_profile", {"definition": deepcopy(document["infer"])}
        )
    }


def _train_spec(
    builder: _Builder, run: RunRecord, revision: str, experiment_revision: str
) -> tuple[str, str]:
    options = split_legacy_variant(run.snapshot["variant"]).options
    train_options = builder.add(
        "option_set",
        {
            "scope": "train",
            **{field: options[field] for field in ("dataset", "metrics", "train")},
        },
    )
    comparison = comparison_digest(
        {
            "variant_revision": revision,
            "train_options": train_options,
            "backend_identity": run.request["identity_fingerprint"],
            "device": run.request["exec"]["device"],
        }
    )
    return (
        builder.add(
            "run_spec",
            {
                "action": "train",
                "experiment_revision": experiment_revision,
                "train_options": train_options,
                "variant_revision": revision,
                "seed": run.request["target"]["seed"],
                "backend_identity": run.request["identity_fingerprint"],
                "device": run.request["exec"]["device"],
                "comparison_hash": comparison,
            },
        ),
        comparison,
    )


def _attempt(
    builder: _Builder,
    run: RunRecord,
    run_spec: str,
    attempt_hashes: dict[str, str],
) -> str:
    retry_of = run.request["retry_of"]
    retry_address = (
        f"{run.request['experiment']}/{run.request['variant']}/{retry_of}"
        if retry_of is not None
        else None
    )
    retry_parent = attempt_hashes.get(retry_address) if retry_address else None
    if retry_address is not None and retry_parent is None:
        raise ContractError(f"migration retry parent is unavailable: {retry_address}")
    options = builder.add(
        "option_set",
        {
            "scope": "full",
            "document": split_legacy_variant(run.snapshot["variant"]).options,
        },
    )
    evidence = builder.blob_bytes(
        canonical_json_bytes(
            {"schema_version": 1, "snapshot": run.snapshot, "request": run.request}
        )
        + b"\n"
    )
    if retry_parent is not None:
        parent = builder.objects[retry_parent].payload
        if parent["run_spec"] != run_spec or parent["option_set"] != options:
            raise ContractError("legacy retry disagrees with parent execution meaning")
    return builder.add(
        "attempt",
        {
            "run_spec": run_spec,
            "option_set": options,
            "record_evidence": evidence,
            "tracker_backends": list(
                validate_tracker(run.snapshot["variant"]["tracker"])
            ),
            "nonce": _legacy_nonce(f"attempt:{run.address}"),
            "retry_parent": retry_parent,
            "created_at": run.request["created_at"],
            "legacy_evidence": {
                "address": run.address,
                "source_digest": run.request["source_digest"],
                "tracker_run_id": run.state["tracker_run_id"],
            },
        },
    )


def _event_payload(
    run: RunRecord,
    attempt: str,
    parent: str | None,
    artifacts: list[dict[str, Any]],
    result_object: str | None,
    *,
    status: str,
) -> dict[str, Any]:
    allocated = status == "allocated"
    return {
        "attempt": attempt,
        "parent": parent,
        "status": status,
        "reason": None if allocated else run.state["reason"],
        "tracker_run_id": None if allocated else run.state["tracker_run_id"],
        "result": None if allocated else deepcopy(run.state["result"]),
        "result_object": None if allocated else result_object,
        "best_checkpoint": None if allocated else run.state["best_checkpoint"],
        "last_checkpoint": None if allocated else run.state["last_checkpoint"],
        "artifacts": artifacts,
        "recorded_at": (
            run.request["created_at"] if allocated else run.state["updated_at"]
        ),
    }


def _artifact_objects(builder: _Builder, run: RunRecord) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for relative_root in ("worker.log", "metrics", "artifacts"):
        root = run.path / relative_root
        if not os.path.lexists(root):
            continue
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in sorted(paths, key=lambda item: item.as_posix().encode("utf-8")):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ContractError(f"migration artifact contains a symlink: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(
                    f"migration artifact contains a non-regular file: {path}"
                )
            blob = builder.blob(path)
            payload = builder.objects[blob].payload
            result.append(
                {
                    "path": PurePosixPath(*path.relative_to(run.path).parts).as_posix(),
                    "blob": blob,
                    "content_hash": payload["content_hash"],
                    "size": payload["size"],
                }
            )
    return result


def _result_object(
    builder: _Builder,
    run: RunRecord,
    attempt: str,
    run_spec: str,
    artifacts: list[dict[str, Any]],
    model_hashes: dict[str, str],
) -> str | None:
    if run.state["status"] != "done":
        return None
    if run.request["action"] == "train":
        address = (
            f"{run.request['experiment']}/{run.request['variant']}/"
            f"{run.state['result']['model_id']}"
        )
        return model_hashes.get(address)
    spec = builder.objects[run_spec].payload
    if run.request["action"] == "eval":
        metrics = run.path / str(run.state["result"]["metrics"])
        try:
            document = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractError(
                f"migration Eval result is invalid: {metrics}"
            ) from error
        return builder.add(
            "eval_result",
            {
                "attempt": attempt,
                "model": spec["model"],
                "evaluation_case": spec["evaluation_case"],
                "document": document,
                "artifacts": artifacts,
            },
        )
    return builder.add(
        "export_result",
        {
            "attempt": attempt,
            "model": spec["model"],
            "export_profile": spec["export_profile"],
            "artifacts": [
                item
                for item in artifacts
                if str(item["path"]).startswith("artifacts/export/")
            ],
        },
    )


def _generated_entries(root: Path, catalog: str):
    if not os.path.lexists(root):
        return []
    result = []
    for experiment in sorted(root.iterdir(), key=lambda item: item.name.encode()):
        if experiment.name.startswith(".") or experiment.name == "index.db":
            continue
        if experiment.is_symlink() or not experiment.is_dir():
            continue
        for variant in sorted(
            experiment.iterdir(), key=lambda item: item.name.encode()
        ):
            if variant.name.startswith(".") or not variant.is_dir():
                continue
            owner = variant / catalog
            if not owner.exists() or owner.is_symlink() or not owner.is_dir():
                continue
            for entry in sorted(owner.iterdir(), key=lambda item: item.name.encode()):
                if entry.name.startswith("."):
                    continue
                result.append((experiment.name, variant.name, entry.name, entry))
    return result


def _authority_digests(
    repository: RepositoryPaths,
    malformed: list[dict[str, str]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for root in (repository.experiments, repository.outputs):
        if not os.path.lexists(root):
            continue
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(repository.root).as_posix()
            if path.name.startswith(".hkdl-index.sqlite3"):
                continue
            try:
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    malformed.append(
                        {"path": relative, "error": "symlink is unsupported"}
                    )
                elif stat.S_ISREG(metadata.st_mode):
                    result[relative] = hash_file(path)[0]
            except OSError as error:
                malformed.append({"path": relative, "error": str(error)})
    return result


def _comparison_binding_transactions(
    train_runs: list[RunRecord],
    run_revisions: dict[str, str],
    comparison_hashes: dict[str, str | None],
) -> list[list[BindingOperation]]:
    active: dict[tuple[str, str], str] = {}
    transactions: list[list[BindingOperation]] = []
    for run in sorted(train_runs, key=_run_order):
        comparison = comparison_hashes.get(run.address)
        if comparison is None:
            continue
        scope = comparison_group_scope(run_revisions[run.address])
        group = str(run.request["target"]["training_group"])
        key = (scope, group)
        previous = active.get(key)
        if previous == comparison:
            continue
        if previous is None:
            batch = [BindingOperation("bind", scope, group, comparison)]
        else:
            batch = [
                BindingOperation("unbind", scope, group, previous),
                BindingOperation("bind", scope, group, comparison),
            ]
        transactions.append(batch)
        active[key] = comparison
    return transactions


def _operations_result(
    transaction_batches: list[list[BindingOperation]],
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for batch in transaction_batches:
        for operation in batch:
            key = (operation.scope, operation.name)
            if operation.action == "bind":
                if key in result:
                    raise ContractError("migration binding plan contains a collision")
                result[key] = operation.target
            else:
                if result.get(key) != operation.target:
                    raise ContractError("migration unbind plan is invalid")
                del result[key]
    return result


def _unique_documents(
    documents: list[dict[str, Any]], *, ignored: set[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for document in documents:
        semantic = {key: value for key, value in document.items() if key not in ignored}
        encoded = canonical_json_bytes(semantic)
        if encoded not in seen:
            result.append(document)
            seen.add(encoded)
    return result


def _matching_authored_source(
    variant: VariantRecord | None, digest: str
) -> Path | None:
    if variant is None:
        return None
    path = variant.path / "src"
    return path if compute_source_digest(path) == digest else None


def _variant_created_at(
    key: tuple[str, str],
    runs: dict[tuple[str, str], list[RunRecord]],
    experiments: dict[str, ExperimentRecord],
) -> str:
    records = sorted(runs.get(key, []), key=_run_order)
    if records:
        return str(records[0].request["created_at"])
    experiment = experiments.get(key[0])
    return (
        str(experiment.document["created_at"])
        if experiment is not None
        else IMPORT_TIMESTAMP
    )


def _run_order(run: RunRecord) -> tuple[bytes, bytes, int]:
    return (
        str(run.request["experiment"]).encode("utf-8"),
        str(run.request["variant"]).encode("utf-8"),
        int(str(run.request["run_id"]).removeprefix("run-")),
    )


def _legacy_nonce(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _additional_bytes(store: ObjectStore, builder: _Builder) -> int:
    total = 0
    for digest, record in builder.objects.items():
        if not os.path.lexists(store.object_path(digest)):
            total += len(canonical_json_bytes(record.envelope)) + 1
    for digest, blob in builder.blobs.items():
        if not os.path.lexists(store.blob_path(digest)):
            total += blob.size
    return total
