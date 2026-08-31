"""Variant-managed action Runs and immutable Model records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .authoring import ExperimentRecord, VariantRecord
from .config import (
    ContractError,
    dump_yaml,
    load_yaml_file,
)
from .run_contracts import (
    ACTIONS,
    EVAL_COMPONENTS,
    EXPORT_COMPONENTS,
    MAX_SEED,
    MODEL_ID_PATTERN,
    RUN_ID_PATTERN,
    RUN_STATUSES,
    TRACKER_ID_PATTERN,  # noqa: F401 - legacy direct import compatibility
    TERMINAL_STATUSES,
    TRAIN_COMPONENTS,
    evaluation_case,
    fingerprint_document,
    metric_spec,
    validate_evaluation,
    validate_evaluation_readiness,
    validate_export_readiness,
    validate_model,
    validate_request,
    validate_snapshot,
    validate_state,
    validate_tracker,
    validate_training_readiness,
    _identity,
    _model_id,
    _run_id,
)
from .research_json import dump_json, split_legacy_experiment, split_legacy_variant
from .v2.maintenance import workspace_operation
from .v2.leases import authoring_write
from .storage import (
    AlreadyExistsError,
    NotFoundError,
    OwnershipError,
    RepositoryPaths,
    atomic_replace,
    atomic_write_new,
    directory_lock,
)


@dataclass(frozen=True)
class RunRecord:
    path: Path
    address: str
    snapshot: dict[str, Any]
    request: dict[str, Any]
    state: dict[str, Any]
    graph_identity: dict[str, Any] | None = None
    event_hash: str | None = None


@dataclass(frozen=True)
class ModelRecord:
    path: Path
    address: str
    document: dict[str, Any]
    graph_hash: str | None = None


class RunStore:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
        nonce: Callable[[int], bytes] | None = None,
        graph_reads: bool = True,
    ):
        self.repository = repository
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._nonce = nonce or secrets.token_bytes
        self.graph_reads = graph_reads

    def graph_reader(self):
        from .v2.reader import current_reader

        return current_reader(self.repository)

    def run_lease(self, record: RunRecord):
        if _v2_is_active(self.repository):
            from .v2.execution import GraphRecorder
            from .v2.leases import attempt_lease

            identity = GraphRecorder(self.repository).identity(record)
            return attempt_lease(self.repository, identity.attempt_hash, record.path)
        from .storage import try_directory_lock

        return try_directory_lock(record.path)

    def freeze(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        source_digest: str,
    ) -> dict[str, Any]:
        snapshot = {
            "schema_version": 1,
            "experiment": deepcopy(experiment.document),
            "variant": deepcopy(variant.document),
            "provenance": {
                "variant_file": f"{variant.document['name']}/variant.yaml",
                "source_digest": source_digest,
                "vcs_revision": None,
                "frozen_at": _utc_timestamp(self._now()),
            },
        }
        validate_snapshot(snapshot)
        return snapshot

    @workspace_operation
    @authoring_write
    def allocate(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        *,
        action: str,
        target: Mapping[str, Any],
        exec_info: Mapping[str, Any],
        source_digest: str,
        identity_fingerprint: str,
        retry_of: str | None = None,
        snapshot: dict[str, Any] | None = None,
        catalog_validator: (
            Callable[[list[RunRecord], list[ModelRecord]], None] | None
        ) = None,
    ) -> RunRecord:
        if _v2_is_active(self.repository) and retry_of is None:
            from .v2.graph import V2Graph

            graph = V2Graph(self.repository)
            graph.assert_experiment_clean(str(experiment.document["name"]), experiment)
            graph.assert_variant_clean(str(experiment.document["name"]), variant)
        root = self.variant_root(
            experiment.document["name"],
            variant.document["name"],
            create=True,
        )
        runs = root / "runs"
        _ensure_directory(runs)
        with directory_lock(root):
            if catalog_validator is not None:
                catalog_validator(
                    self.scan(
                        experiment=experiment.document["name"],
                        variant=variant.document["name"],
                    ),
                    self.scan_models(
                        experiment=experiment.document["name"],
                        variant=variant.document["name"],
                    ),
                )
            run_id = f"run-{self._next_run_number(experiment, variant, runs):03d}"
            timestamp = _utc_timestamp(self._now())
            if snapshot is None:
                snapshot = self.freeze(experiment, variant, source_digest)
            request = {
                "schema_version": 1,
                "run_id": run_id,
                "experiment": experiment.document["name"],
                "variant": variant.document["name"],
                "action": action,
                "target": deepcopy(dict(target)),
                "retry_of": retry_of,
                "source_digest": source_digest,
                "identity_fingerprint": identity_fingerprint,
                "exec": deepcopy(dict(exec_info)),
                "created_at": timestamp,
            }
            state = {
                "schema_version": 1,
                "run_id": run_id,
                "action": action,
                "status": "allocated",
                "reason": None,
                "tracker_run_id": None,
                "result": None,
                "best_checkpoint": None,
                "last_checkpoint": None,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            validate_snapshot(snapshot)
            validate_request(request)
            validate_state(state)
            candidate = Path(tempfile.mkdtemp(prefix=f".{run_id}.candidate-", dir=runs))
            try:
                (candidate / "metrics").mkdir()
                (candidate / "artifacts/checkpoints").mkdir(parents=True)
                atomic_write_new(candidate / "snapshot.yaml", dump_yaml(snapshot))
                if _v2_is_active(self.repository):
                    readable = candidate / "snapshot"
                    readable.mkdir()
                    split = split_legacy_variant(snapshot["variant"])
                    atomic_write_new(
                        readable / "experiment.json",
                        dump_json(split_legacy_experiment(snapshot["experiment"])),
                    )
                    atomic_write_new(readable / "code.json", dump_json(split.code))
                    atomic_write_new(
                        readable / "options.json", dump_json(split.options)
                    )
                atomic_write_new(candidate / "request.json", _json_text(request))
                atomic_write_new(candidate / "state.json", _json_text(state))
                target_path = runs / run_id
                if os.path.lexists(target_path):
                    raise AlreadyExistsError(f"Run already exists: {run_id}")
                os.rename(candidate, target_path)
                _fsync_directory(runs)
                record = RunRecord(
                    target_path,
                    f"{request['experiment']}/{request['variant']}/{run_id}",
                    snapshot,
                    request,
                    state,
                )
                _record_v2_allocation(self.repository, record)
                return (
                    self.load(request["experiment"], request["variant"], run_id)
                    if _v2_is_active(self.repository)
                    else record
                )
            finally:
                if candidate.exists():
                    shutil.rmtree(candidate)

    @workspace_operation
    def update_state(self, record: RunRecord, **changes: Any) -> RunRecord:
        if record.state["status"] in TERMINAL_STATUSES:
            raise ContractError(f"terminal Run is sealed: {record.address}")
        state = dict(record.state)
        state.update(changes)
        state["updated_at"] = _utc_timestamp(self._now())
        validate_state(state)
        atomic_replace(record.path / "state.json", _json_text(state))
        updated = RunRecord(
            record.path,
            record.address,
            record.snapshot,
            record.request,
            state,
        )
        _record_v2_state(self.repository, updated)
        return (
            self.load(
                record.request["experiment"],
                record.request["variant"],
                record.request["run_id"],
            )
            if _v2_is_active(self.repository)
            else updated
        )

    @workspace_operation
    def load(self, experiment: str, variant: str, run_id: str) -> RunRecord:
        _identity(experiment, "Experiment")
        _identity(variant, "Variant")
        _run_id(run_id)
        if self.graph_reads and _v2_is_active(self.repository):
            return self.graph_reader().run(experiment, variant, run_id)
        projection_names = _v2_projection_names(
            self.repository,
            experiment,
            variant,
        )
        if projection_names is not None:
            for physical_experiment in _v2_experiment_projection_names(
                self.repository, experiment
            ):
                for projection_name in projection_names:
                    try:
                        record = self._load_run_physical(
                            physical_experiment,
                            projection_name,
                            run_id,
                        )
                    except NotFoundError:
                        continue
                    return _v2_readdress_run(
                        self.repository, record, variant, experiment
                    )
            raise NotFoundError(f"Run not found: {experiment}/{variant}/{run_id}")
        return self._load_run_physical(experiment, variant, run_id)

    def _load_run_physical(
        self,
        experiment: str,
        variant: str,
        run_id: str,
    ) -> RunRecord:
        root = self.variant_root(experiment, variant, create=False)
        path = root / "runs" / run_id
        if not os.path.lexists(path):
            raise NotFoundError(f"Run not found: {experiment}/{variant}/{run_id}")
        _existing_directory(path)
        snapshot = load_yaml_file(path / "snapshot.yaml")
        request = _load_json(path / "request.json")
        state = _load_json(path / "state.json")
        validate_snapshot(snapshot)
        validate_request(request)
        validate_state(state)
        address = f"{experiment}/{variant}/{run_id}"
        if (
            request["run_id"] != run_id
            or request["experiment"] != experiment
            or request["variant"] != variant
            or state["run_id"] != run_id
            or state["action"] != request["action"]
            or snapshot["experiment"]["name"] != experiment
            or snapshot["variant"]["name"] != variant
            or snapshot["provenance"]["source_digest"] != request["source_digest"]
        ):
            raise OwnershipError(f"Run ownership mismatch: {address}")
        return RunRecord(path, address, snapshot, request, state)

    @workspace_operation
    def scan(
        self,
        *,
        experiment: str | None = None,
        variant: str | None = None,
    ) -> list[RunRecord]:
        if experiment is not None:
            _identity(experiment, "Experiment")
        if variant is not None:
            _identity(variant, "Variant")
        if self.graph_reads and _v2_is_active(self.repository):
            return self.graph_reader().runs(experiment=experiment, variant=variant)
        outputs = self.repository.outputs
        if not os.path.lexists(outputs):
            return []
        _existing_directory(outputs)
        if _v2_is_active(self.repository):
            return self._scan_v2(experiment=experiment, variant=variant)
        records: list[RunRecord] = []
        for experiment_entry in _catalog_directories(outputs, ignore={"index.db"}):
            _identity(experiment_entry.name, "Run Experiment")
            if experiment is not None and experiment_entry.name != experiment:
                continue
            for variant_entry in _catalog_directories(experiment_entry):
                _identity(variant_entry.name, "Run Variant")
                if variant is not None and variant_entry.name != variant:
                    continue
                self._reject_legacy_layout(variant_entry)
                runs = variant_entry / "runs"
                if not os.path.lexists(runs):
                    continue
                _existing_directory(runs)
                for run_entry in _catalog_directories(runs):
                    _run_id(run_entry.name)
                    records.append(
                        self.load(
                            experiment_entry.name,
                            variant_entry.name,
                            run_entry.name,
                        )
                    )
        return sorted(records, key=_run_sort_key)

    def _scan_v2(
        self,
        *,
        experiment: str | None,
        variant: str | None,
    ) -> list[RunRecord]:
        if experiment is not None and variant is not None:
            _v2_projection_names(self.repository, experiment, variant)
        records: list[RunRecord] = []
        addresses: set[str] = set()
        for experiment_entry in _catalog_directories(
            self.repository.outputs, ignore={"index.db"}
        ):
            _identity(experiment_entry.name, "Run Experiment")
            current_experiment = _v2_current_experiment_name(
                self.repository, experiment_entry.name
            )
            if current_experiment is None or (
                experiment is not None and current_experiment != experiment
            ):
                continue
            for variant_entry in _catalog_directories(experiment_entry):
                _identity(variant_entry.name, "Run Variant")
                current_name = _v2_current_variant_name(
                    self.repository,
                    current_experiment,
                    variant_entry.name,
                )
                if current_name is None or (
                    variant is not None and current_name != variant
                ):
                    continue
                self._reject_legacy_layout(variant_entry)
                runs = variant_entry / "runs"
                if not os.path.lexists(runs):
                    continue
                _existing_directory(runs)
                for run_entry in _catalog_directories(runs):
                    _run_id(run_entry.name)
                    physical = self._load_run_physical(
                        experiment_entry.name,
                        variant_entry.name,
                        run_entry.name,
                    )
                    record = _v2_readdress_run(
                        self.repository,
                        physical,
                        current_name,
                        current_experiment,
                    )
                    if record.address in addresses:
                        raise ContractError(
                            f"duplicate v2 Run projection: {record.address}"
                        )
                    addresses.add(record.address)
                    records.append(record)
        return sorted(records, key=_run_sort_key)

    def allocate_model(
        self,
        record: RunRecord,
        *,
        checkpoint: str,
        checkpoint_digest: str,
    ) -> ModelRecord:
        if record.request["action"] != "train":
            raise ContractError("only a Train Run can produce a Model")
        target = record.request["target"]
        root = self.variant_root(
            record.request["experiment"],
            record.request["variant"],
            create=True,
        )
        models = root / "models"
        _ensure_directory(models)
        with directory_lock(root):
            for existing in self.scan_models(
                experiment=record.request["experiment"],
                variant=record.request["variant"],
            ):
                if (
                    existing.document["training_group"] == target["training_group"]
                    and existing.document["seed"] == target["seed"]
                ):
                    raise AlreadyExistsError(
                        "Model already exists for Training Group and seed"
                    )
            timestamp = _utc_timestamp(self._now())
            model_id = self._new_model_id(
                record.request["experiment"],
                record.request["variant"],
                models,
            )
            document = {
                "schema_version": 1,
                "model_id": model_id,
                "experiment": record.request["experiment"],
                "variant": record.request["variant"],
                "training_group": target["training_group"],
                "seed": target["seed"],
                "device": record.request["exec"]["device"],
                "training_fingerprint": record.request["identity_fingerprint"],
                "producer_run": record.request["run_id"],
                "checkpoint": {
                    "path": f"runs/{record.request['run_id']}/{checkpoint}",
                    "digest": checkpoint_digest,
                },
                "created_at": timestamp,
            }
            validate_model(document)
            candidate = Path(
                tempfile.mkdtemp(prefix=f".{model_id}.candidate-", dir=models)
            )
            try:
                atomic_write_new(candidate / "model.json", _json_text(document))
                target_path = models / model_id
                if os.path.lexists(target_path):
                    raise AlreadyExistsError(f"Model already exists: {model_id}")
                os.rename(candidate, target_path)
                _fsync_directory(models)
                model = ModelRecord(
                    target_path,
                    f"{document['experiment']}/{document['variant']}/{model_id}",
                    document,
                )
                _record_v2_model(self.repository, record, model)
                return (
                    self.load_model_manifest(
                        document["experiment"], document["variant"], model_id
                    )
                    if _v2_is_active(self.repository)
                    else model
                )
            finally:
                if candidate.exists():
                    shutil.rmtree(candidate)

    @workspace_operation
    def load_model_manifest(
        self,
        experiment: str,
        variant: str,
        model_id: str,
    ) -> ModelRecord:
        _identity(experiment, "Experiment")
        _identity(variant, "Variant")
        _model_id(model_id)
        if self.graph_reads and _v2_is_active(self.repository):
            return self.graph_reader().model(experiment, variant, model_id)
        projection_names = _v2_projection_names(
            self.repository,
            experiment,
            variant,
        )
        if projection_names is not None:
            for physical_experiment in _v2_experiment_projection_names(
                self.repository, experiment
            ):
                for projection_name in projection_names:
                    try:
                        model = self._load_model_manifest_physical(
                            physical_experiment,
                            projection_name,
                            model_id,
                        )
                    except NotFoundError:
                        continue
                    return _v2_readdress_model(
                        self.repository, model, variant, experiment
                    )
            raise NotFoundError(f"Model not found: {experiment}/{variant}/{model_id}")
        return self._load_model_manifest_physical(experiment, variant, model_id)

    def _load_model_manifest_physical(
        self,
        experiment: str,
        variant: str,
        model_id: str,
    ) -> ModelRecord:
        root = self.variant_root(experiment, variant, create=False)
        path = root / "models" / model_id
        if not os.path.lexists(path):
            raise NotFoundError(f"Model not found: {experiment}/{variant}/{model_id}")
        _existing_directory(path)
        document = _load_json(path / "model.json")
        validate_model(document)
        if (
            document["model_id"] != model_id
            or document["experiment"] != experiment
            or document["variant"] != variant
        ):
            raise OwnershipError(
                f"Model ownership mismatch: {experiment}/{variant}/{model_id}"
            )
        return ModelRecord(path, f"{experiment}/{variant}/{model_id}", document)

    def load_model(self, experiment: str, variant: str, model_id: str) -> ModelRecord:
        model = self.load_model_manifest(experiment, variant, model_id)
        checkpoint = self.resolve_model_checkpoint(model)
        if _file_digest(checkpoint) != model.document["checkpoint"]["digest"]:
            raise ContractError("Model checkpoint digest changed")
        return model

    def _load_model_physical(
        self,
        experiment: str,
        variant: str,
        model_id: str,
    ) -> ModelRecord:
        model = self._load_model_manifest_physical(experiment, variant, model_id)
        checkpoint = self.resolve_model_checkpoint(model)
        if _file_digest(checkpoint) != model.document["checkpoint"]["digest"]:
            raise ContractError("Model checkpoint digest changed")
        return model

    @workspace_operation
    def scan_model_manifests(
        self,
        *,
        experiment: str,
        variant: str,
    ) -> list[ModelRecord]:
        return self._scan_model_catalog(
            experiment=experiment,
            variant=variant,
            load=self.load_model_manifest,
        )

    @workspace_operation
    def scan_models(
        self,
        *,
        experiment: str,
        variant: str,
    ) -> list[ModelRecord]:
        return self._scan_model_catalog(
            experiment=experiment,
            variant=variant,
            load=self.load_model,
        )

    def _scan_model_catalog(
        self,
        *,
        experiment: str,
        variant: str,
        load: Callable[[str, str, str], ModelRecord],
    ) -> list[ModelRecord]:
        if self.graph_reads and _v2_is_active(self.repository):
            models = self.graph_reader().models(experiment, variant)
            if load.__name__ == "load_model":
                for model in models:
                    self.resolve_model_checkpoint(model)
            return models
        result: list[ModelRecord] = []
        projection_names = _v2_projection_names(
            self.repository,
            experiment,
            variant,
        )
        names = projection_names if projection_names is not None else (variant,)
        seen: set[str] = set()
        physical_experiments = (
            _v2_experiment_projection_names(self.repository, experiment)
            if projection_names is not None
            else (experiment,)
        )
        for physical_experiment in physical_experiments:
            for projection_name in names:
                try:
                    root = self.variant_root(
                        physical_experiment, projection_name, create=False
                    )
                except NotFoundError:
                    continue
                models = root / "models"
                if not os.path.lexists(models):
                    continue
                _existing_directory(models)
                for entry in _catalog_directories(models):
                    _model_id(entry.name)
                    if entry.name in seen:
                        raise ContractError(
                            f"duplicate v2 Model projection: "
                            f"{experiment}/{variant}/{entry.name}"
                        )
                    seen.add(entry.name)
                    physical = (
                        self._load_model_manifest_physical(
                            physical_experiment,
                            projection_name,
                            entry.name,
                        )
                        if load.__name__ == "load_model_manifest"
                        else self._load_model_physical(
                            physical_experiment,
                            projection_name,
                            entry.name,
                        )
                    )
                    result.append(
                        _v2_readdress_model(
                            self.repository, physical, variant, experiment
                        )
                        if projection_names is not None
                        else physical
                    )
        return sorted(
            result,
            key=lambda item: (
                item.document["created_at"],
                item.document["model_id"].encode("utf-8"),
            ),
        )

    def resolve_model_checkpoint(self, model: ModelRecord) -> Path:
        if model.graph_hash is not None:
            reader = self.graph_reader()
            payload = reader.object(model.graph_hash, "model")
            blob = reader.object(payload["checkpoint_blob"], "blob")
            if blob["content_hash"] != model.document["checkpoint"]["digest"]:
                raise ContractError("v2 Model checkpoint identity mismatch")
            return reader.graph.store.verify_blob(payload["checkpoint_blob"])
        root = model.path.parents[1]
        relative = PurePosixPath(model.document["checkpoint"]["path"])
        path = root.joinpath(*relative.parts)
        _contained_regular_file(root, path, "Model checkpoint")
        return path

    def resolve_worker_log(self, record: RunRecord) -> Path:
        if (
            record.event_hash is not None
            and record.state["status"] in TERMINAL_STATUSES
        ):
            return self.graph_reader().artifact(record, "worker.log")
        path = record.path / "worker.log"
        if not os.path.lexists(path):
            raise NotFoundError(f"Run log not found: {record.address}")
        _contained_regular_file(record.path, path, "Run worker log")
        return path

    def evaluation_document(
        self,
        record: RunRecord,
        *,
        values: Mapping[str, Any],
        artifacts: Sequence[str],
    ) -> dict[str, Any]:
        case = record.request["target"]["evaluation_case"]
        metrics = metric_spec(record.snapshot["variant"], case)
        document = {
            "schema_version": 1,
            "model_id": record.request["target"]["model_id"],
            "evaluation_case": case,
            "case_fingerprint": record.request["identity_fingerprint"],
            "primary": {
                "name": metrics["primary"],
                "value": values.get(metrics["primary"]),
            },
            "values": dict(values),
            "artifacts": list(artifacts),
            "evaluated_at": _utc_timestamp(self._now()),
        }
        validate_evaluation(document, record.snapshot, record.request)
        return document

    def load_evaluation(self, record: RunRecord) -> dict[str, Any]:
        if record.request["action"] != "eval":
            raise ContractError("Run is not an evaluation")
        if record.event_hash is not None:
            if record.state["status"] != "done":
                raise ContractError("Eval Run has no completed result")
            return self.graph_reader().evaluation(record)
        path = record.path / "metrics/eval.json"
        if not os.path.lexists(path):
            raise ContractError(f"completed Eval Run has no metrics: {record.address}")
        document = _load_json(path)
        validate_evaluation(document, record.snapshot, record.request)
        return document

    def load_training_metric_summary(
        self,
        record: RunRecord,
        *,
        required: bool | None = None,
    ) -> dict[str, Any]:
        if record.request["action"] != "train":
            return {}
        if "local" not in validate_tracker(record.snapshot["variant"]["tracker"]):
            return {}
        if required is None:
            required = record.state["status"] == "done"
        if (
            record.event_hash is not None
            and record.state["status"] in TERMINAL_STATUSES
        ):
            try:
                path = self.graph_reader().artifact(
                    record, "metrics/train-summary.json"
                )
            except NotFoundError:
                if required:
                    raise ContractError(
                        f"completed Train Run has no metric summary: {record.address}"
                    ) from None
                return {}
            summary = _load_json(path)
            _validate_training_metric_summary(summary)
            return deepcopy(summary["metrics"])
        path = record.path / "metrics/train-summary.json"
        if not os.path.lexists(path):
            if required:
                raise ContractError(
                    f"completed Train Run has no metric summary: {record.address}"
                )
            return {}
        _contained_regular_file(record.path, path, "Train metric summary")
        summary = _load_json(path)
        _validate_training_metric_summary(summary)
        return deepcopy(summary["metrics"])

    def load_training_metrics(self, record: RunRecord) -> dict[str, Any]:
        if (
            record.event_hash is not None
            and record.state["status"] in TERMINAL_STATUSES
        ):
            from .v2.execution import GraphRecorder

            return GraphRecorder(self.repository).load_training_metrics(record)
        chunk = self.load_training_metric_chunk(record, offset=0)
        events = chunk["events"]
        partial = chunk["partial"]
        summary_document = _training_metric_summary(events)
        summary_path = record.path / "metrics/train-summary.json"
        if os.path.lexists(summary_path):
            _contained_regular_file(record.path, summary_path, "Train metric summary")
            persisted = _load_json(summary_path)
            _validate_training_metric_summary(persisted)
            if persisted != summary_document:
                raise ContractError("Train metric summary disagrees with history")
        elif record.state["status"] == "done":
            raise ContractError(
                f"completed Train Run has no metric summary: {record.address}"
            )
        if record.state["status"] == "done" and partial:
            raise ContractError("completed Train metric history is partial")
        return {
            "events": events,
            "partial": partial,
            "summary": summary_document["metrics"],
        }

    def load_training_metric_chunk(
        self,
        record: RunRecord,
        *,
        offset: int,
    ) -> dict[str, Any]:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ContractError("Train metric history offset is invalid")
        if record.request["action"] != "train":
            raise ContractError("Run is not a Train Run")
        if "local" not in validate_tracker(record.snapshot["variant"]["tracker"]):
            raise ContractError("Train Run does not use local tracking")
        history_path = record.path / "metrics/train.jsonl"
        captured = (
            record.event_hash is not None
            and record.state["status"] in TERMINAL_STATUSES
        )
        if captured:
            try:
                history_path = self.graph_reader().artifact(
                    record, "metrics/train.jsonl"
                )
            except NotFoundError:
                if offset or record.state["status"] == "done":
                    raise ContractError("Train metric history is unavailable") from None
                return {"events": [], "offset": 0, "partial": False}
        if not os.path.lexists(history_path):
            if offset:
                raise ContractError("Train metric history disappeared while following")
            if record.state["status"] == "done":
                raise ContractError(
                    f"completed Train Run has no metric history: {record.address}"
                )
            return {"events": [], "offset": 0, "partial": False}
        if not captured:
            _contained_regular_file(record.path, history_path, "Train metric history")
        events: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        partial = False
        next_offset = offset
        try:
            with history_path.open("rb") as handle:
                if os.fstat(handle.fileno()).st_size < offset:
                    raise ContractError("Train metric history was truncated")
                handle.seek(offset)
                for line in handle:
                    if not line.endswith(b"\n"):
                        partial = True
                        break
                    row = _load_training_metric_event(line)
                    key = (row["name"], row["step"])
                    if key in seen:
                        raise ContractError(
                            "Train metric history contains a duplicate step"
                        )
                    seen.add(key)
                    events.append(row)
                    next_offset = handle.tell()
        except OSError as error:
            raise ContractError("Train metric history is unavailable") from error
        return {
            "events": events,
            "offset": next_offset,
            "partial": partial,
        }

    def validate_completed_training_metrics(
        self,
        record: RunRecord,
        result: Any,
    ) -> dict[str, str]:
        uses_local = "local" in validate_tracker(record.snapshot["variant"]["tracker"])
        if not uses_local:
            if result is not None:
                raise ContractError("Train worker returned unexpected metric files")
            return {}
        if not isinstance(result, dict) or result != {
            "history": "metrics/train.jsonl",
            "summary": "metrics/train-summary.json",
        }:
            raise ContractError("Train worker metric files are invalid")
        metrics = self.load_training_metrics(record)
        if metrics["partial"]:
            raise ContractError("completed Train metric history is partial")
        return {
            relative: _file_digest(record.path / relative)
            for relative in result.values()
        }

    def direct_retry(self, record: RunRecord) -> RunRecord | None:
        matches = [
            candidate
            for candidate in self.scan(
                experiment=record.request["experiment"],
                variant=record.request["variant"],
            )
            if candidate.request["retry_of"] == record.request["run_id"]
        ]
        if len(matches) > 1:
            raise ContractError("Run has multiple direct retries")
        return matches[0] if matches else None

    def variant_root(
        self,
        experiment: str,
        variant: str,
        *,
        create: bool,
    ) -> Path:
        _identity(experiment, "Experiment")
        _identity(variant, "Variant")
        outputs = self.repository.outputs
        if create:
            _ensure_directory(outputs)
            experiment_root = outputs / experiment
            _ensure_directory(experiment_root)
            root = experiment_root / variant
            _ensure_directory(root)
        else:
            root = outputs / experiment / variant
            if not os.path.lexists(root):
                raise NotFoundError(f"Variant output not found: {experiment}/{variant}")
            _existing_directory(root)
        self._reject_legacy_layout(root)
        return root

    @staticmethod
    def json_text(document: Mapping[str, Any]) -> str:
        return _json_text(document)

    def _next_run_number(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        runs: Path,
    ) -> int:
        projected = _v2_next_run_number(
            self.repository,
            str(experiment.document["name"]),
            str(variant.document["name"]),
        )
        if projected is not None:
            return projected
        maximum = 0
        for entry in os.scandir(runs):
            if entry.name.startswith("."):
                continue
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise ContractError(f"invalid Run catalog entry: {entry.name}")
            _run_id(entry.name)
            maximum = max(maximum, int(entry.name.removeprefix("run-")))
        return maximum + 1

    def _new_model_id(self, experiment: str, variant: str, models: Path) -> str:
        for _ in range(16):
            timestamp = str(
                self._now().timestamp_ns()
                if hasattr(self._now(), "timestamp_ns")
                else int(self._now().timestamp() * 1_000_000_000)
            )
            payload = b"\0".join(
                (
                    experiment.encode("utf-8"),
                    variant.encode("utf-8"),
                    timestamp.encode("ascii"),
                    self._nonce(16),
                )
            )
            model_id = f"model-{hashlib.sha256(payload).hexdigest()[:32]}"
            if not os.path.lexists(models / model_id):
                return model_id
        raise AlreadyExistsError("unable to allocate a unique Model ID")

    @staticmethod
    def _reject_legacy_layout(root: Path) -> None:
        if not os.path.lexists(root):
            return
        for entry in os.scandir(root):
            if RUN_ID_PATTERN.fullmatch(entry.name):
                raise ContractError(
                    f"legacy pipeline Run layout is unsupported: {entry.path}"
                )


def _run_sort_key(record: RunRecord) -> tuple[Any, ...]:
    return (
        record.request["experiment"].encode("utf-8"),
        record.request["variant"].encode("utf-8"),
        int(record.request["run_id"].removeprefix("run-")),
        record.request["run_id"].encode("utf-8"),
    )


def _json_text(document: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def _load_json(path: Path) -> dict[str, Any]:
    _existing_regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON document must be a mapping: {path}")
    return value


def _load_training_metric_event(line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("Train metric history contains invalid JSON") from error
    _validate_training_metric_event(value)
    return value


def _validate_training_metric_event(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "name",
        "step",
        "value",
    }:
        raise ContractError("Train metric event fields are invalid")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != 1
    ):
        raise ContractError("Train metric event schema version is invalid")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ContractError("Train metric name is invalid")
    if (
        isinstance(value["step"], bool)
        or not isinstance(value["step"], int)
        or value["step"] < 0
    ):
        raise ContractError("Train metric step is invalid")
    if (
        isinstance(value["value"], bool)
        or not isinstance(value["value"], (int, float))
        or not math.isfinite(float(value["value"]))
    ):
        raise ContractError("Train metric value is invalid")


def _training_metric_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    last: dict[str, tuple[int, Any]] = {}
    for event in events:
        name = event["name"]
        counts[name] = counts.get(name, 0) + 1
        last[name] = (event["step"], event["value"])
    return {
        "schema_version": 1,
        "events": len(events),
        "metrics": {
            name: {
                "count": counts[name],
                "last_step": last[name][0],
                "last_value": last[name][1],
            }
            for name in sorted(counts, key=lambda item: item.encode("utf-8"))
        },
    }


def _validate_training_metric_summary(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "events",
        "metrics",
    }:
        raise ContractError("Train metric summary fields are invalid")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != 1
    ):
        raise ContractError("Train metric summary schema version is invalid")
    if (
        isinstance(value["events"], bool)
        or not isinstance(value["events"], int)
        or value["events"] < 0
        or not isinstance(value["metrics"], dict)
    ):
        raise ContractError("Train metric summary is invalid")
    total = 0
    for name, metric in value["metrics"].items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(metric, dict)
            or set(metric) != {"count", "last_step", "last_value"}
        ):
            raise ContractError("Train metric summary entry is invalid")
        if (
            isinstance(metric["count"], bool)
            or not isinstance(metric["count"], int)
            or metric["count"] < 1
            or isinstance(metric["last_step"], bool)
            or not isinstance(metric["last_step"], int)
            or metric["last_step"] < 0
            or isinstance(metric["last_value"], bool)
            or not isinstance(metric["last_value"], (int, float))
            or not math.isfinite(float(metric["last_value"]))
        ):
            raise ContractError("Train metric summary entry is invalid")
        total += metric["count"]
    if total != value["events"]:
        raise ContractError("Train metric summary event count is invalid")


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o755)
    except FileExistsError:
        pass
    _existing_directory(path)


def _existing_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"required directory is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise ContractError(f"required directory is invalid: {path}")


def _existing_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"required file is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ContractError(f"required file is invalid: {path}")


def _contained_regular_file(root: Path, path: Path, location: str) -> None:
    try:
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ContractError(f"{location} is outside its owner") from error
    _existing_regular_file(path)


def _catalog_directories(path: Path, *, ignore: set[str] | None = None) -> list[Path]:
    ignored = ignore or set()
    result: list[Path] = []
    for entry in os.scandir(path):
        if entry.name.startswith(".") or entry.name in ignored:
            continue
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ContractError(f"invalid catalog entry: {entry.path}")
        result.append(Path(entry.path))
    return sorted(result, key=lambda item: item.name.encode("utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_digest(path: Path) -> str:
    _existing_regular_file(path)
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _record_v2_allocation(repository: RepositoryPaths, record: RunRecord) -> None:
    from .v2.execution import GraphRecorder

    recorder = GraphRecorder(repository)
    if recorder.active():
        recorder.record_allocation(record)


def _record_v2_state(repository: RepositoryPaths, record: RunRecord) -> None:
    from .v2.execution import GraphRecorder

    recorder = GraphRecorder(repository)
    if recorder.active():
        recorder.record_state(record)


def _record_v2_model(
    repository: RepositoryPaths,
    record: RunRecord,
    model: ModelRecord,
) -> None:
    from .v2.execution import GraphRecorder

    recorder = GraphRecorder(repository)
    if recorder.active():
        recorder.record_model(record, model)


def _v2_is_active(repository: RepositoryPaths) -> bool:
    from .v2.execution import GraphRecorder

    return GraphRecorder(repository).active()


def _v2_projection_names(
    repository: RepositoryPaths,
    experiment: str,
    variant: str,
) -> tuple[str, ...] | None:
    from .v2.execution import GraphRecorder

    recorder = GraphRecorder(repository)
    if not recorder.active():
        return None
    return recorder.projection_variant_names(experiment, variant)


def _v2_experiment_projection_names(
    repository: RepositoryPaths,
    experiment: str,
) -> tuple[str, ...]:
    from .v2.execution import GraphRecorder

    return GraphRecorder(repository).projection_experiment_names(experiment)


def _v2_current_experiment_name(
    repository: RepositoryPaths,
    projection_name: str,
) -> str | None:
    from .v2.execution import GraphRecorder

    recorder = GraphRecorder(repository)
    if not recorder.active():
        return projection_name
    return recorder.current_experiment_name_for_projection(projection_name)


def _v2_current_variant_name(
    repository: RepositoryPaths,
    experiment: str,
    projection_name: str,
) -> str | None:
    from .v2.execution import GraphRecorder

    recorder = GraphRecorder(repository)
    if not recorder.active():
        return projection_name
    return recorder.current_variant_name_for_projection(experiment, projection_name)


def _v2_readdress_run(
    repository: RepositoryPaths,
    record: RunRecord,
    variant: str,
    experiment: str | None = None,
) -> RunRecord:
    from .v2.execution import GraphRecorder

    return GraphRecorder(repository).readdress_run(record, variant, experiment)


def _v2_readdress_model(
    repository: RepositoryPaths,
    model: ModelRecord,
    variant: str,
    experiment: str | None = None,
) -> ModelRecord:
    from .v2.execution import GraphRecorder

    return GraphRecorder(repository).readdress_model(model, variant, experiment)


def _v2_next_run_number(
    repository: RepositoryPaths,
    experiment: str,
    variant: str,
) -> int | None:
    from .v2.execution import GraphRecorder

    recorder = GraphRecorder(repository)
    if not recorder.active():
        return None
    return recorder.next_run_number(experiment, variant)


__all__ = [
    "ACTIONS",
    "EVAL_COMPONENTS",
    "EXPORT_COMPONENTS",
    "MAX_SEED",
    "MODEL_ID_PATTERN",
    "RUN_ID_PATTERN",
    "RUN_STATUSES",
    "TERMINAL_STATUSES",
    "TRAIN_COMPONENTS",
    "ModelRecord",
    "RunRecord",
    "RunStore",
    "evaluation_case",
    "fingerprint_document",
    "metric_spec",
    "validate_evaluation",
    "validate_evaluation_readiness",
    "validate_export_readiness",
    "validate_model",
    "validate_request",
    "validate_snapshot",
    "validate_state",
    "validate_training_readiness",
]
