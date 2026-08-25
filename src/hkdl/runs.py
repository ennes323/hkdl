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


@dataclass(frozen=True)
class ModelRecord:
    path: Path
    address: str
    document: dict[str, Any]


class RunStore:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
        nonce: Callable[[int], bytes] | None = None,
    ):
        self.repository = repository
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._nonce = nonce or secrets.token_bytes

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
            run_id = f"run-{self._next_run_number(runs):03d}"
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
                atomic_write_new(candidate / "request.json", _json_text(request))
                atomic_write_new(candidate / "state.json", _json_text(state))
                target_path = runs / run_id
                if os.path.lexists(target_path):
                    raise AlreadyExistsError(f"Run already exists: {run_id}")
                os.rename(candidate, target_path)
                _fsync_directory(runs)
                return RunRecord(
                    target_path,
                    f"{request['experiment']}/{request['variant']}/{run_id}",
                    snapshot,
                    request,
                    state,
                )
            finally:
                if candidate.exists():
                    shutil.rmtree(candidate)

    def update_state(self, record: RunRecord, **changes: Any) -> RunRecord:
        if record.state["status"] in TERMINAL_STATUSES:
            raise ContractError(f"terminal Run is sealed: {record.address}")
        state = dict(record.state)
        state.update(changes)
        state["updated_at"] = _utc_timestamp(self._now())
        validate_state(state)
        atomic_replace(record.path / "state.json", _json_text(state))
        return RunRecord(
            record.path,
            record.address,
            record.snapshot,
            record.request,
            state,
        )

    def load(self, experiment: str, variant: str, run_id: str) -> RunRecord:
        _identity(experiment, "Experiment")
        _identity(variant, "Variant")
        _run_id(run_id)
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
        outputs = self.repository.outputs
        if not os.path.lexists(outputs):
            return []
        _existing_directory(outputs)
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
                return ModelRecord(
                    target_path,
                    f"{document['experiment']}/{document['variant']}/{model_id}",
                    document,
                )
            finally:
                if candidate.exists():
                    shutil.rmtree(candidate)

    def load_model_manifest(
        self,
        experiment: str,
        variant: str,
        model_id: str,
    ) -> ModelRecord:
        _identity(experiment, "Experiment")
        _identity(variant, "Variant")
        _model_id(model_id)
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
        try:
            root = self.variant_root(experiment, variant, create=False)
        except NotFoundError:
            return []
        models = root / "models"
        if not os.path.lexists(models):
            return []
        _existing_directory(models)
        result: list[ModelRecord] = []
        for entry in _catalog_directories(models):
            _model_id(entry.name)
            result.append(load(experiment, variant, entry.name))
        return sorted(
            result,
            key=lambda item: (
                item.document["created_at"],
                item.document["model_id"].encode("utf-8"),
            ),
        )

    def resolve_model_checkpoint(self, model: ModelRecord) -> Path:
        root = self.variant_root(
            model.document["experiment"],
            model.document["variant"],
            create=False,
        )
        relative = PurePosixPath(model.document["checkpoint"]["path"])
        path = root.joinpath(*relative.parts)
        _contained_regular_file(root, path, "Model checkpoint")
        return path

    def resolve_worker_log(self, record: RunRecord) -> Path:
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
        if not os.path.lexists(history_path):
            if offset:
                raise ContractError("Train metric history disappeared while following")
            if record.state["status"] == "done":
                raise ContractError(
                    f"completed Train Run has no metric history: {record.address}"
                )
            return {"events": [], "offset": 0, "partial": False}
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

    @staticmethod
    def _next_run_number(runs: Path) -> int:
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
