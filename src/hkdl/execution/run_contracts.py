"""Pure contracts shared by Variant-managed Runs and their consumers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from hkdl.authoring.config import (
    DIGEST_PATTERN,
    NAME_PATTERN,
    validate_experiment,
    validate_variant,
)
from hkdl.errors import ContractError
from hkdl.storage.storage import NotFoundError

RUN_ID_PATTERN = re.compile(r"run-[0-9]+")
MODEL_ID_PATTERN = re.compile(r"model-[0-9a-f]{32}")
TRACKER_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]*:[^\s:][^\s]*")
TRAIN_COMPONENTS = frozenset({"model", "loss", "optimizer", "dataloader", "trainer"})
EVAL_COMPONENTS = frozenset({"model", "dataloader", "evaluator"})
EXPORT_COMPONENTS = frozenset({"exporter"})
ACTIONS = frozenset({"train", "eval", "export"})
TRAINING_METRIC_EVENT_FIELDS = frozenset({"schema_version", "name", "step", "value"})
RUN_STATUSES = frozenset(
    {"allocated", "running", "done", "failed", "interrupted", "abandoned"}
)
TERMINAL_STATUSES = frozenset({"done", "failed", "interrupted", "abandoned"})
MAX_SEED = (1 << 63) - 1


def validate_tracker(value: Any) -> tuple[str, ...]:
    tracker = _mapping(value, "variant.tracker")
    if set(tracker) != {"backend"}:
        raise ContractError("tracker must contain only backend")
    backend = tracker["backend"]
    if isinstance(backend, str):
        if backend == "none":
            return ()
        if backend in {"local", "mlflow"}:
            return (backend,)
    elif isinstance(backend, (list, tuple)):
        if (
            backend
            and all(
                isinstance(item, str) and item in {"local", "mlflow"}
                for item in backend
            )
            and len(backend) == len(set(backend))
        ):
            return tuple(item for item in ("local", "mlflow") if item in backend)
    raise ContractError(
        "tracker.backend must be none, local, mlflow, or a non-empty unique list "
        "of local and mlflow"
    )


def validate_training_readiness(
    experiment: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, str]:
    del experiment
    validate_tracker(variant["tracker"])
    return _required_components(variant, TRAIN_COMPONENTS, "training")


def validate_evaluation_readiness(
    experiment: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    case: str = "default",
) -> dict[str, str]:
    del experiment
    validate_tracker(variant["tracker"])
    evaluation_case(variant, case)
    metric_spec(variant, case)
    return _required_components(variant, EVAL_COMPONENTS, "evaluation")


def validate_export_readiness(
    experiment: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, str]:
    del experiment
    validate_tracker(variant["tracker"])
    return _required_components(variant, EXPORT_COMPONENTS, "export")


def evaluation_case(variant: Mapping[str, Any], case: str) -> dict[str, Any]:
    _identity(case, "Evaluation Case")
    evaluation = _mapping(variant["eval"], "variant.eval")
    cases = evaluation.get("cases")
    if cases is None:
        if case != "default":
            raise NotFoundError(f"Evaluation Case not found: {case}")
        return deepcopy(dict(evaluation))
    cases = _mapping(cases, "variant.eval.cases")
    if case not in cases:
        raise NotFoundError(f"Evaluation Case not found: {case}")
    return deepcopy(dict(_mapping(cases[case], f"variant.eval.cases.{case}")))


def metric_spec(variant: Mapping[str, Any], case: str) -> dict[str, Any]:
    selected_case = evaluation_case(variant, case)
    value = selected_case.get("metrics", variant["metrics"])
    metrics = _mapping(value, f"evaluation case {case} metrics")
    primary = metrics.get("primary")
    report = metrics.get("report")
    if not isinstance(primary, str) or not NAME_PATTERN.fullmatch(primary):
        raise ContractError("metrics.primary is invalid")
    if (
        not isinstance(report, (list, tuple))
        or not report
        or any(
            not isinstance(name, str) or not NAME_PATTERN.fullmatch(name)
            for name in report
        )
        or len(report) != len(set(report))
        or list(report).count(primary) != 1
    ):
        raise ContractError(
            "metrics.report must be unique and contain metrics.primary once"
        )
    return {"primary": primary, "report": list(report)}


def validate_snapshot(document: dict[str, Any]) -> None:
    _exact_fields(document, {"schema_version", "experiment", "variant", "provenance"})
    _integer_one(document["schema_version"], "snapshot.schema_version")
    experiment = _mapping(document["experiment"], "snapshot.experiment")
    variant = _mapping(document["variant"], "snapshot.variant")
    validate_experiment(experiment)
    validate_variant(variant)
    provenance = _mapping(document["provenance"], "snapshot.provenance")
    if set(provenance) != {
        "variant_file",
        "source_digest",
        "vcs_revision",
        "frozen_at",
    }:
        raise ContractError("snapshot.provenance fields are invalid")
    validate_owned_path(provenance["variant_file"], "snapshot.provenance.variant_file")
    _digest(provenance["source_digest"], "snapshot.provenance.source_digest")
    if provenance["vcs_revision"] is not None:
        raise ContractError("snapshot.provenance.vcs_revision must be null")
    _timestamp(provenance["frozen_at"], "snapshot.provenance.frozen_at")


def validate_request(document: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "run_id",
        "experiment",
        "variant",
        "action",
        "target",
        "retry_of",
        "source_digest",
        "identity_fingerprint",
        "exec",
        "created_at",
    }
    _exact_fields(document, expected)
    _integer_one(document["schema_version"], "request.schema_version")
    _run_id(document["run_id"])
    _identity(document["experiment"], "request Experiment")
    _identity(document["variant"], "request Variant")
    action = document["action"]
    if action not in ACTIONS:
        raise ContractError("request.action is invalid")
    target = _mapping(document["target"], "request.target")
    if action == "train":
        _exact_fields(target, {"training_group", "seed"})
        _identity(target["training_group"], "Training Group")
        _seed(target["seed"])
    elif action == "eval":
        _exact_fields(
            target,
            {"training_group", "seed", "model_id", "evaluation_case"},
        )
        _identity(target["training_group"], "Training Group")
        _seed(target["seed"])
        _model_id(target["model_id"])
        _identity(target["evaluation_case"], "Evaluation Case")
    else:
        _exact_fields(target, {"model_id"})
        _model_id(target["model_id"])
    retry_of = document["retry_of"]
    if retry_of is not None:
        _run_id(retry_of)
        if retry_of == document["run_id"]:
            raise ContractError("request.retry_of cannot reference itself")
    _digest(document["source_digest"], "request.source_digest")
    _digest(document["identity_fingerprint"], "request.identity_fingerprint")
    exec_info = _mapping(document["exec"], "request.exec")
    _exact_fields(exec_info, {"seed", "device"})
    _seed(exec_info["seed"])
    if (
        not isinstance(exec_info["device"], str)
        or not exec_info["device"]
        or exec_info["device"] == "auto"
    ):
        raise ContractError("request.exec.device must be concrete")
    _timestamp(document["created_at"], "request.created_at")
    _json_compatible(document)


def validate_state(document: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "run_id",
        "action",
        "status",
        "reason",
        "tracker_run_id",
        "result",
        "best_checkpoint",
        "last_checkpoint",
        "created_at",
        "updated_at",
    }
    _exact_fields(document, expected)
    _integer_one(document["schema_version"], "state.schema_version")
    _run_id(document["run_id"])
    if document["action"] not in ACTIONS:
        raise ContractError("state.action is invalid")
    status = document["status"]
    if status not in RUN_STATUSES:
        raise ContractError("state.status is invalid")
    reason = document["reason"]
    if status in {"failed", "interrupted", "abandoned"}:
        if not isinstance(reason, str) or not reason:
            raise ContractError("stopped state requires a reason")
    elif reason is not None:
        raise ContractError("non-stopped state reason must be null")
    tracker_run_id = document["tracker_run_id"]
    if tracker_run_id is not None and (
        not isinstance(tracker_run_id, str)
        or not TRACKER_ID_PATTERN.fullmatch(tracker_run_id)
    ):
        raise ContractError("state.tracker_run_id is invalid")
    result = document["result"]
    if status == "done":
        result = _mapping(result, "state.result")
        if document["action"] == "train":
            _exact_fields(result, {"model_id"})
            _model_id(result["model_id"])
        elif document["action"] == "eval":
            _exact_fields(result, {"metrics"})
            validate_owned_path(result["metrics"], "state.result.metrics")
        else:
            _exact_fields(result, {"export"})
            validate_owned_path(result["export"], "state.result.export")
    elif result is not None:
        raise ContractError("non-completed state.result must be null")
    for field in ("best_checkpoint", "last_checkpoint"):
        value = document[field]
        if value is not None:
            validate_owned_path(value, f"state.{field}")
    if document["action"] != "train" and (
        document["best_checkpoint"] is not None
        or document["last_checkpoint"] is not None
    ):
        raise ContractError("only training Runs may record checkpoints")
    for field in ("created_at", "updated_at"):
        _timestamp(document[field], f"state.{field}")
    _json_compatible(document)


def validate_model(document: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "model_id",
        "experiment",
        "variant",
        "training_group",
        "seed",
        "device",
        "training_fingerprint",
        "producer_run",
        "checkpoint",
        "created_at",
    }
    _exact_fields(document, expected)
    _integer_one(document["schema_version"], "model.schema_version")
    _model_id(document["model_id"])
    _identity(document["experiment"], "model Experiment")
    _identity(document["variant"], "model Variant")
    _identity(document["training_group"], "Training Group")
    _seed(document["seed"])
    if not isinstance(document["device"], str) or not document["device"]:
        raise ContractError("model.device is invalid")
    _digest(document["training_fingerprint"], "model.training_fingerprint")
    _run_id(document["producer_run"])
    checkpoint = _mapping(document["checkpoint"], "model.checkpoint")
    _exact_fields(checkpoint, {"path", "digest"})
    _variant_output_path(checkpoint["path"], "model.checkpoint.path")
    _digest(checkpoint["digest"], "model.checkpoint.digest")
    _timestamp(document["created_at"], "model.created_at")
    _json_compatible(document)


def validate_evaluation(
    document: dict[str, Any],
    snapshot: Mapping[str, Any],
    request: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "schema_version",
        "model_id",
        "evaluation_case",
        "case_fingerprint",
        "primary",
        "values",
        "artifacts",
        "evaluated_at",
    }
    _exact_fields(document, expected)
    _integer_one(document["schema_version"], "evaluation.schema_version")
    _model_id(document["model_id"])
    _identity(document["evaluation_case"], "evaluation case")
    _digest(document["case_fingerprint"], "evaluation.case_fingerprint")
    if request is not None:
        target = request["target"]
        if (
            document["model_id"] != target["model_id"]
            or document["evaluation_case"] != target["evaluation_case"]
            or document["case_fingerprint"] != request["identity_fingerprint"]
        ):
            raise ContractError("evaluation ownership mismatch")
    metrics = metric_spec(snapshot["variant"], document["evaluation_case"])
    values = _mapping(document["values"], "evaluation.values")
    if set(values) != set(metrics["report"]):
        raise ContractError("evaluation values do not match metrics.report")
    for value in values.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ContractError("evaluation metric must be a finite JSON number")
    primary = _mapping(document["primary"], "evaluation.primary")
    _exact_fields(primary, {"name", "value"})
    if (
        primary["name"] != metrics["primary"]
        or primary["value"] != values[metrics["primary"]]
    ):
        raise ContractError("evaluation primary metric is inconsistent")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(set(artifacts)):
        raise ContractError("evaluation artifacts must be a unique list")
    for artifact in artifacts:
        validate_owned_path(artifact, "evaluation artifact")
    _timestamp(document["evaluated_at"], "evaluation.evaluated_at")


def fingerprint_document(document: Mapping[str, Any]) -> str:
    _json_compatible(document)
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def training_metric_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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


def validate_training_metric_event(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != TRAINING_METRIC_EVENT_FIELDS:
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


def validate_training_metric_summary(value: Any) -> None:
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


def _required_components(
    variant: Mapping[str, Any],
    required: frozenset[str],
    action: str,
) -> dict[str, str]:
    components = _mapping(variant["components"], "variant.components")
    missing = required - components.keys()
    if missing:
        raise ContractError(
            f"{action} components are missing: {', '.join(sorted(missing))}"
        )
    return {kind: components[kind] for kind in sorted(required)}


def _exact_fields(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ContractError("document fields are invalid")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{location} must be a mapping")
    return value


def _identity(value: Any, location: str) -> None:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ContractError(f"{location} name is invalid")


def _run_id(value: Any) -> None:
    if not isinstance(value, str) or not RUN_ID_PATTERN.fullmatch(value):
        raise ContractError("invalid Run ID")


def _model_id(value: Any) -> None:
    if not isinstance(value, str) or not MODEL_ID_PATTERN.fullmatch(value):
        raise ContractError("invalid Model ID")


def _seed(value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_SEED
    ):
        raise ContractError(f"seed must be an integer from 0 to {MAX_SEED}")


def _digest(value: Any, location: str) -> None:
    if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
        raise ContractError(f"{location} digest is invalid")


def _timestamp(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location} timestamp is invalid")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{location} timestamp is invalid") from error


def validate_owned_path(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location} path is invalid")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise ContractError(f"{location} path is invalid")


def _variant_output_path(value: Any, location: str) -> None:
    validate_owned_path(value, location)
    parsed = PurePosixPath(value)
    if not parsed.parts or parsed.parts[0] != "runs":
        raise ContractError(f"{location} must reference a Run artifact")


def _integer_one(value: Any, location: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ContractError(f"{location} must be integer 1")


def _json_compatible(value: Any) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("document contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_compatible(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("document mapping keys must be strings")
            _json_compatible(item)
        return
    raise ContractError("document contains a non-JSON value")


__all__ = [
    "ACTIONS",
    "EVAL_COMPONENTS",
    "EXPORT_COMPONENTS",
    "MAX_SEED",
    "MODEL_ID_PATTERN",
    "RUN_ID_PATTERN",
    "RUN_STATUSES",
    "TERMINAL_STATUSES",
    "TRACKER_ID_PATTERN",
    "TRAIN_COMPONENTS",
    "TRAINING_METRIC_EVENT_FIELDS",
    "evaluation_case",
    "fingerprint_document",
    "metric_spec",
    "training_metric_summary",
    "validate_owned_path",
    "validate_training_metric_event",
    "validate_training_metric_summary",
    "validate_evaluation",
    "validate_evaluation_readiness",
    "validate_export_readiness",
    "validate_model",
    "validate_request",
    "validate_snapshot",
    "validate_state",
    "validate_tracker",
    "validate_training_readiness",
]
