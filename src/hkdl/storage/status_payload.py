"""Validate legacy cached status payloads independently of SQLite storage."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from hkdl.authoring.config import DIGEST_PATTERN, NAME_PATTERN
from hkdl.execution.run_contracts import (
    ACTIONS,
    MAX_SEED,
    MODEL_ID_PATTERN,
    RUN_ID_PATTERN,
    RUN_STATUSES,
    TRACKER_ID_PATTERN,
)


class _IndexError(RuntimeError):
    """A legacy index value violates the cached status contract."""


def validate_payload(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("name"), str)
        or not isinstance(value.get("training_groups"), list)
    ):
        raise _IndexError("invalid status payload")
    if not NAME_PATTERN.fullmatch(value["name"]):
        raise _IndexError("invalid status Variant name")
    for group in value["training_groups"]:
        _mapping_fields(group, {"name", "seeds", "aggregates"}, "status group")
        if not isinstance(group["name"], str) or not NAME_PATTERN.fullmatch(
            group["name"]
        ):
            raise _IndexError("invalid status Training Group")
        if not isinstance(group["seeds"], list) or not isinstance(
            group["aggregates"], list
        ):
            raise _IndexError("invalid status group lists")
        for seed in group["seeds"]:
            _mapping_fields(seed, {"seed", "model", "runs"}, "status seed")
            _bounded_integer(seed["seed"], minimum=0, maximum=MAX_SEED, location="seed")
            if not isinstance(seed["runs"], list):
                raise _IndexError("invalid status Run list")
            if seed["model"] is not None:
                _validate_model_payload(seed["model"])
            for run in seed["runs"]:
                _validate_run_payload(run)
        for aggregate in group["aggregates"]:
            _validate_aggregate_payload(aggregate)
    return value


def timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise _IndexError("invalid cached timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _IndexError("invalid cached timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _IndexError("invalid cached timestamp")
    return parsed.astimezone(timezone.utc)


def _mapping_fields(value: object, fields: set[str], location: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise _IndexError(f"invalid {location}")


def _validate_model_payload(value: object) -> None:
    fields = {
        "model_id",
        "producer_run",
        "device",
        "training_fingerprint",
        "created_at",
    }
    _mapping_fields(value, fields, "status Model")
    assert isinstance(value, dict)
    if not isinstance(value["model_id"], str) or not MODEL_ID_PATTERN.fullmatch(
        value["model_id"]
    ):
        raise _IndexError("invalid status Model ID")
    if not isinstance(value["producer_run"], str) or not RUN_ID_PATTERN.fullmatch(
        value["producer_run"]
    ):
        raise _IndexError("invalid status producer Run")
    if not isinstance(value["device"], str):
        raise _IndexError("invalid status Model device")
    if not isinstance(
        value["training_fingerprint"], str
    ) or not DIGEST_PATTERN.fullmatch(value["training_fingerprint"]):
        raise _IndexError("invalid status training fingerprint")
    timestamp(value["created_at"])


def _validate_run_payload(value: object) -> None:
    fields = {
        "run_id",
        "action",
        "status",
        "retry_of",
        "model_id",
        "evaluation_case",
        "primary",
        "values",
        "artifacts",
        "reason",
        "tracker_run_id",
        "tracker_backends",
        "metric_summary",
        "created_at",
        "updated_at",
        "elapsed_seconds",
        "device",
        "configured_batch_size",
        "configured_steps",
        "configured_epochs",
        "best_checkpoint",
        "last_checkpoint",
    }
    _mapping_fields(value, fields, "status Run")
    assert isinstance(value, dict)
    _validate_run_lifecycle(value)
    _validate_run_target(value)
    _validate_run_observability(value)


def _validate_run_lifecycle(value: dict[str, Any]) -> None:
    if not isinstance(value["run_id"], str) or not RUN_ID_PATTERN.fullmatch(
        value["run_id"]
    ):
        raise _IndexError("invalid cached Run ID")
    if (
        not isinstance(value["action"], str)
        or value["action"] not in ACTIONS
        or not isinstance(value["status"], str)
        or value["status"] not in RUN_STATUSES
    ):
        raise _IndexError("invalid cached Run lifecycle")
    if value["status"] in {"failed", "interrupted", "abandoned"}:
        _required_string(value["reason"], "stopped Run reason")
    elif value["reason"] is not None:
        raise _IndexError("non-stopped cached Run reason must be null")
    _optional_pattern(value["tracker_run_id"], TRACKER_ID_PATTERN, "tracker Run ID")


def _validate_run_target(value: dict[str, Any]) -> None:
    _optional_pattern(value["retry_of"], RUN_ID_PATTERN, "retry Run")
    _optional_pattern(value["model_id"], MODEL_ID_PATTERN, "Model ID")
    _optional_name(value["evaluation_case"], "Evaluation Case")
    for field in ("best_checkpoint", "last_checkpoint"):
        if value[field] is not None:
            _owned_path(value[field], field)
    if value["action"] != "train" and (
        value["best_checkpoint"] is not None or value["last_checkpoint"] is not None
    ):
        raise _IndexError("non-Train cached Run has checkpoints")
    if value["action"] == "train":
        if value["model_id"] is not None or value["evaluation_case"] is not None:
            raise _IndexError("invalid cached Train target")
    elif value["action"] == "eval":
        if value["model_id"] is None or value["evaluation_case"] is None:
            raise _IndexError("invalid cached Eval target")
    elif value["model_id"] is None or value["evaluation_case"] is not None:
        raise _IndexError("invalid cached Export target")


def _validate_run_observability(value: dict[str, Any]) -> None:
    if not isinstance(value["device"], str):
        raise _IndexError("invalid cached Run device")
    if not isinstance(value["tracker_backends"], list) or value["tracker_backends"] != [
        item for item in ("local", "mlflow") if item in value["tracker_backends"]
    ]:
        raise _IndexError("invalid cached tracker backends")
    if not isinstance(value["artifacts"], list):
        raise _IndexError("invalid cached artifacts")
    for artifact in value["artifacts"]:
        _owned_path(artifact, "evaluation artifact")
    _validate_primary(value["primary"])
    _validate_finite_mapping(value["values"], "evaluation values")
    _validate_metric_summary(value["metric_summary"])
    timestamp(value["created_at"])
    timestamp(value["updated_at"])
    _bounded_integer(value["elapsed_seconds"], minimum=0, location="elapsed time")
    for name in ("configured_batch_size", "configured_steps", "configured_epochs"):
        if value[name] is not None:
            _bounded_integer(value[name], minimum=1, location=name)


def _validate_primary(value: object) -> None:
    if value is None:
        return
    _mapping_fields(value, {"name", "value"}, "status primary metric")
    assert isinstance(value, dict)
    _required_string(value["name"], "primary metric name")
    if value["value"] is not None:
        _finite_number(value["value"], "primary metric value")


def _validate_finite_mapping(value: object, location: str) -> None:
    if not isinstance(value, dict):
        raise _IndexError(f"invalid cached {location}")
    for name, number in value.items():
        if not isinstance(name, str) or not name:
            raise _IndexError(f"invalid cached {location} name")
        _finite_number(number, location)


def _validate_metric_summary(value: object) -> None:
    if not isinstance(value, dict):
        raise _IndexError("invalid cached metric summary")
    for name, metric in value.items():
        if not isinstance(name, str) or not name:
            raise _IndexError("invalid cached metric summary name")
        _mapping_fields(metric, {"count", "last_step", "last_value"}, "metric summary")
        assert isinstance(metric, dict)
        _bounded_integer(metric["count"], minimum=1, location="metric count")
        _bounded_integer(metric["last_step"], minimum=0, location="metric step")
        _finite_number(metric["last_value"], "metric value")


def _validate_aggregate_payload(value: object) -> None:
    fields = {"evaluation_case", "metric", "eligible", "count", "mean", "sample_std"}
    _mapping_fields(value, fields, "status aggregate")
    assert isinstance(value, dict)
    _optional_name(value["evaluation_case"], "aggregate Evaluation Case", required=True)
    _required_string(value["metric"], "aggregate metric")
    eligible = _bounded_integer(
        value["eligible"], minimum=0, location="aggregate eligible"
    )
    count = _bounded_integer(value["count"], minimum=0, location="aggregate count")
    if count > eligible:
        raise _IndexError("aggregate count exceeds eligible Models")
    _finite_number(value["mean"], "aggregate mean")
    if value["sample_std"] is not None:
        sample_std = _finite_number(value["sample_std"], "aggregate sample_std")
        if sample_std < 0:
            raise _IndexError("aggregate sample_std is negative")


def _bounded_integer(
    value: object, *, minimum: int, location: str, maximum: int | None = None
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise _IndexError(f"invalid cached {location}")
    return value


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _IndexError(f"invalid cached {location}")
    try:
        number = float(value)
    except OverflowError as error:
        raise _IndexError(f"invalid cached {location}") from error
    if not math.isfinite(number):
        raise _IndexError(f"invalid cached {location}")
    return number


def _optional_pattern(value: object, pattern: object, location: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise _IndexError(f"invalid cached {location}")


def _optional_name(value: object, location: str, *, required: bool = False) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise _IndexError(f"invalid cached {location}")


def _required_string(value: object, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise _IndexError(f"invalid cached {location}")


def _owned_path(value: object, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise _IndexError(f"invalid cached {location}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise _IndexError(f"invalid cached {location}")


__all__ = ["validate_payload"]
