"""Disposable SQLite projection for file-authoritative status reads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import DIGEST_PATTERN, NAME_PATTERN, ContractError
from .run_contracts import (
    ACTIONS,
    MAX_SEED,
    MODEL_ID_PATTERN,
    RUN_ID_PATTERN,
    RUN_STATUSES,
    TRACKER_ID_PATTERN,
    _identity,
)
from .storage import (
    STATUS_INDEX_FILENAME,
    LockUnavailableError,
    RepositoryPaths,
    try_directory_lock,
)

INDEX_APPLICATION_ID = 0x484B444C
INDEX_SCHEMA_VERSION = 1
INDEX_FILENAME = STATUS_INDEX_FILENAME
INDEX_RESERVED_PREFIX = INDEX_FILENAME


class IndexFailure(RuntimeError):
    """A requested index operation could not be completed."""


@dataclass(frozen=True)
class IndexReport:
    state: str
    path: str
    schema_version: int | None
    variants: int
    runs: int
    models: int
    detail: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "path": self.path,
            "schema_version": self.schema_version,
            "variants": self.variants,
            "runs": self.runs,
            "models": self.models,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class VariantInventory:
    experiment: str
    variant: str
    fingerprint: str
    run_ids: tuple[str, ...]
    model_ids: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.experiment, self.variant


@dataclass(frozen=True)
class _ProjectionRow:
    inventory: VariantInventory
    payload: dict[str, Any] | None


class _IndexError(RuntimeError):
    pass


class _IncompatibleIndex(_IndexError):
    pass


class _FallbackOnlyIndex(_IndexError):
    pass


ProjectionLoader = Callable[[VariantInventory], dict[str, Any] | None]


class StatusIndex:
    def __init__(self, repository: RepositoryPaths):
        self.repository = repository
        self.path = repository.outputs / INDEX_FILENAME

    def inspect(self) -> IndexReport:
        try:
            inventories = _inventories(self.repository)
        except ContractError as error:
            return self._report(
                "unavailable",
                (),
                detail=f"authority inventory failed: {error}",
            )
        if not os.path.lexists(self.path):
            return self._report("absent", inventories)
        if self.path.is_symlink() or not self.path.is_file():
            return self._report(
                "unavailable",
                inventories,
                detail="index path is not a regular file",
            )
        try:
            with _connect_readonly(self.path) as connection:
                version = _schema_state(connection)
                if version != INDEX_SCHEMA_VERSION:
                    state = "upgradeable" if _upgradeable(version) else "incompatible"
                    return self._report(state, inventories, schema_version=version)
                rows = _read_rows(connection)
        except _IncompatibleIndex as error:
            return self._report("incompatible", inventories, detail=str(error))
        except sqlite3.OperationalError as error:
            state = (
                "unavailable" if "unable to open" in str(error).lower() else "corrupt"
            )
            return self._report(state, inventories, detail=str(error))
        except (sqlite3.DatabaseError, OSError, _IndexError) as error:
            return self._report("corrupt", inventories, detail=str(error))

        expected = {item.key: item.fingerprint for item in inventories}
        observed = {key: row.inventory.fingerprint for key, row in rows.items()}
        state = "ready" if observed == expected else "stale"
        return self._report(state, inventories, schema_version=version)

    def query(
        self,
        *,
        experiment: str | None,
        variant: str | None,
        observed_at: datetime,
        load: ProjectionLoader,
    ) -> dict[str, Any]:
        inventories = _inventories(
            self.repository,
            experiment=experiment,
            variant=variant,
        )
        if not os.path.lexists(self.repository.outputs):
            return {"experiments": []}

        try:
            rows = self._query_rows(
                inventories,
                experiment=experiment,
                variant=variant,
                load=load,
            )
            after = _inventories(
                self.repository,
                experiment=experiment,
                variant=variant,
            )
            if inventories != after:
                rows = [_ProjectionRow(item, load(item)) for item in after]
                self._best_effort_update(
                    rows,
                    set(),
                    after,
                    experiment=experiment,
                    variant=variant,
                )
            return _compose(rows, observed_at)
        except _FallbackOnlyIndex:
            rows = [_ProjectionRow(item, load(item)) for item in inventories]
            return _compose(rows, observed_at)
        except _IndexError:
            rows = [_ProjectionRow(item, load(item)) for item in inventories]
            self._best_effort_replace(
                rows,
                inventories,
                experiment=experiment,
                variant=variant,
            )
            return _compose(rows, observed_at)

    def rebuild(self, load: ProjectionLoader) -> IndexReport:
        try:
            before = _inventories(self.repository)
            rows = [_ProjectionRow(item, load(item)) for item in before]
            after = _inventories(self.repository)
            if before != after:
                raise ContractError("status authority changed during index rebuild")
            self.repository.outputs.mkdir(exist_ok=True)
            if (
                self.repository.outputs.is_symlink()
                or not self.repository.outputs.is_dir()
            ):
                raise ContractError("outputs must be a real directory")
            with try_directory_lock(self.repository.outputs):
                _replace_database(self.path, rows)
        except LockUnavailableError as error:
            raise IndexFailure("index is busy") from error
        except (OSError, sqlite3.Error, _IndexError) as error:
            raise IndexFailure(str(error)) from error
        return self._report(
            "ready",
            after,
            schema_version=INDEX_SCHEMA_VERSION,
        )

    def _query_rows(
        self,
        inventories: list[VariantInventory],
        *,
        experiment: str | None,
        variant: str | None,
        load: ProjectionLoader,
    ) -> list[_ProjectionRow]:
        if not os.path.lexists(self.path):
            raise _IndexError("index is absent")
        if self.path.is_symlink() or not self.path.is_file():
            raise _IndexError("index path is not a regular file")
        try:
            with _connect_readonly(self.path) as connection:
                version = _schema_state(connection)
                if version != INDEX_SCHEMA_VERSION:
                    raise _FallbackOnlyIndex(f"unsupported index schema: {version}")
                existing = _read_rows(connection)
        except _IncompatibleIndex as error:
            raise _FallbackOnlyIndex(str(error)) from error
        except (sqlite3.Error, OSError) as error:
            raise _IndexError(str(error)) from error

        selected: list[_ProjectionRow] = []
        changed: list[_ProjectionRow] = []
        for inventory in inventories:
            row = existing.get(inventory.key)
            if row is not None and row.inventory.fingerprint == inventory.fingerprint:
                selected.append(_ProjectionRow(inventory, row.payload))
                continue
            projected = _ProjectionRow(inventory, load(inventory))
            selected.append(projected)
            changed.append(projected)

        selected_keys = {item.key for item in inventories}
        deleted = {
            key
            for key in existing
            if _in_scope(key, experiment=experiment, variant=variant)
            and key not in selected_keys
        }
        if changed or deleted:
            self._best_effort_update(
                changed,
                deleted,
                inventories,
                experiment=experiment,
                variant=variant,
            )
        return selected

    def _best_effort_replace(
        self,
        rows: list[_ProjectionRow],
        before: list[VariantInventory],
        *,
        experiment: str | None,
        variant: str | None,
    ) -> None:
        try:
            after = _inventories(
                self.repository,
                experiment=experiment,
                variant=variant,
            )
            if before != after:
                return
            with try_directory_lock(self.repository.outputs):
                if os.path.lexists(self.path) and (
                    self.path.is_symlink() or not self.path.is_file()
                ):
                    return
                _replace_database(self.path, rows)
        except (
            ContractError,
            LockUnavailableError,
            OSError,
            sqlite3.Error,
            _IndexError,
        ):
            return

    def _best_effort_update(
        self,
        changed: list[_ProjectionRow],
        deleted: set[tuple[str, str]],
        before: list[VariantInventory],
        *,
        experiment: str | None,
        variant: str | None,
    ) -> None:
        try:
            after = _inventories(
                self.repository,
                experiment=experiment,
                variant=variant,
            )
            if before != after:
                return
            with try_directory_lock(self.repository.outputs):
                with _connect_readonly(self.path) as connection:
                    if _schema_state(connection) != INDEX_SCHEMA_VERSION:
                        return
                    rows = _read_rows(connection)
                for row in changed:
                    rows[row.inventory.key] = row
                for key in deleted:
                    rows.pop(key, None)
                _replace_database(self.path, list(rows.values()))
        except (
            ContractError,
            LockUnavailableError,
            OSError,
            sqlite3.Error,
            _IndexError,
        ):
            return

    def _report(
        self,
        state: str,
        inventories: list[VariantInventory] | tuple[()],
        *,
        schema_version: int | None = None,
        detail: str | None = None,
    ) -> IndexReport:
        return IndexReport(
            state=state,
            path=f"outputs/{INDEX_FILENAME}",
            schema_version=schema_version,
            variants=len(inventories),
            runs=sum(len(item.run_ids) for item in inventories),
            models=sum(len(item.model_ids) for item in inventories),
            detail=detail,
        )


def _inventories(
    repository: RepositoryPaths,
    *,
    experiment: str | None = None,
    variant: str | None = None,
) -> list[VariantInventory]:
    if experiment is not None:
        _identity(experiment, "Experiment")
    if variant is not None:
        _identity(variant, "Variant")
    if variant is not None and experiment is None:
        raise ContractError("Variant status filter requires Experiment")
    outputs = repository.outputs
    if not os.path.lexists(outputs):
        return []
    _real_directory(outputs, "outputs")
    result: list[VariantInventory] = []
    for experiment_entry in _catalog_directories(outputs, ignore={"index.db"}):
        _identity(experiment_entry.name, "Run Experiment")
        if experiment is not None and experiment_entry.name != experiment:
            continue
        for variant_entry in _catalog_directories(experiment_entry):
            _identity(variant_entry.name, "Run Variant")
            if variant is not None and variant_entry.name != variant:
                continue
            result.append(
                _variant_inventory(
                    experiment_entry.name,
                    variant_entry.name,
                    variant_entry,
                )
            )
    return result


def _variant_inventory(
    experiment: str,
    variant: str,
    root: Path,
) -> VariantInventory:
    try:
        entries = list(os.scandir(root))
    except OSError as error:
        raise ContractError(f"cannot scan status authority {root}: {error}") from error
    for entry in entries:
        if RUN_ID_PATTERN.fullmatch(entry.name):
            raise ContractError(
                f"legacy pipeline Run layout is unsupported: {entry.path}"
            )

    tokens: list[list[object]] = []
    run_ids: list[str] = []
    runs = root / "runs"
    if os.path.lexists(runs):
        _real_directory(runs, "Run catalog")
        for entry in _catalog_directories(runs):
            if not RUN_ID_PATTERN.fullmatch(entry.name):
                raise ContractError("invalid Run ID")
            run_ids.append(entry.name)
            for relative in (
                "snapshot.yaml",
                "request.json",
                "state.json",
                "metrics/eval.json",
                "metrics/train-summary.json",
            ):
                tokens.append(
                    _file_token(entry / relative, f"runs/{entry.name}/{relative}")
                )

    model_ids: list[str] = []
    models = root / "models"
    if os.path.lexists(models):
        _real_directory(models, "Model catalog")
        for entry in _catalog_directories(models):
            if not MODEL_ID_PATTERN.fullmatch(entry.name):
                raise ContractError("invalid Model ID")
            model_ids.append(entry.name)
            tokens.append(
                _file_token(entry / "model.json", f"models/{entry.name}/model.json")
            )

    document = {
        "experiment": experiment,
        "variant": variant,
        "run_ids": run_ids,
        "model_ids": model_ids,
        "files": tokens,
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return VariantInventory(
        experiment,
        variant,
        f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        tuple(run_ids),
        tuple(model_ids),
    )


def _file_token(path: Path, relative: str) -> list[object]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return [relative, "missing"]
    except OSError as error:
        raise ContractError(
            f"cannot inspect status authority {path}: {error}"
        ) from error
    return [
        relative,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_dev,
        metadata.st_ino,
    ]


def _catalog_directories(path: Path, *, ignore: set[str] | None = None) -> list[Path]:
    ignored = ignore or set()
    result: list[Path] = []
    try:
        entries = list(os.scandir(path))
    except OSError as error:
        raise ContractError(f"cannot scan status authority {path}: {error}") from error
    for entry in entries:
        if entry.name.startswith(".") or entry.name in ignored:
            continue
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ContractError(f"invalid catalog entry: {entry.path}")
        result.append(Path(entry.path))
    return sorted(result, key=lambda item: item.name.encode("utf-8"))


def _real_directory(path: Path, location: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"cannot inspect {location}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"{location} must be a real directory: {path}")


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.1)
    connection.row_factory = sqlite3.Row
    return connection


def _connect_writable(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=0.1)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _schema_state(connection: sqlite3.Connection) -> int:
    application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if application_id != INDEX_APPLICATION_ID:
        raise _IncompatibleIndex(f"unexpected SQLite application_id: {application_id}")
    if version == INDEX_SCHEMA_VERSION:
        expected = {"metadata", "variant_projection"}
        observed = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not expected.issubset(observed):
            raise _IndexError("index schema tables are missing")
    return version


def _upgradeable(version: int) -> bool:
    migrations: dict[int, Callable[[sqlite3.Connection], None]] = {}
    while version < INDEX_SCHEMA_VERSION:
        migration = migrations.get(version)
        if migration is None:
            return False
        version += 1
    return version == INDEX_SCHEMA_VERSION


def _read_rows(
    connection: sqlite3.Connection,
) -> dict[tuple[str, str], _ProjectionRow]:
    result: dict[tuple[str, str], _ProjectionRow] = {}
    try:
        rows = connection.execute(
            "SELECT experiment, variant, fingerprint, run_count, model_count, "
            "run_ids_json, model_ids_json, payload_json, payload_digest "
            "FROM variant_projection"
        )
        for row in rows:
            run_ids = _string_list(json.loads(row["run_ids_json"]), "Run IDs")
            model_ids = _string_list(json.loads(row["model_ids_json"]), "Model IDs")
            if row["run_count"] != len(run_ids) or row["model_count"] != len(model_ids):
                raise _IndexError("index catalog counts disagree")
            payload_json = str(row["payload_json"])
            if row["payload_digest"] != _digest_text(payload_json):
                raise _IndexError("status payload digest changed")
            payload_value = json.loads(payload_json)
            payload = _validate_payload(payload_value)
            inventory = VariantInventory(
                str(row["experiment"]),
                str(row["variant"]),
                str(row["fingerprint"]),
                tuple(run_ids),
                tuple(model_ids),
            )
            if payload is not None and payload["name"] != inventory.variant:
                raise _IndexError("status payload Variant ownership mismatch")
            if inventory.key in result:
                raise _IndexError("duplicate index Variant")
            result[inventory.key] = _ProjectionRow(inventory, payload)
    except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as error:
        raise _IndexError(f"invalid index row: {error}") from error
    return result


def _string_list(value: object, location: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _IndexError(f"{location} must be a string list")
    return value


def _validate_payload(value: object) -> dict[str, Any] | None:
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
    _timestamp(value["created_at"])


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
    _timestamp(value["created_at"])
    _timestamp(value["updated_at"])
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
        _mapping_fields(
            metric,
            {"count", "last_step", "last_value"},
            "metric summary",
        )
        assert isinstance(metric, dict)
        _bounded_integer(metric["count"], minimum=1, location="metric count")
        _bounded_integer(metric["last_step"], minimum=0, location="metric step")
        _finite_number(metric["last_value"], "metric value")


def _validate_aggregate_payload(value: object) -> None:
    fields = {
        "evaluation_case",
        "metric",
        "eligible",
        "count",
        "mean",
        "sample_std",
    }
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
    value: object,
    *,
    minimum: int,
    location: str,
    maximum: int | None = None,
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


def _replace_database(path: Path, rows: list[_ProjectionRow]) -> None:
    if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
        raise _IndexError("index path is not a regular file")
    target_sidecars = [
        Path(f"{path}{suffix}") for suffix in ("-journal", "-wal", "-shm")
    ]
    for sidecar in target_sidecars:
        if not os.path.lexists(sidecar):
            continue
        metadata = sidecar.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise _IndexError(f"index sidecar is not a regular file: {sidecar.name}")
    for sidecar in target_sidecars:
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
    descriptor, candidate_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.name}.candidate-",
    )
    os.close(descriptor)
    candidate = Path(candidate_name)
    sidecars = [
        Path(f"{candidate}-journal"),
        Path(f"{candidate}-wal"),
        Path(f"{candidate}-shm"),
    ]
    try:
        os.chmod(candidate, 0o644)
        with _connect_writable(candidate) as connection:
            connection.execute(f"PRAGMA application_id={INDEX_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={INDEX_SCHEMA_VERSION}")
            connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE variant_projection (
                    experiment TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    run_count INTEGER NOT NULL,
                    model_count INTEGER NOT NULL,
                    run_ids_json TEXT NOT NULL,
                    model_ids_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    PRIMARY KEY (experiment, variant)
                ) WITHOUT ROWID;
                """
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('projection', 'status')"
            )
            for row in rows:
                _write_row(connection, row)
            connection.commit()
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise _IndexError("candidate index failed quick_check")
        with candidate.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(candidate, path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        for item in [candidate, *sidecars]:
            try:
                item.unlink()
            except FileNotFoundError:
                pass


def _write_row(connection: sqlite3.Connection, row: _ProjectionRow) -> None:
    inventory = row.inventory
    payload_json = _json(row.payload)
    connection.execute(
        """
        INSERT INTO variant_projection(
            experiment, variant, fingerprint, run_count, model_count,
            run_ids_json, model_ids_json, payload_json, payload_digest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment, variant) DO UPDATE SET
            fingerprint = excluded.fingerprint,
            run_count = excluded.run_count,
            model_count = excluded.model_count,
            run_ids_json = excluded.run_ids_json,
            model_ids_json = excluded.model_ids_json,
            payload_json = excluded.payload_json,
            payload_digest = excluded.payload_digest
        """,
        (
            inventory.experiment,
            inventory.variant,
            inventory.fingerprint,
            len(inventory.run_ids),
            len(inventory.model_ids),
            _json(inventory.run_ids),
            _json(inventory.model_ids),
            payload_json,
            _digest_text(payload_json),
        ),
    )


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _digest_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _compose(rows: list[_ProjectionRow], observed_at: datetime) -> dict[str, Any]:
    try:
        experiments: dict[str, dict[str, Any]] = {}
        for row in sorted(
            rows,
            key=lambda item: (
                item.inventory.experiment.encode("utf-8"),
                item.inventory.variant.encode("utf-8"),
            ),
        ):
            if row.payload is None:
                continue
            payload = json.loads(_json(row.payload))
            _validate_payload(payload)
            _refresh_elapsed(payload, observed_at)
            experiment = experiments.setdefault(
                row.inventory.experiment,
                {"name": row.inventory.experiment, "variants": []},
            )
            experiment["variants"].append(payload)
        return {"experiments": list(experiments.values())}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise _IndexError(f"invalid cached status payload: {error}") from error


def _refresh_elapsed(payload: dict[str, Any], observed_at: datetime) -> None:
    for group in payload["training_groups"]:
        for seed in group["seeds"]:
            for run in seed["runs"]:
                created = _timestamp(run["created_at"])
                end = (
                    observed_at
                    if run["status"] in {"allocated", "running"}
                    else _timestamp(run["updated_at"])
                )
                run["elapsed_seconds"] = max(0, int((end - created).total_seconds()))


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise _IndexError("invalid cached timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _IndexError("invalid cached timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _IndexError("invalid cached timestamp")
    return parsed.astimezone(timezone.utc)


def _in_scope(
    key: tuple[str, str],
    *,
    experiment: str | None,
    variant: str | None,
) -> bool:
    return (experiment is None or key[0] == experiment) and (
        variant is None or key[1] == variant
    )


__all__ = [
    "INDEX_APPLICATION_ID",
    "INDEX_FILENAME",
    "INDEX_RESERVED_PREFIX",
    "INDEX_SCHEMA_VERSION",
    "IndexFailure",
    "IndexReport",
    "StatusIndex",
    "VariantInventory",
]
