"""Standalone Variant worker with a stdlib-only diagnostics helper."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlparse

if __package__:
    from ._diagnostics import report_exception
else:
    from _diagnostics import report_exception

KINDS = frozenset(
    {"model", "loss", "optimizer", "dataloader", "trainer", "evaluator", "exporter"}
)
NAMES = re.compile(r"[a-z][a-z0-9_-]*")
TRAIN_COMPONENTS = frozenset({"model", "loss", "optimizer", "dataloader", "trainer"})
EVAL_COMPONENTS = frozenset({"model", "dataloader", "evaluator"})
EXPORT_COMPONENTS = frozenset({"exporter"})
ACTION_COMPONENTS = {
    "train": TRAIN_COMPONENTS,
    "eval": EVAL_COMPONENTS,
    "export": EXPORT_COMPONENTS,
}
sys.dont_write_bytecode = True


class TrackerOwnershipError(RuntimeError):
    """The external tracker identity is ambiguous or owned elsewhere."""


def _normalize_scalar(name: Any, value: Any, step: Any) -> tuple[str, float, int]:
    """Validate shared scalar input before tracker-specific duplicate handling."""
    if (
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
    ):
        raise ValueError("invalid tracker scalar")
    return name, float(value), step


@dataclass(frozen=True)
class ExecInfo:
    seed: int
    device: str


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints: Path
    metrics: Path
    results: Path
    export: Path


@dataclass(frozen=True)
class RunContext:
    cfg: Mapping[str, Any]
    paths: RunPaths
    tracker: Any
    exec: ExecInfo
    resume_from: Path | None
    components: Mapping[str, Any]
    _attempt_path: Path | None = None

    def report_checkpoint(
        self,
        last_checkpoint: Path,
        best_checkpoint: Path | None = None,
    ) -> None:
        if self._attempt_path is None:
            return
        last = _contained_relative_checkpoint(self.paths, Path(last_checkpoint))
        best = (
            _contained_relative_checkpoint(self.paths, Path(best_checkpoint))
            if best_checkpoint is not None
            else None
        )
        journal = _load_journal(self._attempt_path)
        if journal["phase"] != "running" or journal["action"] != "train":
            raise ValueError("checkpoint report does not own the active attempt")
        journal["checkpoint"] = {"best": best, "last": last}
        _write_journal(self._attempt_path, journal)


class NoopTracker:
    def log_scalar(self, name: str, value: float, step: int) -> None:
        del name, value, step


class ScalarTracker:
    def __init__(self, sinks: Sequence[Any]):
        self._sinks = sinks
        self._history: dict[str, dict[int, float]] = {}

    def log_scalar(self, name: str, value: float, step: int) -> None:
        name, numeric_value, step = _normalize_scalar(name, value, step)
        history = self._history.setdefault(name, {})
        if step in history:
            if history[step] != numeric_value:
                raise ValueError("tracker metric step has a conflicting value")
            return
        for sink in self._sinks:
            sink.log_scalar(name, numeric_value, step)
        history[step] = numeric_value


class LocalTracker:
    def __init__(self, metrics: Path):
        if metrics.is_symlink() or not metrics.is_dir():
            raise ValueError("Run metrics directory is invalid")
        self._history_path = metrics / "train.jsonl"
        self._summary_path = metrics / "train-summary.json"
        if os.path.lexists(self._summary_path):
            raise ValueError("local tracker summary already exists")
        descriptor = os.open(
            self._history_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o644,
        )
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8")
        self._counts: dict[str, int] = {}
        self._last: dict[str, tuple[int, float]] = {}
        self._closed = False

    def log_scalar(self, name: str, value: float, step: int) -> None:
        row = {
            "schema_version": 1,
            "name": name,
            "step": step,
            "value": value,
        }
        json.dump(
            row,
            self._handle,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._handle.write("\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._counts[name] = self._counts.get(name, 0) + 1
        self._last[name] = (step, value)

    def close(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        summary = {
            "schema_version": 1,
            "events": sum(self._counts.values()),
            "metrics": {
                name: {
                    "count": self._counts[name],
                    "last_step": self._last[name][0],
                    "last_value": self._last[name][1],
                }
                for name in sorted(self._counts, key=lambda item: item.encode("utf-8"))
            },
        }
        _write_json_new(self._summary_path, summary)
        self._closed = True


class MlflowTracker:
    def __init__(self, client: Any, run_id: str):
        self._client = client
        self._run_id = run_id
        self._history: dict[str, dict[int, float]] = {}

    def log_scalar(self, name: str, value: float, step: int) -> None:
        name, numeric_value, step = _normalize_scalar(name, value, step)
        history = self._history.get(name)
        if history is None:
            history = {}
            for metric in self._client.get_metric_history(self._run_id, name):
                if metric.step in history:
                    raise TrackerOwnershipError(
                        "MLflow metric history contains duplicate steps"
                    )
                history[metric.step] = float(metric.value)
            self._history[name] = history
        if step in history:
            if history[step] != numeric_value:
                raise TrackerOwnershipError(
                    "MLflow metric step disagrees with the resumed Run"
                )
            return
        self._client.log_metric(
            self._run_id,
            name,
            numeric_value,
            step=step,
            synchronous=True,
        )
        history[step] = numeric_value


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    request_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    request: Any = None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = _dispatch(request)
    except KeyboardInterrupt:
        result = {"status": "interrupted"}
    except Exception as error:
        phases = {
            "validate": "worker.validate",
            "train": "worker.train",
            "evaluate": "worker.evaluate",
            "export": "worker.export",
            "tracker": "worker.tracker",
            "tracker_metrics": "worker.tracker_metrics",
            "tracker_finish": "worker.tracker_finish",
        }
        operation = request.get("operation") if isinstance(request, dict) else None
        phase = (
            phases.get(operation, "worker") if isinstance(operation, str) else "worker"
        )
        report_exception(phase, error)
        if isinstance(error, TrackerOwnershipError):
            status = "ownership_conflict"
        elif isinstance(request, dict) and request.get("operation") == "validate":
            status = "contract_error"
        else:
            status = "execution_error"
        result = {"status": status, "error_type": type(error).__name__}
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    operation = request["operation"]
    if operation in {"tracker", "tracker_metrics", "tracker_finish"}:
        return _dispatch_tracker_operation(request)
    protocol = _prepare_action_protocol(request)

    if operation == "validate":
        return _handle_validate(request, protocol)

    if operation not in {"train", "evaluate", "export"}:
        raise ValueError("unknown worker operation")
    components, context, tracker_end = _prepare_action_context(request, protocol)
    if operation == "train":
        return _handle_train(request, components, context, tracker_end)
    if operation == "evaluate":
        return _handle_evaluate(request, components, context)
    return _handle_export(request, components, context)


@dataclass(frozen=True)
class ActionProtocol:
    entrypoint: Any
    selected: dict[str, str]
    action: str
    required: frozenset[str]
    cfg: Mapping[str, Any]


def _dispatch_tracker_operation(request: dict[str, Any]) -> dict[str, Any]:
    operation = request["operation"]
    if operation == "tracker":
        return _ensure_mlflow_run(request)
    if operation == "tracker_metrics":
        return _log_mlflow_metrics(request)
    return _finish_mlflow_run(request)


def _prepare_action_protocol(request: dict[str, Any]) -> ActionProtocol:
    source = Path(request["source"])
    entrypoint = _load_entrypoint(source)
    registry = _registry(entrypoint)
    selected = request["selected"]
    action = request["action"]
    _tracker_backends(request)
    required = ACTION_COMPONENTS[action]
    _resolve(registry, selected, required)
    return ActionProtocol(
        entrypoint, selected, action, required, _freeze(request["cfg"])
    )


def _handle_validate(
    request: dict[str, Any],
    protocol: ActionProtocol,
) -> dict[str, Any]:
    try:
        _validate_tracker_environment(request["cfg"], request)
    except ValueError as error:
        report_exception("worker.validate", error)
        return {"status": "contract_error", "error_type": type(error).__name__}
    except Exception as error:
        report_exception("worker.validate", error)
        return {"status": "execution_error", "error_type": type(error).__name__}
    try:
        normalized = protocol.entrypoint.validate(
            protocol.action,
            protocol.cfg,
            MappingProxyType(dict(protocol.selected)),
            MappingProxyType(dict(request["exec"])),
        )
        exec_info, identity = _normalized_validation(
            normalized,
            request["exec"],
            request.get("identity_fallback", {}),
        )
    except Exception as error:
        report_exception("worker.validate", error)
        return {"status": "contract_error", "error_type": type(error).__name__}
    return {"status": "ok", "exec": exec_info, "identity": identity}


def _prepare_action_context(
    request: dict[str, Any],
    protocol: ActionProtocol,
) -> tuple[Mapping[str, Any], RunContext, Any]:
    components = protocol.entrypoint.assemble(
        protocol.action,
        protocol.cfg,
        MappingProxyType(dict(protocol.selected)),
    )
    if not isinstance(components, Mapping) or set(components) != protocol.required:
        raise ValueError(f"{protocol.action} assembly returned invalid components")
    run_dir = Path(request["run_dir"]).absolute()
    export_dir = Path(
        request.get("export_dir", run_dir / "artifacts/export")
    ).absolute()
    tracker, tracker_end = _training_tracker(request)
    context = RunContext(
        cfg=protocol.cfg,
        paths=RunPaths(
            run_dir=run_dir,
            checkpoints=run_dir / "artifacts/checkpoints",
            metrics=run_dir / "metrics",
            results=Path(
                request.get("results_dir", run_dir / "artifacts/results")
            ).absolute(),
            export=export_dir,
        ),
        tracker=tracker,
        exec=ExecInfo(**request["exec"]),
        resume_from=(
            Path(request["resume_from"]).absolute()
            if request.get("resume_from") is not None
            else None
        ),
        components=MappingProxyType(dict(components)),
        _attempt_path=(
            Path(request["attempt_path"]).absolute()
            if request.get("attempt_path") is not None
            else None
        ),
    )
    return components, context, tracker_end


def _handle_train(
    request: dict[str, Any],
    components: Mapping[str, Any],
    context: RunContext,
    tracker_end: Any,
) -> dict[str, Any]:
    try:
        result = components["trainer"].fit(context)
        if isinstance(result, Mapping):
            best = result["best_checkpoint"]
            last = result["last_checkpoint"]
        else:
            best = result.best_checkpoint
            last = result.last_checkpoint
        response = {
            "status": "ok",
            "best_checkpoint": str(best),
            "last_checkpoint": str(last),
        }
        tracker_end(None)
        if "local" in _tracker_backends(request):
            response["metrics"] = {
                "history": "metrics/train.jsonl",
                "summary": "metrics/train-summary.json",
            }
        _record_worker_done(context._attempt_path, response)
        return response
    except KeyboardInterrupt:
        tracker_end("KILLED")
        raise
    except BaseException:
        tracker_end("FAILED")
        raise


def _handle_evaluate(
    request: dict[str, Any],
    components: Mapping[str, Any],
    context: RunContext,
) -> dict[str, Any]:
    """Accept structured results while retaining legacy flat scalar mappings."""
    result = components["evaluator"].evaluate(context, Path(request["checkpoint"]))
    if not isinstance(result, Mapping):
        raise ValueError("evaluator result must be a mapping")
    # A scalar metric named "values" still belongs to the legacy flat form.
    if set(result) == {"values", "files"} or (
        set(result) == {"values"} and isinstance(result["values"], Mapping)
    ):
        response = _evaluation_response(result)
    else:
        response = {"status": "ok", "values": dict(result), "files": []}
    _record_worker_done(context._attempt_path, response)
    return response


def _evaluation_response(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize structured output; omitted files mean no result artifacts."""
    values = result["values"]
    files = result.get("files", ())
    if not isinstance(values, Mapping):
        raise ValueError("evaluator values must be a mapping")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ValueError("evaluator files must be a sequence")
    return {
        "status": "ok",
        "values": dict(values),
        "files": [str(path) for path in files],
    }


def _handle_export(
    request: dict[str, Any],
    components: Mapping[str, Any],
    context: RunContext,
) -> dict[str, Any]:
    result = components["exporter"].export(context, Path(request["checkpoint"]))
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise ValueError("exporter result must be a sequence")
    response = {"status": "ok", "files": [str(path) for path in result]}
    _record_worker_done(context._attempt_path, response)
    return response


def _validate_tracker_environment(cfg: dict[str, Any], request: dict[str, Any]) -> None:
    del cfg
    if "mlflow" not in _tracker_backends(request):
        return
    uri = _tracking_uri(Path(request["repository_root"]))
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    client.get_experiment_by_name("__hkdl_preflight__")


def _ensure_mlflow_run(request: dict[str, Any]) -> dict[str, Any]:
    cfg = request["cfg"]
    if "mlflow" not in _tracker_backends(request):
        raise ValueError("tracker operation requires mlflow")
    uri = _tracking_uri(Path(request["repository_root"]))
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment_name = cfg["experiment"]["name"]
    experiment_id = _mlflow_experiment_id(client, experiment_name)
    run_id = Path(request["run_dir"]).name
    variant_name = cfg["variant"]["name"]
    run_key = _run_key(cfg, run_id)
    external_run_id = _owned_mlflow_run(
        client,
        experiment_id,
        run_key,
        request.get("current_tracker_run_id"),
    )
    if external_run_id is None:
        tags = _mlflow_tags(request, cfg, run_key, run_id, variant_name)
        created = client.create_run(
            experiment_id,
            tags=tags,
            run_name=f"{variant_name}/{run_id}",
        )
        external_run_id = created.info.run_id
    return {"status": "ok", "tracker_run_id": f"mlflow:{external_run_id}"}


def _mlflow_experiment_id(client: Any, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(experiment_name)
    except Exception:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise
        return experiment.experiment_id


def _owned_mlflow_run(
    client: Any,
    experiment_id: str,
    run_key: str,
    current: Any,
) -> str | None:
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`hkdl.run_key` = '{run_key}'",
        max_results=2,
    )
    if len(runs) > 1:
        raise TrackerOwnershipError("duplicate MLflow ownership identity")
    if current is None:
        return runs[0].info.run_id if runs else None
    if not isinstance(current, str) or not current.startswith("mlflow:"):
        raise ValueError("invalid persisted MLflow identity")
    expected = current.removeprefix("mlflow:")
    if len(runs) != 1 or runs[0].info.run_id != expected:
        raise TrackerOwnershipError("persisted MLflow ownership mismatch")
    return expected


def _mlflow_tags(
    request: dict[str, Any],
    cfg: dict[str, Any],
    run_key: str,
    run_id: str,
    variant_name: str,
) -> dict[str, str]:
    template = cfg["variant"]["template"]
    metadata = request.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("tracker metadata must be a mapping")
    tags = {
        "hkdl.run_key": run_key,
        "hkdl.address": f"{cfg['experiment']['name']}/{variant_name}/{run_id}",
        "hkdl.experiment": cfg["experiment"]["name"],
        "hkdl.variant": variant_name,
        "hkdl.run_id": run_id,
        "hkdl.source_digest": cfg["provenance"]["source_digest"],
        "hkdl.template.name": template["name"],
        "hkdl.template.version": template["version"],
    }
    for key in (
        "action",
        "training_group",
        "seed",
        "model_id",
        "evaluation_case",
        "retry_of",
        "experiment_hash",
        "experiment_revision_hash",
        "variant_hash",
        "variant_revision_hash",
        "option_set_hash",
        "run_spec_hash",
        "attempt_hash",
        "comparison_hash",
    ):
        value = metadata.get(key)
        if value is not None:
            tags[f"hkdl.{key}"] = str(value)
    return tags


def _log_mlflow_metrics(request: dict[str, Any]) -> dict[str, Any]:
    import mlflow

    uri = _tracking_uri(Path(request["repository_root"]))
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run_id = _external_tracker_id(request["tracker_run_id"])
    values = request["values"]
    if not isinstance(values, Mapping):
        raise ValueError("tracker values must be a mapping")
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("invalid tracker metric")
        client.log_metric(run_id, name, float(value), step=0, synchronous=True)
    return {"status": "ok"}


def _finish_mlflow_run(request: dict[str, Any]) -> dict[str, Any]:
    import mlflow

    uri = _tracking_uri(Path(request["repository_root"]))
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run_id = _external_tracker_id(request["tracker_run_id"])
    status = request["run_status"]
    if status not in {"FINISHED", "FAILED", "KILLED"}:
        raise ValueError("invalid tracker terminal status")
    client.set_terminated(run_id, status=status)
    return {"status": "ok"}


def _external_tracker_id(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("mlflow:"):
        raise ValueError("invalid tracker identity")
    return value.removeprefix("mlflow:")


# The returned callback closes the local metric sink when present. A non-null status
# also terminates MLflow; a null status leaves final completion to the parent
# after receipt publication.
def _training_tracker(request: dict[str, Any]) -> tuple[Any, Any]:
    """Create training metric sinks and their idempotent cleanup callback.

    The worker closes local metrics; the parent owns successful MLflow termination.
    """
    backends = _tracker_backends(request)
    if request["action"] != "train" or not backends:
        return NoopTracker(), lambda status: None
    sinks: list[Any] = []
    local = None
    if request["action"] == "train" and "local" in backends:
        local = LocalTracker(Path(request["run_dir"]) / "metrics")
        sinks.append(local)
    mlflow_module = None
    if "mlflow" in backends:
        tracker_run_id = request.get("tracker_run_id")
        if not isinstance(tracker_run_id, str) or not tracker_run_id.startswith(
            "mlflow:"
        ):
            raise ValueError("MLflow tracker identity is unavailable")
        uri = _tracking_uri(Path(request["repository_root"]))
        import mlflow

        mlflow.set_tracking_uri(uri)
        external_run_id = tracker_run_id.removeprefix("mlflow:")
        mlflow.start_run(run_id=external_run_id)
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        sinks.append(MlflowTracker(client, external_run_id))
        mlflow_module = mlflow
    ended = False

    def end(status: str | None) -> None:
        nonlocal ended
        if ended:
            return
        if local is not None:
            local.close()
        if status is not None and mlflow_module is not None:
            mlflow_module.end_run(status=status)
        ended = True

    return ScalarTracker(sinks), end


def _tracker_backends(request: Mapping[str, Any]) -> tuple[str, ...]:
    value = request.get("tracker_backends")
    if (
        not isinstance(value, (list, tuple))
        or any(
            not isinstance(item, str) or item not in {"local", "mlflow"}
            for item in value
        )
        or len(value) != len(set(value))
        or tuple(value) != tuple(item for item in ("local", "mlflow") if item in value)
    ):
        raise ValueError("invalid tracker backend protocol")
    return tuple(value)


def _tracking_uri(repository_root: Path) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI is required")
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            raise ValueError("MLflow HTTP tracking URI is invalid")
        return uri
    if (
        parsed.scheme != "sqlite"
        or not uri.startswith("sqlite:////")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("MLflow tracking URI is unsupported")
    database = Path(unquote(parsed.path))
    if not database.is_absolute():
        raise ValueError("MLflow SQLite path must be absolute")
    repository = repository_root.resolve(strict=True)
    resolved_database = database.resolve(strict=False)
    try:
        resolved_database.relative_to(repository)
    except ValueError:
        return uri
    raise ValueError("MLflow SQLite database must be outside the repository")


def _run_key(cfg: dict[str, Any], run_id: str) -> str:
    payload = {
        "experiment": cfg["experiment"]["name"],
        "variant": cfg["variant"]["name"],
        "run_id": run_id,
        "source_digest": cfg["provenance"]["source_digest"],
        "frozen_at": cfg["provenance"]["frozen_at"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _record_worker_done(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    journal = _load_journal(path)
    if journal["phase"] != "running":
        raise ValueError("attempt journal is not running")
    journal["phase"] = "worker_done"
    journal["result"] = result
    _write_journal(path, journal)


def _contained_relative_checkpoint(paths: RunPaths, path: Path) -> str:
    if not path.is_absolute():
        raise ValueError("reported checkpoint must be absolute")
    try:
        relative_input = path.relative_to(paths.run_dir)
        current = paths.run_dir
        for part in relative_input.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("reported checkpoint contains a symlink")
        resolved = path.resolve(strict=True)
        resolved.relative_to(paths.checkpoints.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("reported checkpoint is outside its root") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError("reported checkpoint is not a regular file")
    return resolved.relative_to(paths.run_dir.resolve(strict=True)).as_posix()


def _load_journal(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt journal is invalid")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("attempt journal must be a mapping")
    return document


def _write_journal(path: Path, document: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.replacement-",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_new(path: Path, document: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.candidate-",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(path_descriptor)
        finally:
            os.close(path_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _load_entrypoint(source: Path) -> Any:
    registry_path = source / "registry.py"
    if not registry_path.is_file() or registry_path.is_symlink():
        raise ValueError("registry.py is unavailable")
    module_name = f"_hkdl_variant_{id(registry_path)}"
    spec = importlib.util.spec_from_file_location(module_name, registry_path)
    if spec is None or spec.loader is None:
        raise ValueError("registry.py cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.entrypoint()


def _registry(entrypoint: Any) -> dict[tuple[str, str], Any]:
    registry: dict[tuple[str, str], Any] = {}
    for registration in entrypoint.registry():
        if not isinstance(registration, (tuple, list)) or len(registration) != 3:
            raise ValueError("invalid registry registration")
        kind, name, provider = registration
        if kind not in KINDS or not isinstance(name, str) or not NAMES.fullmatch(name):
            raise ValueError("invalid registry registration")
        key = (kind, name)
        if key in registry:
            raise ValueError("duplicate registry registration")
        registry[key] = provider
    return registry


def _resolve(
    registry: Mapping[tuple[str, str], Any],
    selected: Any,
    required: frozenset[str],
) -> None:
    if not isinstance(selected, dict) or set(selected) != required:
        raise ValueError("action selection contains invalid components")
    for kind, name in selected.items():
        if (kind, name) not in registry:
            raise ValueError(f"unresolved component: {kind}/{name}")


def _normalized_validation(
    normalized: Any,
    requested: Any,
    fallback: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(normalized, Mapping) and set(normalized) == {"exec", "identity"}:
        exec_info = _normalized_exec(normalized["exec"], requested)
        identity = normalized["identity"]
    else:
        exec_info = _normalized_exec(normalized, requested)
        identity = fallback
    if not isinstance(identity, Mapping):
        raise ValueError("validate identity must be a mapping")
    identity = _plain_json(identity)
    return exec_info, identity


def _normalized_exec(normalized: Any, requested: Any) -> dict[str, Any]:
    if not isinstance(normalized, Mapping) or set(normalized) != {"seed", "device"}:
        raise ValueError("validate must return seed and device")
    seed = normalized["seed"]
    device = normalized["device"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != requested["seed"]:
        raise ValueError("validate changed the seed")
    if not isinstance(device, str) or not device or device == "auto":
        raise ValueError("validate returned a non-concrete device")
    return {"seed": seed, "device": device}


def _plain_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("identity keys must be strings")
            result[key] = _plain_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    raise ValueError("identity must be JSON-compatible")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
