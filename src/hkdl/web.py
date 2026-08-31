"""Single-Experiment local inspection and authoring-commit web view."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from .authoring import Authoring, ExperimentRecord, VariantRecord
from .config import NAME_PATTERN, ContractError
from .research_json import split_legacy_variant
from .run_contracts import (
    TERMINAL_STATUSES,
    RUN_ID_PATTERN,
    evaluation_case,
    fingerprint_document,
    metric_spec,
    validate_tracker,
)
from .runs import RunRecord, RunStore
from .status import Status, _without_identities
from .status_index import IndexFailure
from .storage import NotFoundError, OwnershipError, RepositoryPaths
from .v2.authoring import V2Authoring
from .v2.execution import GraphRecorder
from .v2.reader import current_reader, graph_observation
from .v2.maintenance import WorkspaceBusy, workspace_access
from .v2.graph import (
    CURRENT_REVISION_NAME,
    DirtyDraftError,
    V2Graph,
    attempt_event_scope,
    variant_model_scope,
    variant_run_scope,
)
from .web_authoring import (
    _json_changes as _authoring_json_changes,
    build_authoring_state as _build_authoring_state,
)
from .web_runs import captured_run, worker_log_view


LOOPBACK_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
MIN_COMPARISON_RUNS = 2
MAX_COMPARISON_RUNS = 4
MAX_COMPARISON_POINTS = 2_000
MAX_MUTATION_BODY_BYTES = 4_096
MUTATION_HEADER = "X-HKDL-Action"
MUTATION_HEADER_VALUE = "commit"


class WebFailure(RuntimeError):
    """Local web server setup or serving failure."""


class WebRequestError(ValueError):
    """A web query or mutation violates the public request contract."""


@graph_observation
def build_authoring_state(
    repository: RepositoryPaths,
    experiment_name: str,
    *,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    return _build_authoring_state(repository, experiment_name, now=now)


def _json_changes(
    before: Any,
    after: Any,
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    return _authoring_json_changes(before, after, prefix=prefix)


@graph_observation
def build_comparison(
    repository: RepositoryPaths,
    experiment_name: str,
    run_addresses: Sequence[str],
    *,
    metric_name: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build a server-owned comparison from authoritative Run snapshots."""
    observed_at = (now or (lambda: datetime.now(timezone.utc)))()
    Authoring(repository).load_experiment(experiment_name)
    selections = _comparison_selections(run_addresses)
    store = RunStore(repository)
    graph = GraphRecorder(repository)
    records = [
        _comparison_record(store, graph, experiment_name, variant, run_id)
        for variant, run_id in selections
    ]
    metric_documents = [_comparison_metrics(store, graph, record) for record in records]
    available_metrics = sorted(
        {
            str(event["name"])
            for document in metric_documents
            for event in document["events"]
        },
        key=lambda value: value.encode("utf-8"),
    )
    selected_metric = _selected_metric(
        metric_name,
        available_metrics,
        metric_documents,
    )
    runs = [
        _comparison_run_view(record, document, selected_metric)
        for record, document in zip(records, metric_documents, strict=True)
    ]
    return {
        "observed_at": observed_at.isoformat(),
        "experiment": experiment_name,
        "selected_metric": selected_metric,
        "available_metrics": available_metrics,
        "runs": runs,
        "input_differences": _input_differences(records),
        "evaluation_groups": _evaluation_groups(store, graph, records),
    }


def _comparison_selections(values: Sequence[str]) -> list[tuple[str, str]]:
    if not MIN_COMPARISON_RUNS <= len(values) <= MAX_COMPARISON_RUNS:
        raise WebRequestError(
            f"comparison requires {MIN_COMPARISON_RUNS} to {MAX_COMPARISON_RUNS} Runs"
        )
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if not isinstance(value, str) or value.count("/") != 1:
            raise WebRequestError("comparison Run must be VARIANT/RUN")
        variant, run_id = value.split("/", 1)
        if not NAME_PATTERN.fullmatch(variant) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise WebRequestError("comparison Run must be VARIANT/RUN")
        selection = (variant, run_id)
        if selection in seen:
            raise WebRequestError("comparison Runs must be unique")
        seen.add(selection)
        result.append(selection)
    return result


@graph_observation
def build_run_detail(
    repository: RepositoryPaths,
    experiment_name: str,
    run_address: str,
    *,
    metric_name: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Inspect one Train/Eval/Export without changing inputs or lifecycle."""
    if not isinstance(run_address, str) or run_address.count("/") != 1:
        raise WebRequestError("Run must be VARIANT/RUN")
    variant, run_id = run_address.split("/")
    if not NAME_PATTERN.fullmatch(variant) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise WebRequestError("Run must be VARIANT/RUN")
    Authoring(repository).load_experiment(experiment_name)
    store, graph = RunStore(repository), GraphRecorder(repository)
    record, snapshots = captured_run(
        store.load(experiment_name, variant, run_id), graph
    )
    action = record.request["action"]
    metrics = None
    if action == "train":
        document = _comparison_metrics(store, graph, record)
        names = sorted(
            {event["name"] for event in document["events"]},
            key=lambda name: name.encode("utf-8"),
        )
        selected = _selected_metric(metric_name, names, [document])
        metrics = {
            "available_metrics": names,
            "selected_metric": selected,
            "series": _comparison_run_view(record, document, selected),
        }
    elif metric_name is not None:
        raise WebRequestError("metric selection requires a Train Run")
    return {
        "observed_at": (now or (lambda: datetime.now(timezone.utc)))().isoformat(),
        "experiment": experiment_name,
        "run": {
            "variant": variant,
            "run_id": run_id,
            "action": action,
            "status": record.state["status"],
            "reason": record.state["reason"],
            "created_at": record.request["created_at"],
            "updated_at": record.state["updated_at"],
            "seed": record.request["target"].get("seed"),
            "device": record.request["exec"]["device"],
        },
        "snapshots": snapshots,
        "metrics": metrics,
        "log": worker_log_view(store, graph, record),
    }


def _comparison_record(
    store: RunStore,
    graph: GraphRecorder,
    experiment: str,
    variant: str,
    run_id: str,
) -> RunRecord:
    record = store.load(experiment, variant, run_id)
    if graph.active():
        record = graph.authoritative_record(record)
    if record.request["action"] != "train":
        raise WebRequestError("comparison supports Train Runs only")
    return record


def _comparison_metrics(
    store: RunStore,
    graph: GraphRecorder,
    record: RunRecord,
) -> dict[str, Any]:
    trackers = validate_tracker(record.snapshot["variant"]["tracker"])
    if "local" not in trackers:
        return {
            "availability": "tracking_disabled",
            "partial": False,
            "events": [],
        }
    metrics = (
        graph.load_training_metrics(record)
        if graph.active() and record.state["status"] in TERMINAL_STATUSES
        else store.load_training_metrics(record)
    )
    return {
        "availability": "available" if metrics["events"] else "not_recorded",
        "partial": bool(metrics["partial"]),
        "events": list(metrics["events"]),
    }


def _selected_metric(
    requested: str | None,
    available: list[str],
    documents: list[dict[str, Any]],
) -> str | None:
    if requested is not None:
        if (
            not requested
            or len(requested) > 128
            or any(ord(character) < 32 for character in requested)
        ):
            raise WebRequestError("comparison metric is invalid")
        if requested not in available:
            raise WebRequestError("comparison metric is not available")
        return requested
    if not available:
        return None
    per_run = [
        {str(event["name"]) for event in document["events"]} for document in documents
    ]
    common = set.intersection(*per_run) if per_run else set()
    return min(common or set(available), key=lambda value: value.encode("utf-8"))


def _comparison_run_view(
    record: RunRecord,
    metrics: dict[str, Any],
    metric_name: str | None,
) -> dict[str, Any]:
    points = sorted(
        (
            {"step": int(event["step"]), "value": event["value"]}
            for event in metrics["events"]
            if event["name"] == metric_name
        ),
        key=lambda item: item["step"],
    )
    rendered = _downsample_points(points, limit=MAX_COMPARISON_POINTS)
    target = record.request["target"]
    availability = str(metrics["availability"])
    if metric_name is not None and not points and availability == "available":
        availability = "metric_not_recorded"
    return {
        "variant": record.request["variant"],
        "run_id": record.request["run_id"],
        "status": record.state["status"],
        "seed": target.get("seed"),
        "comparison_group": target.get("training_group"),
        "availability": availability,
        "partial": metrics["partial"],
        "last": points[-1] if points else None,
        "points_total": len(points),
        "points_returned": len(rendered),
        "points": rendered,
    }


def _downsample_points(
    points: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Preserve endpoints and each deterministic bucket's extrema."""
    if len(points) <= limit:
        return deepcopy(points)
    bucket_count = max(1, (limit - 2) // 2)
    interior = points[1:-1]
    bucket_size = max(1, (len(interior) + bucket_count - 1) // bucket_count)
    selected = [points[0]]
    for start in range(0, len(interior), bucket_size):
        bucket = interior[start : start + bucket_size]
        low = min(bucket, key=lambda item: (item["value"], item["step"]))
        high = max(bucket, key=lambda item: (item["value"], -item["step"]))
        selected.extend(
            sorted(
                {low["step"]: low, high["step"]: high}.values(),
                key=lambda item: item["step"],
            )
        )
    selected.append(points[-1])
    return (
        deepcopy(selected[: limit - 1] + [points[-1]])
        if len(selected) > limit
        else deepcopy(selected)
    )


def _input_differences(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    inputs = []
    for record in records:
        split = split_legacy_variant(record.snapshot["variant"])
        inputs.append({"code": split.code, "options": split.options})
    flattened = [_flatten_document(document) for document in inputs]
    observed_paths = {path for document in flattened for path in document}
    paths = sorted(
        (
            path
            for path in observed_paths
            if not any(candidate.startswith(f"{path}.") for candidate in observed_paths)
        ),
        key=lambda value: value.encode("utf-8"),
    )
    result = []
    for path in paths:
        cells = [
            {"present": path in document, "value": deepcopy(document.get(path))}
            for document in flattened
        ]
        comparable = [
            (cell["present"], cell["value"] if cell["present"] else None)
            for cell in cells
        ]
        if any(value != comparable[0] for value in comparable[1:]):
            result.append({"path": path, "values": cells})
    return result


def _flatten_document(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        if not value and prefix:
            result[prefix] = {}
        for key in sorted(value, key=lambda item: str(item).encode("utf-8")):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_document(value[key], prefix=path))
        return result
    return {prefix: deepcopy(value)}


def _evaluation_groups(
    store: RunStore,
    graph: GraphRecorder,
    records: Sequence[RunRecord],
) -> list[dict[str, Any]]:
    if graph.active():
        return _evaluation_groups_v2(graph, records)
    return _evaluation_groups_legacy(store, records)


def _evaluation_groups_v2(
    graph: GraphRecorder,
    records: Sequence[RunRecord],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_variant: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        key = (
            str(record.request["experiment"]),
            str(record.request["variant"]),
        )
        by_variant.setdefault(key, []).append(record)
    for (experiment, variant), selected in by_variant.items():
        experiment_hash = graph.graph.experiment_hash(experiment)
        variant_hash = graph.graph.variant_hash(experiment_hash, variant)
        selected_attempts = {
            graph.identity(record).attempt_hash: record for record in selected
        }
        models: dict[str, RunRecord] = {}
        for model_hash in graph.graph.bindings.names(
            variant_model_scope(variant_hash)
        ).values():
            model = graph.graph.store.load(model_hash)
            producing_attempt = model.payload.get("producing_attempt")
            if model.kind != "model" or not isinstance(producing_attempt, str):
                raise ContractError("v2 Model comparison payload is invalid")
            owner = selected_attempts.get(producing_attempt)
            if owner is not None:
                models[model_hash] = owner
        if not models:
            continue
        for run_id, attempt_hash in graph.graph.bindings.names(
            variant_run_scope(variant_hash)
        ).items():
            attempt = graph.graph.store.load(attempt_hash)
            if attempt.kind != "attempt":
                raise ContractError("v2 Run binding does not reference an Attempt")
            run_spec_hash = attempt.payload.get("run_spec")
            if not isinstance(run_spec_hash, str):
                raise ContractError("v2 Attempt RunSpec reference is invalid")
            run_spec = graph.graph.store.load(run_spec_hash)
            if run_spec.kind != "run_spec" or run_spec.payload.get("action") != "eval":
                continue
            model_hash = run_spec.payload.get("model")
            case_hash = run_spec.payload.get("evaluation_case")
            if not isinstance(model_hash, str) or not isinstance(case_hash, str):
                raise ContractError("v2 Eval RunSpec is invalid")
            train = models.get(model_hash)
            if train is None:
                continue
            event_hash = graph.graph.bindings.resolve(
                attempt_event_scope(attempt_hash),
                CURRENT_REVISION_NAME,
            )
            event = graph.graph.store.load(event_hash)
            if (
                event.kind != "attempt_event"
                or event.payload.get("attempt") != attempt_hash
            ):
                raise ContractError("v2 Attempt event ownership mismatch")
            status = event.payload.get("status")
            if not isinstance(status, str):
                raise ContractError("v2 Attempt event status is invalid")
            label = "Evaluation condition"
            values: dict[str, float | int] = {}
            result_hash = event.payload.get("result_object")
            if status == "done" and result_hash is None:
                raise ContractError("completed v2 Eval has no result object")
            if status != "done" and result_hash is not None:
                raise ContractError("non-completed v2 Eval has a result object")
            if result_hash is not None:
                if not isinstance(result_hash, str):
                    raise ContractError("v2 Eval result reference is invalid")
                result = graph.graph.store.load(result_hash)
                if (
                    result.kind != "eval_result"
                    or result.payload.get("attempt") != attempt_hash
                    or result.payload.get("model") != model_hash
                    or result.payload.get("evaluation_case") != case_hash
                ):
                    raise ContractError("v2 Eval result ownership mismatch")
                document = result.payload.get("document")
                if not isinstance(document, dict):
                    raise ContractError("v2 Eval result document is invalid")
                raw_label = document.get("evaluation_case")
                if isinstance(raw_label, str) and raw_label:
                    label = raw_label
                raw_values = document.get("values")
                if not isinstance(raw_values, dict) or any(
                    not isinstance(name, str)
                    or not name
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for name, value in raw_values.items()
                ):
                    raise ContractError("v2 Eval result values are invalid")
                values = dict(raw_values)
            grouped.setdefault((label, case_hash), []).append(
                {
                    "variant": variant,
                    "train_run_id": train.request["run_id"],
                    "eval_run_id": run_id,
                    "status": status,
                    "values": values,
                }
            )
    return _evaluation_group_documents(grouped)


def _evaluation_groups_legacy(
    store: RunStore,
    records: Sequence[RunRecord],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    models_by_variant: dict[str, list[Any]] = {}
    evaluations_by_variant: dict[str, list[RunRecord]] = {}
    for record in records:
        variant = str(record.request["variant"])
        experiment = str(record.request["experiment"])
        if variant not in models_by_variant:
            models_by_variant[variant] = store.scan_model_manifests(
                experiment=experiment,
                variant=variant,
            )
        models = models_by_variant[variant]
        model_ids = {
            str(model.document["model_id"])
            for model in models
            if model.document["producer_run"] == record.request["run_id"]
        }
        if not model_ids:
            continue
        if variant not in evaluations_by_variant:
            variant_runs = store.scan(experiment=experiment, variant=variant)
            evaluations_by_variant[variant] = [
                candidate
                for candidate in variant_runs
                if candidate.request["action"] == "eval"
            ]
        for evaluation in evaluations_by_variant[variant]:
            if evaluation.request["target"].get("model_id") not in model_ids:
                continue
            case = str(evaluation.request["target"]["evaluation_case"])
            values: dict[str, Any] = {}
            if evaluation.state["status"] == "done":
                values = dict(store.load_evaluation(evaluation)["values"])
            identity = fingerprint_document(
                {
                    "definition": evaluation_case(evaluation.snapshot["variant"], case),
                    "metrics": metric_spec(evaluation.snapshot["variant"], case),
                }
            )
            grouped.setdefault((case, identity), []).append(
                {
                    "variant": variant,
                    "train_run_id": record.request["run_id"],
                    "eval_run_id": evaluation.request["run_id"],
                    "status": evaluation.state["status"],
                    "values": values,
                }
            )
    return _evaluation_group_documents(grouped)


def _evaluation_group_documents(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    case_counts = Counter(case for case, _ in grouped)
    case_indexes: Counter[str] = Counter()
    result = []
    for case, identity in sorted(
        grouped,
        key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")),
    ):
        case_indexes[case] += 1
        result.append(
            {
                "case": case,
                "condition": (case_indexes[case] if case_counts[case] > 1 else None),
                "results": sorted(
                    grouped[(case, identity)],
                    key=lambda item: (
                        item["variant"].encode("utf-8"),
                        item["train_run_id"].encode("utf-8"),
                        item["eval_run_id"].encode("utf-8"),
                    ),
                ),
            }
        )
    return result


@graph_observation
def build_overview(
    repository: RepositoryPaths,
    experiment_name: str,
    *,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build one atomic observation without merging authored and generated identity."""
    observed_at = (now or (lambda: datetime.now(timezone.utc)))()
    authoring = Authoring(repository)
    experiment = authoring.load_experiment(experiment_name)
    variants = authoring.list_variants(experiment_name)
    generated_history = Status(repository, now=lambda: observed_at).query(
        experiment=experiment_name, include_identities=True
    )
    template = cast(dict[str, object], experiment.document["template"])
    graph = V2Graph(repository)
    experiment_payload = {
        "name": experiment.document["name"],
        "type": experiment.document["type"],
        "template": {"name": template["name"]},
    }
    experiment_view_payload = {
        **experiment_payload,
        "question": experiment.document["question"],
        "commit_state": _experiment_state(graph, experiment),
    }
    graph_active = graph.is_active()
    reader = current_reader(repository) if graph_active else None
    history_variants = {
        item["variant_hash"] if graph_active else item["name"]: item
        for history_experiment in generated_history["experiments"]
        for item in history_experiment["variants"]
        if history_experiment["name"] == experiment_name
    }
    authored_variants = {}
    labels = {key: item["name"] for key, item in history_variants.items()}
    for variant in variants:
        name = str(variant.document["name"])
        try:
            key = reader.entities(experiment_name, name)[1] if reader else name
        except NotFoundError:
            key = f"draft:{name}"
        authored_variants[key] = variant
        labels[key] = name
    names = sorted(
        set(authored_variants) | set(history_variants),
        key=lambda key: labels[key].encode(),
    )
    nested_variants = [
        _variant_view(
            repository,
            graph,
            experiment_name,
            authored_variants.get(name),
            history_variants.get(name),
            labels[name],
        )
        for name in names
    ]
    experiment_view = _without_identities(
        {**experiment_view_payload, "variants": nested_variants}
    )
    return {
        "observed_at": observed_at.isoformat(),
        "experiment": experiment_view,
        "authored_catalog": {
            "experiment": experiment_payload,
            "variants": [{"name": variant.document["name"]} for variant in variants],
        },
        "generated_history": _without_identities(generated_history),
    }


def _experiment_state(graph: V2Graph, experiment: ExperimentRecord) -> str:
    if not graph.is_active():
        return "uncommitted"
    try:
        graph.assert_experiment_clean(str(experiment.document["name"]), experiment)
    except (DirtyDraftError, NotFoundError):
        return "uncommitted"
    return "clean"


def _variant_view(
    repository: RepositoryPaths,
    graph: V2Graph,
    experiment_name: str,
    authored: VariantRecord | None,
    history: dict[str, Any] | None,
    name: str,
) -> dict[str, Any]:
    history = history or {"training_groups": []}
    if authored is None:
        return {
            "name": name,
            "history_only": True,
            "code_state": "locked",
            "code": {},
            "options": {},
            "runs": _lineage_runs(repository, experiment_name, name, history),
        }
    variant = authored
    code = (
        dict(variant.code_document)
        if variant.code_document is not None
        else dict(split_legacy_variant(variant.document).code)
    )
    options = (
        dict(variant.options_document)
        if variant.options_document is not None
        else dict(split_legacy_variant(variant.document).options)
    )
    state = "uncommitted"
    if graph.is_active():
        try:
            graph.assert_variant_clean(experiment_name, variant)
        except (DirtyDraftError, NotFoundError):
            try:
                experiment_hash = graph.experiment_hash(experiment_name)
                variant_hash = graph.variant_hash(experiment_hash, name)
                state = (
                    "locked"
                    if graph.has_active_execution(variant_hash)
                    else "uncommitted"
                )
            except NotFoundError:
                state = "uncommitted"
        else:
            state = "clean"
    return {
        "name": name,
        "code_state": state,
        "code": code,
        "options": options,
        "runs": _lineage_runs(repository, experiment_name, name, history),
    }


def _lineage_runs(
    repository: RepositoryPaths,
    experiment: str,
    variant: str,
    history: dict[str, Any],
) -> list[dict[str, Any]]:
    if V2Graph(repository).is_active():
        return _graph_lineage_runs(repository, experiment, variant, history)
    store = RunStore(repository)
    result: list[dict[str, Any]] = []
    for group in history.get("training_groups", []):
        for seed in group.get("seeds", []):
            model = seed.get("model")
            runs = list(seed.get("runs", []))
            downstream = [item for item in runs if item.get("action") != "train"]
            train_runs = [item for item in runs if item.get("action") == "train"]
            if not train_runs and model is not None:
                train_runs = [
                    {
                        "run_id": model.get("producer_run"),
                        "action": "train",
                        "status": "done",
                    }
                ]
            for run in train_runs:
                run_view = {
                    **run,
                    "name": run.get("run_id", "Train"),
                    "comparison_group": group.get("name"),
                    "seed": seed.get("seed"),
                }
                run_id = run.get("run_id")
                if isinstance(run_id, str):
                    try:
                        record = store.load(experiment, variant, run_id)
                        split = split_legacy_variant(record.snapshot["variant"])
                        run_view["code_snapshot"] = split.code
                        run_view["options_snapshot"] = split.options
                    except (ContractError, NotFoundError):
                        pass
                if model is not None:
                    run_view["model"] = {
                        **model,
                        "name": model.get("model_id", "Model"),
                        "evaluations": [
                            item for item in downstream if item.get("action") == "eval"
                        ],
                        "exports": [
                            item
                            for item in downstream
                            if item.get("action") == "export"
                        ],
                    }
                result.append(run_view)
            if not train_runs:
                result.extend(downstream)
    return result


def _graph_lineage_runs(repository, experiment, variant, history):
    reader = current_reader(repository)
    summaries = {}
    model_summaries = {}
    for group in history.get("training_groups", []):
        for seed in group.get("seeds", []):
            for summary in seed.get("runs", []):
                summaries[summary["attempt_hash"]] = summary
            if seed.get("model") is not None:
                model = seed["model"]
                model_summaries[model["model_hash"]] = model
    records = reader.runs(experiment=experiment, variant=variant)
    downstream = {}
    for record in records:
        if record.request["action"] != "train":
            spec = reader.object(record.graph_identity["run_spec_hash"], "run_spec")
            downstream.setdefault(spec["model"], []).append(
                summaries[record.graph_identity["attempt_hash"]]
            )
    models_by_producer = {}
    for model in reader.models(experiment, variant):
        payload = reader.object(model.graph_hash, "model")
        models_by_producer[payload["producing_attempt"]] = model
    result = []
    trains = sorted(
        (r for r in records if r.request["action"] == "train"),
        key=lambda r: (
            r.request["target"]["training_group"].encode(),
            r.request["target"]["seed"],
            int(r.request["run_id"].split("-")[1]),
        ),
    )
    for record in trains:
        identity = record.graph_identity["attempt_hash"]
        split = split_legacy_variant(record.snapshot["variant"])
        view = {
            **summaries[identity],
            "name": record.request["run_id"],
            "comparison_group": record.request["target"]["training_group"],
            "seed": record.request["target"]["seed"],
            "code_snapshot": split.code,
            "options_snapshot": split.options,
        }
        model = models_by_producer.get(identity)
        if model is not None:
            children = downstream.get(model.graph_hash, [])
            view["model"] = {
                **model_summaries[model.graph_hash],
                "name": model.document["model_id"],
                "evaluations": [item for item in children if item["action"] == "eval"],
                "exports": [item for item in children if item["action"] == "export"],
            }
        result.append(view)
    return _without_identities(result)


class _WebApplication:
    def __init__(
        self,
        repository: RepositoryPaths,
        experiment_name: str,
        *,
        now: Callable[[], datetime] | None,
    ):
        self.repository = repository
        self.experiment_name = experiment_name
        self.now = now
        try:
            root = resources.files("hkdl.web_ui")
            self.assets = {
                "/": (
                    "text/html; charset=utf-8",
                    root.joinpath("index.html").read_bytes(),
                ),
                "/assets/app.css": (
                    "text/css; charset=utf-8",
                    root.joinpath("app.css").read_bytes(),
                ),
                "/assets/app.js": (
                    "text/javascript; charset=utf-8",
                    root.joinpath("app.js").read_bytes(),
                ),
                "/assets/favicon.svg": (
                    "image/svg+xml",
                    root.joinpath("favicon.svg").read_bytes(),
                ),
            }
        except (FileNotFoundError, OSError) as error:
            raise WebFailure(f"web UI assets are unavailable: {error}") from error

    def overview(self) -> dict[str, Any]:
        return build_overview(
            self.repository,
            self.experiment_name,
            now=self.now,
        )

    def comparison(
        self,
        run_addresses: Sequence[str],
        *,
        metric_name: str | None,
    ) -> dict[str, Any]:
        return build_comparison(
            self.repository,
            self.experiment_name,
            run_addresses,
            metric_name=metric_name,
            now=self.now,
        )

    def authoring(self) -> dict[str, Any]:
        return build_authoring_state(
            self.repository,
            self.experiment_name,
            now=self.now,
        )

    def run_detail(self, address: str, metric_name: str | None) -> dict[str, Any]:
        return build_run_detail(
            self.repository,
            self.experiment_name,
            address,
            metric_name=metric_name,
            now=self.now,
        )

    def commit_experiment(self) -> dict[str, Any]:
        identity = V2Authoring(Authoring(self.repository)).commit_experiment(
            self.experiment_name
        )
        return {
            "kind": "experiment",
            "name": self.experiment_name,
            "changed": identity.changed,
            "authoring": self.authoring(),
        }

    def commit_variant(self, variant: str) -> dict[str, Any]:
        identity = V2Authoring(Authoring(self.repository)).commit_variant(
            self.experiment_name,
            variant,
        )
        return {
            "kind": "variant",
            "name": variant,
            "changed": identity.changed,
            "authoring": self.authoring(),
        }


class _WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: _WebApplication,
    ):
        self.application = application
        super().__init__(server_address, _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def application(self) -> _WebApplication:
        return cast(_WebServer, self.server).application

    def do_GET(self) -> None:  # noqa: N802
        self._handle(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle(head_only=True)

    def do_POST(self) -> None:  # noqa: N802
        try:
            with workspace_access(self.application.repository):
                self._handle_post()
        except WorkspaceBusy as error:
            self.close_connection = True
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "workspace_maintenance",
                str(error),
                head_only=False,
            )

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _handle(self, *, head_only: bool) -> None:
        try:
            with workspace_access(self.application.repository):
                self._handle_admitted(head_only=head_only)
        except WorkspaceBusy as error:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "workspace_maintenance",
                str(error),
                head_only=head_only,
            )

    def _handle_admitted(self, *, head_only: bool) -> None:
        target = urlsplit(self.path)
        path = target.path
        if path == "/api/v1/overview":
            self._overview(head_only=head_only)
            return
        if path == "/api/v1/comparison":
            self._comparison(target.query, head_only=head_only)
            return
        if path == "/api/v1/run":
            self._run_detail(target.query, head_only=head_only)
            return
        if path == "/api/v1/authoring":
            self._authoring(head_only=head_only)
            return
        if path in {
            "/api/v1/authoring/experiment/commit",
            "/api/v1/authoring/variant/commit",
        }:
            self._method_not_allowed(allow="POST", head_only=head_only)
            return
        asset = self.application.assets.get(path)
        if asset is not None:
            content_type, body = asset
            self._respond(
                HTTPStatus.OK,
                body,
                content_type=content_type,
                head_only=head_only,
            )
            return
        self._error(
            HTTPStatus.NOT_FOUND,
            "not_found",
            "resource not found",
            head_only=head_only,
        )

    def _overview(self, *, head_only: bool) -> None:
        try:
            document = self.application.overview()
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                str(error),
                head_only=head_only,
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=head_only,
            )
            return
        except IndexFailure as error:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "status_unavailable",
                str(error),
                head_only=head_only,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _comparison(self, raw_query: str, *, head_only: bool) -> None:
        try:
            query = parse_qs(raw_query, keep_blank_values=True, max_num_fields=8)
        except ValueError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=head_only,
            )
            return
        try:
            unexpected = set(query) - {"run", "metric"}
            if unexpected or len(query.get("metric", [])) > 1:
                raise WebRequestError("comparison query fields are invalid")
            document = self.application.comparison(
                query.get("run", []),
                metric_name=(query.get("metric") or [None])[0],
            )
        except WebRequestError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=head_only,
            )
            return
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                str(error),
                head_only=head_only,
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=head_only,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _run_detail(self, raw_query: str, *, head_only: bool) -> None:
        try:
            query = parse_qs(raw_query, keep_blank_values=True, max_num_fields=2)
        except ValueError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=head_only,
            )
            return
        try:
            if (
                set(query) - {"run", "metric"}
                or len(query.get("run", [])) != 1
                or len(query.get("metric", [])) > 1
            ):
                raise WebRequestError("Run detail query fields are invalid")
            document = self.application.run_detail(
                query["run"][0], (query.get("metric") or [None])[0]
            )
        except WebRequestError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=head_only,
            )
            return
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND, "not_found", str(error), head_only=head_only
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=head_only,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _authoring(self, *, head_only: bool) -> None:
        try:
            document = self.application.authoring()
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                str(error),
                head_only=head_only,
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=head_only,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _handle_post(self) -> None:
        path = urlsplit(self.path).path
        if path not in {
            "/api/v1/authoring/experiment/commit",
            "/api/v1/authoring/variant/commit",
        }:
            self._method_not_allowed()
            return
        try:
            payload = self._mutation_document()
            if path == "/api/v1/authoring/experiment/commit":
                if payload:
                    raise WebRequestError(
                        "Experiment commit body must be an empty object"
                    )
                document = self.application.commit_experiment()
            else:
                if set(payload) != {"variant"} or not isinstance(
                    payload.get("variant"), str
                ):
                    raise WebRequestError(
                        "Variant commit body must contain only a Variant name"
                    )
                variant = str(payload["variant"])
                if not NAME_PATTERN.fullmatch(variant):
                    raise WebRequestError("Variant commit name is invalid")
                document = self.application.commit_variant(variant)
        except WebRequestError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=False,
            )
            return
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                str(error),
                head_only=False,
            )
            return
        except DirtyDraftError as error:
            self._error(
                HTTPStatus.CONFLICT,
                "conflict",
                str(error),
                head_only=False,
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=False,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=False,
        )

    def _mutation_document(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            raise WebRequestError("Commit request transfer encoding is unsupported")
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError as error:
            self.close_connection = True
            raise WebRequestError("Commit request Content-Length is invalid") from error
        if length < 0 or length > MAX_MUTATION_BODY_BYTES:
            self.close_connection = True
            raise WebRequestError("Commit request body is too large")
        encoded = self.rfile.read(length)
        if self.headers.get(MUTATION_HEADER) != MUTATION_HEADER_VALUE:
            raise WebRequestError(
                "Commit request is missing its same-origin action header"
            )
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise WebRequestError("Commit request must use application/json")
        try:
            decoded = encoded.decode("utf-8")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WebRequestError("Commit request body is not valid JSON") from error
        if not isinstance(payload, dict):
            raise WebRequestError("Commit request body must be an object")
        return cast(dict[str, Any], payload)

    def _method_not_allowed(
        self,
        *,
        allow: str = "GET, HEAD",
        head_only: bool = False,
    ) -> None:
        if (
            self.headers.get("Content-Length") not in (None, "0")
            or self.headers.get("Transfer-Encoding") is not None
        ):
            self.close_connection = True
        body = _json_bytes(
            {
                "error": {
                    "code": "method_not_allowed",
                    "message": f"method not allowed; use {allow}",
                }
            }
        )
        self._respond(
            HTTPStatus.METHOD_NOT_ALLOWED,
            body,
            content_type="application/json; charset=utf-8",
            extra_headers={"Allow": allow},
            head_only=head_only,
        )

    def _error(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        *,
        head_only: bool,
    ) -> None:
        self._respond(
            status,
            _json_bytes({"error": {"code": code, "message": message}}),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _respond(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str,
        head_only: bool,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; base-uri 'none'; "
            "form-action 'none'",
        )
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)


def create_server(
    repository: RepositoryPaths,
    experiment_name: str,
    *,
    port: int = DEFAULT_WEB_PORT,
    now: Callable[[], datetime] | None = None,
) -> _WebServer:
    """Validate the selected Experiment, then bind a loopback-only server."""
    build_overview(repository, experiment_name, now=now)
    application = _WebApplication(repository, experiment_name, now=now)
    try:
        return _WebServer((LOOPBACK_HOST, port), application)
    except OSError as error:
        raise WebFailure(
            f"could not bind http://{LOOPBACK_HOST}:{port}: {error}"
        ) from error


def serve(
    repository: RepositoryPaths,
    experiment_name: str,
    *,
    port: int = DEFAULT_WEB_PORT,
) -> None:
    """Serve one selected Experiment until interrupted."""
    server = create_server(repository, experiment_name, port=port)
    actual_port = server.server_port
    print(
        f"serving {experiment_name} at http://{LOOPBACK_HOST}:{actual_port}/",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
