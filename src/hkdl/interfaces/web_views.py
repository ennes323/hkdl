"""Read-only web view construction from one captured workspace observation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, cast

from hkdl.authoring.authoring import Authoring
from hkdl.authoring.authoring_records import ExperimentRecord, VariantRecord
from hkdl.authoring.config import NAME_PATTERN
from hkdl.authoring.research_json import split_legacy_variant
from hkdl.errors import ContractError
from hkdl.execution.run_contracts import (
    RUN_ID_PATTERN,
    evaluation_case,
    fingerprint_document,
    metric_spec,
    validate_tracker,
)
from hkdl.execution.run_records import RunRecord
from hkdl.storage.graph.graph import DirtyDraftError, V2Graph
from hkdl.storage.graph.reader import GraphReader, current_reader, graph_observation
from hkdl.storage.runs import RunStore
from hkdl.storage.status import Status, _without_identities
from hkdl.storage.storage import NotFoundError, RepositoryPaths

from .web_runs import captured_run, worker_log_view

MIN_COMPARISON_RUNS = 2
MAX_COMPARISON_RUNS = 4
MAX_COMPARISON_POINTS = 2_000


class WebRequestError(ValueError):
    """A web query or mutation violates the public request contract."""


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
    reader = current_reader(repository) if V2Graph(repository).is_active() else None
    records = [
        _comparison_record(store, reader, experiment_name, variant, run_id)
        for variant, run_id in selections
    ]
    metric_documents = [_comparison_metrics(store, record) for record in records]
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
        "evaluation_groups": _evaluation_groups(store, reader, records),
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
    store = RunStore(repository)
    reader = current_reader(repository) if V2Graph(repository).is_active() else None
    record, snapshots = captured_run(
        store.load(experiment_name, variant, run_id), reader
    )
    action = record.request["action"]
    metrics = None
    if action == "train":
        document = _comparison_metrics(store, record)
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
        "log": worker_log_view(store, record),
    }


def _comparison_record(
    store: RunStore,
    reader: GraphReader | None,
    experiment: str,
    variant: str,
    run_id: str,
) -> RunRecord:
    record = store.load(experiment, variant, run_id)
    if reader is not None:
        record = reader.authoritative_record(record)
    if record.request["action"] != "train":
        raise WebRequestError("comparison supports Train Runs only")
    return record


def _comparison_metrics(
    store: RunStore,
    record: RunRecord,
) -> dict[str, Any]:
    trackers = validate_tracker(record.snapshot["variant"]["tracker"])
    if "local" not in trackers:
        return {
            "availability": "tracking_disabled",
            "partial": False,
            "events": [],
        }
    metrics = store.load_training_metrics(record)
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
    return sorted(
        _input_changes([(True, document) for document in inputs]),
        key=lambda item: item["path"].encode("utf-8"),
    )


def _input_changes(
    values: Sequence[tuple[bool, Any]],
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Compare complete JSON nodes before descending into shared object shapes.

    A mapping/leaf change stays at its parent so neither captured value is lost.
    Comparison preserves JSON categories, including inside lists and objects.
    """
    if all(
        present == values[0][0] and _same_json_value(value, values[0][1])
        for present, value in values
    ):
        return []
    if all(present and isinstance(value, Mapping) for present, value in values):
        keys = {key for _, value in values for key in value}
        if keys:
            result = []
            for key in sorted(keys, key=lambda item: item.encode("utf-8")):
                # Bracket notation keeps literal dots and delimiters in a key
                # distinct from structural path separators.
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
                    path = f"{prefix}.{key}" if prefix else key
                else:
                    path = f"{prefix}[{json.dumps(key, ensure_ascii=False)}]"
                children = [
                    (True, value[key]) if present and key in value else (False, None)
                    for present, value in values
                ]
                result.extend(_input_changes(children, prefix=path))
            return result
    return [
        {
            "path": prefix,
            "values": [
                {"present": present, "value": deepcopy(value)}
                for present, value in values
            ],
        }
    ]


def _same_json_value(before: Any, after: Any) -> bool:
    """Compare JSON categories without treating booleans as numeric values."""
    if type(before) is not type(after):
        return (
            type(before) in (int, float)
            and type(after) in (int, float)
            and before == after
        )
    if isinstance(before, Mapping):
        return before.keys() == after.keys() and all(
            _same_json_value(value, after[key]) for key, value in before.items()
        )
    if isinstance(before, list):
        return len(before) == len(after) and all(
            _same_json_value(left, right) for left, right in zip(before, after)
        )
    return before == after


def _evaluation_groups(
    store: RunStore,
    reader: GraphReader | None,
    records: Sequence[RunRecord],
) -> list[dict[str, Any]]:
    if reader is not None:
        return _evaluation_groups_v2(reader, records)
    return _evaluation_groups_legacy(store, records)


def _evaluation_groups_v2(
    reader: GraphReader,
    records: Sequence[RunRecord],
) -> list[dict[str, Any]]:
    """Group validated Eval facts using the existing presentation rules."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in reader.training_evaluations(records):
        document = row.document or {}
        raw_label = document.get("evaluation_case")
        label = (
            raw_label
            if isinstance(raw_label, str) and raw_label
            else "Evaluation condition"
        )
        grouped.setdefault((label, row.case_hash), []).append(
            {
                "variant": row.variant,
                "train_run_id": row.train_run_id,
                "eval_run_id": row.eval_run_id,
                "status": row.status,
                "values": dict(document.get("values", {})),
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
