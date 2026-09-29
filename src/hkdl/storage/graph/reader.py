"""One immutable binding observation, independent of generated Run documents.

Paths on compatibility records are operational locators, never catalog or
identity authority. Only old Attempts without captured evidence may consult
the isolated legacy reader. A broken captured reference never falls back.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

from hkdl.authoring.research_json import load_json_file, split_legacy_variant
from hkdl.errors import ContractError
from hkdl.execution.run_contracts import (
    TRAINING_METRIC_EVENT_FIELDS,
    training_metric_summary,
    validate_evaluation,
    validate_model,
    validate_request,
    validate_snapshot,
    validate_state,
    validate_tracker,
    validate_training_metric_event,
    validate_training_metric_summary,
)
from hkdl.execution.run_records import ModelRecord, RunRecord, run_sort_key
from hkdl.storage.runs import RunStore
from hkdl.storage.storage import NotFoundError, RepositoryPaths

from .graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    attempt_event_scope,
    entity_revision_scope,
    experiment_variant_scope,
    variant_model_scope,
    variant_run_scope,
    workspace_experiment_scope,
)
from .maintenance import workspace_access
from .records import ExecutionIdentity, TrainingEvaluation

_OBSERVATION: ContextVar[GraphReader | None] = ContextVar(
    "hkdl_graph_observation", default=None
)


def current_reader(repository: RepositoryPaths) -> GraphReader:
    reader = _OBSERVATION.get()
    return (
        reader
        if reader is not None and reader.repository == repository
        else GraphReader(repository)
    )


def graph_observation(function):
    """Pin one HEAD across a web observation or Status method (including joins)."""

    @wraps(function)
    def observed(owner, *args, **kwargs):
        repository = (
            owner if isinstance(owner, RepositoryPaths) else owner.store.repository
        )
        with workspace_access(repository):
            if not V2Graph(repository).is_active():
                return function(owner, *args, **kwargs)
            reader = current_reader(repository)
            token = _OBSERVATION.set(reader)
            try:
                return function(owner, *args, **kwargs)
            finally:
                _OBSERVATION.reset(token)

    return observed


class GraphReader:
    def __init__(self, repository: RepositoryPaths):
        self.repository = repository
        self.graph = V2Graph(repository)
        self.head = self.graph.bindings.head()
        self.bindings = (
            self.graph.bindings.bindings(self.head) if self.head is not None else {}
        )
        self._bindings_by_scope: dict[str, dict[str, str]] = {}
        self._names_by_target: dict[tuple[str, str], list[str]] = {}
        for (scope, name), digest in self.bindings.items():
            self._bindings_by_scope.setdefault(scope, {})[name] = digest
            self._names_by_target.setdefault((scope, digest), []).append(name)
        self._objects: dict[str, Any] = {}
        self._runs: dict[tuple[str, str, str], RunRecord] = {}
        self._source_digests: dict[str, str] = {}
        self.legacy = RunStore(repository, graph_reads=False)

    def identity(self, record: RunRecord) -> ExecutionIdentity:
        if record.graph_identity is not None:
            return ExecutionIdentity(**record.graph_identity)
        experiment_hash = self.resolve(
            workspace_experiment_scope(), str(record.request["experiment"])
        )
        variant_hash = self.resolve(
            experiment_variant_scope(experiment_hash),
            str(record.request["variant"]),
        )
        attempt_hash = self.resolve(
            variant_run_scope(variant_hash),
            str(record.request["run_id"]),
        )
        attempt = self.graph.store.load(attempt_hash)
        if attempt.kind != "attempt":
            raise ContractError("v2 Run binding does not reference an Attempt")
        run_spec_hash = str(attempt.payload["run_spec"])
        raw_option_set = attempt.payload.get("option_set")
        option_set_hash = str(raw_option_set) if raw_option_set is not None else None
        run_spec = self.graph.store.load(run_spec_hash)
        revision_field = {
            "train": "variant_revision",
            "eval": "evaluator_revision",
            "export": "exporter_revision",
        }[str(run_spec.payload["action"])]
        revision_hash = str(run_spec.payload[revision_field])
        comparison_hash = run_spec.payload.get("comparison_hash")
        return ExecutionIdentity(
            experiment_hash,
            str(
                run_spec.payload.get(
                    "experiment_revision",
                    self.resolve(
                        entity_revision_scope(experiment_hash), CURRENT_REVISION_NAME
                    ),
                )
            ),
            variant_hash,
            revision_hash,
            option_set_hash,
            run_spec_hash,
            attempt_hash,
            str(comparison_hash) if comparison_hash is not None else None,
        )

    def model_hash(self, experiment: str, variant: str, model_id: str) -> str:
        experiment_hash = self.resolve(workspace_experiment_scope(), experiment)
        variant_hash = self.resolve(experiment_variant_scope(experiment_hash), variant)
        return self.resolve(variant_model_scope(variant_hash), model_id)

    def authoritative_record(self, record: RunRecord) -> RunRecord:
        """Reuse a selected event; otherwise reconstruct from this observation."""
        if record.event_hash is not None:
            if record.graph_identity is not None:
                return record
            return replace(record, graph_identity=self.identity(record).as_dict())
        return self.run(
            str(record.request["experiment"]),
            str(record.request["variant"]),
            str(record.request["run_id"]),
        )

    def captured_inputs(self, record: RunRecord) -> dict[str, Any]:
        """Describe the already reconstructed immutable Code and Options."""
        record = self.authoritative_record(record)
        split = split_legacy_variant(record.snapshot["variant"])
        return {
            "source": (
                "content_addressed"
                if self.identity(record).option_set_hash is not None
                else "legacy_snapshot"
            ),
            "code": split.code,
            "options": split.options,
        }

    def load_training_metrics(self, record: RunRecord) -> dict[str, Any]:
        if record.request["action"] != "train":
            raise ContractError("Run is not a Train Run")
        if "local" not in validate_tracker(record.snapshot["variant"]["tracker"]):
            raise ContractError("Train Run does not use local tracking")
        try:
            path = self.artifact(record, "metrics/train.jsonl")
        except NotFoundError:
            if record.state["status"] == "done":
                raise ContractError(
                    f"completed Train Run has no metric history: {record.address}"
                )
            return {"partial": False, "events": [], "summary": {}}
        events: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ContractError("v2 Train metric history is unavailable") from error
        partial = bool(content) and not content.endswith(b"\n")
        lines = content.splitlines(keepends=True)
        if partial:
            lines = lines[:-1]
        for raw_line in lines:
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ContractError(
                    "v2 Train metric history is invalid JSON"
                ) from error
            if (
                not isinstance(event, dict)
                or set(event) != TRAINING_METRIC_EVENT_FIELDS
            ):
                raise ContractError("v2 Train metric event fields are invalid")
            try:
                validate_training_metric_event(event)
            except ContractError:
                raise ContractError("v2 Train metric event is invalid") from None
            key = (event["name"], event["step"])
            if key in seen:
                raise ContractError("v2 Train metric event is invalid")
            seen.add(key)
            events.append(event)
        summary_document = training_metric_summary(events)
        try:
            summary_path = self.artifact(record, "metrics/train-summary.json")
        except NotFoundError:
            if record.state["status"] == "done":
                raise ContractError(
                    f"completed Train Run has no metric summary: {record.address}"
                )
        else:
            try:
                persisted = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ContractError("v2 Train metric summary is unavailable") from error
            validate_training_metric_summary(persisted)
            if persisted != summary_document:
                raise ContractError("Train metric summary disagrees with history")
        if record.state["status"] == "done" and partial:
            raise ContractError("completed Train metric history is partial")
        return {
            "partial": partial,
            "events": events,
            "summary": summary_document["metrics"],
        }

    # Join selected Train attempts through Models to pinned Eval events/results.
    def training_evaluations(
        self, records: Sequence[RunRecord]
    ) -> list[TrainingEvaluation]:
        """Validate graph ownership without widening the selected result scope."""
        observations: list[TrainingEvaluation] = []
        by_variant: dict[tuple[str, str], list[RunRecord]] = {}
        for record in records:
            key = (
                str(record.request["experiment"]),
                str(record.request["variant"]),
            )
            by_variant.setdefault(key, []).append(record)
        for (experiment, variant), selected in by_variant.items():
            _, variant_hash = self.entities(experiment, variant)
            selected_attempts = {
                self.resolve(
                    variant_run_scope(variant_hash), str(record.request["run_id"])
                ): record
                for record in selected
            }
            models: dict[str, RunRecord] = {}
            for model_hash in self.names(variant_model_scope(variant_hash)).values():
                model = self.graph.store.load(model_hash)
                producing_attempt = model.payload.get("producing_attempt")
                if model.kind != "model" or not isinstance(producing_attempt, str):
                    raise ContractError("v2 Model comparison payload is invalid")
                owner = selected_attempts.get(producing_attempt)
                if owner is not None:
                    models[model_hash] = owner
            if not models:
                continue
            for run_id, attempt_hash in self.names(
                variant_run_scope(variant_hash)
            ).items():
                attempt = self.graph.store.load(attempt_hash)
                if attempt.kind != "attempt":
                    raise ContractError("v2 Run binding does not reference an Attempt")
                run_spec_hash = attempt.payload.get("run_spec")
                if not isinstance(run_spec_hash, str):
                    raise ContractError("v2 Attempt RunSpec reference is invalid")
                run_spec = self.graph.store.load(run_spec_hash)
                if (
                    run_spec.kind != "run_spec"
                    or run_spec.payload.get("action") != "eval"
                ):
                    continue
                model_hash = run_spec.payload.get("model")
                case_hash = run_spec.payload.get("evaluation_case")
                if not isinstance(model_hash, str) or not isinstance(case_hash, str):
                    raise ContractError("v2 Eval RunSpec is invalid")
                train = models.get(model_hash)
                if train is None:
                    continue
                event_hash = self.resolve(
                    attempt_event_scope(attempt_hash),
                    CURRENT_REVISION_NAME,
                )
                event = self.graph.store.load(event_hash)
                if (
                    event.kind != "attempt_event"
                    or event.payload.get("attempt") != attempt_hash
                ):
                    raise ContractError("v2 Attempt event ownership mismatch")
                status = event.payload.get("status")
                if not isinstance(status, str):
                    raise ContractError("v2 Attempt event status is invalid")
                document = None
                result_hash = event.payload.get("result_object")
                if status == "done" and result_hash is None:
                    raise ContractError("completed v2 Eval has no result object")
                if status != "done" and result_hash is not None:
                    raise ContractError("non-completed v2 Eval has a result object")
                if result_hash is not None:
                    if not isinstance(result_hash, str):
                        raise ContractError("v2 Eval result reference is invalid")
                    result = self.graph.store.load(result_hash)
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
                observations.append(
                    TrainingEvaluation(
                        variant,
                        str(train.request["run_id"]),
                        run_id,
                        status,
                        case_hash,
                        deepcopy(document),
                    )
                )
        return observations

    def source_digest(self, digest: str) -> str:
        if digest in self._source_digests:
            return self._source_digests[digest]
        source = self.object(digest, "source_tree")
        if source.get("availability") == "unavailable":
            result = source["legacy_digest"]
        elif source.get("availability") == "available":
            framed = hashlib.sha256()
            seen = set()
            for entry in sorted(
                source["files"], key=lambda item: item["path"].encode("utf-8")
            ):
                path = PurePosixPath(entry["path"])
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.as_posix() != entry["path"]
                    or entry["path"] in seen
                ):
                    raise ContractError("v2 source tree path is invalid")
                seen.add(entry["path"])
                blob = self.object(entry["blob"], "blob")
                if any(
                    blob[field] != entry[field] for field in ("content_hash", "size")
                ):
                    raise ContractError("v2 source tree blob evidence disagrees")
                content = self.graph.store.verify_blob(entry["blob"]).read_bytes()
                relative = entry["path"].encode("utf-8")
                framed.update(struct.pack(">Q", len(relative)))
                framed.update(relative)
                framed.update(struct.pack(">Q", len(content)))
                framed.update(content)
            result = f"sha256:{framed.hexdigest()}"
        else:
            raise ContractError("v2 source availability is invalid")
        self._source_digests[digest] = result
        return result

    def object(self, digest: str, kind: str) -> dict[str, Any]:
        if digest not in self._objects:
            self._objects[digest] = self.graph.store.load(digest)
        record = self._objects[digest]
        if record.kind != kind:
            raise ContractError(f"expected v2 {kind}, found {record.kind}")
        return deepcopy(record.payload)

    def names(self, scope: str) -> dict[str, str]:
        return dict(self._bindings_by_scope.get(scope, {}))

    def resolve(self, scope: str, name: str) -> str:
        try:
            return self.bindings[scope, name]
        except KeyError as error:
            raise NotFoundError(f"name binding not found: {scope}/{name}") from error

    def name(self, scope: str, digest: str) -> str:
        names = self._names_by_target.get((scope, digest), [])
        if len(names) != 1:
            raise ContractError(f"v2 active binding is missing or ambiguous: {scope}")
        return names[0]

    def entities(self, experiment: str, variant: str) -> tuple[str, str]:
        experiment_hash = self.resolve(workspace_experiment_scope(), experiment)
        self.object(experiment_hash, "experiment")
        variant_hash = self.resolve(experiment_variant_scope(experiment_hash), variant)
        if self.object(variant_hash, "variant")["experiment"] != experiment_hash:
            raise ContractError("v2 Variant ownership mismatch")
        return experiment_hash, variant_hash

    def run(self, experiment: str, variant: str, name: str) -> RunRecord:
        key = experiment, variant, name
        if key in self._runs:
            return self._runs[key]
        experiment_hash, variant_hash = self.entities(experiment, variant)
        attempt_hash = self.resolve(variant_run_scope(variant_hash), name)
        attempt = self.object(attempt_hash, "attempt")
        spec_hash = attempt["run_spec"]
        spec = self.object(spec_hash, "run_spec")
        event_hash = self.resolve(
            attempt_event_scope(attempt_hash), CURRENT_REVISION_NAME
        )
        event = self.object(event_hash, "attempt_event")
        if event["attempt"] != attempt_hash:
            raise ContractError("v2 Attempt event ownership mismatch")
        revision_field = {
            "train": "variant_revision",
            "eval": "evaluator_revision",
            "export": "exporter_revision",
        }.get(spec["action"])
        if revision_field is None:
            raise ContractError("unsupported v2 Run action")
        revision_hash = spec[revision_field]
        revision = self.object(revision_hash, "variant_revision")
        if revision["variant"] != variant_hash:
            raise ContractError("v2 Run Code ownership mismatch")
        evidence_hash = attempt.get("record_evidence")
        if evidence_hash is None:
            address = attempt.get("legacy_evidence", {}).get("address")
            if not isinstance(address, str) or len(address.split("/")) != 3:
                raise ContractError("legacy v2 Attempt has no exact record locator")
            legacy = self.legacy._load_run_physical(*address.split("/"))
            snapshot, request, path = (
                deepcopy(legacy.snapshot),
                deepcopy(legacy.request),
                legacy.path,
            )
        else:
            evidence = load_json_file(self.graph.store.verify_blob(evidence_hash))
            if (
                set(evidence) != {"schema_version", "snapshot", "request"}
                or evidence["schema_version"] != 1
            ):
                raise ContractError("v2 captured Run evidence is invalid")
            snapshot, request = (
                deepcopy(evidence["snapshot"]),
                deepcopy(evidence["request"]),
            )
            validate_snapshot(snapshot)
            validate_request(request)
            path = (
                self.repository.outputs
                / request["experiment"]
                / request["variant"]
                / "runs"
                / request["run_id"]
            )
        if request["action"] != spec["action"]:
            raise ContractError("v2 captured action disagrees with RunSpec")
        if snapshot["provenance"]["source_digest"] != request["source_digest"]:
            raise ContractError("v2 captured source ownership mismatch")
        if self.source_digest(revision["source_tree"]) != request["source_digest"]:
            raise ContractError("v2 Code source provenance disagrees with captured Run")
        if evidence_hash is not None:
            if any(
                snapshot["variant"][field] != revision[field]
                for field in ("components", "template")
            ):
                raise ContractError("captured Code disagrees with its revision")
            if (
                request["exec"]["device"] != spec["device"]
                or request["created_at"] != attempt["created_at"]
            ):
                raise ContractError("captured execution disagrees with its Attempt")
            if spec["action"] == "train" and (
                request["target"]["seed"] != spec["seed"]
                or request["identity_fingerprint"] != spec["backend_identity"]
            ):
                raise ContractError("captured Train meaning disagrees with RunSpec")
        experiment_revision = spec.get("experiment_revision") or self.resolve(
            entity_revision_scope(experiment_hash), CURRENT_REVISION_NAME
        )
        if spec.get("experiment_revision") is not None:
            experiment_payload = self.object(experiment_revision, "experiment_revision")
            if experiment_payload["experiment"] != experiment_hash:
                raise ContractError("v2 Run Experiment ownership mismatch")
            if evidence_hash is not None and any(
                snapshot["experiment"][field] != experiment_payload[field]
                for field in ("type", "question", "template")
            ):
                raise ContractError("captured Experiment disagrees with its revision")
            snapshot["experiment"].update(
                {
                    field: experiment_payload[field]
                    for field in ("type", "question", "template")
                }
            )
        snapshot["variant"]["components"] = deepcopy(revision["components"])
        snapshot["variant"]["template"] = deepcopy(revision["template"])
        option_hash = attempt.get("option_set")
        if option_hash is not None:
            options = self.object(option_hash, "option_set")
            if options.get("scope") != "full":
                raise ContractError("Run Options reference is not a full OptionSet")
            if evidence_hash is not None and any(
                snapshot["variant"].get(key) != value
                for key, value in options["document"].items()
                if key != "schema_version"
            ):
                raise ContractError("captured Options disagree with OptionSet")
            snapshot["variant"].update(
                {k: v for k, v in options["document"].items() if k != "schema_version"}
            )
        if "tracker_backends" in attempt:
            snapshot["variant"]["tracker"] = {
                "backend": attempt["tracker_backends"] or "none"
            }
            validate_tracker(snapshot["variant"]["tracker"])
        request.update(
            experiment=experiment,
            variant=variant,
            run_id=name,
            created_at=attempt["created_at"],
        )
        request["exec"]["device"] = spec["device"]
        snapshot["experiment"]["name"] = experiment
        snapshot["variant"]["name"] = variant
        parent = attempt.get("retry_parent")
        request["retry_of"] = (
            self.name(variant_run_scope(variant_hash), parent) if parent else None
        )
        if spec["action"] == "train":
            request["target"]["seed"] = spec["seed"]
            request["exec"]["seed"] = spec["seed"]
            request["identity_fingerprint"] = spec["backend_identity"]
        else:
            model = self.object(spec["model"], "model")
            if (
                self.object(model["variant_revision"], "variant_revision")["variant"]
                != variant_hash
            ):
                raise ContractError("v2 Run Model ownership mismatch")
            self.name(variant_run_scope(variant_hash), model["producing_attempt"])
            request["target"]["model_id"] = self.name(
                variant_model_scope(variant_hash), spec["model"]
            )
            request["exec"]["seed"] = model["seed"]
            if spec["action"] == "eval":
                request["target"]["seed"] = model["seed"]
        state = {
            "schema_version": 1,
            "run_id": name,
            "action": spec["action"],
            "created_at": attempt["created_at"],
            "updated_at": event["recorded_at"],
            **{
                field: deepcopy(event[field])
                for field in (
                    "status",
                    "reason",
                    "tracker_run_id",
                    "result",
                    "best_checkpoint",
                    "last_checkpoint",
                )
            },
        }
        if state["status"] == "done" and spec["action"] == "train":
            state["result"] = {
                "model_id": self.name(
                    variant_model_scope(variant_hash), event["result_object"]
                )
            }
        validate_snapshot(snapshot)
        validate_request(request)
        validate_state(state)
        identity = {
            "experiment_hash": experiment_hash,
            "experiment_revision_hash": experiment_revision,
            "variant_hash": variant_hash,
            "variant_revision_hash": revision_hash,
            "option_set_hash": option_hash,
            "run_spec_hash": spec_hash,
            "attempt_hash": attempt_hash,
            "comparison_hash": spec.get("comparison_hash"),
        }
        result = RunRecord(
            path,
            f"{experiment}/{variant}/{name}",
            snapshot,
            request,
            state,
            identity,
            event_hash,
        )
        self._runs[key] = result
        return result

    def runs(
        self, *, experiment: str | None = None, variant: str | None = None
    ) -> list[RunRecord]:
        result = []
        for exp_name, exp_hash in self.names(workspace_experiment_scope()).items():
            if experiment is not None and experiment != exp_name:
                continue
            for var_name, var_hash in self.names(
                experiment_variant_scope(exp_hash)
            ).items():
                if variant is not None and variant != var_name:
                    continue
                for name in self.names(variant_run_scope(var_hash)):
                    result.append(self.run(exp_name, var_name, name))
        if experiment is not None and variant is not None:
            self.entities(experiment, variant)
        return sorted(result, key=run_sort_key)

    def model(self, experiment: str, variant: str, name: str) -> ModelRecord:
        _, variant_hash = self.entities(experiment, variant)
        digest = self.resolve(variant_model_scope(variant_hash), name)
        payload = self.object(digest, "model")
        relative = payload.get("checkpoint_relative_path")
        if relative is None:
            producer = self.object(payload["producing_attempt"], "attempt")
            if producer.get("record_evidence") is not None:
                raise ContractError(
                    "captured v2 Model is missing checkpoint path evidence"
                )
            legacy = self.legacy.load_model_manifest(experiment, variant, name)
            return ModelRecord(legacy.path, legacy.address, legacy.document, digest)
        producer = self.name(
            variant_run_scope(variant_hash), payload["producing_attempt"]
        )
        run = self.run(experiment, variant, producer)
        if (
            run.request["action"] != "train"
            or payload["variant_revision"]
            != run.graph_identity["variant_revision_hash"]
            or payload["seed"] != run.request["target"]["seed"]
            or payload["comparison_hash"] != run.graph_identity["comparison_hash"]
        ):
            raise ContractError("v2 Model producer ownership mismatch")
        document = {
            "schema_version": 1,
            "model_id": name,
            "experiment": experiment,
            "variant": variant,
            "training_group": run.request["target"]["training_group"],
            "seed": payload["seed"],
            "device": run.request["exec"]["device"],
            "training_fingerprint": run.request["identity_fingerprint"],
            "producer_run": producer,
            "created_at": payload["created_at"],
            "checkpoint": {
                "path": f"runs/{producer}/{relative}",
                "digest": payload["checkpoint_content_hash"],
            },
        }
        validate_model(document)
        return ModelRecord(
            run.path.parents[1] / "models" / name,
            f"{experiment}/{variant}/{name}",
            document,
            digest,
        )

    def models(self, experiment: str, variant: str) -> list[ModelRecord]:
        _, entity = self.entities(experiment, variant)
        return sorted(
            (
                self.model(experiment, variant, name)
                for name in self.names(variant_model_scope(entity))
            ),
            key=lambda model: (
                model.document["created_at"],
                model.document["model_id"].encode(),
            ),
        )

    def artifact(self, record: RunRecord, relative: str):
        record = self.authoritative_record(record)
        event = self.object(record.event_hash, "attempt_event")
        if event["attempt"] != record.graph_identity["attempt_hash"]:
            raise ContractError("v2 artifact event ownership mismatch")
        matches = [ref for ref in event["artifacts"] if ref["path"] == relative]
        if not matches:
            raise NotFoundError(f"Run artifact not found: {record.address}/{relative}")
        if len(matches) != 1:
            raise ContractError("duplicate v2 Run artifact path")
        blob = self.object(matches[0]["blob"], "blob")
        if any(
            matches[0].get(field) != blob[field] for field in ("content_hash", "size")
        ):
            raise ContractError("v2 artifact evidence disagrees with blob")
        return self.graph.store.verify_blob(matches[0]["blob"])

    def evaluation(self, record: RunRecord) -> dict[str, Any]:
        event = self.object(record.event_hash, "attempt_event")
        result = self.object(event["result_object"], "eval_result")
        spec = self.object(record.graph_identity["run_spec_hash"], "run_spec")
        if (
            result["attempt"] != record.graph_identity["attempt_hash"]
            or result["model"] != spec["model"]
            or result["evaluation_case"] != spec["evaluation_case"]
        ):
            raise ContractError("v2 evaluation ownership mismatch")
        document = deepcopy(result["document"])
        document["model_id"] = record.request["target"]["model_id"]
        validate_evaluation(document, record.snapshot, record.request)
        return document
