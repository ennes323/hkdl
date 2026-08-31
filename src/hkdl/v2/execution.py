"""Write-through v2 graph recording for Train, Eval, Export, and retry."""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from ..config import ContractError
from ..run_contracts import (
    TERMINAL_STATUSES,
    evaluation_case,
    metric_spec,
    validate_tracker,
)
from ..runs import (
    ModelRecord,
    RunRecord,
    _training_metric_summary,
    _validate_training_metric_summary,
)
from ..storage import NotFoundError, RepositoryPaths
from .bindings import BindingOperation
from .objects import canonical_json_bytes
from .graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    attempt_event_scope,
    comparison_digest,
    comparison_group_scope,
    experiment_variant_scope,
    variant_model_scope,
    variant_run_scope,
    workspace_experiment_scope,
)


@dataclass(frozen=True)
class ExecutionIdentity:
    experiment_hash: str
    experiment_revision_hash: str
    variant_hash: str
    variant_revision_hash: str
    option_set_hash: str | None
    run_spec_hash: str
    attempt_hash: str
    comparison_hash: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "experiment_hash": self.experiment_hash,
            "experiment_revision_hash": self.experiment_revision_hash,
            "variant_hash": self.variant_hash,
            "variant_revision_hash": self.variant_revision_hash,
            "option_set_hash": self.option_set_hash,
            "run_spec_hash": self.run_spec_hash,
            "attempt_hash": self.attempt_hash,
            "comparison_hash": self.comparison_hash,
        }


class GraphRecorder:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
        nonce: Callable[[], str] | None = None,
    ):
        self.repository = repository
        self.graph = V2Graph(repository, now=now, nonce=nonce)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._nonce = nonce or (lambda: secrets.token_hex(16))

    def active(self) -> bool:
        return self.graph.is_active()

    def record_allocation(self, record: RunRecord) -> ExecutionIdentity:
        (
            experiment_hash,
            experiment_revision_hash,
            variant_hash,
            revision_hash,
        ) = self._variant_identity(record)
        action = str(record.request["action"])
        comparison_hash: str | None = None
        options = _options_document(record.snapshot["variant"])
        full_options = self.graph.store.put(
            "option_set", {"scope": "full", "document": options}
        )
        retry_parent = record.request["retry_of"]
        retry_hash = (
            self.graph.bindings.resolve(variant_run_scope(variant_hash), retry_parent)
            if retry_parent is not None
            else None
        )
        if retry_hash is not None:
            parent_attempt = self.graph.store.load(retry_hash)
            run_spec = self.graph.store.load(str(parent_attempt.payload["run_spec"]))
            parent_options = parent_attempt.payload.get("option_set")
            if parent_options is not None:
                full_options = self.graph.store.load(str(parent_options))
            comparison = run_spec.payload.get("comparison_hash")
            comparison_hash = str(comparison) if comparison is not None else None
            revision_field = {
                "train": "variant_revision",
                "eval": "evaluator_revision",
                "export": "exporter_revision",
            }[str(run_spec.payload["action"])]
            revision_hash = str(run_spec.payload[revision_field])
            experiment_revision_hash = str(
                run_spec.payload.get("experiment_revision", experiment_revision_hash)
            )
        elif action == "train":
            train_options = self.graph.store.put(
                "option_set",
                {
                    "scope": "train",
                    "dataset": deepcopy(options["dataset"]),
                    "metrics": deepcopy(options["metrics"]),
                    "train": deepcopy(options["train"]),
                },
            )
            comparison_payload = {
                "variant_revision": revision_hash,
                "train_options": train_options.digest,
                "backend_identity": record.request["identity_fingerprint"],
                "device": record.request["exec"]["device"],
            }
            comparison_hash = comparison_digest(comparison_payload)
            run_spec_payload = {
                "action": "train",
                "experiment_revision": experiment_revision_hash,
                "variant_revision": revision_hash,
                "train_options": train_options.digest,
                "seed": record.request["target"]["seed"],
                "backend_identity": record.request["identity_fingerprint"],
                "device": record.request["exec"]["device"],
                "comparison_hash": comparison_hash,
            }
            run_spec = self.graph.store.put("run_spec", run_spec_payload)
        elif action == "eval":
            model_hash = self.graph.bindings.resolve(
                variant_model_scope(variant_hash),
                str(record.request["target"]["model_id"]),
            )
            case_name = str(record.request["target"]["evaluation_case"])
            case_hash = self.graph.store.put(
                "evaluation_case",
                {
                    "definition": evaluation_case(
                        record.snapshot["variant"], case_name
                    ),
                    "metrics": metric_spec(record.snapshot["variant"], case_name),
                },
            )
            run_spec_payload = {
                "action": "eval",
                "experiment_revision": experiment_revision_hash,
                "model": model_hash,
                "evaluation_case": case_hash.digest,
                "evaluator_revision": revision_hash,
                "device": record.request["exec"]["device"],
            }
            run_spec = self.graph.store.put("run_spec", run_spec_payload)
        elif action == "export":
            model_hash = self.graph.bindings.resolve(
                variant_model_scope(variant_hash),
                str(record.request["target"]["model_id"]),
            )
            profile_hash = self.graph.store.put(
                "export_profile", {"definition": deepcopy(options["infer"])}
            )
            run_spec_payload = {
                "action": "export",
                "experiment_revision": experiment_revision_hash,
                "model": model_hash,
                "export_profile": profile_hash.digest,
                "exporter_revision": revision_hash,
                "device": record.request["exec"]["device"],
            }
            run_spec = self.graph.store.put("run_spec", run_spec_payload)
        else:
            raise ContractError(f"unsupported v2 Run action: {action}")
        evidence = self.graph.store.put_blob_bytes(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "snapshot": record.snapshot,
                    "request": record.request,
                }
            )
            + b"\n",
            media_type="application/json",
        )
        attempt = self.graph.store.put(
            "attempt",
            {
                "run_spec": run_spec.digest,
                "record_evidence": evidence.digest,
                "option_set": full_options.digest,
                "tracker_backends": list(
                    validate_tracker(record.snapshot["variant"].get("tracker", {}))
                ),
                "nonce": self._nonce(),
                "retry_parent": retry_hash,
                "created_at": record.request["created_at"],
                "legacy_evidence": {
                    "address": record.address,
                    "source_digest": record.request["source_digest"],
                },
            },
        )
        event = self._event(record, attempt.digest, parent=None)
        operations = [
            BindingOperation(
                "bind",
                variant_run_scope(variant_hash),
                str(record.request["run_id"]),
                attempt.digest,
            ),
            BindingOperation(
                "bind",
                attempt_event_scope(attempt.digest),
                CURRENT_REVISION_NAME,
                event,
            ),
        ]
        if comparison_hash is not None:
            group_name = str(record.request["target"]["training_group"])
            groups = self.graph.bindings.names(comparison_group_scope(revision_hash))
            existing = groups.get(group_name)
            if existing is not None and existing != comparison_hash:
                raise ContractError(f"comparison group meaning changed: {group_name}")
            if existing is None:
                operations.append(
                    BindingOperation(
                        "bind",
                        comparison_group_scope(revision_hash),
                        group_name,
                        comparison_hash,
                    )
                )
        self.graph.bindings.commit(operations)
        return ExecutionIdentity(
            experiment_hash,
            experiment_revision_hash,
            variant_hash,
            revision_hash,
            full_options.digest,
            run_spec.digest,
            attempt.digest,
            comparison_hash,
        )

    def record_state(self, record: RunRecord) -> ExecutionIdentity:
        identity = self.identity(record)
        event_scope = attempt_event_scope(identity.attempt_hash)
        current_event = self.graph.bindings.resolve(event_scope, CURRENT_REVISION_NAME)
        current_payload = self.graph.store.load(current_event).payload
        if current_payload["status"] in TERMINAL_STATUSES:
            if current_payload["status"] != record.state["status"]:
                raise ContractError("terminal v2 Attempt event is sealed")
            return identity
        event = self._event(record, identity.attempt_hash, parent=current_event)
        if event == current_event:
            return identity
        self.graph.bindings.commit(
            [
                BindingOperation(
                    "unbind",
                    event_scope,
                    CURRENT_REVISION_NAME,
                    current_event,
                ),
                BindingOperation(
                    "bind",
                    event_scope,
                    CURRENT_REVISION_NAME,
                    event,
                ),
            ]
        )
        return identity

    def record_model(self, run: RunRecord, model: ModelRecord) -> str:
        identity = self.identity(run)
        checkpoint = (
            self.repository.outputs
            / run.request["experiment"]
            / run.request["variant"]
            / model.document["checkpoint"]["path"]
        )
        blob = self.graph.store.put_blob(checkpoint)
        run_spec = self.graph.store.load(identity.run_spec_hash).payload
        model_object = self.graph.store.put(
            "model",
            {
                "producing_attempt": identity.attempt_hash,
                "experiment_revision": identity.experiment_revision_hash,
                "variant_revision": identity.variant_revision_hash,
                "train_options": run_spec.get("train_options"),
                "comparison_hash": run_spec["comparison_hash"],
                "seed": model.document["seed"],
                "checkpoint_blob": blob.digest,
                "checkpoint_content_hash": blob.payload["content_hash"],
                "checkpoint_relative_path": model.document["checkpoint"]["path"].split(
                    "/", 2
                )[2],
                "created_at": model.document["created_at"],
            },
        )
        scope = variant_model_scope(identity.variant_hash)
        name = str(model.document["model_id"])
        existing = self.graph.bindings.names(scope).get(name)
        if existing is None:
            self.graph.bindings.bind(scope, name, model_object.digest)
        elif existing != model_object.digest:
            raise ContractError("v2 Model binding disagrees with published Model")
        return model_object.digest

    def identity(self, record: RunRecord) -> ExecutionIdentity:
        if record.graph_identity is not None:
            return ExecutionIdentity(**record.graph_identity)
        experiment_hash = self.graph.experiment_hash(str(record.request["experiment"]))
        variant_hash = self.graph.variant_hash(
            experiment_hash,
            str(record.request["variant"]),
        )
        attempt_hash = self.graph.bindings.resolve(
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
                    "experiment_revision", self.graph.current_revision(experiment_hash)
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
        experiment_hash = self.graph.experiment_hash(experiment)
        variant_hash = self.graph.variant_hash(experiment_hash, variant)
        return self.graph.bindings.resolve(variant_model_scope(variant_hash), model_id)

    def authoritative_record(self, record: RunRecord) -> RunRecord:
        if record.event_hash is not None:
            return record
        identity = self.identity(record)
        event_hash = self.graph.bindings.resolve(
            attempt_event_scope(identity.attempt_hash),
            CURRENT_REVISION_NAME,
        )
        event = self.graph.store.load(event_hash)
        if (
            event.kind != "attempt_event"
            or event.payload["attempt"] != identity.attempt_hash
        ):
            raise ContractError("v2 Attempt event ownership mismatch")
        state = dict(record.state)
        state.update(
            status=event.payload["status"],
            reason=event.payload["reason"],
            tracker_run_id=event.payload["tracker_run_id"],
            result=deepcopy(event.payload["result"]),
            best_checkpoint=event.payload["best_checkpoint"],
            last_checkpoint=event.payload["last_checkpoint"],
            updated_at=event.payload["recorded_at"],
        )
        return RunRecord(
            record.path,
            record.address,
            record.snapshot,
            record.request,
            state,
        )

    def artifact_path(self, record: RunRecord, relative: str) -> Path:
        if record.event_hash is not None:
            from .reader import current_reader

            return current_reader(self.repository).artifact(record, relative)
        identity = self.identity(record)
        event_hash = record.event_hash or self.graph.bindings.resolve(
            attempt_event_scope(identity.attempt_hash),
            CURRENT_REVISION_NAME,
        )
        event = self.graph.store.load(event_hash)
        for reference in event.payload["artifacts"]:
            if reference["path"] == relative:
                return self.graph.store.verify_blob(str(reference["blob"]))
        raise NotFoundError(f"Run artifact not found: {record.address}/{relative}")

    def load_training_metrics(self, record: RunRecord) -> dict[str, Any]:
        if record.request["action"] != "train":
            raise ContractError("Run is not a Train Run")
        if "local" not in validate_tracker(record.snapshot["variant"]["tracker"]):
            raise ContractError("Train Run does not use local tracking")
        try:
            path = self.artifact_path(record, "metrics/train.jsonl")
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
            if not isinstance(event, dict) or set(event) != {
                "schema_version",
                "name",
                "step",
                "value",
            }:
                raise ContractError("v2 Train metric event fields are invalid")
            name = event["name"]
            step = event["step"]
            value = event["value"]
            if (
                event["schema_version"] != 1
                or not isinstance(name, str)
                or not name
                or isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (name, step) in seen
            ):
                raise ContractError("v2 Train metric event is invalid")
            seen.add((name, step))
            events.append(event)
        summary_document = _training_metric_summary(events)
        try:
            summary_path = self.artifact_path(record, "metrics/train-summary.json")
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
            _validate_training_metric_summary(persisted)
            if persisted != summary_document:
                raise ContractError("Train metric summary disagrees with history")
        if record.state["status"] == "done" and partial:
            raise ContractError("completed Train metric history is partial")
        return {
            "partial": partial,
            "events": events,
            "summary": summary_document["metrics"],
        }

    def assert_run_projection_complete(self, records: list[RunRecord]) -> None:
        observed: dict[tuple[str, str], set[str]] = {}
        for record in records:
            key = (
                str(record.request["experiment"]),
                str(record.request["variant"]),
            )
            observed.setdefault(key, set()).add(str(record.request["run_id"]))
        for (experiment, variant), run_ids in observed.items():
            experiment_hash = self.graph.experiment_hash(experiment)
            variant_hash = self.graph.variant_hash(experiment_hash, variant)
            bound = set(self.graph.bindings.names(variant_run_scope(variant_hash)))
            if bound != run_ids:
                raise ContractError(
                    f"v2 Run projection is incomplete for {experiment}/{variant}"
                )

    def projection_variant_names(
        self, experiment: str, variant: str
    ) -> tuple[str, ...]:
        experiment_hash = self.graph.experiment_hash(experiment)
        scope = experiment_variant_scope(experiment_hash)
        variant_hash = self.graph.variant_hash(experiment_hash, variant)
        return self.graph.bindings.historical_names(scope, variant_hash)

    def projection_experiment_names(self, experiment: str) -> tuple[str, ...]:
        experiment_hash = self.graph.experiment_hash(experiment)
        return self.graph.bindings.historical_names(
            workspace_experiment_scope(), experiment_hash
        )

    def current_experiment_name_for_projection(
        self, projection_name: str
    ) -> str | None:
        scope = workspace_experiment_scope()
        current = self.graph.bindings.names(scope)
        if projection_name in current:
            return projection_name
        target = self.graph.bindings.historical_target(scope, projection_name)
        if target is None:
            return None
        matches = [name for name, value in current.items() if value == target]
        if len(matches) != 1:
            raise ContractError("v2 Experiment entity has no unique active name")
        return matches[0]

    def current_variant_name_for_projection(
        self,
        experiment: str,
        projection_name: str,
    ) -> str | None:
        experiment_hash = self.graph.experiment_hash(experiment)
        scope = experiment_variant_scope(experiment_hash)
        current = self.graph.bindings.names(scope)
        if projection_name in current:
            return projection_name
        target = self.graph.bindings.historical_target(scope, projection_name)
        if target is None:
            return None
        matches = [name for name, value in current.items() if value == target]
        if len(matches) != 1:
            raise ContractError("v2 Variant entity has no unique active name")
        return matches[0]

    def readdress_run(
        self, record: RunRecord, variant: str, experiment: str | None = None
    ) -> RunRecord:
        request = deepcopy(record.request)
        snapshot = deepcopy(record.snapshot)
        experiment = experiment or str(request["experiment"])
        request["experiment"] = experiment
        snapshot["experiment"]["name"] = experiment
        request["variant"] = variant
        snapshot["variant"]["name"] = variant
        return RunRecord(
            record.path,
            f"{experiment}/{variant}/{request['run_id']}",
            snapshot,
            request,
            dict(record.state),
            record.graph_identity,
            record.event_hash,
        )

    def readdress_model(
        self, model: ModelRecord, variant: str, experiment: str | None = None
    ) -> ModelRecord:
        document = deepcopy(model.document)
        experiment = experiment or str(document["experiment"])
        document["experiment"] = experiment
        document["variant"] = variant
        return ModelRecord(
            model.path,
            f"{experiment}/{variant}/{document['model_id']}",
            document,
            model.graph_hash,
        )

    def next_run_number(self, experiment: str, variant: str) -> int:
        experiment_hash = self.graph.experiment_hash(experiment)
        variant_hash = self.graph.variant_hash(experiment_hash, variant)
        scope = variant_run_scope(variant_hash)
        numbers = [
            int(name.removeprefix("run-"))
            for name in self.graph.bindings.historical_scope_names(scope)
            if name.startswith("run-")
        ]
        return max(numbers, default=0) + 1

    def _variant_identity(self, record: RunRecord) -> tuple[str, str, str, str]:
        experiment = str(record.request["experiment"])
        variant = str(record.request["variant"])
        experiment_hash = self.graph.experiment_hash(experiment)
        experiment_revision_hash = self.graph.current_revision(experiment_hash)
        variant_hash = self.graph.variant_hash(experiment_hash, variant)
        revision_hash = self.graph.current_revision(variant_hash)
        source_tree = self.graph.store.load(revision_hash).payload["source_tree"]
        source_payload = self.graph.store.load(str(source_tree)).payload
        legacy_digest = source_payload.get("legacy_digest")
        if (
            legacy_digest is not None
            and legacy_digest != record.request["source_digest"]
        ):
            raise ContractError("v2 revision source disagrees with legacy Run")
        return (
            experiment_hash,
            experiment_revision_hash,
            variant_hash,
            revision_hash,
        )

    def _event(
        self, record: RunRecord, attempt_hash: str, *, parent: str | None
    ) -> str:
        references = (
            self._artifact_references(record)
            if record.state["status"] in TERMINAL_STATUSES
            else []
        )
        result_hash = self._result(record, references)
        payload = {
            "attempt": attempt_hash,
            "parent": parent,
            "status": record.state["status"],
            "reason": record.state["reason"],
            "tracker_run_id": record.state["tracker_run_id"],
            "result": deepcopy(record.state["result"]),
            "result_object": result_hash,
            "best_checkpoint": record.state["best_checkpoint"],
            "last_checkpoint": record.state["last_checkpoint"],
            "artifacts": references,
            "recorded_at": record.state["updated_at"],
        }
        return self.graph.store.put("attempt_event", payload).digest

    def _artifact_references(self, record: RunRecord) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for relative_root in ("worker.log", "metrics", "artifacts"):
            root = record.path / relative_root
            if not os.path.lexists(root):
                continue
            paths = [root] if root.is_file() else list(root.rglob("*"))
            for path in sorted(paths, key=lambda item: item.as_posix().encode("utf-8")):
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise ContractError("Run artifact contains a symlink")
                if stat.S_ISDIR(metadata.st_mode):
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise ContractError("Run artifact contains a non-regular file")
                relative = PurePosixPath(
                    *path.relative_to(record.path).parts
                ).as_posix()
                blob = self.graph.store.put_blob(path)
                references.append(
                    {
                        "path": relative,
                        "blob": blob.digest,
                        "content_hash": blob.payload["content_hash"],
                        "size": blob.payload["size"],
                    }
                )
        return references

    def _result(
        self,
        record: RunRecord,
        references: list[dict[str, Any]],
    ) -> str | None:
        if record.state["status"] != "done":
            return None
        identity = self.identity(record)
        if record.request["action"] == "train":
            model_id = record.state["result"]["model_id"]
            return self.model_hash(
                str(record.request["experiment"]),
                str(record.request["variant"]),
                str(model_id),
            )
        if record.request["action"] == "eval":
            document = _load_json(record.path / record.state["result"]["metrics"])
            run_spec = self.graph.store.load(identity.run_spec_hash).payload
            return self.graph.store.put(
                "eval_result",
                {
                    "attempt": identity.attempt_hash,
                    "model": run_spec["model"],
                    "evaluation_case": run_spec["evaluation_case"],
                    "document": document,
                    "artifacts": references,
                },
            ).digest
        run_spec = self.graph.store.load(identity.run_spec_hash).payload
        return self.graph.store.put(
            "export_result",
            {
                "attempt": identity.attempt_hash,
                "model": run_spec["model"],
                "export_profile": run_spec["export_profile"],
                "artifacts": [
                    reference
                    for reference in references
                    if str(reference["path"]).startswith("artifacts/export/")
                ],
            },
        ).digest


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load v2 result evidence: {path}") from error
    if not isinstance(document, dict):
        raise ContractError("v2 result evidence must be a mapping")
    return document


def _options_document(variant: dict[str, Any]) -> dict[str, Any]:
    """Return the research options carried by either authored schema."""

    return {
        "schema_version": 2,
        "dataset": deepcopy(variant["dataset"]),
        "metrics": deepcopy(variant["metrics"]),
        "train": deepcopy(variant["train"]),
        "eval": deepcopy(variant["eval"]),
        "infer": deepcopy(variant["infer"]),
    }
