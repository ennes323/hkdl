"""Schema-declared graph edges, independent of SQLite and graph services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hkdl.authoring.config import DIGEST_PATTERN
from hkdl.errors import ContractError


@dataclass(frozen=True)
class ObjectReference:
    """One typed, payload-declared edge in the v2 object graph."""

    source: str
    source_kind: str
    target: str
    target_kind: str
    relation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "target": self.target,
            "target_kind": self.target_kind,
            "relation": self.relation,
        }


def extract_references(records: list[Any]) -> tuple[ObjectReference, ...]:
    """Extract only schema-known digest fields from object payloads.

    Payloads are intentionally not scanned recursively for strings that look
    like hashes.  A hash in a metric, legacy evidence, or user-defined data is
    not a graph edge unless the object kind and field explicitly define it as
    one.  This keeps the index stable as payloads gain ordinary metadata.
    """

    by_digest = {record.digest: record for record in records}
    references: set[ObjectReference] = set()
    for record in records:
        _extract_record_references(record, by_digest, references)
    return tuple(
        sorted(
            references,
            key=lambda ref: (ref.source, ref.relation, ref.target, ref.target_kind),
        )
    )


def _extract_record_references(
    record: Any,
    by_digest: dict[str, Any],
    references: set[ObjectReference],
) -> None:
    kind = record.kind
    payload = record.payload
    if not isinstance(payload, Mapping):
        raise ContractError(f"v2 {kind} payload is invalid")

    if kind == "experiment_revision":
        _add_reference(
            record, payload, "experiment", "experiment", by_digest, references
        )
        _add_reference(
            record, payload, "parent", "experiment_revision", by_digest, references
        )
        return

    if kind == "variant":
        _add_reference(
            record, payload, "experiment", "experiment", by_digest, references
        )
        return

    if kind == "variant_revision":
        _add_reference(record, payload, "variant", "variant", by_digest, references)
        _add_reference(
            record, payload, "parent", "variant_revision", by_digest, references
        )
        _add_reference(
            record,
            payload,
            "derivation_parent",
            "variant_revision",
            by_digest,
            references,
        )
        _add_reference(
            record,
            payload,
            "merge_parent",
            "variant_revision",
            by_digest,
            references,
        )
        _add_reference(
            record, payload, "source_tree", "source_tree", by_digest, references
        )
        _add_mapping_references(
            record,
            payload,
            "evaluation_cases",
            "evaluation_case",
            by_digest,
            references,
        )
        _add_mapping_references(
            record,
            payload,
            "export_profiles",
            "export_profile",
            by_digest,
            references,
        )
        return

    if kind == "source_tree":
        files = payload.get("files")
        if files is None:
            return
        if not isinstance(files, list):
            raise ContractError("v2 source_tree files are invalid")
        _add_list_references(record, files, "blob", "blob", by_digest, references)
        return

    if kind == "run_spec":
        # The action determines which of these fields are meaningful, but the
        # field names themselves are the authority for reference extraction.
        # Unknown action-specific fields are deliberately ignored.
        for field, target_kind in (
            ("experiment_revision", "experiment_revision"),
            ("variant_revision", "variant_revision"),
            ("evaluator_revision", "variant_revision"),
            ("exporter_revision", "variant_revision"),
            ("option_set", "option_set"),
            ("train_options", "option_set"),
            ("options", "option_set"),
            ("model", "model"),
            ("evaluation_case", "evaluation_case"),
            ("export_profile", "export_profile"),
        ):
            _add_reference(record, payload, field, target_kind, by_digest, references)
        return

    if kind == "attempt":
        _add_reference(record, payload, "run_spec", "run_spec", by_digest, references)
        _add_reference(
            record, payload, "option_set", "option_set", by_digest, references
        )
        _add_reference(
            record, payload, "record_evidence", "blob", by_digest, references
        )
        _add_reference(
            record, payload, "retry_parent", "attempt", by_digest, references
        )
        return

    if kind == "attempt_event":
        _add_reference(record, payload, "attempt", "attempt", by_digest, references)
        _add_reference(
            record, payload, "parent", "attempt_event", by_digest, references
        )
        result_object = payload.get("result_object")
        if result_object is not None:
            target_kind = _result_object_kind(record, result_object, by_digest)
            _add_reference_value(
                record,
                "result_object",
                result_object,
                target_kind,
                by_digest,
                references,
            )
        artifacts = payload.get("artifacts")
        if artifacts is not None:
            _add_artifact_references(record, artifacts, by_digest, references)
        return

    if kind == "model":
        _add_reference(
            record,
            payload,
            "experiment_revision",
            "experiment_revision",
            by_digest,
            references,
        )
        _add_reference(
            record, payload, "train_options", "option_set", by_digest, references
        )
        _add_reference(
            record,
            payload,
            "producing_attempt",
            "attempt",
            by_digest,
            references,
        )
        _add_reference(
            record,
            payload,
            "variant_revision",
            "variant_revision",
            by_digest,
            references,
        )
        _add_reference(
            record,
            payload,
            "checkpoint_blob",
            "blob",
            by_digest,
            references,
        )
        return

    if kind == "eval_result":
        _add_reference(record, payload, "attempt", "attempt", by_digest, references)
        _add_reference(record, payload, "model", "model", by_digest, references)
        _add_reference(
            record,
            payload,
            "evaluation_case",
            "evaluation_case",
            by_digest,
            references,
        )
        artifacts = payload.get("artifacts")
        if artifacts is not None:
            _add_artifact_references(record, artifacts, by_digest, references)
        return

    if kind == "export_result":
        _add_reference(record, payload, "attempt", "attempt", by_digest, references)
        _add_reference(record, payload, "model", "model", by_digest, references)
        _add_reference(
            record,
            payload,
            "export_profile",
            "export_profile",
            by_digest,
            references,
        )
        artifacts = payload.get("artifacts")
        if artifacts is not None:
            _add_artifact_references(record, artifacts, by_digest, references)
        return

    if kind == "binding_transaction":
        # Binding operations are the source of active roots, not object-graph
        # edges.  Only the chain parent is a typed object reference.
        _add_reference(
            record,
            payload,
            "previous",
            "binding_transaction",
            by_digest,
            references,
        )


def _add_reference(
    record: Any,
    payload: Mapping[str, Any],
    field: str,
    target_kind: str,
    by_digest: dict[str, Any],
    references: set[ObjectReference],
) -> None:
    if field not in payload or payload[field] is None:
        return
    _add_reference_value(
        record,
        field,
        payload[field],
        target_kind,
        by_digest,
        references,
    )


def _add_mapping_references(
    record: Any,
    payload: Mapping[str, Any],
    field: str,
    target_kind: str,
    by_digest: dict[str, Any],
    references: set[ObjectReference],
) -> None:
    values = payload.get(field)
    if values is None:
        return
    if not isinstance(values, Mapping):
        raise ContractError(f"v2 {record.kind} {field} references are invalid")
    for name, target in values.items():
        if not isinstance(name, str):
            raise ContractError(f"v2 {record.kind} {field} names are invalid")
        _add_reference_value(
            record,
            field,
            target,
            target_kind,
            by_digest,
            references,
        )


def _add_list_references(
    record: Any,
    values: list[Any],
    field: str,
    target_kind: str,
    by_digest: dict[str, Any],
    references: set[ObjectReference],
) -> None:
    for value in values:
        if not isinstance(value, Mapping) or field not in value:
            raise ContractError(f"v2 {record.kind} {field} references are invalid")
        _add_reference_value(
            record,
            field,
            value[field],
            target_kind,
            by_digest,
            references,
        )


def _add_artifact_references(
    record: Any,
    artifacts: Any,
    by_digest: dict[str, Any],
    references: set[ObjectReference],
) -> None:
    if not isinstance(artifacts, list):
        raise ContractError(f"v2 {record.kind} artifacts are invalid")
    _add_list_references(record, artifacts, "blob", "blob", by_digest, references)


def _add_reference_value(
    record: Any,
    field: str,
    value: Any,
    target_kind: str,
    by_digest: dict[str, Any],
    references: set[ObjectReference],
) -> None:
    target = _validate_digest(value, f"v2 {record.kind} {field} reference")
    existing = by_digest.get(target)
    if existing is not None and existing.kind != target_kind:
        raise ContractError(
            f"v2 {record.kind} {field} target kind disagrees: "
            f"expected {target_kind}, got {existing.kind}"
        )
    references.add(
        ObjectReference(record.digest, record.kind, target, target_kind, field)
    )


def _result_object_kind(
    record: Any,
    value: Any,
    by_digest: dict[str, Any],
) -> str:
    """Infer an Attempt event result type from its Attempt's RunSpec."""

    attempt_hash = record.payload.get("attempt")
    attempt_digest = _validate_digest(
        attempt_hash, "v2 attempt_event attempt reference"
    )
    attempt = by_digest.get(attempt_digest)
    if attempt is None or attempt.kind != "attempt":
        raise ContractError("v2 attempt_event Attempt is unavailable")
    run_spec_hash = _validate_digest(
        attempt.payload.get("run_spec"), "v2 Attempt RunSpec reference"
    )
    run_spec = by_digest.get(run_spec_hash)
    if run_spec is None or run_spec.kind != "run_spec":
        raise ContractError("v2 Attempt RunSpec is unavailable")
    action = run_spec.payload.get("action")
    target_kind = {
        "train": "model",
        "eval": "eval_result",
        "export": "export_result",
    }.get(action)
    existing = by_digest.get(_validate_digest(value, "v2 result object reference"))
    if target_kind is None:
        if existing is not None and existing.kind in {
            "model",
            "eval_result",
            "export_result",
        }:
            return existing.kind
        raise ContractError("v2 attempt_event action cannot type result object")
    if existing is not None and existing.kind != target_kind:
        raise ContractError(
            "v2 attempt_event result object kind disagrees with RunSpec action"
        )
    return target_kind


def _validate_digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
        raise ContractError(f"{location} is invalid")
    return value


# Preserve the established Python type location.
ObjectReference.__module__ = "hkdl.storage.graph.projection"
