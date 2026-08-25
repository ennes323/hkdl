"""Ephemeral preflight plans for ordinary actions and retries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .authoring import ExperimentRecord, VariantRecord
from .config import ContractError
from .run_contracts import (
    evaluation_case,
    fingerprint_document,
    metric_spec,
    validate_evaluation_readiness,
    validate_export_readiness,
    validate_training_readiness,
)
from .runs import ModelRecord, RunRecord


@dataclass(frozen=True)
class ExecutionPlan:
    target: dict[str, Any]
    selected: dict[str, str]
    preflight: dict[str, Any]
    fingerprint: str


class ExecutionPlanner:
    def __init__(self, preflight: Callable[..., dict[str, Any]]):
        self._preflight = preflight

    def train(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        *,
        source_digest: str,
        snapshot: dict[str, Any],
        training_group: str,
        seed: int,
        device: str,
        python: Any,
        environment_descriptor: int,
    ) -> ExecutionPlan:
        selected = validate_training_readiness(experiment.document, variant.document)
        target = {"training_group": training_group, "seed": seed}
        fallback = {
            "dataset": variant.document["dataset"],
            "train": variant.document["train"],
            "components": selected,
        }
        preflight = self._preflight(
            python,
            variant,
            action="train",
            snapshot=snapshot,
            selected=selected,
            seed=seed,
            device=device,
            fallback=fallback,
            target=target,
            environment_descriptor=environment_descriptor,
        )
        fingerprint = action_fingerprint(
            action="train",
            source_digest=source_digest,
            selected=selected,
            identity=preflight["identity"],
            device=preflight["exec"]["device"],
        )
        return ExecutionPlan(target, selected, preflight, fingerprint)

    def evaluation(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        model: ModelRecord,
        *,
        source_digest: str,
        snapshot: dict[str, Any],
        training_group: str,
        evaluation_case_name: str,
        selected: dict[str, str] | None = None,
        case_document: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        device: str,
        python: Any,
        environment_descriptor: int,
    ) -> ExecutionPlan:
        selected = selected or validate_evaluation_readiness(
            experiment.document, variant.document, case=evaluation_case_name
        )
        target = {
            "training_group": training_group,
            "seed": model.document["seed"],
            "model_id": model.document["model_id"],
            "evaluation_case": evaluation_case_name,
        }
        fallback = {
            "case": case_document
            if case_document is not None
            else evaluation_case(variant.document, evaluation_case_name),
            "metrics": metrics
            if metrics is not None
            else metric_spec(variant.document, evaluation_case_name),
            "components": selected,
        }
        preflight = self._preflight(
            python,
            variant,
            action="eval",
            snapshot=snapshot,
            selected=selected,
            seed=model.document["seed"],
            device=device,
            fallback=fallback,
            target=target,
            environment_descriptor=environment_descriptor,
        )
        fingerprint = action_fingerprint(
            action="eval",
            source_digest=source_digest,
            selected=selected,
            identity=preflight["identity"],
        )
        return ExecutionPlan(target, selected, preflight, fingerprint)

    def export(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        model: ModelRecord,
        *,
        source_digest: str,
        snapshot: dict[str, Any],
        device: str,
        python: Any,
        environment_descriptor: int,
    ) -> ExecutionPlan:
        selected = validate_export_readiness(experiment.document, variant.document)
        target = {"model_id": model.document["model_id"]}
        preflight = self._preflight(
            python,
            variant,
            action="export",
            snapshot=snapshot,
            selected=selected,
            seed=model.document["seed"],
            device=device,
            fallback={
                "infer": variant.document["infer"],
                "components": selected,
            },
            target=target,
            environment_descriptor=environment_descriptor,
        )
        fingerprint = action_fingerprint(
            action="export",
            source_digest=source_digest,
            selected=selected,
            identity=preflight["identity"],
        )
        return ExecutionPlan(target, selected, preflight, fingerprint)

    def retry(
        self,
        record: RunRecord,
        variant: VariantRecord,
        *,
        source_digest: str,
        python: Any,
        environment_descriptor: int,
    ) -> ExecutionPlan:
        selected = selected_for_request(record)
        preflight = self._preflight(
            python,
            variant,
            action=record.request["action"],
            snapshot=record.snapshot,
            selected=selected,
            seed=record.request["exec"]["seed"],
            device=record.request["exec"]["device"],
            fallback=fallback_for_request(record, selected),
            target=record.request["target"],
            environment_descriptor=environment_descriptor,
        )
        fingerprint = action_fingerprint(
            action=record.request["action"],
            source_digest=source_digest,
            selected=selected,
            identity=preflight["identity"],
            device=(
                preflight["exec"]["device"]
                if record.request["action"] == "train"
                else None
            ),
        )
        if (
            fingerprint != record.request["identity_fingerprint"]
            or preflight["exec"] != record.request["exec"]
        ):
            raise ContractError("retry preflight differs from the original Run")
        return ExecutionPlan(
            dict(record.request["target"]),
            selected,
            preflight,
            fingerprint,
        )


def action_fingerprint(
    *,
    action: str,
    source_digest: str,
    selected: Mapping[str, str],
    identity: Mapping[str, Any],
    device: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "action": action,
        "source_digest": source_digest,
        "components": dict(selected),
        "identity": dict(identity),
    }
    if action == "train":
        payload["device"] = device
    return fingerprint_document(payload)


def selected_for_request(record: RunRecord) -> dict[str, str]:
    experiment = record.snapshot["experiment"]
    variant = record.snapshot["variant"]
    if record.request["action"] == "train":
        return validate_training_readiness(experiment, variant)
    if record.request["action"] == "eval":
        return validate_evaluation_readiness(
            experiment,
            variant,
            case=record.request["target"]["evaluation_case"],
        )
    return validate_export_readiness(experiment, variant)


def fallback_for_request(
    record: RunRecord,
    selected: Mapping[str, str],
) -> dict[str, Any]:
    variant = record.snapshot["variant"]
    if record.request["action"] == "train":
        return {
            "dataset": variant["dataset"],
            "train": variant["train"],
            "components": dict(selected),
        }
    if record.request["action"] == "eval":
        case = record.request["target"]["evaluation_case"]
        return {
            "case": evaluation_case(variant, case),
            "metrics": metric_spec(variant, case),
            "components": dict(selected),
        }
    return {"infer": variant["infer"], "components": dict(selected)}
