"""Variant-managed action execution, Model publication, and retry."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from hkdl.authoring.authoring import Authoring
from hkdl.authoring.authoring_records import ExperimentRecord, VariantRecord
from hkdl.errors import ContractError
from hkdl.runtime._diagnostics import report_exception, report_message
from hkdl.runtime.runtime import (
    RuntimeFailure,
    RuntimeInterrupted,
    RuntimeOwnershipConflict,
    VariantRuntime,
)
from hkdl.storage.graph.maintenance import workspace_operation
from hkdl.storage.runs import RunStore
from hkdl.storage.settings import WorkspaceSettingsStore, normalize_tracker
from hkdl.storage.storage import (
    LockUnavailableError,
    RepositoryPaths,
    compute_source_digest,
    try_directory_lock,
)

from .attempts import (
    action_candidate,
    load_attempt,
    new_attempt,
    remove_attempt,
    validate_attempt_owner,
    write_attempt,
)
from .execution_planner import ExecutionPlanner, fallback_for_request
from .receipt_committer import ReceiptCommitter
from .run_contracts import (
    TERMINAL_STATUSES,
    evaluation_case,
    metric_spec,
    validate_evaluation_readiness,
    validate_tracker,
)
from .run_records import ModelRecord, RunRecord


class ExecutionFailure(RuntimeError):
    def __init__(self, action: str, message: str, *, address: str | None = None):
        super().__init__(message)
        self.action = action
        self.address = address


class ExecutionInterrupted(KeyboardInterrupt):
    def __init__(self, action: str, address: str):
        super().__init__(address)
        self.action = action
        self.address = address


class LifecycleConflict(RuntimeError):
    """The requested Variant-managed mutation conflicts with durable state."""


class _EvaluationCellOccupied(Exception):
    """A matching eval-all cell appeared before locked allocation."""


class RunExecution:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        runtime: VariantRuntime | None = None,
        store: RunStore | None = None,
    ):
        self.repository = repository
        self.authoring = Authoring(repository)
        self.runtime = runtime or VariantRuntime(repository)
        self.store = store or RunStore(repository)
        self.planner = ExecutionPlanner(self._preflight)
        self.receipts = ReceiptCommitter(repository, self.store, self.authoring)

    @workspace_operation
    def train(
        self,
        experiment_name: str,
        variant_name: str,
        training_group: str,
        *,
        seed: int = 0,
        device: str = "auto",
        tracker: str | None = None,
    ) -> RunRecord:
        variant = self.authoring.check_variant(experiment_name, variant_name)
        self._require_committed_variant(experiment_name, variant)
        backends = self._tracker_backends(variant, tracker)
        environment_variant = self._with_tracker(variant, backends)
        try:
            with self.runtime.acquire_environment(environment_variant) as environment:
                return self._train(
                    experiment_name,
                    variant_name,
                    training_group,
                    python=environment.python,
                    environment_descriptor=environment.descriptor,
                    seed=seed,
                    device=device,
                    tracker=backends,
                )
        except RuntimeFailure as error:
            raise ExecutionFailure("train", str(error)) from error

    def _train(
        self,
        experiment_name: str,
        variant_name: str,
        training_group: str,
        *,
        python: Path,
        environment_descriptor: int,
        seed: int,
        device: str,
        tracker: tuple[str, ...],
    ) -> RunRecord:
        experiment, variant, python, source_digest, snapshot = self._prepare(
            experiment_name,
            variant_name,
            action="train",
            python=python,
            tracker=tracker,
        )
        plan = self.planner.train(
            experiment,
            variant,
            source_digest=source_digest,
            snapshot=snapshot,
            training_group=training_group,
            seed=seed,
            device=device,
            python=python,
            environment_descriptor=environment_descriptor,
        )
        target = plan.target
        fingerprint = plan.fingerprint

        def validate_slot(
            runs: list[RunRecord],
            models: list[ModelRecord],
        ) -> None:
            del models
            same_group = [
                item
                for item in runs
                if item.request["action"] == "train"
                and item.request["target"]["training_group"] == training_group
            ]
            if any(
                item.request["identity_fingerprint"] != fingerprint
                for item in same_group
            ):
                raise LifecycleConflict(
                    f"Training Group fingerprint changed: {training_group}"
                )
            if any(item.request["target"]["seed"] == seed for item in same_group):
                raise LifecycleConflict(
                    f"Training seed already has an execution: {training_group}/{seed}"
                )

        record = self.store.allocate(
            experiment,
            variant,
            action="train",
            target=target,
            exec_info=plan.preflight["exec"],
            source_digest=source_digest,
            identity_fingerprint=fingerprint,
            snapshot=snapshot,
            catalog_validator=validate_slot,
            validate_model_catalog=False,
        )
        return self._execute(
            record,
            variant,
            python=python,
            environment_descriptor=environment_descriptor,
            selected=plan.selected,
            expected_identity=plan.preflight["identity"],
        )

    @workspace_operation
    def evaluate(
        self,
        experiment_name: str,
        variant_name: str,
        training_group: str,
        evaluation_case_name: str,
        *,
        seed: int | str | None = None,
        device: str = "auto",
    ) -> list[RunRecord]:
        variant = self.authoring.check_variant(experiment_name, variant_name)
        self._require_committed_variant(experiment_name, variant)
        backends = self._tracker_backends(variant, None)
        environment_variant = self._with_tracker(variant, backends)
        try:
            with self.runtime.acquire_environment(environment_variant) as environment:
                return self._evaluate(
                    experiment_name,
                    variant_name,
                    training_group,
                    evaluation_case_name,
                    python=environment.python,
                    environment_descriptor=environment.descriptor,
                    seed=seed,
                    device=device,
                    tracker=backends,
                )
        except RuntimeFailure as error:
            raise ExecutionFailure("eval", str(error)) from error

    def _evaluate(
        self,
        experiment_name: str,
        variant_name: str,
        training_group: str,
        evaluation_case_name: str,
        *,
        python: Path,
        environment_descriptor: int,
        seed: int | str | None,
        device: str,
        tracker: tuple[str, ...],
    ) -> list[RunRecord]:
        experiment, variant, python, source_digest, snapshot = self._prepare(
            experiment_name,
            variant_name,
            action="eval",
            python=python,
            tracker=tracker,
        )
        selected_models = self._evaluation_models(
            experiment_name,
            variant_name,
            training_group,
            seed,
        )

        selected = validate_evaluation_readiness(
            experiment.document,
            variant.document,
            case=evaluation_case_name,
        )
        case_document = evaluation_case(variant.document, evaluation_case_name)
        metrics = metric_spec(variant.document, evaluation_case_name)
        existing_cells = self._evaluation_cells(experiment_name, variant_name)
        results: list[RunRecord] = []
        for model in selected_models:
            if self._skip_existing_evaluation(
                existing_cells,
                model.document["model_id"],
                evaluation_case_name,
                allow_skip=seed == "all",
            ):
                continue
            plan = self.planner.evaluation(
                experiment,
                variant,
                model,
                source_digest=source_digest,
                snapshot=snapshot,
                training_group=training_group,
                evaluation_case_name=evaluation_case_name,
                selected=selected,
                case_document=case_document,
                metrics=metrics,
                device=device,
                python=python,
                environment_descriptor=environment_descriptor,
            )
            target = plan.target
            fingerprint = plan.fingerprint

            def validate_slot(
                runs: list[RunRecord],
                models: list[ModelRecord],
            ) -> None:
                del models
                same_case = [
                    item
                    for item in runs
                    if item.request["action"] == "eval"
                    and item.request["target"]["evaluation_case"]
                    == evaluation_case_name
                ]
                if any(
                    item.request["identity_fingerprint"] != fingerprint
                    for item in same_case
                ):
                    raise LifecycleConflict(
                        f"Evaluation Case fingerprint changed: {evaluation_case_name}"
                    )
                if any(
                    item.request["target"]["model_id"] == model.document["model_id"]
                    for item in same_case
                ):
                    if seed == "all":
                        raise _EvaluationCellOccupied
                    raise LifecycleConflict(
                        "Model and Evaluation Case already have an execution"
                    )

            try:
                record = self.store.allocate(
                    experiment,
                    variant,
                    action="eval",
                    target=target,
                    exec_info=plan.preflight["exec"],
                    source_digest=source_digest,
                    identity_fingerprint=fingerprint,
                    snapshot=snapshot,
                    catalog_validator=validate_slot,
                    validate_model_catalog=False,
                )
            except _EvaluationCellOccupied:
                continue
            results.append(
                self._execute(
                    record,
                    variant,
                    python=python,
                    environment_descriptor=environment_descriptor,
                    selected=plan.selected,
                    expected_identity=plan.preflight["identity"],
                    model=model,
                )
            )
        return results

    @workspace_operation
    def export(
        self,
        experiment_name: str,
        variant_name: str,
        model_id: str,
        *,
        device: str = "auto",
    ) -> RunRecord:
        self.store.load_model(experiment_name, variant_name, model_id)
        variant = self.authoring.check_variant(experiment_name, variant_name)
        self._require_committed_variant(experiment_name, variant)
        backends = self._tracker_backends(variant, None)
        environment_variant = self._with_tracker(variant, backends)
        try:
            with self.runtime.acquire_environment(environment_variant) as environment:
                return self._export(
                    experiment_name,
                    variant_name,
                    model_id,
                    python=environment.python,
                    environment_descriptor=environment.descriptor,
                    device=device,
                    tracker=backends,
                )
        except RuntimeFailure as error:
            raise ExecutionFailure("export", str(error)) from error

    def _export(
        self,
        experiment_name: str,
        variant_name: str,
        model_id: str,
        *,
        python: Path,
        environment_descriptor: int,
        device: str,
        tracker: tuple[str, ...],
    ) -> RunRecord:
        model = self.store.load_model(experiment_name, variant_name, model_id)
        experiment, variant, python, source_digest, snapshot = self._prepare(
            experiment_name,
            variant_name,
            action="export",
            python=python,
            tracker=tracker,
        )
        plan = self.planner.export(
            experiment,
            variant,
            model,
            source_digest=source_digest,
            snapshot=snapshot,
            device=device,
            python=python,
            environment_descriptor=environment_descriptor,
        )
        target = plan.target
        fingerprint = plan.fingerprint

        def validate_slot(
            runs: list[RunRecord],
            models: list[ModelRecord],
        ) -> None:
            del models
            if any(
                item.request["action"] == "export"
                and item.request["target"]["model_id"] == model_id
                for item in runs
            ):
                raise LifecycleConflict(f"Model already has an export: {model_id}")

        record = self.store.allocate(
            experiment,
            variant,
            action="export",
            target=target,
            exec_info=plan.preflight["exec"],
            source_digest=source_digest,
            identity_fingerprint=fingerprint,
            snapshot=snapshot,
            catalog_validator=validate_slot,
            validate_model_catalog=False,
        )
        return self._execute(
            record,
            variant,
            python=python,
            environment_descriptor=environment_descriptor,
            selected=plan.selected,
            expected_identity=plan.preflight["identity"],
            model=model,
        )

    @workspace_operation
    def retry(
        self,
        experiment_name: str,
        variant_name: str,
        run_id: str,
        *,
        tracker: str | None = None,
    ) -> RunRecord:
        self.store.load(experiment_name, variant_name, run_id)
        variant = self.authoring.check_variant(experiment_name, variant_name)
        self._require_committed_variant(experiment_name, variant)
        backends = self._tracker_backends(variant, tracker)
        environment_variant = self._with_tracker(variant, backends)
        try:
            with self.runtime.acquire_environment(environment_variant) as environment:
                return self._retry(
                    experiment_name,
                    variant_name,
                    run_id,
                    python=environment.python,
                    environment_descriptor=environment.descriptor,
                    tracker=backends,
                )
        except RuntimeFailure as error:
            raise ExecutionFailure("retry", str(error)) from error

    def _retry(
        self,
        experiment_name: str,
        variant_name: str,
        run_id: str,
        *,
        python: Path,
        environment_descriptor: int,
        tracker: tuple[str, ...],
    ) -> RunRecord:
        original = self.store.load(experiment_name, variant_name, run_id)
        original, abandoned_tracker, recovered = self._prepare_retry_parent(original)
        if recovered:
            return original
        current_variant = self.authoring.check_variant(
            experiment_name,
            variant_name,
        )
        current_digest = compute_source_digest(current_variant.path / "src")
        if current_digest != original.request["source_digest"]:
            raise ContractError("Variant source changed since the original Run")
        self._finish_abandoned_tracker(
            original,
            current_variant,
            python,
            environment_descriptor,
            abandoned_tracker,
            tracker,
        )
        retry_snapshot = deepcopy(original.snapshot)
        retry_snapshot["variant"]["tracker"] = _tracker_document(tracker)
        runtime_variant = VariantRecord(
            current_variant.path,
            current_variant.experiment,
            retry_snapshot["variant"],
            current_variant.authored_schema_version,
            current_variant.code_document,
            current_variant.options_document,
        )
        original = RunRecord(
            original.path,
            original.address,
            retry_snapshot,
            original.request,
            original.state,
            original.graph_identity,
            original.event_hash,
        )
        plan = self.planner.retry(
            original,
            runtime_variant,
            source_digest=current_digest,
            python=python,
            environment_descriptor=environment_descriptor,
        )
        experiment = ExperimentRecord(
            self.repository.experiments / experiment_name,
            original.snapshot["experiment"],
        )
        retry = self._allocate_retry(
            original,
            experiment,
            runtime_variant,
            source_digest=current_digest,
            fingerprint=plan.fingerprint,
        )
        return self._execute_retry(
            original,
            retry,
            runtime_variant,
            python=python,
            environment_descriptor=environment_descriptor,
            selected=plan.selected,
            expected_identity=plan.preflight["identity"],
        )

    def _prepare_retry_parent(
        self,
        original: RunRecord,
    ) -> tuple[RunRecord, str | None, bool]:
        try:
            with self.store.run_lease(original):
                original = self.store.load(
                    original.request["experiment"],
                    original.request["variant"],
                    original.request["run_id"],
                )
                if self.store.direct_retry(original) is not None:
                    raise LifecycleConflict(
                        f"Run already has a retry: {original.address}"
                    )
                if original.state["status"] == "done":
                    raise LifecycleConflict(
                        f"completed Run cannot be retried: {original.address}"
                    )
                if original.state["status"] not in {"allocated", "running"}:
                    return original, None, False
                journal = load_attempt(original.path / ".attempt.json")
                if journal is not None and journal["phase"] == "ready":
                    return self.receipts.commit_receipt(original), None, True
                _remove_attempt_candidate(original, journal)
                original = self.store.update_state(
                    original,
                    status="abandoned",
                    reason="AbandonedExecution",
                    **_retry_checkpoint_changes(journal),
                )
                remove_attempt(original.path / ".attempt.json")
                return original, original.state["tracker_run_id"], False
        except LockUnavailableError as error:
            raise LifecycleConflict(f"Run is busy: {original.address}") from error

    def _finish_abandoned_tracker(
        self,
        original: RunRecord,
        variant: VariantRecord,
        python: Path,
        environment_descriptor: int,
        tracker_run_id: str | None,
        tracker: tuple[str, ...],
    ) -> None:
        if tracker_run_id is None:
            return
        parent_variant = self._with_tracker(
            variant, tuple(validate_tracker(original.snapshot["variant"]["tracker"]))
        )

        def finish(python: Path, environment_descriptor: int) -> None:
            with try_directory_lock(original.path) as descriptor:
                self.runtime.finish_tracker(
                    python,
                    parent_variant,
                    tracker_run_id=tracker_run_id,
                    status="KILLED",
                    lock_descriptor=descriptor,
                    environment_descriptor=environment_descriptor,
                )

        try:
            if "mlflow" in tracker:
                finish(python, environment_descriptor)
            else:
                with self.runtime.acquire_environment(parent_variant) as environment:
                    finish(environment.python, environment.descriptor)
        except RuntimeFailure as error:
            raise ExecutionFailure(
                original.request["action"],
                type(error).__name__,
                address=original.address,
            ) from error

    def _allocate_retry(
        self,
        original: RunRecord,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        *,
        source_digest: str,
        fingerprint: str,
    ) -> RunRecord:
        def validate_retry(
            runs: list[RunRecord],
            models: list[ModelRecord],
        ) -> None:
            del models
            if any(
                item.request["retry_of"] == original.request["run_id"] for item in runs
            ):
                raise LifecycleConflict(f"Run already has a retry: {original.address}")

        return self.store.allocate(
            experiment,
            variant,
            action=original.request["action"],
            target=original.request["target"],
            exec_info=original.request["exec"],
            source_digest=source_digest,
            identity_fingerprint=fingerprint,
            retry_of=original.request["run_id"],
            snapshot=original.snapshot,
            catalog_validator=validate_retry,
            validate_model_catalog=False,
        )

    def _execute_retry(
        self,
        original: RunRecord,
        retry: RunRecord,
        variant: VariantRecord,
        *,
        python: Path,
        environment_descriptor: int,
        selected: dict[str, str],
        expected_identity: dict[str, Any],
    ) -> RunRecord:
        if retry.request["action"] == "train":
            with _checkpoint_input(
                self._resume_checkpoint(original),
                original.state["last_checkpoint"],
                retry.path,
            ) as checkpoint:
                return self._execute(
                    retry,
                    variant,
                    python=python,
                    environment_descriptor=environment_descriptor,
                    selected=selected,
                    expected_identity=expected_identity,
                    resume_from=checkpoint,
                )
        model = self.store.load_model(
            retry.request["experiment"],
            retry.request["variant"],
            retry.request["target"]["model_id"],
        )
        return self._execute(
            retry,
            variant,
            python=python,
            environment_descriptor=environment_descriptor,
            selected=selected,
            expected_identity=expected_identity,
            model=model,
        )

    def _prepare(
        self,
        experiment_name: str,
        variant_name: str,
        *,
        action: str,
        python: Path,
        tracker: tuple[str, ...],
    ) -> tuple[ExperimentRecord, VariantRecord, Path, str, dict[str, Any]]:
        variant = self.authoring.check_variant(experiment_name, variant_name)
        experiment = self.authoring.load_experiment(experiment_name)
        variant = self._with_tracker(variant, tracker)
        source_digest = compute_source_digest(variant.path / "src")
        snapshot = self.store.freeze(experiment, variant, source_digest)
        return experiment, variant, python, source_digest, snapshot

    @staticmethod
    def _with_tracker(
        variant: VariantRecord, backends: tuple[str, ...]
    ) -> VariantRecord:
        document = deepcopy(variant.document)
        document["tracker"] = _tracker_document(backends)
        return replace(variant, document=document)

    def _tracker_backends(
        self, variant: VariantRecord, override: str | None
    ) -> tuple[str, ...]:
        if override is not None:
            return normalize_tracker(override)
        if variant.authored_schema_version == 2:
            return WorkspaceSettingsStore(self.repository.root).load().tracker_backends
        return tuple(validate_tracker(variant.document.get("tracker", {})))

    def _require_committed_variant(
        self,
        experiment_name: str,
        variant: VariantRecord,
    ) -> None:
        from hkdl.storage.graph.graph import V2Graph

        graph = V2Graph(self.repository)
        if graph.is_active():
            graph.assert_experiment_clean(
                experiment_name,
                self.authoring.load_experiment(experiment_name),
            )
            graph.assert_variant_clean(experiment_name, variant)

    def _preflight(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        environment_descriptor: int,
        action: str,
        snapshot: dict[str, Any],
        selected: dict[str, str],
        seed: int,
        device: str,
        fallback: dict[str, Any],
        target: dict[str, Any],
        diagnostic_phase: str = "preflight",
    ) -> dict[str, Any]:
        try:
            result = self.runtime.preflight(
                python,
                variant,
                action=action,
                cfg=snapshot,
                selected=selected,
                seed=seed,
                device=device,
                identity_fallback=deepcopy(fallback),
                runtime_target=deepcopy(target),
                environment_descriptor=environment_descriptor,
            )
            identity = result.get("identity")
            if not isinstance(identity, dict):
                raise ContractError("Variant preflight identity is invalid")
            return {"exec": result["exec"], "identity": identity}
        except Exception as error:
            # Postflight is reported by the Run-owned execution boundary.
            if diagnostic_phase == "preflight":
                report_exception("preflight", error)
            raise

    # Flow: lease the Run, journal the attempt, attach tracking, invoke the worker,
    # revalidate identity, publish its durable receipt, then finish tracking.
    # Failures before a ready receipt stop the Run and clean its candidate; a ready
    # receipt remains intact for recovery.
    def _execute(
        self,
        record: RunRecord,
        variant: VariantRecord,
        *,
        python: Path,
        environment_descriptor: int,
        selected: dict[str, str],
        expected_identity: dict[str, Any],
        model: ModelRecord | None = None,
        resume_from: Path | None = None,
    ) -> RunRecord:
        """Execute and publish one journaled action while holding the Run lease.

        Failures before receipt readiness stop the attempt; ready receipts remain
        recoverable.
        """
        action = record.request["action"]
        phase = "setup"
        try:
            with self.store.run_lease(record) as lock_descriptor:
                record, attempt_path, candidate = self._start_attempt(
                    record,
                    lock_descriptor=lock_descriptor,
                )
                phase = "tracker.setup"
                tracker_run_id = self._ensure_tracker(
                    record,
                    variant,
                    python,
                    attempt_path,
                    lock_descriptor,
                    environment_descriptor,
                )
                record = self.store.load(
                    record.request["experiment"],
                    record.request["variant"],
                    record.request["run_id"],
                )
                phase = "action"
                result = self._invoke_action(
                    record,
                    variant,
                    python=python,
                    environment_descriptor=environment_descriptor,
                    selected=selected,
                    model=model,
                    resume_from=resume_from,
                    tracker_run_id=tracker_run_id,
                    attempt_path=attempt_path,
                    candidate=candidate,
                    lock_descriptor=lock_descriptor,
                )
                phase = "postflight"
                self._record_worker_result(attempt_path, result)
                self._verify_identity_after_execution(
                    record,
                    variant,
                    python=python,
                    selected=selected,
                    expected_identity=expected_identity,
                    environment_descriptor=environment_descriptor,
                )
                phase = "tracker.metrics"
                self._log_evaluation_metrics(
                    action,
                    result,
                    variant,
                    python=python,
                    tracker_run_id=tracker_run_id,
                    lock_descriptor=lock_descriptor,
                    environment_descriptor=environment_descriptor,
                )
                phase = "publication"
                completed = self.receipts.complete_from_journal(record, attempt_path)
                phase = "tracker.finish"
                self.runtime.finish_tracker(
                    python,
                    variant,
                    tracker_run_id=tracker_run_id,
                    status="FINISHED",
                    lock_descriptor=lock_descriptor,
                    environment_descriptor=environment_descriptor,
                )
                return completed
        except LockUnavailableError as error:
            raise LifecycleConflict(f"Run is busy: {record.address}") from error
        except (RuntimeInterrupted, KeyboardInterrupt) as error:
            self._stop_attempt(
                record,
                variant,
                python,
                environment_descriptor,
                status="interrupted",
                reason="interrupted",
            )
            raise ExecutionInterrupted(action, record.address) from error
        except RuntimeOwnershipConflict as error:
            _report_execution_failure(phase, error, record)
            self._stop_attempt(
                record,
                variant,
                python,
                environment_descriptor,
                status="failed",
                reason="TrackerOwnershipConflict",
            )
            raise LifecycleConflict(
                f"Tracker ownership conflict: {record.address}"
            ) from error
        except Exception as error:
            _report_execution_failure(phase, error, record)
            self._stop_attempt(
                record,
                variant,
                python,
                environment_descriptor,
                status="failed",
                reason=type(error).__name__,
            )
            raise ExecutionFailure(
                action,
                f"{phase}: {type(error).__name__}",
                address=record.address,
            ) from error

    def _invoke_action(
        self,
        record: RunRecord,
        variant: VariantRecord,
        *,
        python: Path,
        environment_descriptor: int,
        selected: dict[str, str],
        model: ModelRecord | None,
        resume_from: Path | None,
        tracker_run_id: str | None,
        attempt_path: Path,
        candidate: Path,
        lock_descriptor: int,
    ) -> dict[str, Any]:
        action = record.request["action"]
        common = {
            "cfg": record.snapshot,
            "selected": selected,
            "exec_info": record.request["exec"],
            "run_dir": record.path,
            "tracker_run_id": tracker_run_id,
            "attempt_path": attempt_path,
            "lock_descriptor": lock_descriptor,
            "runtime_target": record.request["target"],
            "environment_descriptor": environment_descriptor,
        }
        if action == "train":
            return self.runtime.train(
                python,
                variant,
                resume_from=resume_from,
                **common,
            )
        if model is None:
            raise ContractError(f"{action.title()} Run has no Model")
        with _checkpoint_input(
            self.store.resolve_model_checkpoint(model),
            model.document["checkpoint"]["path"],
            record.path,
        ) as checkpoint:
            if action == "eval":
                return self.runtime.evaluate(
                    python,
                    variant,
                    checkpoint=checkpoint,
                    results_dir=candidate,
                    **common,
                )
            return self.runtime.export(
                python,
                variant,
                export_dir=candidate,
                checkpoint=checkpoint,
                **common,
            )

    def _log_evaluation_metrics(
        self,
        action: str,
        result: dict[str, Any],
        variant: VariantRecord,
        *,
        python: Path,
        tracker_run_id: str | None,
        lock_descriptor: int,
        environment_descriptor: int,
    ) -> None:
        if action != "eval":
            return
        values = result.get("values")
        if not isinstance(values, dict):
            raise RuntimeFailure("Evaluator values must be a mapping")
        self.runtime.log_tracker_metrics(
            python,
            variant,
            tracker_run_id=tracker_run_id,
            values=values,
            lock_descriptor=lock_descriptor,
            environment_descriptor=environment_descriptor,
        )

    def _start_attempt(
        self,
        record: RunRecord,
        *,
        lock_descriptor: int,
    ) -> tuple[RunRecord, Path, Path]:
        attempt_path = record.path / ".attempt.json"
        if os.path.lexists(attempt_path):
            raise ContractError("Run already has an attempt journal")
        candidate_relative = action_candidate(record.request["action"])
        candidate = (
            record.path / candidate_relative
            if candidate_relative is not None
            else record.path
        )
        if candidate_relative is not None:
            if os.path.lexists(candidate):
                raise ContractError("action candidate already exists")
            candidate.mkdir()
        journal = new_attempt(
            action=record.request["action"],
            tracker_run_id=record.state["tracker_run_id"],
            candidate=candidate_relative,
        )
        write_attempt(
            attempt_path,
            journal,
            directory_descriptor=lock_descriptor,
        )
        try:
            record = self.store.update_state(
                record,
                status="running",
                reason=None,
            )
        except BaseException:
            remove_attempt(attempt_path)
            if candidate_relative is not None:
                shutil.rmtree(candidate)
            raise
        return record, attempt_path, candidate

    def _ensure_tracker(
        self,
        record: RunRecord,
        variant: VariantRecord,
        python: Path,
        attempt_path: Path,
        lock_descriptor: int,
        environment_descriptor: int,
    ) -> str | None:
        tracker_run_id = record.state["tracker_run_id"]
        if "mlflow" in validate_tracker(record.snapshot["variant"]["tracker"]):
            target = record.request["target"]
            metadata = {
                "action": record.request["action"],
                "training_group": target.get("training_group"),
                "seed": target.get("seed"),
                "model_id": target.get("model_id"),
                "evaluation_case": target.get("evaluation_case"),
                "retry_of": record.request["retry_of"],
            }
            from hkdl.storage.graph.execution import GraphRecorder

            recorder = GraphRecorder(self.repository)
            if recorder.active():
                metadata.update(recorder.identity(record).as_dict())
            tracker_run_id = self.runtime.ensure_tracker(
                python,
                variant,
                cfg=record.snapshot,
                run_dir=record.path,
                current_tracker_run_id=tracker_run_id,
                lock_descriptor=lock_descriptor,
                metadata=metadata,
                environment_descriptor=environment_descriptor,
            )
            record = self.store.update_state(
                record,
                tracker_run_id=tracker_run_id,
            )
            journal = load_attempt(attempt_path)
            if journal is None:
                raise ContractError("attempt journal disappeared")
            journal["tracker_run_id"] = tracker_run_id
            write_attempt(attempt_path, journal)
        return tracker_run_id

    def _verify_identity_after_execution(
        self,
        record: RunRecord,
        variant: VariantRecord,
        *,
        python: Path,
        environment_descriptor: int,
        selected: dict[str, str],
        expected_identity: dict[str, Any],
    ) -> None:
        if (
            compute_source_digest(variant.path / "src")
            != record.request["source_digest"]
        ):
            raise RuntimeFailure("Variant source changed during execution")
        result = self._preflight(
            python,
            variant,
            action=record.request["action"],
            snapshot=record.snapshot,
            selected=selected,
            seed=record.request["exec"]["seed"],
            device=record.request["exec"]["device"],
            fallback=fallback_for_request(record, selected),
            target=record.request["target"],
            diagnostic_phase="postflight",
            environment_descriptor=environment_descriptor,
        )
        if (
            result["exec"] != record.request["exec"]
            or result["identity"] != expected_identity
        ):
            raise RuntimeFailure("Variant action identity changed during execution")

    def _stop_attempt(
        self,
        record: RunRecord,
        variant: VariantRecord,
        python: Path,
        environment_descriptor: int,
        *,
        status: str,
        reason: str,
    ) -> None:
        try:
            current = self.store.load(
                record.request["experiment"],
                record.request["variant"],
                record.request["run_id"],
            )
        except Exception:
            return
        try:
            with self.store.run_lease(current):
                current = self.store.load(
                    record.request["experiment"],
                    record.request["variant"],
                    record.request["run_id"],
                )
                if current.state["status"] in TERMINAL_STATUSES:
                    return
                attempt_path = current.path / ".attempt.json"
                journal = load_attempt(attempt_path)
                if journal is not None and journal["phase"] == "ready":
                    return
                _remove_attempt_candidate(current, journal)
                current = self.store.update_state(
                    current,
                    **_stopped_state_changes(journal, status, reason),
                )
                remove_attempt(attempt_path)
        except LockUnavailableError as error:
            raise LifecycleConflict(f"Run is busy: {record.address}") from error
        self._finish_stopped_tracker(
            current,
            variant,
            python,
            environment_descriptor,
            status,
        )

    def _finish_stopped_tracker(
        self,
        record: RunRecord,
        variant: VariantRecord,
        python: Path,
        environment_descriptor: int,
        status: str,
    ) -> None:
        tracker = record.state["tracker_run_id"]
        if tracker is None:
            return
        try:
            with try_directory_lock(record.path) as descriptor:
                self.runtime.finish_tracker(
                    python,
                    variant,
                    tracker_run_id=tracker,
                    status="KILLED" if status == "interrupted" else "FAILED",
                    lock_descriptor=descriptor,
                    environment_descriptor=environment_descriptor,
                )
        except Exception as error:
            report_exception("tracker.cleanup", error)

    def _record_worker_result(
        self,
        attempt_path: Path,
        result: dict[str, Any],
    ) -> None:
        journal = load_attempt(attempt_path)
        if journal is None:
            raise ContractError("attempt journal disappeared")
        if journal["phase"] == "running":
            journal["phase"] = "worker_done"
            journal["result"] = result
            write_attempt(attempt_path, journal)
        elif journal["phase"] != "worker_done":
            raise ContractError("attempt journal phase is invalid")

    def _resume_checkpoint(self, record: RunRecord) -> Path | None:
        relative = record.state["last_checkpoint"]
        if relative is None:
            return None
        if record.event_hash is not None:
            return self.store.graph_reader().artifact(record, relative)
        path = record.path / relative
        if path.is_symlink() or not path.is_file():
            raise ContractError("retry checkpoint is invalid")
        try:
            path.resolve(strict=True).relative_to(
                (record.path / "artifacts/checkpoints").resolve(strict=True)
            )
        except (OSError, ValueError) as error:
            raise ContractError("retry checkpoint is outside its root") from error
        return path.resolve()

    def _evaluation_cells(
        self,
        experiment: str,
        variant: str,
    ) -> set[tuple[str, str]]:
        return {
            (
                record.request["target"]["model_id"],
                record.request["target"]["evaluation_case"],
            )
            for record in self.store.scan(experiment=experiment, variant=variant)
            if record.request["action"] == "eval"
        }

    def _evaluation_models(
        self,
        experiment: str,
        variant: str,
        training_group: str,
        seed: int | str | None,
    ) -> list[ModelRecord]:
        models = [
            model
            for model in self.store.scan_models(experiment=experiment, variant=variant)
            if model.document["training_group"] == training_group
        ]
        if not models:
            raise LifecycleConflict(f"Training Group has no Models: {training_group}")
        if seed is None and len(models) != 1:
            raise LifecycleConflict(
                f"Training Group has multiple Models; specify --seed: {training_group}"
            )
        if seed not in {None, "all"}:
            models = [model for model in models if model.document["seed"] == seed]
            if not models:
                raise LifecycleConflict(
                    f"Training Group seed has no Model: {training_group}/{seed}"
                )
        return sorted(models, key=lambda item: item.document["seed"])

    def _skip_existing_evaluation(
        self,
        existing_cells: set[tuple[str, str]],
        model_id: str,
        case: str,
        *,
        allow_skip: bool,
    ) -> bool:
        if (model_id, case) not in existing_cells:
            return False
        if allow_skip:
            return True
        raise LifecycleConflict("Model and Evaluation Case already have an execution")


def _report_execution_failure(
    phase: str, error: BaseException, record: RunRecord
) -> None:
    report_exception(phase, error)
    try:
        log = record.path / "worker.log"
        if phase == "action" and log.is_file() and not log.is_symlink():
            request = record.request
            report_message(
                phase,
                "worker log: hkdl run logs "
                f"{request['experiment']} {request['variant']} {request['run_id']}",
            )
    except Exception:
        pass


def _retry_checkpoint_changes(journal: dict[str, Any] | None) -> dict[str, Any]:
    if journal is None or journal["action"] != "train":
        return {}
    checkpoint = journal["checkpoint"]
    changes: dict[str, Any] = {}
    if checkpoint["best"] is not None:
        changes["best_checkpoint"] = checkpoint["best"]
    if checkpoint["last"] is not None:
        changes["last_checkpoint"] = checkpoint["last"]
    return changes


def _stopped_state_changes(
    journal: dict[str, Any] | None,
    status: str,
    reason: str,
) -> dict[str, Any]:
    changes = {"status": status, "reason": reason}
    changes.update(_retry_checkpoint_changes(journal))
    return changes


def _remove_attempt_candidate(
    record: RunRecord,
    journal: dict[str, Any] | None,
) -> None:
    relative = action_candidate(record.request["action"])
    if journal is not None:
        validate_attempt_owner(journal, record.request["action"])
    if relative is None:
        return
    candidate = record.path / relative
    if candidate.parent.is_symlink() or not candidate.parent.is_dir():
        raise ContractError("attempt candidate parent must be a real directory")
    if not os.path.lexists(candidate):
        return
    if candidate.is_dir() and not candidate.is_symlink():
        shutil.rmtree(candidate)
    else:
        candidate.unlink()
    descriptor = os.open(candidate.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tracker_document(backends: tuple[str, ...]) -> dict[str, object]:
    if not backends:
        value: object = "none"
    elif len(backends) == 1:
        value = backends[0]
    else:
        value = list(backends)
    return {"backend": value}


__all__ = [
    "ExecutionFailure",
    "ExecutionInterrupted",
    "LifecycleConflict",
    "RunExecution",
]


@contextmanager
def _checkpoint_input(
    source: Path | None, relative: str | None, run_dir: Path
) -> Iterator[Path | None]:
    """Lend the worker a named copy without exposing immutable blob storage."""
    if source is None:
        yield None
        return
    if relative is None:
        raise ContractError("checkpoint has no original path")
    with TemporaryDirectory(prefix=".hkdl-input-", dir=run_dir) as directory:
        checkpoint = Path(directory) / Path(relative).name
        shutil.copyfile(source, checkpoint)
        checkpoint.chmod(0o400)
        yield checkpoint
