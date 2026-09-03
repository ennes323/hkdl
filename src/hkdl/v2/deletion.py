"""Planning and recoverable execution for logical v2 deletion.

The v2 object store is immutable.  Deleting a Run or Experiment therefore
means removing the current name bindings that make it reachable and moving
its authored/generated projections to a recoverable trash transaction.
Objects, blobs, and the append-only binding history are deliberately retained.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config import DIGEST_PATTERN, ContractError
from ..run_contracts import RUN_STATUSES, TERMINAL_STATUSES
from ..runs import RunStore
from ..storage import (
    LockUnavailableError,
    NotFoundError,
    RepositoryPaths,
    atomic_replace,
    atomic_write_new,
)
from .bindings import BindingHeadConflict, BindingOperation
from .graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    attempt_event_scope,
    entity_revision_scope,
    evaluation_case_scope,
    experiment_variant_scope,
    export_profile_scope,
    comparison_group_scope,
    variant_model_scope,
    variant_run_scope,
    workspace_experiment_scope,
)
from .objects import ObjectRecord, object_digest
from .projection import GraphProjection
from .leases import attempt_lease, lease_held, entity_guard
from .maintenance import workspace_operation


_SCOPE_HASH = re.compile(
    r"^(?P<prefix>[a-z-]+)/(?P<digest>[0-9a-f]{64})/(?P<suffix>[a-z-]+)$"
)
_JOURNAL_SCHEMA = 1
_JOURNAL_PHASES = frozenset(
    {"planned", "moving", "moved", "binding_committed", "completed", "rolled_back"}
)


class DeletionConflict(RuntimeError):
    """A deletion is unsafe or requires an explicit deletion option."""

    def __init__(
        self,
        message: str,
        plan: "RunDeletionPlan | ExperimentDeletionPlan | VariantDeletionPlan | None" = None,
    ):
        super().__init__(message)
        self.plan = plan


RunDeletionConflict = DeletionConflict


class DeletionFailure(RuntimeError):
    """A confirmed deletion hit an operational failure."""

    def __init__(self, message: str, *, state: str, journal: Path | None = None):
        super().__init__(message)
        self.state = state
        self.journal = journal


@dataclass(frozen=True)
class DeletionRun:
    """One active Run binding and its typed v2 references."""

    experiment: str
    variant: str
    run_id: str
    address: str
    variant_hash: str
    attempt_hash: str
    run_spec_hash: str
    event_hash: str
    action: str
    status: str
    retry_parent: str | None
    path: Path | None = None
    event_chain: tuple[str, ...] = ()
    result_hashes: tuple[str, ...] = ()
    blob_hashes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "variant": self.variant,
            "run_id": self.run_id,
            "address": self.address,
            "variant_hash": self.variant_hash,
            "attempt_hash": self.attempt_hash,
            "run_spec_hash": self.run_spec_hash,
            "event_hash": self.event_hash,
            "action": self.action,
            "status": self.status,
            "retry_parent": self.retry_parent,
            "event_chain": list(self.event_chain),
            "result_hashes": list(self.result_hashes),
            "blob_hashes": list(self.blob_hashes),
            "path": str(self.path) if self.path is not None else None,
        }


@dataclass(frozen=True)
class DeletionModel:
    """One active Model binding produced by a selected Attempt."""

    experiment: str
    variant: str
    variant_hash: str
    model_id: str
    model_hash: str
    producing_attempt: str
    checkpoint_blob: str | None
    path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "variant": self.variant,
            "variant_hash": self.variant_hash,
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "producing_attempt": self.producing_attempt,
            "checkpoint_blob": self.checkpoint_blob,
            "path": str(self.path) if self.path is not None else None,
        }


@dataclass(frozen=True)
class RunDeletionPlan:
    """Deterministic read-only deletion plan.

    ``runs`` is the complete Run closure that would be removed when
    ``cascade`` is true.  A plan with extra runs or Models is blocked unless
    cascade is explicitly enabled.  The digest excludes physical paths so a
    rename or a checkout at a different absolute path does not change the
    graph plan.
    """

    experiment: str
    variant: str
    run_id: str
    variant_hash: str
    root_attempt_hash: str
    before_head: str | None
    runs: tuple[DeletionRun, ...]
    models: tuple[DeletionModel, ...]
    attempt_hashes: tuple[str, ...]
    run_spec_hashes: tuple[str, ...]
    event_hashes: tuple[str, ...]
    result_hashes: tuple[str, ...]
    blob_hashes: tuple[str, ...]
    operations: tuple[BindingOperation, ...]
    dependencies: tuple[str, ...]
    held_leases: tuple[str, ...]
    stale_nonterminal: tuple[str, ...]
    conflicts: tuple[str, ...]
    cascade: bool
    force_stale: bool
    plan_digest: str

    @property
    def ready(self) -> bool:
        return not self.conflicts

    @property
    def blocked(self) -> bool:
        return not self.ready

    @property
    def root(self) -> str:
        return f"{self.experiment}/{self.variant}/{self.run_id}"

    @property
    def retry_descendants(self) -> tuple[DeletionRun, ...]:
        return tuple(item for item in self.runs if item.address != self.root)

    @property
    def run_attempts(self) -> tuple[str, ...]:
        return self.attempt_hashes

    @property
    def model_hashes(self) -> tuple[str, ...]:
        return tuple(item.model_hash for item in self.models)

    @property
    def result_objects(self) -> tuple[str, ...]:
        return self.result_hashes

    @property
    def artifact_blobs(self) -> tuple[str, ...]:
        return self.blob_hashes

    @property
    def binding_operations(self) -> tuple[BindingOperation, ...]:
        return self.operations

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": {
                "experiment": self.experiment,
                "variant": self.variant,
                "run_id": self.run_id,
                "address": self.root,
                "variant_hash": self.variant_hash,
                "attempt_hash": self.root_attempt_hash,
            },
            "before_head": self.before_head,
            "runs": [item.as_dict() for item in self.runs],
            "models": [item.as_dict() for item in self.models],
            "attempt_hashes": list(self.attempt_hashes),
            "run_spec_hashes": list(self.run_spec_hashes),
            "event_hashes": list(self.event_hashes),
            "result_hashes": list(self.result_hashes),
            "blob_hashes": list(self.blob_hashes),
            "operations": [item.as_dict() for item in self.operations],
            "dependencies": list(self.dependencies),
            "held_leases": list(self.held_leases),
            "stale_nonterminal": list(self.stale_nonterminal),
            "conflicts": list(self.conflicts),
            "cascade": self.cascade,
            "force_stale": self.force_stale,
            "ready": self.ready,
            "plan_digest": self.plan_digest,
        }

    def digest_payload(self) -> dict[str, Any]:
        """Return the path-independent content used to calculate the digest."""

        document = self.as_dict()
        document.pop("plan_digest", None)
        for run in document["runs"]:
            run.pop("path", None)
        for model in document["models"]:
            model.pop("path", None)
        return document


@dataclass(frozen=True)
class RunDeletionResult:
    """Published deletion transaction result."""

    transaction: str
    plan_digest: str
    binding_transaction: str
    journal: Path
    moved: tuple[str, ...]
    missing: tuple[str, ...]
    unbound: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction": self.transaction,
            "plan_digest": self.plan_digest,
            "binding_transaction": self.binding_transaction,
            "journal": str(self.journal),
            "moved": list(self.moved),
            "missing": list(self.missing),
            "unbound": list(self.unbound),
        }


DeletionResult = RunDeletionResult


@dataclass(frozen=True)
class ExperimentDeletionBlocker:
    """One actionable reason an Experiment cannot be deleted."""

    code: str
    address: str
    status: str | None
    lease: str | None
    reason: str
    next_action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "address": self.address,
            "status": self.status,
            "lease": self.lease,
            "reason": self.reason,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class ExperimentDeletionPlan:
    """Deterministic read-only plan for deleting one complete Experiment."""

    experiment: str
    experiment_hash: str
    before_head: str | None
    variants: tuple[str, ...]
    variant_hashes: tuple[str, ...]
    runs: tuple[DeletionRun, ...]
    models: tuple[DeletionModel, ...]
    operations: tuple[BindingOperation, ...]
    blockers: tuple[ExperimentDeletionBlocker, ...]
    authored_path: str
    output_paths: tuple[str, ...]
    plan_digest: str

    @property
    def ready(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "experiment_hash": self.experiment_hash,
            "before_head": self.before_head,
            "variants": list(self.variants),
            "variant_hashes": list(self.variant_hashes),
            "runs": [item.as_dict() for item in self.runs],
            "models": [item.as_dict() for item in self.models],
            "operations": [item.as_dict() for item in self.operations],
            "blockers": [item.as_dict() for item in self.blockers],
            "authored_path": self.authored_path,
            "output_paths": list(self.output_paths),
            "ready": self.ready,
            "plan_digest": self.plan_digest,
        }

    def digest_payload(self) -> dict[str, Any]:
        document = self.as_dict()
        document.pop("plan_digest", None)
        for run in document["runs"]:
            run.pop("path", None)
        for model in document["models"]:
            model.pop("path", None)
        return document


ExperimentDeletionResult = RunDeletionResult


@dataclass(frozen=True)
class DerivedVariant:
    """One active Variant below the selected Variant in effective lineage."""

    experiment: str
    variant: str
    address: str
    relation: str
    depth: int
    current_parent: str
    effective_parent_after: str | None

    @property
    def reconnected(self) -> bool:
        return self.depth == 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "variant": self.variant,
            "address": self.address,
            "relation": self.relation,
            "depth": self.depth,
            "current_parent": self.current_parent,
            "effective_parent_after": self.effective_parent_after,
            "reconnected": self.reconnected,
        }


@dataclass(frozen=True)
class VariantDeletionPlan:
    """Deterministic plan for deleting one complete active Variant owner."""

    experiment: str
    variant: str
    experiment_hash: str
    variant_hash: str
    before_head: str | None
    revision_hashes: tuple[str, ...]
    runs: tuple[DeletionRun, ...]
    models: tuple[DeletionModel, ...]
    operations: tuple[BindingOperation, ...]
    blockers: tuple[ExperimentDeletionBlocker, ...]
    derived_variants: tuple[DerivedVariant, ...]
    authored_path: str
    output_paths: tuple[str, ...]
    plan_digest: str

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def requires_lineage_confirmation(self) -> bool:
        return bool(self.derived_variants)

    @property
    def direct_children(self) -> tuple[DerivedVariant, ...]:
        return tuple(item for item in self.derived_variants if item.reconnected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "variant": self.variant,
            "experiment_hash": self.experiment_hash,
            "variant_hash": self.variant_hash,
            "before_head": self.before_head,
            "revision_hashes": list(self.revision_hashes),
            "runs": [item.as_dict() for item in self.runs],
            "models": [item.as_dict() for item in self.models],
            "operations": [item.as_dict() for item in self.operations],
            "blockers": [item.as_dict() for item in self.blockers],
            "derived_variants": [item.as_dict() for item in self.derived_variants],
            "authored_path": self.authored_path,
            "output_paths": list(self.output_paths),
            "ready": self.ready,
            "requires_lineage_confirmation": self.requires_lineage_confirmation,
            "plan_digest": self.plan_digest,
        }

    def digest_payload(self) -> dict[str, Any]:
        document = self.as_dict()
        document.pop("plan_digest", None)
        for run in document["runs"]:
            run.pop("path", None)
        for model in document["models"]:
            model.pop("path", None)
        return document


VariantDeletionResult = RunDeletionResult


@dataclass(frozen=True)
class _VariantContext:
    experiment: str
    variant: str
    variant_hash: str


class RunDeletionService:
    """Plan and apply one graph-rooted Run deletion."""

    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        graph: V2Graph | None = None,
        store: RunStore | None = None,
        now: Callable[[], datetime] | None = None,
        nonce: Callable[[], str] | None = None,
    ):
        self.repository = repository
        self.graph = graph or V2Graph(repository, now=now, nonce=nonce)
        self.store = store or RunStore(repository, now=now)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._nonce = nonce or (lambda: secrets.token_hex(16))
        self.trash_root = repository.root / ".hkdl/trash/deletions"

    @workspace_operation
    def plan(
        self,
        experiment: str,
        variant: str,
        run_id: str,
        *,
        cascade: bool = False,
        force_stale: bool = False,
    ) -> RunDeletionPlan:
        """Build a deterministic, non-mutating deletion plan."""

        if not self.graph.is_active():
            raise ContractError("HKDL v2 is not active")
        self._assert_no_pending_deletion()
        bindings = self.graph.bindings.bindings()
        contexts = self._contexts(bindings)
        experiment_hash = self.graph.experiment_hash(experiment)
        variant_hash = self.graph.variant_hash(experiment_hash, variant)
        root_binding = (variant_run_scope(variant_hash), run_id)
        try:
            root_attempt = bindings[root_binding]
        except KeyError as error:
            raise NotFoundError(
                f"Run not found: {experiment}/{variant}/{run_id}"
            ) from error

        runs = self._runs(bindings, contexts)
        by_attempt = {item.attempt_hash: item for item in runs}
        if root_attempt not in by_attempt:
            raise ContractError("v2 Run binding does not reference an active Run")

        selected_attempts = {root_attempt}
        changed = True
        while changed:
            changed = False
            for item in runs:
                if (
                    item.retry_parent in selected_attempts
                    and item.attempt_hash not in selected_attempts
                ):
                    selected_attempts.add(item.attempt_hash)
                    changed = True
            selected_models = {
                item.model_hash
                for item in self._models(bindings, contexts)
                if item.producing_attempt in selected_attempts
            }
            for item in runs:
                if (
                    item.attempt_hash not in selected_attempts
                    and item.action in {"eval", "export"}
                    and self._run_model_hash(item) in selected_models
                ):
                    selected_attempts.add(item.attempt_hash)
                    changed = True

        all_models = self._models(bindings, contexts)
        selected_runs = tuple(
            sorted(
                (item for item in runs if item.attempt_hash in selected_attempts),
                key=lambda item: item.address.encode("utf-8"),
            )
        )
        selected_models = tuple(
            sorted(
                (
                    item
                    for item in all_models
                    if item.producing_attempt in selected_attempts
                ),
                key=lambda item: item.model_hash,
            )
        )
        active_train_comparisons = {
            self._run_comparison(item)
            for item in runs
            if item.attempt_hash not in selected_attempts and item.action == "train"
        }
        selected_comparisons = {
            self._run_comparison(item)
            for item in selected_runs
            if item.action == "train"
        }
        operations = self._operations(
            bindings,
            selected_runs,
            selected_models,
            selected_comparisons,
            active_train_comparisons,
        )
        attempt_hashes = tuple(sorted(selected_attempts))
        run_spec_hashes = tuple(sorted({item.run_spec_hash for item in selected_runs}))
        event_hashes = tuple(
            sorted({event for item in selected_runs for event in item.event_chain})
        )
        result_hashes = tuple(
            sorted({result for item in selected_runs for result in item.result_hashes})
        )
        blob_hashes = tuple(
            sorted(
                {blob for item in selected_runs for blob in item.blob_hashes}
                | {
                    blob
                    for item in selected_models
                    if (blob := item.checkpoint_blob) is not None
                }
            )
        )
        dependencies = tuple(
            sorted(
                [
                    f"run:{item.address}"
                    for item in selected_runs
                    if item.attempt_hash != root_attempt
                ]
                + [f"model:{item.model_hash}" for item in selected_models],
                key=lambda item: item.encode("utf-8"),
            )
        )
        held_leases = self._held_leases(selected_runs)
        stale_nonterminal = tuple(
            item.address
            for item in selected_runs
            if item.status not in TERMINAL_STATUSES and item.address not in held_leases
        )
        conflicts: list[str] = []
        if dependencies and not cascade:
            conflicts.append(
                "deletion has dependencies; rerun with cascade: "
                + ", ".join(dependencies)
            )
        if held_leases:
            conflicts.append("Run lease is held: " + ", ".join(held_leases))
        if stale_nonterminal and not force_stale:
            conflicts.append(
                "stale nonterminal Run requires force_stale: "
                + ", ".join(stale_nonterminal)
            )

        provisional = RunDeletionPlan(
            experiment=experiment,
            variant=variant,
            run_id=run_id,
            variant_hash=variant_hash,
            root_attempt_hash=root_attempt,
            before_head=self.graph.bindings.head(),
            runs=selected_runs,
            models=selected_models,
            attempt_hashes=attempt_hashes,
            run_spec_hashes=run_spec_hashes,
            event_hashes=event_hashes,
            result_hashes=result_hashes,
            blob_hashes=blob_hashes,
            operations=operations,
            dependencies=dependencies,
            held_leases=held_leases,
            stale_nonterminal=stale_nonterminal,
            conflicts=tuple(conflicts),
            cascade=cascade,
            force_stale=force_stale,
            plan_digest="",
        )
        digest = object_digest("binding_transaction", provisional.digest_payload())
        return RunDeletionPlan(
            experiment=provisional.experiment,
            variant=provisional.variant,
            run_id=provisional.run_id,
            variant_hash=provisional.variant_hash,
            root_attempt_hash=provisional.root_attempt_hash,
            before_head=provisional.before_head,
            runs=provisional.runs,
            models=provisional.models,
            attempt_hashes=provisional.attempt_hashes,
            run_spec_hashes=provisional.run_spec_hashes,
            event_hashes=provisional.event_hashes,
            result_hashes=provisional.result_hashes,
            blob_hashes=provisional.blob_hashes,
            operations=provisional.operations,
            dependencies=provisional.dependencies,
            held_leases=provisional.held_leases,
            stale_nonterminal=provisional.stale_nonterminal,
            conflicts=provisional.conflicts,
            cascade=provisional.cascade,
            force_stale=provisional.force_stale,
            plan_digest=digest,
        )

    def delete(
        self,
        experiment: str,
        variant: str,
        run_id: str,
        *,
        cascade: bool = False,
        force_stale: bool = False,
        yes: bool = True,
        dry_run: bool = False,
    ) -> RunDeletionPlan | RunDeletionResult:
        """Plan or execute one deletion.

        The service API treats invocation as explicit confirmation; callers
        exposing a CLI should map ``--yes`` to this ``yes`` argument and use
        ``dry_run`` for previews.
        """

        plan = self.plan(
            experiment,
            variant,
            run_id,
            cascade=cascade,
            force_stale=force_stale,
        )
        if dry_run:
            return plan
        if not yes:
            raise DeletionConflict("deletion confirmation is required", plan)
        return self.execute(plan)

    apply = delete

    @workspace_operation
    def execute(self, plan: RunDeletionPlan, *, yes: bool = True) -> RunDeletionResult:
        experiment = self.graph.store.load(plan.variant_hash).payload["experiment"]
        with entity_guard(
            self.repository,
            [("experiment", experiment), ("variant", plan.variant_hash)],
        ):
            return self._execute_checked(plan, yes=yes)

    def _execute_checked(
        self, plan: RunDeletionPlan, *, yes: bool
    ) -> RunDeletionResult:
        """Apply a previously inspected plan after revalidating its digest."""

        if not yes:
            raise DeletionConflict("deletion confirmation is required", plan)
        current = self.plan(
            plan.experiment,
            plan.variant,
            plan.run_id,
            cascade=plan.cascade,
            force_stale=plan.force_stale,
        )
        if current.plan_digest != plan.plan_digest:
            raise DeletionConflict("deletion plan is stale; rebuild the plan", current)
        if not current.ready:
            raise DeletionConflict(
                "deletion plan is blocked: " + "; ".join(current.conflicts), current
            )
        return self._execute(current)

    @workspace_operation
    def recover(self) -> tuple[dict[str, Any], ...]:
        """Recover unfinished trash transactions after a process crash."""

        if not os.path.lexists(self.trash_root):
            return ()
        _require_directory(self.trash_root, "deletion trash root")
        recovered: list[dict[str, Any]] = []
        for transaction_dir in sorted(
            self.trash_root.iterdir(), key=lambda item: item.name
        ):
            if transaction_dir.is_symlink() or not transaction_dir.is_dir():
                raise ContractError("deletion trash contains an invalid transaction")
            journal = transaction_dir / "journal.json"
            if not os.path.lexists(journal):
                continue
            document = _load_json(journal, "deletion journal")
            self._validate_journal(document, transaction_dir.name)
            phase = document.get("phase")
            if phase in {"completed", "rolled_back"}:
                continue
            if phase not in _JOURNAL_PHASES:
                raise ContractError("deletion journal phase is invalid")
            before_head = document.get("before_head")
            committed_head = document.get("binding_transaction")
            current_head = self.graph.bindings.head()
            if committed_head is None and current_head != before_head:
                if self._head_matches_journal(current_head, document):
                    committed_head = current_head
                    document["binding_transaction"] = current_head
            if committed_head is not None and current_head == committed_head:
                self._finish_moved(document, transaction_dir)
                GraphProjection(self.graph).rebuild()
                self._write_journal(document, journal, phase="completed")
                item = {
                    "transaction": transaction_dir.name,
                    "action": "completed",
                    "phase": "completed",
                }
                if (experiment := _journal_experiment(document)) is not None:
                    item["experiment"] = experiment
                if (variant := _journal_variant(document)) is not None:
                    item["experiment"], item["variant"] = variant
                recovered.append(item)
                continue
            if current_head != before_head:
                raise DeletionConflict(
                    f"deletion journal has an unexpected binding HEAD: {transaction_dir.name}"
                )
            self._restore_entries(document, transaction_dir)
            self._write_journal(document, journal, phase="rolled_back")
            item = {
                "transaction": transaction_dir.name,
                "action": "rolled_back",
                "phase": "rolled_back",
            }
            if (experiment := _journal_experiment(document)) is not None:
                item["experiment"] = experiment
            if (variant := _journal_variant(document)) is not None:
                item["experiment"], item["variant"] = variant
            recovered.append(item)
        return tuple(recovered)

    recover_pending = recover

    def _execute(self, plan: RunDeletionPlan) -> RunDeletionResult:
        if plan.before_head != self.graph.bindings.head():
            raise DeletionConflict(
                "binding HEAD changed while preparing deletion", plan
            )
        transaction = self._transaction_name(plan)
        transaction_dir = self.trash_root / transaction
        _ensure_directory(self.trash_root)
        try:
            transaction_dir.mkdir(mode=0o755)
        except FileExistsError as error:
            raise DeletionConflict(
                "deletion transaction already exists", plan
            ) from error
        if transaction_dir.is_symlink() or not transaction_dir.is_dir():
            raise ContractError("deletion transaction path is invalid")
        journal = transaction_dir / "journal.json"
        entries = self._journal_entries(plan, transaction_dir)
        document = {
            "schema_version": _JOURNAL_SCHEMA,
            "phase": "planned",
            "transaction": transaction,
            "plan_digest": plan.plan_digest,
            "before_head": plan.before_head,
            "binding_transaction": None,
            "operations": [item.as_dict() for item in plan.operations],
            "entries": entries,
        }
        self._write_journal(document, journal, phase="planned")
        commit_started = False
        try:
            with self._run_locks(plan.runs):
                if self.graph.bindings.head() != plan.before_head:
                    raise DeletionConflict(
                        "binding HEAD changed while locking Runs", plan
                    )
                self._move_entries(document, transaction_dir, journal)
                if self.graph.bindings.head() != plan.before_head:
                    raise DeletionConflict(
                        "binding HEAD changed after projection move", plan
                    )
                commit_started = True
                try:
                    binding_transaction = self.graph.bindings.commit(
                        list(plan.operations), expected_head=plan.before_head
                    )
                except BindingHeadConflict as error:
                    commit_started = False
                    raise DeletionConflict(
                        "binding HEAD changed before deletion commit", plan
                    ) from error
                document["binding_transaction"] = binding_transaction
                self._write_journal(document, journal, phase="binding_committed")
                self._verify_unbound(plan)
                GraphProjection(self.graph).rebuild()
                self._write_journal(document, journal, phase="completed")
        except BaseException:
            if document.get("binding_transaction") is None:
                try:
                    if not commit_started:
                        self._restore_entries(document, transaction_dir)
                        self._write_journal(document, journal, phase="rolled_back")
                    else:
                        current_head = self.graph.bindings.head()
                        if current_head == document.get("before_head"):
                            self._restore_entries(document, transaction_dir)
                            self._write_journal(document, journal, phase="rolled_back")
                        elif current_head is not None and self._head_matches_journal(
                            current_head, document
                        ):
                            document["binding_transaction"] = current_head
                except BaseException:
                    # Keep the journal in its current phase for later recovery.
                    pass
            raise

        moved = tuple(
            str(entry["source"]) for entry in entries if entry.get("moved") is True
        )
        missing = tuple(
            str(entry["source"]) for entry in entries if entry.get("missing") is True
        )
        return RunDeletionResult(
            transaction=transaction,
            plan_digest=plan.plan_digest,
            binding_transaction=str(document["binding_transaction"]),
            journal=journal,
            moved=moved,
            missing=missing,
            unbound=tuple(item.as_dict() for item in plan.operations),
        )

    def _contexts(
        self,
        bindings: dict[tuple[str, str], str],
    ) -> dict[str, _VariantContext]:
        experiments = {
            target: name
            for (scope, name), target in bindings.items()
            if scope == workspace_experiment_scope()
        }
        contexts: dict[str, _VariantContext] = {}
        for (scope, name), target in bindings.items():
            if not scope.startswith("experiment/") or not scope.endswith("/variants"):
                continue
            pieces = scope.split("/")
            if len(pieces) != 3 or pieces[0] != "experiment":
                raise ContractError("v2 Variant binding scope is invalid")
            experiment_hash = _digest_from_hex(pieces[1])
            experiment_name = experiments.get(experiment_hash)
            if experiment_name is None:
                raise ContractError("v2 Variant binding has no Experiment name")
            variant = self.graph.store.load(target)
            if (
                variant.kind != "variant"
                or variant.payload.get("experiment") != experiment_hash
            ):
                raise ContractError("v2 Variant binding ownership mismatch")
            existing = contexts.get(target)
            context = _VariantContext(experiment_name, name, target)
            if existing is not None and existing != context:
                raise ContractError("v2 Variant entity has multiple active names")
            contexts[target] = context
        return contexts

    def _runs(
        self,
        bindings: dict[tuple[str, str], str],
        contexts: dict[str, _VariantContext],
    ) -> tuple[DeletionRun, ...]:
        result: list[DeletionRun] = []
        for (scope, run_id), attempt_hash in bindings.items():
            variant_hash = _scoped_hash(scope, "variant", "runs")
            if variant_hash is None:
                continue
            context = contexts.get(variant_hash)
            if context is None:
                raise ContractError("v2 Run binding has no active Variant")
            attempt = self._typed(attempt_hash, "attempt")
            run_spec_hash = _required_digest(
                attempt.payload.get("run_spec"), "Attempt run_spec"
            )
            run_spec = self._typed(run_spec_hash, "run_spec")
            action = run_spec.payload.get("action")
            if action not in {"train", "eval", "export"}:
                raise ContractError("v2 RunSpec action is invalid")
            event_scope = attempt_event_scope(attempt_hash)
            event_hash = bindings.get((event_scope, CURRENT_REVISION_NAME))
            if event_hash is None:
                raise ContractError("v2 Attempt has no current lifecycle event")
            event_chain, result_hashes, blob_hashes = self._event_chain(
                event_hash, attempt_hash
            )
            event = self._typed(event_hash, "attempt_event")
            status = event.payload.get("status")
            if status not in RUN_STATUSES:
                raise ContractError("v2 Attempt event status is invalid")
            retry_parent = attempt.payload.get("retry_parent")
            if retry_parent is not None:
                retry_parent = _required_digest(retry_parent, "Attempt retry_parent")
            path = self._run_path(context, run_id)
            result.append(
                DeletionRun(
                    experiment=context.experiment,
                    variant=context.variant,
                    run_id=run_id,
                    address=f"{context.experiment}/{context.variant}/{run_id}",
                    variant_hash=variant_hash,
                    attempt_hash=attempt_hash,
                    run_spec_hash=run_spec_hash,
                    event_hash=event_hash,
                    action=action,
                    status=status,
                    retry_parent=retry_parent,
                    path=path,
                    event_chain=event_chain,
                    result_hashes=result_hashes,
                    blob_hashes=blob_hashes,
                )
            )
        return tuple(result)

    def _models(
        self,
        bindings: dict[tuple[str, str], str],
        contexts: dict[str, _VariantContext],
    ) -> tuple[DeletionModel, ...]:
        result: list[DeletionModel] = []
        for (scope, model_id), model_hash in bindings.items():
            variant_hash = _scoped_hash(scope, "variant", "models")
            if variant_hash is None:
                continue
            context = contexts.get(variant_hash)
            if context is None:
                raise ContractError("v2 Model binding has no active Variant")
            model = self._typed(model_hash, "model")
            producing_attempt = _required_digest(
                model.payload.get("producing_attempt"), "Model producing_attempt"
            )
            checkpoint_blob = model.payload.get("checkpoint_blob")
            if checkpoint_blob is not None:
                checkpoint_blob = _required_digest(
                    checkpoint_blob, "Model checkpoint_blob"
                )
                self._typed(checkpoint_blob, "blob")
            path = self._model_path(context, model_id)
            result.append(
                DeletionModel(
                    experiment=context.experiment,
                    variant=context.variant,
                    variant_hash=variant_hash,
                    model_id=model_id,
                    model_hash=model_hash,
                    producing_attempt=producing_attempt,
                    checkpoint_blob=checkpoint_blob,
                    path=path,
                )
            )
        return tuple(result)

    def _event_chain(
        self,
        head: str,
        attempt_hash: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        chain: list[str] = []
        results: set[str] = set()
        blobs: set[str] = set()
        current: str | None = head
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ContractError("v2 Attempt event chain contains a cycle")
            seen.add(current)
            event = self._typed(current, "attempt_event")
            if event.payload.get("attempt") != attempt_hash:
                raise ContractError("v2 Attempt event ownership mismatch")
            chain.append(current)
            result_hash = event.payload.get("result_object")
            if result_hash is not None:
                result_hash = _required_digest(result_hash, "Attempt result_object")
                self._typed(result_hash, None)
                results.add(result_hash)
            artifacts = event.payload.get("artifacts")
            if not isinstance(artifacts, list):
                raise ContractError("v2 Attempt event artifacts are invalid")
            for reference in artifacts:
                if not isinstance(reference, dict):
                    raise ContractError("v2 Attempt artifact reference is invalid")
                blob_hash = _required_digest(
                    reference.get("blob"), "Attempt artifact blob"
                )
                self._typed(blob_hash, "blob")
                blobs.add(blob_hash)
            parent = event.payload.get("parent")
            current = (
                None
                if parent is None
                else _required_digest(parent, "Attempt event parent")
            )
        return tuple(chain), tuple(sorted(results)), tuple(sorted(blobs))

    def _operations(
        self,
        bindings: dict[tuple[str, str], str],
        runs: tuple[DeletionRun, ...],
        models: tuple[DeletionModel, ...],
        selected_comparisons: set[tuple[str, str]],
        active_train_comparisons: set[tuple[str, str]],
    ) -> tuple[BindingOperation, ...]:
        selected_attempts = {item.attempt_hash for item in runs}
        selected_model_hashes = {item.model_hash for item in models}
        operations: list[BindingOperation] = []
        for (scope, name), target in bindings.items():
            if (
                _scoped_hash(scope, "variant", "runs") is not None
                and target in selected_attempts
            ):
                operations.append(BindingOperation("unbind", scope, name, target))
            elif (
                _scoped_hash(scope, "variant", "models") is not None
                and target in selected_model_hashes
            ):
                operations.append(BindingOperation("unbind", scope, name, target))
        for item in runs:
            operations.append(
                BindingOperation(
                    "unbind",
                    attempt_event_scope(item.attempt_hash),
                    CURRENT_REVISION_NAME,
                    item.event_hash,
                )
            )
        if selected_comparisons:
            for (scope, name), target in bindings.items():
                comparison_scope = _comparison_scope(scope)
                if (
                    comparison_scope is not None
                    and (
                        comparison_scope,
                        target,
                    )
                    in selected_comparisons
                    and (
                        comparison_scope,
                        target,
                    )
                    not in active_train_comparisons
                ):
                    operations.append(BindingOperation("unbind", scope, name, target))
        unique: dict[tuple[str, str], BindingOperation] = {}
        for operation in operations:
            unique[(operation.scope, operation.name)] = operation
        return tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.scope.encode("utf-8"),
                    item.name.encode("utf-8"),
                    item.target.encode("utf-8"),
                ),
            )
        )

    def _held_leases(self, runs: tuple[DeletionRun, ...]) -> tuple[str, ...]:
        held: list[str] = []
        for item in sorted(runs, key=lambda value: value.address.encode("utf-8")):
            if lease_held(self.repository, item.attempt_hash, item.path):
                held.append(item.address)
        return tuple(sorted(set(held), key=lambda value: value.encode("utf-8")))

    def _run_model_hash(self, item: DeletionRun) -> str | None:
        spec = self._typed(item.run_spec_hash, "run_spec")
        if item.action not in {"eval", "export"}:
            return None
        value = spec.payload.get("model")
        return _required_digest(value, "RunSpec model")

    def _run_comparison(self, item: DeletionRun) -> tuple[str, str] | None:
        if item.action != "train":
            return None
        spec = self._typed(item.run_spec_hash, "run_spec").payload
        value = spec.get("comparison_hash")
        revision = spec.get("variant_revision")
        if value is None:
            return None
        return (
            _required_digest(revision, "RunSpec variant_revision"),
            _required_digest(value, "RunSpec comparison_hash"),
        )

    def _typed(self, digest: str, kind: str | None) -> ObjectRecord:
        record = self.graph.store.load(digest)
        if kind is not None and record.kind != kind:
            raise ContractError(
                f"v2 object kind mismatch: expected {kind}, got {record.kind}"
            )
        return record

    def _run_path(self, context: _VariantContext, run_id: str) -> Path | None:
        try:
            return self.store.load(context.experiment, context.variant, run_id).path
        except NotFoundError:
            return None

    def _model_path(self, context: _VariantContext, model_id: str) -> Path | None:
        try:
            return self.store.load_model_manifest(
                context.experiment,
                context.variant,
                model_id,
            ).path
        except NotFoundError:
            return None

    def _assert_no_pending_deletion(self) -> None:
        if not os.path.lexists(self.trash_root):
            return
        _require_directory(self.trash_root, "deletion trash root")
        pending: list[str] = []
        for transaction_dir in sorted(
            self.trash_root.iterdir(), key=lambda item: item.name
        ):
            if transaction_dir.is_symlink() or not transaction_dir.is_dir():
                raise ContractError("deletion trash contains an invalid transaction")
            journal = transaction_dir / "journal.json"
            if not os.path.lexists(journal):
                continue
            document = _load_json(journal, "deletion journal")
            self._validate_journal(document, transaction_dir.name)
            if document.get("phase") not in {"completed", "rolled_back"}:
                pending.append(str(journal.relative_to(self.repository.root)))
        if pending:
            raise DeletionConflict(
                "deletion planning is blocked by pending deletion recovery: "
                + ", ".join(pending)
                + "; repeat the original non-dry-run deletion command to recover it"
            )

    def _journal_entries(
        self,
        plan: RunDeletionPlan,
        transaction_dir: Path,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        index = 0
        for item in plan.runs:
            if item.path is not None:
                source = _relative_to_root(item.path, self.repository.root)
                target = _relative_to_root(
                    transaction_dir
                    / "projection"
                    / f"{index:04d}-run-{item.attempt_hash.removeprefix('sha256:')[:16]}",
                    self.repository.root,
                )
                entries.append(
                    {
                        "kind": "run",
                        "address": item.address,
                        "source": source,
                        "target": target,
                        "moved": False,
                        "missing": False,
                    }
                )
                index += 1
        for item in plan.models:
            if item.path is not None:
                source = _relative_to_root(item.path, self.repository.root)
                target = _relative_to_root(
                    transaction_dir
                    / "projection"
                    / f"{index:04d}-model-{item.model_hash.removeprefix('sha256:')[:16]}",
                    self.repository.root,
                )
                entries.append(
                    {
                        "kind": "model",
                        "address": f"{item.experiment}/{item.variant}/{item.model_id}",
                        "source": source,
                        "target": target,
                        "moved": False,
                        "missing": False,
                    }
                )
                index += 1
        return entries

    def _move_entries(
        self, document: dict[str, Any], transaction_dir: Path, journal: Path
    ) -> None:
        self._write_journal(document, journal, phase="moving")
        for entry in document["entries"]:
            source = self.repository.root / str(entry["source"])
            target = self.repository.root / str(entry["target"])
            if entry.get("moved") is True or entry.get("missing") is True:
                continue
            if not os.path.lexists(source):
                entry["missing"] = True
                self._write_journal(document, journal, phase="moving")
                continue
            _require_directory(source, "deletion projection")
            if os.path.lexists(target):
                raise ContractError("deletion trash target already exists")
            _ensure_directory(target.parent)
            os.rename(source, target)
            _fsync_directory(source.parent)
            _fsync_directory(target.parent)
            entry["moved"] = True
            self._write_journal(document, journal, phase="moving")
        self._write_journal(document, journal, phase="moved")

    def _finish_moved(self, document: dict[str, Any], transaction_dir: Path) -> None:
        for entry in document["entries"]:
            if entry.get("missing") is True or entry.get("moved") is True:
                continue
            source = self.repository.root / str(entry["source"])
            target = self.repository.root / str(entry["target"])
            if os.path.lexists(target) and not os.path.lexists(source):
                entry["moved"] = True
            elif os.path.lexists(source) and not os.path.lexists(target):
                _require_directory(source, "deletion projection")
                _ensure_directory(target.parent)
                os.rename(source, target)
                _fsync_directory(source.parent)
                _fsync_directory(target.parent)
                entry["moved"] = True
            elif not os.path.lexists(source) and not os.path.lexists(target):
                entry["missing"] = True
            else:
                raise ContractError("deletion journal has both source and trash target")

    def _restore_entries(self, document: dict[str, Any], transaction_dir: Path) -> None:
        for entry in reversed(document["entries"]):
            if entry.get("missing") is True:
                continue
            source = self.repository.root / str(entry["source"])
            target = self.repository.root / str(entry["target"])
            if os.path.lexists(source) and not os.path.lexists(target):
                entry["moved"] = False
                continue
            if not os.path.lexists(target):
                continue
            if os.path.lexists(source):
                raise ContractError("deletion restore has both source and trash target")
            _require_directory(target, "deletion trash projection")
            _ensure_directory(source.parent)
            os.rename(target, source)
            _fsync_directory(target.parent)
            _fsync_directory(source.parent)
            entry["moved"] = False

    def _verify_unbound(self, plan: RunDeletionPlan) -> None:
        bindings = self.graph.bindings.bindings()
        for operation in plan.operations:
            if (operation.scope, operation.name) in bindings:
                raise ContractError("deletion binding remains active")

    def _head_matches_journal(
        self,
        head: str,
        document: dict[str, Any],
    ) -> bool:
        try:
            transaction = self.graph.store.load(head)
        except ContractError:
            return False
        if transaction.kind != "binding_transaction":
            return False
        payload = transaction.payload
        return payload.get("previous") == document.get("before_head") and payload.get(
            "operations"
        ) == document.get("operations")

    def _validate_journal(self, document: dict[str, Any], transaction: str) -> None:
        if document.get("schema_version") != _JOURNAL_SCHEMA:
            raise ContractError("deletion journal schema is unsupported")
        if document.get("transaction") != transaction:
            raise ContractError("deletion journal transaction is invalid")
        before_head = document.get("before_head")
        if before_head is not None:
            _required_digest(before_head, "deletion journal before_head")
        binding_transaction = document.get("binding_transaction")
        if binding_transaction is not None:
            _required_digest(
                binding_transaction, "deletion journal binding_transaction"
            )
        operations = document.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ContractError("deletion journal operations are invalid")
        if binding_transaction is not None and not self._head_matches_journal(
            binding_transaction, document
        ):
            raise ContractError(
                "deletion journal binding transaction does not match operations"
            )
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise ContractError("deletion journal entries are invalid")
        sources: set[str] = set()
        targets: set[str] = set()
        target_prefix = Path(".hkdl/trash/deletions") / transaction / "projection"
        for entry in entries:
            if not isinstance(entry, dict) or not {"source", "target"}.issubset(entry):
                raise ContractError("deletion journal entry is invalid")
            for key in ("source", "target"):
                value = entry[key]
                if (
                    not isinstance(value, str)
                    or not value
                    or Path(value).is_absolute()
                    or ".." in Path(value).parts
                ):
                    raise ContractError("deletion journal path is invalid")
            source = Path(entry["source"])
            target = Path(entry["target"])
            if len(source.parts) < 2 or source.parts[0] not in {
                "experiments",
                "outputs",
            }:
                raise ContractError("deletion journal source is outside owned roots")
            if (
                len(target.parts) <= len(target_prefix.parts)
                or target.parts[: len(target_prefix.parts)] != target_prefix.parts
            ):
                raise ContractError(
                    "deletion journal target is outside its transaction"
                )
            if str(source) in sources or str(target) in targets:
                raise ContractError("deletion journal contains duplicate paths")
            sources.add(str(source))
            targets.add(str(target))

    def _transaction_name(self, plan: RunDeletionPlan) -> str:
        nonce = str(self._nonce())
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", nonce):
            nonce = secrets.token_hex(8)
        base = f"delete-{plan.plan_digest.removeprefix('sha256:')[:20]}-{nonce}"
        candidate = base
        for _ in range(16):
            if not os.path.lexists(self.trash_root / candidate):
                return candidate
            candidate = f"{base}-{secrets.token_hex(4)}"
        raise DeletionConflict("unable to allocate a unique deletion transaction")

    def _write_journal(
        self,
        document: dict[str, Any],
        journal: Path,
        *,
        phase: str,
    ) -> None:
        if phase not in _JOURNAL_PHASES:
            raise ContractError("deletion journal phase is invalid")
        document["phase"] = phase
        payload = (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        if os.path.lexists(journal):
            atomic_replace(journal, payload)
        else:
            atomic_write_new(journal, payload)

    @contextmanager
    def _run_locks(self, runs: tuple[DeletionRun, ...]) -> Iterator[None]:
        with ExitStack() as stack:
            try:
                for item in sorted(runs, key=lambda item: item.attempt_hash):
                    stack.enter_context(
                        attempt_lease(self.repository, item.attempt_hash, item.path)
                    )
            except LockUnavailableError as error:
                raise DeletionConflict(
                    "Run lease became held during deletion"
                ) from error
            yield


class ExperimentDeletionService(RunDeletionService):
    """Plan and apply deletion of one complete active Experiment closure."""

    owner_label = "Experiment"

    @workspace_operation
    def plan(self, experiment: str) -> ExperimentDeletionPlan:
        if not self.graph.is_active():
            raise ContractError("HKDL v2 is not active")
        self._assert_no_pending_deletion()
        self._assert_no_pending_rename()

        bindings = self.graph.bindings.bindings()
        experiment_hash = self.graph.experiment_hash(experiment)
        experiment_object = self._typed(experiment_hash, "experiment")
        if experiment_object.payload.get("created_at") is None:
            raise ContractError("v2 Experiment object is invalid")

        contexts = self._contexts(bindings)
        selected_contexts = tuple(
            sorted(
                (
                    context
                    for context in contexts.values()
                    if context.experiment == experiment
                ),
                key=lambda item: item.variant.encode("utf-8"),
            )
        )
        variant_hashes = tuple(item.variant_hash for item in selected_contexts)
        variants = tuple(item.variant for item in selected_contexts)
        selected_variant_hashes = set(variant_hashes)

        runs = tuple(
            sorted(
                (
                    item
                    for item in self._runs(bindings, contexts)
                    if item.variant_hash in selected_variant_hashes
                ),
                key=lambda item: item.address.encode("utf-8"),
            )
        )
        models = tuple(
            sorted(
                (
                    item
                    for item in self._models(bindings, contexts)
                    if item.variant_hash in selected_variant_hashes
                ),
                key=lambda item: (
                    item.variant.encode("utf-8"),
                    item.model_id.encode("utf-8"),
                ),
            )
        )

        authored = self.repository.experiments / experiment
        _is_directory(authored)
        historical_names = self.graph.bindings.historical_names(
            workspace_experiment_scope(), experiment_hash
        )
        output_paths = tuple(
            _relative_to_root(self.repository.outputs / name, self.repository.root)
            for name in historical_names
        )
        for path in output_paths:
            _is_directory(self.repository.root / path)

        revision_hashes = self._owned_variant_revisions(selected_variant_hashes)
        scopes = {
            workspace_experiment_scope(),
            entity_revision_scope(experiment_hash),
            experiment_variant_scope(experiment_hash),
        }
        for variant_hash in variant_hashes:
            scopes.update(
                {
                    entity_revision_scope(variant_hash),
                    variant_run_scope(variant_hash),
                    variant_model_scope(variant_hash),
                }
            )
        for revision_hash in revision_hashes:
            scopes.update(
                {
                    evaluation_case_scope(revision_hash),
                    export_profile_scope(revision_hash),
                    comparison_group_scope(revision_hash),
                }
            )
        for item in runs:
            scopes.add(attempt_event_scope(item.attempt_hash))

        operations = tuple(
            BindingOperation("unbind", scope, name, target)
            for (scope, name), target in sorted(
                bindings.items(),
                key=lambda item: (
                    item[0][0].encode("utf-8"),
                    item[0][1].encode("utf-8"),
                    item[1],
                ),
            )
            if scope in scopes
            and not (
                scope == workspace_experiment_scope()
                and (name != experiment or target != experiment_hash)
            )
        )
        if not any(
            item.scope == workspace_experiment_scope()
            and item.name == experiment
            and item.target == experiment_hash
            for item in operations
        ):
            raise ContractError("v2 Experiment binding is missing from deletion plan")

        blockers = self._experiment_blockers(runs)
        provisional = ExperimentDeletionPlan(
            experiment=experiment,
            experiment_hash=experiment_hash,
            before_head=self.graph.bindings.head(),
            variants=variants,
            variant_hashes=variant_hashes,
            runs=runs,
            models=models,
            operations=operations,
            blockers=blockers,
            authored_path=_relative_to_root(authored, self.repository.root),
            output_paths=output_paths,
            plan_digest="",
        )
        digest = object_digest("binding_transaction", provisional.digest_payload())
        return ExperimentDeletionPlan(
            experiment=provisional.experiment,
            experiment_hash=provisional.experiment_hash,
            before_head=provisional.before_head,
            variants=provisional.variants,
            variant_hashes=provisional.variant_hashes,
            runs=provisional.runs,
            models=provisional.models,
            operations=provisional.operations,
            blockers=provisional.blockers,
            authored_path=provisional.authored_path,
            output_paths=provisional.output_paths,
            plan_digest=digest,
        )

    @workspace_operation
    def execute(self, plan: ExperimentDeletionPlan) -> ExperimentDeletionResult:
        entities = [("experiment", plan.experiment_hash)] + [
            ("variant", digest) for digest in plan.variant_hashes
        ]
        with entity_guard(self.repository, entities):
            current = self.plan(plan.experiment)
            if current.plan_digest != plan.plan_digest:
                raise DeletionConflict(
                    "Experiment deletion plan changed after confirmation", current
                )
            if not current.ready:
                raise DeletionConflict(
                    "Experiment deletion is blocked: "
                    + "; ".join(item.reason for item in current.blockers),
                    current,
                )
            try:
                return super()._execute(current)
            except DeletionConflict as error:
                try:
                    refreshed = self.plan(plan.experiment)
                except (ContractError, NotFoundError):
                    refreshed = current
                raise DeletionConflict(str(error), refreshed) from error
            except Exception as error:
                state, journal = self._failure_state(current)
                raise DeletionFailure(
                    f"Experiment deletion failed ({state}): {error}",
                    state=state,
                    journal=journal,
                ) from error

    def _owned_variant_revisions(self, variant_hashes: set[str]) -> tuple[str, ...]:
        revisions = {
            item.digest
            for item in self.graph.store.iter_records()
            if item.kind == "variant_revision"
            and item.payload.get("variant") in variant_hashes
        }
        for variant_hash in variant_hashes:
            revisions.add(self.graph.current_revision(variant_hash))
        return tuple(sorted(revisions))

    def _assert_no_pending_rename(self) -> None:
        journal_root = self.graph.store.root / "refs/renames"
        if not os.path.lexists(journal_root):
            return
        _require_directory(journal_root, "v2 rename journal root")
        if any(journal_root.iterdir()):
            raise DeletionConflict(
                f"{self.owner_label} deletion is blocked by pending authored rename "
                "recovery; repeat the original Experiment or Variant rename "
                "command before deleting"
            )

    def _experiment_blockers(
        self,
        runs: tuple[DeletionRun, ...],
        *,
        owner_label: str = "Experiment",
    ) -> tuple[ExperimentDeletionBlocker, ...]:
        blockers: list[ExperimentDeletionBlocker] = []
        held = set(self._held_leases(runs))
        for item in runs:
            lease = "held" if item.address in held else "not held"
            if item.status not in TERMINAL_STATUSES:
                blockers.append(
                    ExperimentDeletionBlocker(
                        "RUN_NONTERMINAL",
                        item.address,
                        item.status,
                        lease,
                        f"{owner_label} deletion requires every Run to be terminal.",
                        (
                            f"inspect with `hkdl status {item.experiment} "
                            f"{item.variant} {item.run_id} --full`; resolve the Run "
                            "or explicitly delete the stale Run first"
                        ),
                    )
                )
            if item.address in held:
                blockers.append(
                    ExperimentDeletionBlocker(
                        "RUN_LEASE_HELD",
                        item.address,
                        item.status,
                        "held",
                        "A process still holds this Run identity.",
                        "wait for the process to release the Run, then retry",
                    )
                )
        return tuple(
            sorted(
                blockers,
                key=lambda item: (
                    item.address.encode("utf-8"),
                    item.code.encode("utf-8"),
                ),
            )
        )

    def _journal_entries(
        self,
        plan: ExperimentDeletionPlan,
        transaction_dir: Path,
    ) -> list[dict[str, Any]]:
        result = []
        sources = [
            ("experiment", plan.experiment, plan.authored_path, "authored-experiment")
        ]
        sources.extend(
            (
                "outputs",
                Path(output_path).name,
                output_path,
                f"generated-outputs/{index:04d}",
            )
            for index, output_path in enumerate(plan.output_paths)
        )
        for kind, address, source_path, target_name in sources:
            target = _relative_to_root(
                transaction_dir / "projection" / target_name,
                self.repository.root,
            )
            result.append(
                {
                    "kind": kind,
                    "address": address,
                    "source": source_path,
                    "target": target,
                    "moved": False,
                    "missing": False,
                }
            )
        return result

    def _failure_state(self, plan: ExperimentDeletionPlan) -> tuple[str, Path | None]:
        journals = sorted(
            self.trash_root.glob(
                f"delete-{plan.plan_digest.removeprefix('sha256:')[:20]}-*/journal.json"
            )
        )
        journal = journals[-1] if journals else None
        try:
            active = (
                self.graph.bindings.resolve(
                    workspace_experiment_scope(), plan.experiment
                )
                == plan.experiment_hash
            )
        except NotFoundError:
            active = False
        if not active:
            return "deletion committed; recovery required", journal
        if journal is None:
            return "nothing changed", None
        try:
            phase = _load_json(journal, "deletion journal").get("phase")
        except ContractError:
            phase = None
        if phase == "rolled_back":
            return "rolled back successfully", journal
        return "nothing committed; recovery required", journal


class VariantDeletionService(ExperimentDeletionService):
    """Plan and apply deletion of one complete active Variant closure."""

    owner_label = "Variant"

    @workspace_operation
    def plan(self, experiment: str, variant: str) -> VariantDeletionPlan:
        if not self.graph.is_active():
            raise ContractError("HKDL v2 is not active")
        self._assert_no_pending_deletion()
        self._assert_no_pending_rename()

        bindings = self.graph.bindings.bindings()
        experiment_hash = self.graph.experiment_hash(experiment)
        variant_hash = self.graph.variant_hash(experiment_hash, variant)
        variant_object = self._typed(variant_hash, "variant")
        if variant_object.payload.get("experiment") != experiment_hash:
            raise ContractError("v2 Variant ownership mismatch")

        contexts = self._contexts(bindings)
        context = contexts.get(variant_hash)
        if (
            context is None
            or context.experiment != experiment
            or context.variant != variant
        ):
            raise ContractError("v2 Variant binding is missing from deletion plan")

        runs = tuple(
            sorted(
                (
                    item
                    for item in self._runs(bindings, contexts)
                    if item.variant_hash == variant_hash
                ),
                key=lambda item: item.address.encode("utf-8"),
            )
        )
        models = tuple(
            sorted(
                (
                    item
                    for item in self._models(bindings, contexts)
                    if item.variant_hash == variant_hash
                ),
                key=lambda item: item.model_id.encode("utf-8"),
            )
        )
        self._validate_variant_closure(variant_hash, runs, models, bindings)

        revision_hashes = self._owned_variant_revisions({variant_hash})
        scopes = {
            entity_revision_scope(variant_hash),
            variant_run_scope(variant_hash),
            variant_model_scope(variant_hash),
        }
        for revision_hash in revision_hashes:
            scopes.update(
                {
                    evaluation_case_scope(revision_hash),
                    export_profile_scope(revision_hash),
                    comparison_group_scope(revision_hash),
                }
            )
        for item in runs:
            scopes.add(attempt_event_scope(item.attempt_hash))

        variant_scope = experiment_variant_scope(experiment_hash)
        operations = tuple(
            BindingOperation("unbind", scope, name, target)
            for (scope, name), target in sorted(
                bindings.items(),
                key=lambda item: (
                    item[0][0].encode("utf-8"),
                    item[0][1].encode("utf-8"),
                    item[1],
                ),
            )
            if (
                scope in scopes
                or (
                    scope == variant_scope
                    and name == variant
                    and target == variant_hash
                )
            )
        )
        if not any(
            item.scope == variant_scope
            and item.name == variant
            and item.target == variant_hash
            for item in operations
        ):
            raise ContractError("v2 Variant binding is missing from deletion plan")

        authored = self.repository.experiments / experiment / variant
        _is_directory(authored)
        experiment_names = self.graph.bindings.historical_names(
            workspace_experiment_scope(), experiment_hash
        )
        variant_names = self.graph.bindings.historical_names(
            variant_scope, variant_hash
        )
        output_paths = tuple(
            _relative_to_root(
                self.repository.outputs / experiment_name / variant_name,
                self.repository.root,
            )
            for experiment_name in experiment_names
            for variant_name in variant_names
        )
        for path in output_paths:
            _is_directory(self.repository.root / path)

        blockers = self._experiment_blockers(runs, owner_label="Variant")
        derived_variants = self._derived_variants(variant_hash, contexts)
        provisional = VariantDeletionPlan(
            experiment=experiment,
            variant=variant,
            experiment_hash=experiment_hash,
            variant_hash=variant_hash,
            before_head=self.graph.bindings.head(),
            revision_hashes=revision_hashes,
            runs=runs,
            models=models,
            operations=operations,
            blockers=blockers,
            derived_variants=derived_variants,
            authored_path=_relative_to_root(authored, self.repository.root),
            output_paths=output_paths,
            plan_digest="",
        )
        digest = object_digest("binding_transaction", provisional.digest_payload())
        return VariantDeletionPlan(
            experiment=provisional.experiment,
            variant=provisional.variant,
            experiment_hash=provisional.experiment_hash,
            variant_hash=provisional.variant_hash,
            before_head=provisional.before_head,
            revision_hashes=provisional.revision_hashes,
            runs=provisional.runs,
            models=provisional.models,
            operations=provisional.operations,
            blockers=provisional.blockers,
            derived_variants=provisional.derived_variants,
            authored_path=provisional.authored_path,
            output_paths=provisional.output_paths,
            plan_digest=digest,
        )

    @workspace_operation
    def execute(
        self,
        plan: VariantDeletionPlan,
        *,
        allow_lineage_reconnection: bool = False,
    ) -> VariantDeletionResult:
        with entity_guard(
            self.repository,
            [
                ("experiment", plan.experiment_hash),
                ("variant", plan.variant_hash),
            ],
        ):
            current = self.plan(plan.experiment, plan.variant)
            if current.plan_digest != plan.plan_digest:
                raise DeletionConflict(
                    "Variant deletion plan changed after confirmation", current
                )
            if not current.ready:
                raise DeletionConflict(
                    "Variant deletion is blocked: "
                    + "; ".join(item.reason for item in current.blockers),
                    current,
                )
            if current.requires_lineage_confirmation and not allow_lineage_reconnection:
                raise DeletionConflict(
                    "Variant deletion is blocked by active child Variants; "
                    "explicit lineage reconnection confirmation is required",
                    current,
                )
            try:
                return RunDeletionService._execute(self, current)
            except DeletionConflict as error:
                try:
                    refreshed = self.plan(plan.experiment, plan.variant)
                except (ContractError, NotFoundError):
                    refreshed = current
                raise DeletionConflict(str(error), refreshed) from error
            except Exception as error:
                state, journal = self._failure_state(current)
                raise DeletionFailure(
                    f"Variant deletion failed ({state}): {error}",
                    state=state,
                    journal=journal,
                ) from error

    def _validate_variant_closure(
        self,
        variant_hash: str,
        runs: tuple[DeletionRun, ...],
        models: tuple[DeletionModel, ...],
        bindings: dict[tuple[str, str], str],
    ) -> None:
        attempts = {item.attempt_hash for item in runs}
        model_hashes = {item.model_hash for item in models}
        for item in runs:
            spec = self._typed(item.run_spec_hash, "run_spec")
            revision_field = {
                "train": "variant_revision",
                "eval": "evaluator_revision",
                "export": "exporter_revision",
            }[item.action]
            revision_hash = _required_digest(
                spec.payload.get(revision_field), "RunSpec Variant revision"
            )
            revision = self._typed(revision_hash, "variant_revision")
            if revision.payload.get("variant") != variant_hash:
                raise ContractError("v2 Run Code ownership mismatch")
            if item.retry_parent is not None and item.retry_parent not in attempts:
                raise ContractError("v2 Run retry ownership mismatch")
            if item.action in {"eval", "export"}:
                model_hash = _required_digest(
                    spec.payload.get("model"), "RunSpec model"
                )
                if model_hash not in model_hashes:
                    raise ContractError("v2 Run Model ownership mismatch")
        for item in models:
            if item.producing_attempt not in attempts:
                raise ContractError("v2 Model producer ownership mismatch")
            model = self._typed(item.model_hash, "model")
            revision_hash = _required_digest(
                model.payload.get("variant_revision"), "Model Variant revision"
            )
            revision = self._typed(revision_hash, "variant_revision")
            if revision.payload.get("variant") != variant_hash:
                raise ContractError("v2 Model revision ownership mismatch")
        for (scope, _), target in bindings.items():
            run_owner = _scoped_hash(scope, "variant", "runs")
            if (
                target in attempts
                and run_owner is not None
                and run_owner != variant_hash
            ):
                raise ContractError("v2 Attempt has multiple active Variant owners")
            model_owner = _scoped_hash(scope, "variant", "models")
            if (
                target in model_hashes
                and model_owner is not None
                and model_owner != variant_hash
            ):
                raise ContractError("v2 Model has multiple active Variant owners")

    def _derived_variants(
        self,
        variant_hash: str,
        contexts: dict[str, _VariantContext],
    ) -> tuple[DerivedVariant, ...]:
        active = set(contexts)
        current_revisions = {
            digest: self.graph.current_revision(digest) for digest in active
        }
        active_parents = {
            digest: self._effective_active_parent(
                current_revisions[digest], active, excluded=frozenset()
            )
            for digest in active
        }
        self._validate_active_lineage(active_parents)
        result: list[DerivedVariant] = []
        for digest, context in contexts.items():
            if digest == variant_hash:
                continue
            depth = 0
            cursor = digest
            seen: set[str] = set()
            while (parent := active_parents.get(cursor)) is not None:
                if cursor in seen:
                    raise ContractError("v2 active Variant lineage contains a cycle")
                seen.add(cursor)
                depth += 1
                if parent == variant_hash:
                    current_parent = contexts[active_parents[digest]]
                    after_hash = (
                        self._effective_active_parent(
                            current_revisions[digest],
                            active,
                            excluded=frozenset({variant_hash}),
                        )
                        if depth == 1
                        else active_parents[digest]
                    )
                    result.append(
                        DerivedVariant(
                            experiment=context.experiment,
                            variant=context.variant,
                            address=self._context_address(context),
                            relation="direct" if depth == 1 else "transitive",
                            depth=depth,
                            current_parent=self._context_address(current_parent),
                            effective_parent_after=(
                                self._context_address(contexts[after_hash])
                                if after_hash is not None
                                else None
                            ),
                        )
                    )
                    break
                cursor = parent
        return tuple(
            sorted(
                result,
                key=lambda item: (item.depth, item.address.encode("utf-8")),
            )
        )

    @staticmethod
    def _validate_active_lineage(active_parents: dict[str, str | None]) -> None:
        for start in active_parents:
            current: str | None = start
            seen: set[str] = set()
            while current is not None:
                if current in seen:
                    raise ContractError("v2 active Variant lineage contains a cycle")
                seen.add(current)
                current = active_parents.get(current)

    def _effective_active_parent(
        self,
        revision_hash: str,
        active: set[str],
        *,
        excluded: frozenset[str],
    ) -> str | None:
        current = revision_hash
        seen: set[str] = set()
        while True:
            lineage = self._derivation_parent(current)
            if lineage is None:
                return None
            parent_variant, parent_revision = lineage
            if parent_variant in seen:
                raise ContractError("v2 Variant derivation contains a cycle")
            seen.add(parent_variant)
            if parent_variant in active and parent_variant not in excluded:
                return parent_variant
            current = parent_revision

    def _derivation_parent(self, revision_hash: str) -> tuple[str, str] | None:
        current: str | None = revision_hash
        owner: str | None = None
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ContractError("v2 Variant revision history contains a cycle")
            seen.add(current)
            revision = self._typed(current, "variant_revision")
            variant_hash = _required_digest(
                revision.payload.get("variant"), "Variant revision owner"
            )
            if owner is None:
                owner = variant_hash
            elif variant_hash != owner:
                raise ContractError("v2 Variant revision parent ownership mismatch")
            derivation = revision.payload.get("derivation_parent")
            if derivation is not None:
                derivation = _required_digest(derivation, "Variant derivation parent")
                parent = self._typed(derivation, "variant_revision")
                parent_variant = _required_digest(
                    parent.payload.get("variant"), "Variant derivation owner"
                )
                return parent_variant, derivation
            parent = revision.payload.get("parent")
            current = (
                None
                if parent is None
                else _required_digest(parent, "Variant revision parent")
            )
        return None

    @staticmethod
    def _context_address(context: _VariantContext) -> str:
        return f"{context.experiment}/{context.variant}"

    def _journal_entries(
        self,
        plan: VariantDeletionPlan,
        transaction_dir: Path,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        sources = [
            (
                "variant",
                f"{plan.experiment}/{plan.variant}",
                plan.authored_path,
                "authored-variant",
            )
        ]
        sources.extend(
            (
                "outputs",
                output_path,
                output_path,
                f"generated-outputs/{index:04d}",
            )
            for index, output_path in enumerate(plan.output_paths)
        )
        for kind, address, source_path, target_name in sources:
            target = _relative_to_root(
                transaction_dir / "projection" / target_name,
                self.repository.root,
            )
            result.append(
                {
                    "kind": kind,
                    "address": address,
                    "source": source_path,
                    "target": target,
                    "moved": False,
                    "missing": False,
                }
            )
        return result

    def _failure_state(self, plan: VariantDeletionPlan) -> tuple[str, Path | None]:
        journals = sorted(
            self.trash_root.glob(
                f"delete-{plan.plan_digest.removeprefix('sha256:')[:20]}-*/journal.json"
            )
        )
        journal = journals[-1] if journals else None
        try:
            active = (
                self.graph.bindings.resolve(
                    experiment_variant_scope(plan.experiment_hash), plan.variant
                )
                == plan.variant_hash
            )
        except NotFoundError:
            active = False
        if not active:
            return "deletion committed; recovery required", journal
        if journal is None:
            return "nothing changed", None
        try:
            phase = _load_json(journal, "deletion journal").get("phase")
        except ContractError:
            phase = None
        if phase == "rolled_back":
            return "rolled back successfully", journal
        return "nothing committed; recovery required", journal


DeletionService = RunDeletionService
RunDeletionPlanner = RunDeletionService


def _journal_experiment(document: dict[str, Any]) -> str | None:
    for entry in document.get("entries", ()):
        if entry.get("kind") == "experiment" and isinstance(entry.get("address"), str):
            return str(entry["address"])
    return None


def _journal_variant(document: dict[str, Any]) -> tuple[str, str] | None:
    for entry in document.get("entries", ()):
        if entry.get("kind") != "variant" or not isinstance(entry.get("address"), str):
            continue
        pieces = str(entry["address"]).split("/")
        if len(pieces) != 2 or not all(pieces):
            raise ContractError("deletion journal Variant address is invalid")
        return pieces[0], pieces[1]
    return None


def _required_digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
        raise ContractError(f"{location} is invalid")
    return value


def _digest_from_hex(value: str) -> str:
    digest = f"sha256:{value}"
    if not DIGEST_PATTERN.fullmatch(digest):
        raise ContractError("v2 binding scope digest is invalid")
    return digest


def _scoped_hash(scope: str, prefix: str, suffix: str) -> str | None:
    match = _SCOPE_HASH.fullmatch(scope)
    if match is None:
        return None
    if match.group("prefix") != prefix or match.group("suffix") != suffix:
        return None
    return _digest_from_hex(match.group("digest"))


def _comparison_scope(scope: str) -> str | None:
    if not scope.startswith("variant-revision/") or not scope.endswith(
        "/comparison-groups"
    ):
        return None
    pieces = scope.split("/")
    if len(pieces) != 3:
        raise ContractError("v2 comparison binding scope is invalid")
    return _digest_from_hex(pieces[1])


def _is_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise ContractError(f"deletion projection is a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"deletion projection is not a directory: {path}")
    return True


def _require_directory(path: Path, location: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"{location} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"{location} must be a real directory: {path}")


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o755, parents=True, exist_ok=True)
    _require_directory(path, "deletion directory")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as error:
        raise ContractError(f"deletion path escapes repository: {path}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path, location: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"{location} must be a regular non-symlink file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{location} is invalid") from error
    if not isinstance(value, dict):
        raise ContractError(f"{location} must be a mapping")
    return value


__all__ = [
    "DeletionConflict",
    "DeletionResult",
    "DeletionService",
    "DeletionModel",
    "DeletionRun",
    "RunDeletionConflict",
    "RunDeletionPlan",
    "RunDeletionPlanner",
    "RunDeletionResult",
    "RunDeletionService",
]
