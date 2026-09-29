"""Recoverable one-sided promotion of committed Variant Code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hkdl.authoring.authoring import Authoring
from hkdl.authoring.config import DIGEST_PATTERN, NAME_PATTERN
from hkdl.errors import ContractError
from hkdl.storage.storage import (
    NotFoundError,
    atomic_replace,
    atomic_write_new,
    compute_source_digest,
)

from .bindings import BindingHeadConflict, BindingOperation
from .deletion import VariantDeletionPlan, VariantDeletionService
from .graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    entity_revision_scope,
    experiment_variant_scope,
)
from .leases import entity_guard
from .maintenance import (
    PROMOTION_JOURNAL,
    workspace_access,
    workspace_operation,
)
from .objects import object_digest
from .projection import GraphProjection

_TRANSACTION = re.compile(r"promote-[A-Za-z0-9_.-]+")
_PHASES = {
    "planned",
    "candidate_ready",
    "target_moved",
    "target_published",
    "binding_committed",
    "completed",
    "rolled_back",
}


class PromotionConflict(RuntimeError):
    def __init__(self, message: str, plan: VariantPromotionPlan | None = None):
        super().__init__(message)
        self.plan = plan


class PromotionFailure(RuntimeError):
    def __init__(self, message: str, *, state: str, journal: Path | None = None):
        super().__init__(message)
        self.state = state
        self.journal = journal


@dataclass(frozen=True)
class PromotionBlocker:
    code: str
    reason: str
    next_action: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "reason": self.reason,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class VariantPromotionPlan:
    experiment: str
    source: str
    target: str
    experiment_hash: str
    source_variant_hash: str
    target_variant_hash: str
    source_revision_hash: str
    target_revision_hash: str
    base_revision_hash: str | None
    before_head: str | None
    source_tree_hash: str
    target_tree_hash: str
    source_draft_fingerprint: str
    target_draft_fingerprint: str
    changed_paths: tuple[str, ...]
    template_changed: bool
    components_changed: bool
    blockers: tuple[PromotionBlocker, ...]
    source_deletion: VariantDeletionPlan | None
    already_integrated: bool
    plan_digest: str

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def changed(self) -> bool:
        return self.ready

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "source": self.source,
            "target": self.target,
            "experiment_hash": self.experiment_hash,
            "source_variant_hash": self.source_variant_hash,
            "target_variant_hash": self.target_variant_hash,
            "source_revision_hash": self.source_revision_hash,
            "target_revision_hash": self.target_revision_hash,
            "base_revision_hash": self.base_revision_hash,
            "before_head": self.before_head,
            "source_tree_hash": self.source_tree_hash,
            "target_tree_hash": self.target_tree_hash,
            "source_draft_fingerprint": self.source_draft_fingerprint,
            "target_draft_fingerprint": self.target_draft_fingerprint,
            "changed_paths": list(self.changed_paths),
            "template_changed": self.template_changed,
            "components_changed": self.components_changed,
            "blockers": [item.as_dict() for item in self.blockers],
            "source_deletion": (
                None if self.source_deletion is None else self.source_deletion.as_dict()
            ),
            "already_integrated": self.already_integrated,
            "ready": self.ready,
            "changed": self.changed,
            "target_options": "preserved",
            "source_variant": "deleted",
            "plan_digest": self.plan_digest,
        }

    def digest_payload(self) -> dict[str, Any]:
        document = self.as_dict()
        document.pop("plan_digest")
        return document


@dataclass(frozen=True)
class VariantPromotionResult:
    experiment: str
    source: str
    target: str
    changed: bool
    transaction: str | None
    binding_transaction: str | None
    journal: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "source": self.source,
            "target": self.target,
            "changed": self.changed,
            "transaction": self.transaction,
            "binding_transaction": self.binding_transaction,
            "journal": None if self.journal is None else str(self.journal),
            "target_options": "preserved",
            "source_variant": "deleted",
        }


class VariantPromotionService:
    def __init__(self, repository):
        self.repository = repository
        self.authoring = Authoring(repository)
        self.graph = V2Graph(repository)
        self.stable_journal = repository.root / PROMOTION_JOURNAL
        self.trash_root = repository.root / ".hkdl/trash/promotions"

    @workspace_operation
    def plan(self, experiment: str, source: str, target: str) -> VariantPromotionPlan:
        if not self.graph.is_active():
            raise ContractError("HKDL v2 is not active")
        if source == target:
            raise ContractError(
                "Variant promotion requires different source and target"
            )
        if os.path.lexists(self.stable_journal):
            raise PromotionConflict(
                "Variant promotion is blocked by pending recovery; repeat the "
                "original non-dry-run promote command"
            )

        experiment_record = self.authoring.load_experiment(experiment)
        source_record = self.authoring.load_variant(experiment, source)
        target_record = self.authoring.load_variant(experiment, target)
        if (
            experiment_record.authored_schema_version != 2
            or source_record.authored_schema_version != 2
            or target_record.authored_schema_version != 2
        ):
            raise ContractError(
                "Variant promotion requires schema-2 authoring; run "
                "`hkdl migrate --authoring --yes` first"
            )

        experiment_hash = self.graph.experiment_hash(experiment)
        self.graph.assert_experiment_clean(experiment, experiment_record)
        source_hash = self.graph.variant_hash(experiment_hash, source)
        target_hash = self.graph.variant_hash(experiment_hash, target)
        source_revision = self.graph.current_revision(source_hash)
        target_revision = self.graph.current_revision(target_hash)
        source_payload = self._revision(source_revision, owner=source_hash)
        target_payload = self._revision(target_revision, owner=target_hash)
        self._available_tree(source_payload, "source")
        self._available_tree(target_payload, "target")

        blockers: list[PromotionBlocker] = []
        if self.graph.preview_variant_revision(
            source_hash, source_record, parent=source_revision
        ).changed:
            blockers.append(
                PromotionBlocker(
                    "SOURCE_DIRTY",
                    f"Source Variant draft differs from committed Code: {experiment}/{source}",
                    f"commit or restore it with `hkdl variant commit {experiment} {source}`",
                )
            )
        if self.graph.preview_variant_revision(
            target_hash, target_record, parent=target_revision
        ).changed:
            blockers.append(
                PromotionBlocker(
                    "TARGET_DIRTY",
                    f"Target Variant draft differs from committed Code: {experiment}/{target}",
                    f"commit or restore it with `hkdl variant commit {experiment} {target}`",
                )
            )

        base, already_integrated = self._integration_base(
            source_hash,
            source_revision,
            target_hash,
            target_revision,
        )
        if base is None:
            blockers.append(
                PromotionBlocker(
                    "NO_COMMON_BASE",
                    "Source is not derived from or previously promoted into Target",
                    "use a Variant cloned from the Target; unrelated promotion is not supported",
                )
            )
        base_payload = None if base is None else self._revision(base)
        if (
            base_payload is not None
            and not already_integrated
            and self._code_identity(target_payload) != self._code_identity(base_payload)
        ):
            blockers.append(
                PromotionBlocker(
                    "TARGET_DIVERGED",
                    "Target Code changed after the last shared integration point",
                    "reconcile the Target manually; three-way merge is not supported",
                )
            )

        source_identity = self._code_identity(source_payload)
        target_identity = self._code_identity(target_payload)
        no_code_change = source_identity == target_identity
        already_integrated = already_integrated or (base is not None and no_code_change)
        if not already_integrated and self.graph.has_active_execution(target_hash):
            blockers.append(
                PromotionBlocker(
                    "TARGET_EXECUTION_ACTIVE",
                    "Target has active Run or Model bindings and its Code is locked",
                    f"delete the active execution lineage under {experiment}/{target} first",
                )
            )

        source_deletion = None
        if not blockers:
            source_deletion = VariantDeletionService(self.repository).plan(
                experiment, source
            )
            blockers.extend(
                PromotionBlocker(
                    f"SOURCE_{item.code}",
                    f"{item.address}: {item.reason}",
                    item.next_action,
                )
                for item in source_deletion.blockers
            )
            if source_deletion.derived_variants:
                blockers.append(
                    PromotionBlocker(
                        "SOURCE_HAS_ACTIVE_DESCENDANTS",
                        "Source Variant has active derived Variants",
                        "delete or reorganize Source descendants before promotion",
                    )
                )

        comparison = target_payload if base_payload is None else base_payload
        provisional = VariantPromotionPlan(
            experiment=experiment,
            source=source,
            target=target,
            experiment_hash=experiment_hash,
            source_variant_hash=source_hash,
            target_variant_hash=target_hash,
            source_revision_hash=source_revision,
            target_revision_hash=target_revision,
            base_revision_hash=base,
            before_head=self.graph.bindings.head(),
            source_tree_hash=str(source_payload["source_tree"]),
            target_tree_hash=str(target_payload["source_tree"]),
            source_draft_fingerprint=self._code_fingerprint(source_record.path),
            target_draft_fingerprint=compute_source_digest(target_record.path),
            changed_paths=self._changed_paths(comparison, source_payload),
            template_changed=comparison.get("template")
            != source_payload.get("template"),
            components_changed=(
                comparison.get("components") != source_payload.get("components")
            ),
            blockers=tuple(blockers),
            source_deletion=source_deletion,
            already_integrated=already_integrated,
            plan_digest="",
        )
        digest = object_digest("binding_transaction", provisional.digest_payload())
        return VariantPromotionPlan(**{**provisional.__dict__, "plan_digest": digest})

    def execute(self, plan: VariantPromotionPlan) -> VariantPromotionResult:
        with workspace_access(self.repository, exclusive=True, promotion=True):
            entities = [
                ("experiment", plan.experiment_hash),
                ("variant", plan.source_variant_hash),
                ("variant", plan.target_variant_hash),
            ]
            with entity_guard(self.repository, entities):
                current = self.plan(plan.experiment, plan.source, plan.target)
                if current.plan_digest != plan.plan_digest:
                    raise PromotionConflict(
                        "Variant promotion plan changed after confirmation", current
                    )
                if not current.ready:
                    raise PromotionConflict(
                        "Variant promotion is blocked: "
                        + "; ".join(item.reason for item in current.blockers),
                        current,
                    )
                return self._execute(current)

    def recover(
        self, experiment: str, source: str, target: str
    ) -> tuple[dict[str, Any], ...]:
        with workspace_access(self.repository, exclusive=True, promotion=True):
            if not os.path.lexists(self.stable_journal):
                return ()
            document = self._load_journal()
            pending_address = (
                str(document["experiment"]),
                str(document["source"]),
                str(document["target"]),
            )
            requested_address = (experiment, source, target)
            if requested_address != pending_address:
                pending = f"{pending_address[0]}/{pending_address[1]} -> {pending_address[0]}/{pending_address[2]}"
                raise PromotionConflict(
                    "pending Variant promotion belongs to "
                    f"{pending}; repeat that original command"
                )
            VariantDeletionService(self.repository).recover()
            current_head = self.graph.bindings.head()
            committed = document.get("binding_transaction")
            if committed is None:
                committed = self._matching_transaction(current_head, document)
                if committed is not None:
                    document["binding_transaction"] = committed
            if committed is not None and self._head_descends_from(
                current_head, str(committed)
            ):
                self._finish_committed(document)
                if not self._source_deleted(document):
                    self._delete_source_identity(
                        str(document["experiment"]),
                        str(document["source"]),
                        str(document["source_variant_hash"]),
                    )
                self._complete(document)
                action = "completed"
            elif current_head == document["before_head"]:
                if self.graph.bindings.head() != document["before_head"]:
                    raise PromotionConflict(
                        "binding HEAD changed during Variant promotion recovery"
                    )
                self._restore_before_head(document)
                action = "rolled_back"
            else:
                raise PromotionConflict(
                    "Variant promotion journal has an unexpected binding HEAD"
                )
            return (
                {
                    "action": action,
                    "transaction": document["transaction"],
                    "experiment": document["experiment"],
                    "source": document["source"],
                    "target": document["target"],
                },
            )

    # Flow: stage Source code with Target options, replace the Target draft,
    # commit its revision, rebuild the projection, then delete Source and finish.
    # Failures restore when safe or retain journal state for explicit recovery.
    def _execute(self, plan: VariantPromotionPlan) -> VariantPromotionResult:
        """Replace Target Code, commit its revision, then delete Source.

        The journal supports restoration before commit and completion after a
        matching commit; ambiguous failures leave it for :meth:`recover`.
        """

        transaction = f"promote-{secrets.token_hex(16)}"
        self._ensure_trash_root()
        transaction_dir = self.trash_root / transaction
        transaction_dir.mkdir(mode=0o755)
        candidate = transaction_dir / "candidate"
        before = transaction_dir / "before"
        displaced = transaction_dir / "displaced"
        source_path = self.repository.experiments / plan.experiment / plan.source
        target_path = self.repository.experiments / plan.experiment / plan.target
        source_payload = self._revision(
            plan.source_revision_hash, owner=plan.source_variant_hash
        )
        if plan.already_integrated:
            new_revision_hash = plan.target_revision_hash
        else:
            new_payload = {
                "variant": plan.target_variant_hash,
                "parent": plan.target_revision_hash,
                "derivation_parent": None,
                "merge_parent": plan.source_revision_hash,
                "template": source_payload["template"],
                "source_tree": source_payload["source_tree"],
                "components": source_payload["components"],
            }
            new_revision_hash = self.graph.store.put(
                "variant_revision", new_payload
            ).digest
        operations = [
            BindingOperation(
                "unbind",
                entity_revision_scope(plan.target_variant_hash),
                CURRENT_REVISION_NAME,
                plan.target_revision_hash,
            ),
            BindingOperation(
                "bind",
                entity_revision_scope(plan.target_variant_hash),
                CURRENT_REVISION_NAME,
                new_revision_hash,
            ),
        ]
        document: dict[str, Any] = {
            "schema_version": 1,
            "phase": "planned",
            "transaction": transaction,
            "experiment": plan.experiment,
            "source": plan.source,
            "target": plan.target,
            "source_variant_hash": plan.source_variant_hash,
            "target_variant_hash": plan.target_variant_hash,
            "source_revision_hash": plan.source_revision_hash,
            "target_revision_hash": plan.target_revision_hash,
            "new_revision_hash": new_revision_hash,
            "before_head": plan.before_head,
            "binding_transaction": None,
            "plan_digest": plan.plan_digest,
            "source_path": self._relative(source_path),
            "target_path": self._relative(target_path),
            "candidate_path": self._relative(candidate),
            "before_path": self._relative(before),
            "displaced_path": self._relative(displaced),
            "source_fingerprint": plan.source_draft_fingerprint,
            "before_fingerprint": plan.target_draft_fingerprint,
            "candidate_fingerprint": None,
            "operations": [item.as_dict() for item in operations],
        }
        self._write_journal(document)
        try:
            if self._code_fingerprint(source_path) != plan.source_draft_fingerprint:
                raise PromotionConflict(
                    "Source draft changed while preparing promotion"
                )
            if compute_source_digest(target_path) != plan.target_draft_fingerprint:
                raise PromotionConflict(
                    "Target draft changed while preparing promotion"
                )
            candidate.mkdir(mode=0o755)
            shutil.copy2(source_path / "code.json", candidate / "code.json")
            shutil.copy2(target_path / "options.json", candidate / "options.json")
            shutil.copytree(source_path / "src", candidate / "src")
            preview = self.graph.capture_source_tree(candidate / "src", publish=False)
            if preview.digest != plan.source_tree_hash:
                raise PromotionConflict("Source Code changed while preparing promotion")
            document["candidate_fingerprint"] = compute_source_digest(candidate)
            self._write_journal(document, phase="candidate_ready")
            if self._code_fingerprint(source_path) != plan.source_draft_fingerprint:
                raise PromotionConflict(
                    "Source draft changed while preparing promotion"
                )
            if compute_source_digest(target_path) != plan.target_draft_fingerprint:
                raise PromotionConflict(
                    "Target draft changed while preparing promotion"
                )
            os.rename(target_path, before)
            self._fsync(target_path.parent)
            self._write_journal(document, phase="target_moved")
            os.rename(candidate, target_path)
            self._fsync(target_path.parent)
            self._write_journal(document, phase="target_published")
            if not plan.already_integrated:
                promoted = self.authoring.load_variant(plan.experiment, plan.target)
                identity = self.graph.preview_variant_revision(
                    plan.target_variant_hash,
                    promoted,
                    parent=plan.target_revision_hash,
                    merge_parent=plan.source_revision_hash,
                )
                if identity.variant_revision_hash != new_revision_hash:
                    raise ContractError(
                        "promoted draft disagrees with planned revision"
                    )
            try:
                binding_transaction = self.graph.bindings.commit(
                    operations, expected_head=plan.before_head
                )
            except BindingHeadConflict as error:
                raise PromotionConflict(
                    "binding HEAD changed before Variant promotion commit"
                ) from error
            document["binding_transaction"] = binding_transaction
            self._write_journal(document, phase="binding_committed")
            GraphProjection(self.graph).rebuild()
            self._delete_source(plan)
            self._complete(document)
            return VariantPromotionResult(
                plan.experiment,
                plan.source,
                plan.target,
                True,
                transaction,
                binding_transaction,
                transaction_dir / "journal.json",
            )
        except BaseException as error:
            current_head = self.graph.bindings.head()
            rolled_back = False
            if current_head == plan.before_head:
                try:
                    self._restore_before_head(document)
                    rolled_back = True
                except BaseException:
                    pass
            elif self._head_matches(current_head, document):
                document["binding_transaction"] = current_head
                self._write_journal(document)
            if isinstance(error, PromotionConflict):
                raise
            if rolled_back:
                state = "rolled back successfully"
                recovery_journal = transaction_dir / "journal.json"
            elif (
                document.get("binding_transaction") is not None
                and self._head_descends_from(
                    self.graph.bindings.head(),
                    str(document["binding_transaction"]),
                )
            ) or self._matching_transaction(self.graph.bindings.head(), document):
                state = "promotion committed; recovery required"
                recovery_journal = self.stable_journal
            else:
                state = "nothing committed; recovery required"
                recovery_journal = self.stable_journal
            raise PromotionFailure(
                f"Variant promotion failed ({state}): {error}",
                state=state,
                journal=recovery_journal,
            ) from error

    def _delete_source(self, plan: VariantPromotionPlan):
        if plan.source_deletion is None:
            raise ContractError("Variant promotion Source deletion plan is unavailable")
        return self._delete_source_identity(
            plan.experiment, plan.source, plan.source_variant_hash
        )

    def _delete_source_identity(
        self, experiment: str, source: str, source_variant_hash: str
    ):
        deletion = VariantDeletionService(self.repository)
        current = deletion.plan(experiment, source)
        if current.variant_hash != source_variant_hash:
            raise PromotionConflict(
                "Source Variant identity changed before promotion deletion"
            )
        if not current.ready:
            raise PromotionConflict(
                "Source Variant deletion is blocked: "
                + "; ".join(item.reason for item in current.blockers)
            )
        if current.derived_variants:
            raise PromotionConflict(
                "Source Variant gained active descendants before promotion deletion"
            )
        return deletion.execute(current)

    def _integration_base(
        self,
        source_hash: str,
        source_revision: str,
        target_hash: str,
        target_revision: str,
    ) -> tuple[str | None, bool]:
        source_parent_chain = self._parent_chain(source_revision, source_hash)
        target_parent_chain = self._parent_chain(target_revision, target_hash)
        source_parent_set = set(source_parent_chain)
        for revision_hash in target_parent_chain:
            payload = self._revision(revision_hash, owner=target_hash)
            merge_parent = payload.get("merge_parent")
            if merge_parent is None:
                continue
            merge_payload = self._revision(str(merge_parent))
            if merge_payload.get("variant") != source_hash:
                continue
            if str(merge_parent) == source_revision:
                return str(merge_parent), revision_hash == target_revision
            if str(merge_parent) in source_parent_set:
                return str(merge_parent), False

        source_ancestry = self._derivation_ancestry(source_revision)
        for revision_hash in target_parent_chain:
            if revision_hash in source_ancestry:
                return revision_hash, False
        return None, False

    def _parent_chain(self, start: str, owner: str) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ContractError("v2 Variant revision history contains a cycle")
            seen.add(current)
            payload = self._revision(current, owner=owner)
            result.append(current)
            parent = payload.get("parent")
            current = None if parent is None else str(parent)
        return tuple(result)

    def _derivation_ancestry(self, start: str) -> frozenset[str]:
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(current: str) -> None:
            if current in visiting:
                raise ContractError("v2 Variant derivation ancestry contains a cycle")
            if current in visited:
                return
            visiting.add(current)
            payload = self._revision(current)
            owner = payload.get("variant")
            parent = payload.get("parent")
            if parent is not None:
                parent_payload = self._revision(str(parent))
                if parent_payload.get("variant") != owner:
                    raise ContractError("v2 Variant revision parent ownership mismatch")
                visit(str(parent))
            derivation = payload.get("derivation_parent")
            if derivation is not None:
                visit(str(derivation))
            visiting.remove(current)
            visited.add(current)

        visit(start)
        return frozenset(visited)

    def _changed_paths(
        self, before_revision: dict[str, Any], after_revision: dict[str, Any]
    ) -> tuple[str, ...]:
        before = self._tree_files(str(before_revision["source_tree"]))
        after = self._tree_files(str(after_revision["source_tree"]))
        return tuple(
            sorted(
                path
                for path in set(before) | set(after)
                if before.get(path) != after.get(path)
            )
        )

    def _tree_files(self, tree_hash: str) -> dict[str, tuple[str, int]]:
        tree = self.graph.store.load(tree_hash)
        if (
            tree.kind != "source_tree"
            or tree.payload.get("availability") != "available"
        ):
            raise ContractError("Variant promotion requires available source evidence")
        result: dict[str, tuple[str, int]] = {}
        files = tree.payload.get("files")
        if not isinstance(files, list):
            raise ContractError("v2 source tree files are invalid")
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ContractError("v2 source tree entry is invalid")
            path = PurePosixPath(str(item["path"]))
            blob_hash = item.get("blob")
            size = item.get("size")
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != item["path"]
                or item["path"] in result
                or not isinstance(blob_hash, str)
                or not DIGEST_PATTERN.fullmatch(blob_hash)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ContractError("v2 source tree entry is invalid")
            blob = self.graph.store.load(blob_hash)
            if (
                blob.kind != "blob"
                or blob.payload.get("content_hash") != item.get("content_hash")
                or blob.payload.get("size") != size
            ):
                raise ContractError("v2 source tree blob evidence disagrees")
            self.graph.store.verify_blob(blob)
            result[str(item["path"])] = (blob_hash, size)
        return result

    def _available_tree(self, revision: dict[str, Any], label: str) -> None:
        tree_hash = revision.get("source_tree")
        if not isinstance(tree_hash, str) or not DIGEST_PATTERN.fullmatch(tree_hash):
            raise ContractError(f"{label} Variant source tree reference is invalid")
        self._tree_files(tree_hash)

    def _revision(self, digest: str, *, owner: str | None = None) -> dict[str, Any]:
        if not isinstance(digest, str) or not DIGEST_PATTERN.fullmatch(digest):
            raise ContractError("Variant revision digest is invalid")
        record = self.graph.store.load(digest)
        if record.kind != "variant_revision":
            raise ContractError("Variant revision object kind is invalid")
        if owner is not None and record.payload.get("variant") != owner:
            raise ContractError("Variant revision ownership mismatch")
        for field in ("parent", "derivation_parent", "merge_parent"):
            value = record.payload.get(field)
            if value is not None and (
                not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value)
            ):
                raise ContractError(f"Variant revision {field} is invalid")
        return record.payload

    @staticmethod
    def _code_identity(payload: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (
            payload.get("template"),
            payload.get("source_tree"),
            payload.get("components"),
        )

    def _write_journal(
        self, document: dict[str, Any], *, phase: str | None = None
    ) -> None:
        if phase is not None:
            document["phase"] = phase
        if document.get("phase") not in _PHASES:
            raise ContractError("Variant promotion journal phase is invalid")
        text = (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        transaction_journal = (
            self.trash_root / str(document["transaction"]) / "journal.json"
        )
        for path in (transaction_journal, self.stable_journal):
            if os.path.lexists(path):
                existing_transaction = self._journal_transaction(path)
                if existing_transaction != document["transaction"]:
                    raise PromotionConflict(
                        "another Variant promotion owns the recovery journal"
                    )
                atomic_replace(path, text)
            else:
                path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                atomic_write_new(path, text)

    @staticmethod
    def _journal_transaction(path: Path) -> str:
        try:
            metadata = path.lstat()
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractError("Variant promotion journal is invalid") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or not isinstance(document, dict)
            or not isinstance(document.get("transaction"), str)
        ):
            raise ContractError("Variant promotion journal is invalid")
        return str(document["transaction"])

    def _load_journal(self) -> dict[str, Any]:
        try:
            metadata = self.stable_journal.lstat()
            document = json.loads(self.stable_journal.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractError("Variant promotion journal is invalid") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError("Variant promotion journal must be a regular file")
        required = {
            "schema_version",
            "phase",
            "transaction",
            "experiment",
            "source",
            "target",
            "source_variant_hash",
            "target_variant_hash",
            "source_revision_hash",
            "target_revision_hash",
            "new_revision_hash",
            "before_head",
            "binding_transaction",
            "plan_digest",
            "source_path",
            "target_path",
            "candidate_path",
            "before_path",
            "displaced_path",
            "source_fingerprint",
            "before_fingerprint",
            "candidate_fingerprint",
            "operations",
        }
        if (
            not isinstance(document, dict)
            or set(document) != required
            or document.get("schema_version") != 1
            or document.get("phase") not in _PHASES
            or not isinstance(document.get("transaction"), str)
            or not _TRANSACTION.fullmatch(str(document["transaction"]))
        ):
            raise ContractError("Variant promotion journal fields are invalid")
        names = (document["experiment"], document["source"], document["target"])
        if any(
            not isinstance(name, str) or not NAME_PATTERN.fullmatch(name)
            for name in names
        ):
            raise ContractError("Variant promotion journal names are invalid")
        for field in (
            "source_variant_hash",
            "target_variant_hash",
            "source_revision_hash",
            "target_revision_hash",
            "new_revision_hash",
            "plan_digest",
        ):
            value = document.get(field)
            if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
                raise ContractError("Variant promotion journal digest is invalid")
        for field in ("before_head", "binding_transaction"):
            value = document.get(field)
            if value is not None and (
                not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value)
            ):
                raise ContractError("Variant promotion journal HEAD is invalid")
        for field in (
            "source_fingerprint",
            "before_fingerprint",
            "candidate_fingerprint",
        ):
            value = document.get(field)
            if value is not None and (
                not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value)
            ):
                raise ContractError("Variant promotion journal fingerprint is invalid")
        expected_operations = [
            BindingOperation(
                "unbind",
                entity_revision_scope(str(document["target_variant_hash"])),
                CURRENT_REVISION_NAME,
                str(document["target_revision_hash"]),
            ).as_dict(),
            BindingOperation(
                "bind",
                entity_revision_scope(str(document["target_variant_hash"])),
                CURRENT_REVISION_NAME,
                str(document["new_revision_hash"]),
            ).as_dict(),
        ]
        if document.get("operations") != expected_operations:
            raise ContractError("Variant promotion journal operations are invalid")
        binding_transaction = document.get("binding_transaction")
        if binding_transaction is not None and not self._head_matches(
            str(binding_transaction), document
        ):
            raise ContractError(
                "Variant promotion journal binding transaction is invalid"
            )
        source_variant = str(document["source_variant_hash"])
        target_variant = str(document["target_variant_hash"])
        if source_variant == target_variant:
            raise ContractError("Variant promotion journal owners are invalid")
        source_revision = self._revision(
            str(document["source_revision_hash"]), owner=source_variant
        )
        target_revision = self._revision(
            str(document["target_revision_hash"]), owner=target_variant
        )
        if document["new_revision_hash"] == document["target_revision_hash"]:
            if self._code_identity(source_revision) != self._code_identity(
                target_revision
            ):
                raise ContractError("Variant promotion journal no-op is invalid")
        else:
            promoted = self._revision(
                str(document["new_revision_hash"]), owner=target_variant
            )
            if (
                promoted.get("parent") != document["target_revision_hash"]
                or promoted.get("merge_parent") != document["source_revision_hash"]
                or promoted.get("derivation_parent") is not None
            ):
                raise ContractError("Variant promotion journal revision is invalid")
        self._ensure_trash_root()
        transaction_dir = self.trash_root / str(document["transaction"])
        if transaction_dir.is_symlink() or not transaction_dir.is_dir():
            raise ContractError("Variant promotion transaction directory is invalid")
        expected = {
            "source_path": self.repository.experiments
            / str(document["experiment"])
            / str(document["source"]),
            "target_path": self.repository.experiments
            / str(document["experiment"])
            / str(document["target"]),
            "candidate_path": transaction_dir / "candidate",
            "before_path": transaction_dir / "before",
            "displaced_path": transaction_dir / "displaced",
        }
        for field, path in expected.items():
            if document[field] != self._relative(path):
                raise ContractError("Variant promotion journal path is invalid")
        return document

    def _restore_before_head(self, document: dict[str, Any]) -> None:
        target = self.repository.root / str(document["target_path"])
        before = self.repository.root / str(document["before_path"])
        displaced = self.repository.root / str(document["displaced_path"])
        candidate = self.repository.root / str(document["candidate_path"])
        if os.path.lexists(before):
            if compute_source_digest(before) != document["before_fingerprint"]:
                raise ContractError("Variant promotion backup changed during recovery")
            if os.path.lexists(target):
                fingerprint = document.get("candidate_fingerprint")
                if fingerprint is None or compute_source_digest(target) != fingerprint:
                    raise ContractError(
                        "Variant promotion target changed during recovery; inspect journal paths"
                    )
                if os.path.lexists(displaced):
                    raise ContractError(
                        "Variant promotion displaced path already exists"
                    )
                os.rename(target, displaced)
            os.rename(before, target)
            self._fsync(target.parent)
        elif not os.path.lexists(target):
            raise ContractError("Variant promotion recovery cannot find Target draft")
        if os.path.lexists(candidate):
            fingerprint = document.get("candidate_fingerprint")
            if (
                fingerprint is not None
                and compute_source_digest(candidate) != fingerprint
            ):
                raise ContractError(
                    "Variant promotion candidate changed during recovery"
                )
        self._write_journal(document, phase="rolled_back")
        self.stable_journal.unlink()
        self._fsync(self.stable_journal.parent)

    def _finish_committed(self, document: dict[str, Any]) -> None:
        target = self.repository.root / str(document["target_path"])
        candidate = self.repository.root / str(document["candidate_path"])
        fingerprint = document.get("candidate_fingerprint")
        if not os.path.lexists(target) and os.path.lexists(candidate):
            os.rename(candidate, target)
            self._fsync(target.parent)
        if (
            fingerprint is None
            or not os.path.lexists(target)
            or compute_source_digest(target) != fingerprint
        ):
            raise ContractError("committed Variant promotion draft is unavailable")
        current = self.graph.current_revision(str(document["target_variant_hash"]))
        if current != document["new_revision_hash"]:
            raise ContractError("committed Variant promotion revision is not current")
        if document["new_revision_hash"] != document["target_revision_hash"]:
            promoted = self.authoring.load_variant(
                str(document["experiment"]), str(document["target"])
            )
            identity = self.graph.preview_variant_revision(
                str(document["target_variant_hash"]),
                promoted,
                parent=str(document["target_revision_hash"]),
                merge_parent=str(document["source_revision_hash"]),
            )
            if identity.variant_revision_hash != document["new_revision_hash"]:
                raise ContractError(
                    "committed Variant promotion draft disagrees with its revision"
                )
        GraphProjection(self.graph).rebuild()

    def _complete(self, document: dict[str, Any]) -> None:
        self._write_journal(document, phase="completed")
        self.stable_journal.unlink()
        self._fsync(self.stable_journal.parent)

    def _source_deleted(self, document: dict[str, Any]) -> bool:
        experiment_hash = self.graph.experiment_hash(str(document["experiment"]))
        scope = experiment_variant_scope(experiment_hash)
        source = str(document["source"])
        expected = str(document["source_variant_hash"])
        try:
            active = self.graph.bindings.resolve(scope, source)
        except NotFoundError:
            if expected not in self.graph.bindings.historical_targets(scope, source):
                raise PromotionConflict(
                    "Variant promotion Source history is unavailable"
                )
            return True
        if active != expected:
            raise PromotionConflict(
                "Variant promotion Source name belongs to a different entity"
            )
        return False

    def _matching_transaction(
        self, head: str | None, document: dict[str, Any]
    ) -> str | None:
        current = head
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ContractError("v2 binding transaction chain contains a cycle")
            seen.add(current)
            if self._head_matches(current, document):
                return current
            record = self.graph.store.load(current)
            if record.kind != "binding_transaction":
                raise ContractError("v2 binding chain contains a non-transaction")
            previous = record.payload.get("previous")
            current = None if previous is None else str(previous)
        return None

    def _head_descends_from(self, head: str | None, ancestor: str) -> bool:
        current = head
        seen: set[str] = set()
        while current is not None:
            if current == ancestor:
                return True
            if current in seen:
                raise ContractError("v2 binding transaction chain contains a cycle")
            seen.add(current)
            record = self.graph.store.load(current)
            if record.kind != "binding_transaction":
                raise ContractError("v2 binding chain contains a non-transaction")
            previous = record.payload.get("previous")
            current = None if previous is None else str(previous)
        return False

    def _head_matches(self, head: str | None, document: dict[str, Any]) -> bool:
        if head is None:
            return False
        try:
            record = self.graph.store.load(head)
        except (ContractError, NotFoundError):
            return False
        return (
            record.kind == "binding_transaction"
            and record.payload.get("previous") == document.get("before_head")
            and record.payload.get("operations") == document.get("operations")
        )

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.repository.root).as_posix()
        except ValueError as error:
            raise ContractError(
                "Variant promotion path is outside repository"
            ) from error

    def _ensure_trash_root(self) -> None:
        current = self.repository.root
        for component in (".hkdl", "trash", "promotions"):
            current = current / component
            current.mkdir(mode=0o755, exist_ok=True)
            if current.is_symlink() or not current.is_dir():
                raise ContractError("Variant promotion trash root is invalid")

    @staticmethod
    def _code_fingerprint(path: Path) -> str:
        code = path / "code.json"
        try:
            metadata = code.lstat()
            content = code.read_bytes()
        except OSError as error:
            raise ContractError("Variant Code draft is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError("Variant Code draft must be a regular file")
        digest = hashlib.sha256()
        digest.update(content)
        digest.update(compute_source_digest(path / "src").encode("ascii"))
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _fsync(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
