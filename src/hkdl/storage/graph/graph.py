"""Typed graph construction and lookup for HKDL v2 identities."""

from __future__ import annotations

import os
import secrets
import stat
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from hkdl.authoring.authoring_records import ExperimentRecord, VariantRecord
from hkdl.errors import ContractError
from hkdl.execution.run_contracts import evaluation_case, metric_spec
from hkdl.storage.storage import (
    AlreadyExistsError,
    NotFoundError,
    RepositoryPaths,
    atomic_replace,
    atomic_write_new,
)

from .bindings import (
    BindingHeadConflict,
    BindingLog,
    BindingOperation,
)
from .leases import authoring_guard
from .maintenance import workspace_operation
from .objects import (
    ObjectRecord,
    ObjectStore,
    hash_file,
    object_digest,
)

RevisionValidator = Callable[[dict[str, Any], str | None], None]

CURRENT_FORMAT = "v2"
CURRENT_REVISION_NAME = "current"


@dataclass(frozen=True)
class ExperimentIdentity:
    experiment_hash: str
    experiment_revision_hash: str
    changed: bool


@dataclass(frozen=True)
class VariantIdentity:
    experiment_hash: str
    variant_hash: str
    variant_revision_hash: str
    source_tree_hash: str
    evaluation_cases: dict[str, str]
    export_profiles: dict[str, str]
    changed: bool


class V2Graph:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
        nonce: Callable[[], str] | None = None,
    ):
        self.repository = repository
        self.store = ObjectStore(repository.root)
        self.bindings = BindingLog(self.store, now=now, nonce=nonce)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._nonce = nonce or (lambda: secrets.token_hex(16))
        self.current_path = repository.root / ".hkdl/store/CURRENT"

    def is_active(self) -> bool:
        if not os.path.lexists(self.current_path):
            return False
        try:
            metadata = self.current_path.lstat()
            content = self.current_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise ContractError("HKDL store CURRENT is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError("HKDL store CURRENT must be a regular non-symlink file")
        if content != f"{CURRENT_FORMAT}\n":
            raise ContractError("HKDL store CURRENT format is unsupported")
        self.bindings.head()
        return True

    @workspace_operation
    def activate(self) -> None:
        """Atomically select v2 after the caller has verified the complete graph."""

        parent = self.current_path.parent
        parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ContractError("HKDL store root must be a real directory")
        if os.path.lexists(self.current_path):
            atomic_replace(self.current_path, f"{CURRENT_FORMAT}\n")
        else:
            atomic_write_new(self.current_path, f"{CURRENT_FORMAT}\n")
        if not self.is_active():
            raise ContractError("HKDL v2 activation failed")

    def experiment_hash(self, name: str) -> str:
        return self.bindings.resolve(workspace_experiment_scope(), name)

    def variant_hash(self, experiment_hash: str, name: str) -> str:
        return self.bindings.resolve(experiment_variant_scope(experiment_hash), name)

    def current_revision(self, entity_hash: str) -> str:
        return self.bindings.resolve(
            entity_revision_scope(entity_hash), CURRENT_REVISION_NAME
        )

    @workspace_operation
    def commit_experiment(
        self,
        record: ExperimentRecord,
        *,
        creation_nonce: str | None = None,
        validate_revision: RevisionValidator | None = None,
    ) -> ExperimentIdentity:
        with authoring_guard(self.repository, record.document["name"]):
            expected_head = (
                self.bindings.head() if validate_revision is not None else None
            )
            entity = None
            name = str(record.document["name"])
            try:
                entity_hash = self.experiment_hash(name)
            except NotFoundError:
                if not self.bindings.can_bind_name(workspace_experiment_scope(), name):
                    raise AlreadyExistsError(
                        f"Experiment name belongs to another active entity: {name}"
                    )
                publish_entity = (
                    self.store.put if validate_revision is None else self.store.preview
                )
                entity = publish_entity(
                    "experiment",
                    {
                        "created_at": record.document["created_at"],
                        "creation_nonce": creation_nonce or self._nonce(),
                    },
                )
                entity_hash = entity.digest
                current_revision = None
            else:
                try:
                    current_revision = self.current_revision(entity_hash)
                except NotFoundError:
                    current_revision = None

            revision_payload = {
                "experiment": entity_hash,
                "parent": current_revision,
                "type": record.document["type"],
                "question": record.document["question"],
                "template": deepcopy(record.document["template"]),
            }
            self._validate_revision(revision_payload, validate_revision, expected_head)
            if current_revision is not None:
                current_payload = self.store.load(current_revision).payload
                if _same_revision_content(current_payload, revision_payload):
                    return ExperimentIdentity(entity_hash, current_revision, False)
            if entity is not None and validate_revision is not None:
                self.store.put("experiment", entity.payload)
            revision = self.store.put("experiment_revision", revision_payload)

            operations: list[BindingOperation] = []
            if current_revision is None:
                if name not in self.bindings.names(workspace_experiment_scope()):
                    operations.append(
                        BindingOperation(
                            "bind",
                            workspace_experiment_scope(),
                            name,
                            entity_hash,
                        )
                    )
            else:
                operations.append(
                    BindingOperation(
                        "unbind",
                        entity_revision_scope(entity_hash),
                        CURRENT_REVISION_NAME,
                        current_revision,
                    )
                )
            operations.append(
                BindingOperation(
                    "bind",
                    entity_revision_scope(entity_hash),
                    CURRENT_REVISION_NAME,
                    revision.digest,
                )
            )
            self._commit_revision_bindings(operations, validate_revision, expected_head)
            return ExperimentIdentity(entity_hash, revision.digest, True)

    @workspace_operation
    def commit_variant(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        *,
        derivation_parent: str | None = None,
        merge_parent: str | None = None,
        creation_nonce: str | None = None,
        commit_experiment: bool = True,
        validate_revision: RevisionValidator | None = None,
    ) -> VariantIdentity:
        with authoring_guard(
            self.repository, experiment.document["name"], variant.document["name"]
        ):
            expected_head = (
                self.bindings.head() if validate_revision is not None else None
            )
            entity = None
            if commit_experiment and validate_revision is None:
                experiment_identity = self.commit_experiment(experiment)
                experiment_hash = experiment_identity.experiment_hash
            else:
                experiment_hash = self.experiment_hash(str(experiment.document["name"]))
                self.assert_experiment_clean(
                    str(experiment.document["name"]), experiment
                )
            name = str(variant.document["name"])
            variant_scope = experiment_variant_scope(experiment_hash)
            try:
                entity_hash = self.bindings.resolve(variant_scope, name)
            except NotFoundError:
                if not self.bindings.can_bind_name(variant_scope, name):
                    raise AlreadyExistsError(
                        f"Variant name belongs to another active entity: {name}"
                    )
                publish_entity = (
                    self.store.put if validate_revision is None else self.store.preview
                )
                entity = publish_entity(
                    "variant",
                    {
                        "experiment": experiment_hash,
                        "created_at": _timestamp(self._now()),
                        "creation_nonce": creation_nonce or self._nonce(),
                    },
                )
                entity_hash = entity.digest
                current_revision = None
            else:
                try:
                    current_revision = self.current_revision(entity_hash)
                except NotFoundError:
                    current_revision = None

            source_tree = self.capture_source_tree(
                variant.path / "src", publish=validate_revision is None
            )
            revision_payload = self.variant_revision_payload(
                entity_hash,
                variant.document,
                source_tree.digest,
                parent=current_revision,
                derivation_parent=derivation_parent,
                merge_parent=merge_parent,
            )
            self._validate_revision(revision_payload, validate_revision, expected_head)
            if current_revision is not None and _same_revision_content(
                self.store.load(current_revision).payload,
                revision_payload,
            ):
                return VariantIdentity(
                    experiment_hash,
                    entity_hash,
                    current_revision,
                    source_tree.digest,
                    {},
                    {},
                    False,
                )
            if current_revision is not None and self.has_active_execution(entity_hash):
                raise DirtyDraftError(
                    "Variant Code is locked while active Runs or Models exist; "
                    "delete the active execution lineage before committing Code"
                )
            if validate_revision is not None:
                published_source = self.capture_source_tree(
                    variant.path / "src", publish=True
                )
                if published_source.digest != source_tree.digest:
                    raise StalePreviewError()
                if entity is not None:
                    self.store.put("variant", entity.payload)
            revision = self.store.put("variant_revision", revision_payload)

            operations: list[BindingOperation] = []
            if current_revision is None:
                if name not in self.bindings.names(variant_scope):
                    operations.append(
                        BindingOperation("bind", variant_scope, name, entity_hash)
                    )
            else:
                operations.append(
                    BindingOperation(
                        "unbind",
                        entity_revision_scope(entity_hash),
                        CURRENT_REVISION_NAME,
                        current_revision,
                    )
                )
            operations.append(
                BindingOperation(
                    "bind",
                    entity_revision_scope(entity_hash),
                    CURRENT_REVISION_NAME,
                    revision.digest,
                )
            )
            self._commit_revision_bindings(operations, validate_revision, expected_head)
            return VariantIdentity(
                experiment_hash,
                entity_hash,
                revision.digest,
                source_tree.digest,
                {},
                {},
                True,
            )

    def _validate_revision(
        self,
        payload: dict[str, Any],
        validate_revision: RevisionValidator | None,
        expected_head: str | None,
    ) -> None:
        if validate_revision is None:
            return
        if self.bindings.head() != expected_head:
            raise StalePreviewError()
        validate_revision(deepcopy(payload), expected_head)
        if self.bindings.head() != expected_head:
            raise StalePreviewError()

    def _commit_revision_bindings(
        self,
        operations: list[BindingOperation],
        validate_revision: RevisionValidator | None,
        expected_head: str | None,
    ) -> None:
        if validate_revision is None:
            self.bindings.commit(operations)
            return
        try:
            self.bindings.commit(operations, expected_head=expected_head)
        except BindingHeadConflict as error:
            raise StalePreviewError() from error

    def preview_variant_revision(
        self,
        variant_hash: str,
        variant: VariantRecord,
        *,
        parent: str | None,
        derivation_parent: str | None = None,
        merge_parent: str | None = None,
    ) -> VariantIdentity:
        source_tree = self.capture_source_tree(variant.path / "src", publish=False)
        revision_payload = self.variant_revision_payload(
            variant_hash,
            variant.document,
            source_tree.digest,
            parent=parent,
            derivation_parent=derivation_parent,
            merge_parent=merge_parent,
        )
        changed = True
        revision_hash = object_digest(
            "variant_revision",
            revision_payload,
        )
        if parent is not None and _same_revision_content(
            self.store.load(parent).payload,
            revision_payload,
        ):
            revision_hash = parent
            changed = False
        entity = self.store.load(variant_hash)
        experiment_hash = str(entity.payload["experiment"])
        return VariantIdentity(
            experiment_hash,
            variant_hash,
            revision_hash,
            source_tree.digest,
            {},
            {},
            changed,
        )

    def preview_experiment_revision(
        self,
        experiment_hash: str,
        experiment: ExperimentRecord,
        *,
        parent: str | None,
    ) -> ExperimentIdentity:
        payload = {
            "experiment": experiment_hash,
            "parent": parent,
            "type": experiment.document["type"],
            "question": experiment.document["question"],
            "template": deepcopy(experiment.document["template"]),
        }
        digest = object_digest("experiment_revision", payload)
        changed = True
        if parent is not None and _same_revision_content(
            self.store.load(parent).payload, payload
        ):
            digest = parent
            changed = False
        return ExperimentIdentity(experiment_hash, digest, changed)

    def assert_experiment_clean(
        self, name: str, experiment: ExperimentRecord
    ) -> ExperimentIdentity:
        if not self.is_active():
            raise ContractError("HKDL v2 is not active")
        entity = self.experiment_hash(name)
        current = self.current_revision(entity)
        preview = self.preview_experiment_revision(entity, experiment, parent=current)
        if preview.changed:
            raise DirtyDraftError(
                f"Experiment draft differs from committed revision; run "
                f"`hkdl experiment commit {name}`"
            )
        return preview

    def has_active_execution(self, variant_hash: str) -> bool:
        return bool(
            self.bindings.names(variant_run_scope(variant_hash))
            or self.bindings.names(variant_model_scope(variant_hash))
        )

    def assert_variant_clean(
        self, experiment_name: str, variant: VariantRecord
    ) -> VariantIdentity:
        if not self.is_active():
            raise ContractError("HKDL v2 is not active")
        experiment_hash = self.experiment_hash(experiment_name)
        variant_hash = self.variant_hash(experiment_hash, str(variant.document["name"]))
        current = self.current_revision(variant_hash)
        preview = self.preview_variant_revision(variant_hash, variant, parent=current)
        if preview.changed:
            raise DirtyDraftError(
                f"Variant draft differs from committed revision; run "
                f"`hkdl variant commit {experiment_name} {variant.document['name']}`"
            )
        return preview

    def capture_source_tree(self, root: Path, *, publish: bool) -> ObjectRecord:
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ContractError("Variant source tree must be a real directory")
        files: list[dict[str, Any]] = []
        for path in sorted(
            root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
        ):
            relative = PurePosixPath(*path.relative_to(root).parts).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ContractError(f"symlinks are not allowed: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(f"non-regular source entry: {relative}")
            content_hash, size = hash_file(path)
            blob_payload = {
                "content_hash": content_hash,
                "media_type": "application/octet-stream",
                "size": size,
            }
            blob_hash = object_digest("blob", blob_payload)
            if publish:
                published = self.store.put_blob(path)
                if published.digest != blob_hash:
                    raise ContractError(
                        "source blob preview disagrees with publication"
                    )
            files.append(
                {
                    "path": relative,
                    "blob": blob_hash,
                    "content_hash": content_hash,
                    "size": size,
                }
            )
        payload = {"availability": "available", "files": files}
        return (
            self.store.put("source_tree", payload)
            if publish
            else self.store.preview("source_tree", payload)
        )

    def unavailable_source_tree(self, digest: str, *, publish: bool) -> ObjectRecord:
        payload = {"availability": "unavailable", "legacy_digest": digest}
        return (
            self.store.put("source_tree", payload)
            if publish
            else self.store.preview("source_tree", payload)
        )

    @staticmethod
    def variant_revision_payload(
        variant_hash: str,
        document: dict[str, Any],
        source_tree_hash: str,
        *,
        parent: str | None,
        derivation_parent: str | None,
        merge_parent: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "variant": variant_hash,
            "parent": parent,
            "derivation_parent": derivation_parent,
            "template": deepcopy(document["template"]),
            "source_tree": source_tree_hash,
            "components": deepcopy(document["components"]),
        }
        if merge_parent is not None:
            payload["merge_parent"] = merge_parent
        return payload

    def _evaluation_cases(
        self, document: dict[str, Any], *, publish: bool
    ) -> dict[str, str]:
        evaluation = document["eval"]
        names = (
            sorted(evaluation["cases"], key=lambda item: item.encode("utf-8"))
            if isinstance(evaluation.get("cases"), dict)
            else ["default"]
        )
        result: dict[str, str] = {}
        for name in names:
            payload = {
                "definition": evaluation_case(document, name),
                "metrics": metric_spec(document, name),
            }
            record = (
                self.store.put("evaluation_case", payload)
                if publish
                else self.store.preview("evaluation_case", payload)
            )
            result[name] = record.digest
        return result

    def _export_profiles(
        self, document: dict[str, Any], *, publish: bool
    ) -> dict[str, str]:
        payload = {"definition": deepcopy(document["infer"])}
        record = (
            self.store.put("export_profile", payload)
            if publish
            else self.store.preview("export_profile", payload)
        )
        return {"default": record.digest}


class DirtyDraftError(RuntimeError):
    """An action attempted to run against an uncommitted authored draft."""


class StalePreviewError(DirtyDraftError):
    """The candidate no longer matches the changes confirmed by the caller."""

    def __init__(self) -> None:
        super().__init__(
            "The draft or commit state changed since this preview. "
            "Refresh and review before committing again."
        )


def comparison_digest(payload: dict[str, Any]) -> str:
    return object_digest("run_spec", {"comparison": payload})


def workspace_experiment_scope() -> str:
    return "workspace/experiments"


def experiment_variant_scope(experiment_hash: str) -> str:
    return f"experiment/{_hash_segment(experiment_hash)}/variants"


def entity_revision_scope(entity_hash: str) -> str:
    return f"entity/{_hash_segment(entity_hash)}/revisions"


def variant_run_scope(variant_hash: str) -> str:
    return f"variant/{_hash_segment(variant_hash)}/runs"


def variant_model_scope(variant_hash: str) -> str:
    return f"variant/{_hash_segment(variant_hash)}/models"


def evaluation_case_scope(revision_hash: str) -> str:
    return f"variant-revision/{_hash_segment(revision_hash)}/evaluation-cases"


def export_profile_scope(revision_hash: str) -> str:
    return f"variant-revision/{_hash_segment(revision_hash)}/export-profiles"


def comparison_group_scope(revision_hash: str) -> str:
    return f"variant-revision/{_hash_segment(revision_hash)}/comparison-groups"


def attempt_event_scope(attempt_hash: str) -> str:
    return f"attempt/{_hash_segment(attempt_hash)}/events"


def _hash_segment(digest: str) -> str:
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ContractError("invalid graph hash")
    return digest.removeprefix("sha256:")


def _timestamp(now: datetime) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ContractError("v2 graph clock must be timezone-aware")
    return now.isoformat()


def _same_revision_content(first: dict[str, Any], second: dict[str, Any]) -> bool:
    ignored = {"parent", "derivation_parent", "merge_parent"}
    return {key: value for key, value in first.items() if key not in ignored} == {
        key: value for key, value in second.items() if key not in ignored
    }
