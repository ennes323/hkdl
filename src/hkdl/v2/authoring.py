"""V2 commit and rename operations over mutable authored drafts."""

from __future__ import annotations

import copy
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..authoring import Authoring, RESERVED_VARIANT_NAMES
from ..config import NAME_PATTERN, ContractError, dump_yaml
from ..storage import AlreadyExistsError, atomic_replace, atomic_write_new
from .graph import (
    DirtyDraftError,
    ExperimentIdentity,
    V2Graph,
    VariantIdentity,
    experiment_variant_scope,
    workspace_experiment_scope,
)
from .maintenance import workspace_operation
from .leases import authoring_write, lease_held
from .reader import GraphReader
from .maintenance import WorkspaceBusy
from ..run_contracts import TERMINAL_STATUSES


@dataclass(frozen=True)
class RenamePlan:
    experiment: str
    old_name: str
    new_name: str
    variant_hash: str
    binding_head: str | None
    changed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "old_name": self.old_name,
            "new_name": self.new_name,
            "variant_hash": self.variant_hash,
            "binding_head": self.binding_head,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class ExperimentRenamePlan:
    old_name: str
    new_name: str
    experiment_hash: str
    binding_head: str | None
    changed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "old_name": self.old_name,
            "new_name": self.new_name,
            "experiment_hash": self.experiment_hash,
            "binding_head": self.binding_head,
            "changed": self.changed,
        }


class V2Authoring:
    def __init__(self, authoring: Authoring):
        self.authoring = authoring
        self.graph = V2Graph(authoring.repository)
        self.journal_root = self.graph.store.root / "refs/renames"

    @workspace_operation
    @authoring_write
    def commit_experiment(self, name: str) -> ExperimentIdentity:
        self._require_active()
        self.recover_renames()
        record = self.authoring.load_experiment(name)
        experiment_hash = self.graph.experiment_hash(name)
        current = self.graph.store.load(
            self.graph.current_revision(experiment_hash)
        ).payload
        structural_changed = (
            current.get("type") != record.document["type"]
            or current.get("template") != record.document["template"]
        )
        if structural_changed and self.graph.bindings.names(
            experiment_variant_scope(experiment_hash)
        ):
            raise DirtyDraftError(
                "Experiment type and Template family are locked while Variants exist"
            )
        return self.graph.commit_experiment(record)

    @workspace_operation
    @authoring_write
    def commit_variant(self, experiment: str, variant: str) -> VariantIdentity:
        self._require_active()
        self.recover_renames()
        return self.graph.commit_variant(
            self.authoring.load_experiment(experiment),
            self.authoring.load_variant(experiment, variant),
            commit_experiment=False,
        )

    @workspace_operation
    @authoring_write
    def rename_experiment(
        self,
        old_name: str,
        new_name: str,
        *,
        dry_run: bool = False,
    ) -> ExperimentRenamePlan:
        self._require_active()
        if not dry_run:
            self.recover_renames()
        _entity_name(old_name, "Experiment")
        _entity_name(new_name, "Experiment")
        if old_name == new_name:
            raise ContractError("Experiment rename requires a different name")
        record = self.authoring.load_experiment(old_name)
        scope = workspace_experiment_scope()
        experiment_hash = self.graph.bindings.resolve(scope, old_name)
        if new_name in self.graph.bindings.names(scope):
            raise AlreadyExistsError(f"Experiment already exists: {new_name}")
        if not self.graph.bindings.can_bind_name(
            scope, new_name, target=experiment_hash
        ):
            raise AlreadyExistsError(
                f"Experiment name belongs to another active entity: {new_name}"
            )
        target = self.authoring.repository.experiments / new_name
        if os.path.lexists(target):
            raise AlreadyExistsError(f"Experiment draft already exists: {new_name}")
        self._assert_rename_idle(old_name)
        plan = ExperimentRenamePlan(
            old_name,
            new_name,
            experiment_hash,
            self.graph.bindings.head(),
            True,
        )
        if dry_run:
            return plan

        journal = self._write_journal(
            {
                "schema_version": 2,
                "phase": "prepared",
                "kind": "experiment",
                "scope": scope,
                "old_name": old_name,
                "new_name": new_name,
                "entity_hash": experiment_hash,
                "authored_schema_version": record.authored_schema_version,
            }
        )
        try:
            self.graph.bindings.rename(scope, old_name, new_name)
            self._replace_journal(journal, phase="binding_committed")
            os.rename(record.path, target)
            _fsync_directory(self.authoring.repository.experiments)
            if record.authored_schema_version == 1:
                document = copy.deepcopy(record.document)
                document["name"] = new_name
                atomic_replace(target / "experiment.yaml", dump_yaml(document))
            journal.unlink()
            _fsync_directory(self.journal_root)
        except BaseException:
            self._rollback_experiment_rename(journal)
            raise
        return ExperimentRenamePlan(
            old_name,
            new_name,
            experiment_hash,
            self.graph.bindings.head(),
            True,
        )

    @workspace_operation
    @authoring_write
    def rename_variant(
        self,
        experiment: str,
        old_name: str,
        new_name: str,
        *,
        dry_run: bool = False,
    ) -> RenamePlan:
        self._require_active()
        if not dry_run:
            self.recover_renames()
        _variant_name(old_name)
        _variant_name(new_name)
        if old_name == new_name:
            raise ContractError("Variant rename requires a different name")
        experiment_record = self.authoring.load_experiment(experiment)
        variant = self.authoring.load_variant(experiment, old_name)
        experiment_hash = self.graph.experiment_hash(experiment)
        scope = experiment_variant_scope(experiment_hash)
        variant_hash = self.graph.bindings.resolve(scope, old_name)
        if new_name in self.graph.bindings.names(scope):
            raise AlreadyExistsError(f"Variant already exists: {experiment}/{new_name}")
        if not self.graph.bindings.can_bind_name(scope, new_name, target=variant_hash):
            raise AlreadyExistsError(
                f"Variant name belongs to another active entity: {new_name}"
            )
        target = experiment_record.path / new_name
        if os.path.lexists(target):
            raise AlreadyExistsError(
                f"Variant draft already exists: {experiment}/{new_name}"
            )
        self._assert_rename_idle(experiment, old_name)
        plan = RenamePlan(
            experiment,
            old_name,
            new_name,
            variant_hash,
            self.graph.bindings.head(),
            True,
        )
        if dry_run:
            return plan

        journal = self._write_journal(
            {
                "schema_version": 2,
                "phase": "prepared",
                "experiment": experiment,
                "scope": scope,
                "old_name": old_name,
                "new_name": new_name,
                "variant_hash": variant_hash,
            }
        )
        try:
            self.graph.bindings.rename(scope, old_name, new_name)
            self._replace_journal(journal, phase="binding_committed")
            os.rename(variant.path, target)
            _fsync_directory(experiment_record.path)
            if variant.authored_schema_version == 1:
                document = copy.deepcopy(variant.document)
                document["name"] = new_name
                atomic_replace(target / "variant.yaml", dump_yaml(document))
            journal.unlink()
            _fsync_directory(self.journal_root)
        except BaseException:
            self._rollback_rename(journal)
            raise
        return RenamePlan(
            experiment,
            old_name,
            new_name,
            variant_hash,
            self.graph.bindings.head(),
            True,
        )

    def _assert_rename_idle(self, experiment, variant=None):
        repository = self.authoring.repository
        for record in GraphReader(repository).runs(
            experiment=experiment, variant=variant
        ):
            if record.state["status"] not in TERMINAL_STATUSES or lease_held(
                repository, record.graph_identity["attempt_hash"], record.path
            ):
                raise WorkspaceBusy("active or nonterminal Run blocks authored rename")

    @workspace_operation
    def recover_renames(self) -> None:
        if not os.path.lexists(self.journal_root):
            return
        if self.journal_root.is_symlink() or not self.journal_root.is_dir():
            raise ContractError("v2 rename journal root is invalid")
        for journal in sorted(self.journal_root.iterdir(), key=lambda path: path.name):
            document = _load_journal(journal)
            if document.get("kind") == "experiment":
                self._recover_experiment_rename(journal, document)
                continue
            scope = str(document["scope"])
            old = str(document["old_name"])
            new = str(document["new_name"])
            names = self.graph.bindings.names(scope)
            if names.get(new) == document["variant_hash"]:
                self._finish_rename(document)
                journal.unlink()
                _fsync_directory(self.journal_root)
            elif names.get(old) == document["variant_hash"]:
                self._restore_draft(document)
                journal.unlink()
                _fsync_directory(self.journal_root)
            else:
                raise ContractError("v2 rename journal disagrees with active bindings")

    def _recover_experiment_rename(
        self, journal: Path, document: dict[str, Any]
    ) -> None:
        scope = str(document["scope"])
        old = str(document["old_name"])
        new = str(document["new_name"])
        entity_hash = str(document["entity_hash"])
        names = self.graph.bindings.names(scope)
        if names.get(new) == entity_hash:
            self._finish_experiment_rename(document)
        elif names.get(old) == entity_hash:
            self._restore_experiment_draft(document)
        else:
            raise ContractError("v2 Experiment rename journal disagrees with bindings")
        journal.unlink()
        _fsync_directory(self.journal_root)

    def _finish_experiment_rename(self, document: dict[str, Any]) -> None:
        root = self.authoring.repository.experiments
        old = root / str(document["old_name"])
        new = root / str(document["new_name"])
        if os.path.lexists(old) and os.path.lexists(new):
            raise ContractError("Experiment rename recovery found both draft names")
        if os.path.lexists(old):
            os.rename(old, new)
            _fsync_directory(root)
        if not os.path.lexists(new):
            raise ContractError("Experiment rename recovery cannot find the draft")
        if int(document["authored_schema_version"]) == 1:
            raw = _load_experiment_document(new / "experiment.yaml")
            raw["name"] = str(document["new_name"])
            atomic_replace(new / "experiment.yaml", dump_yaml(raw))

    def _restore_experiment_draft(self, document: dict[str, Any]) -> None:
        root = self.authoring.repository.experiments
        old = root / str(document["old_name"])
        new = root / str(document["new_name"])
        if os.path.lexists(old) and os.path.lexists(new):
            raise ContractError("Experiment rename rollback found both draft names")
        if os.path.lexists(new):
            os.rename(new, old)
            _fsync_directory(root)
        if os.path.lexists(old / "experiment.yaml"):
            raw = _load_experiment_document(old / "experiment.yaml")
            raw["name"] = str(document["old_name"])
            atomic_replace(old / "experiment.yaml", dump_yaml(raw))

    def _rollback_experiment_rename(self, journal: Path) -> None:
        if not os.path.lexists(journal):
            return
        document = _load_journal(journal)
        scope = str(document["scope"])
        old = str(document["old_name"])
        new = str(document["new_name"])
        entity_hash = str(document["entity_hash"])
        names = self.graph.bindings.names(scope)
        if names.get(new) == entity_hash and old not in names:
            self.graph.bindings.rename(scope, new, old)
        self._restore_experiment_draft(document)
        journal.unlink()
        _fsync_directory(self.journal_root)

    def _finish_rename(self, document: dict[str, Any]) -> None:
        experiment = self.authoring.load_experiment(str(document["experiment"]))
        old = experiment.path / str(document["old_name"])
        new = experiment.path / str(document["new_name"])
        if os.path.lexists(old) and os.path.lexists(new):
            raise ContractError("v2 rename recovery found both draft names")
        if os.path.lexists(old):
            os.rename(old, new)
            _fsync_directory(experiment.path)
        if not os.path.lexists(new):
            raise ContractError("v2 rename recovery cannot find the draft")
        if os.path.lexists(new / "variant.yaml"):
            raw = _load_variant_document(new / "variant.yaml")
            raw["name"] = str(document["new_name"])
            atomic_replace(new / "variant.yaml", dump_yaml(raw))

    def _restore_draft(self, document: dict[str, Any]) -> None:
        experiment = self.authoring.load_experiment(str(document["experiment"]))
        old = experiment.path / str(document["old_name"])
        new = experiment.path / str(document["new_name"])
        if os.path.lexists(old) and os.path.lexists(new):
            raise ContractError("v2 rename rollback found both draft names")
        if os.path.lexists(new):
            os.rename(new, old)
            _fsync_directory(experiment.path)
        if os.path.lexists(old / "variant.yaml"):
            raw = _load_variant_document(old / "variant.yaml")
            raw["name"] = str(document["old_name"])
            atomic_replace(old / "variant.yaml", dump_yaml(raw))

    def _rollback_rename(self, journal: Path) -> None:
        if not os.path.lexists(journal):
            return
        document = _load_journal(journal)
        scope = str(document["scope"])
        old = str(document["old_name"])
        new = str(document["new_name"])
        names = self.graph.bindings.names(scope)
        if names.get(new) == document["variant_hash"] and old not in names:
            self.graph.bindings.rename(scope, new, old)
        self._restore_draft(document)
        journal.unlink()
        _fsync_directory(self.journal_root)

    def _write_journal(self, document: dict[str, Any]) -> Path:
        self.graph.store._ensure_layout()
        self.journal_root.mkdir(mode=0o755, parents=True, exist_ok=True)
        if self.journal_root.is_symlink() or not self.journal_root.is_dir():
            raise ContractError("v2 rename journal root is invalid")
        path = self.journal_root / f"rename-{secrets.token_hex(16)}.json"
        atomic_write_new(path, _json_text(document))
        return path

    @staticmethod
    def _replace_journal(path: Path, *, phase: str) -> None:
        document = _load_journal(path)
        document["phase"] = phase
        atomic_replace(path, _json_text(document))

    def _require_active(self) -> None:
        if not self.graph.is_active():
            raise ContractError("HKDL v2 is not active; run `hkdl migrate --all` first")


def _variant_name(value: str) -> None:
    _entity_name(value, "Variant")
    if value in RESERVED_VARIANT_NAMES:
        raise ContractError(f"Variant uses a reserved name: {value}")


def _entity_name(value: str, kind: str) -> None:
    if not NAME_PATTERN.fullmatch(value):
        raise ContractError(f"invalid {kind} name")


def _json_text(document: dict[str, Any]) -> str:
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def _load_journal(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("v2 rename journal is invalid") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError("v2 rename journal must be a regular non-symlink file")
    variant_fields = {
        "schema_version",
        "phase",
        "experiment",
        "scope",
        "old_name",
        "new_name",
        "variant_hash",
    }
    experiment_fields = {
        "schema_version",
        "phase",
        "kind",
        "scope",
        "old_name",
        "new_name",
        "entity_hash",
        "authored_schema_version",
    }
    fields = frozenset(document) if isinstance(document, dict) else frozenset()
    if not isinstance(document, dict) or fields not in {
        frozenset(variant_fields),
        frozenset(experiment_fields),
    }:
        raise ContractError("v2 rename journal fields are invalid")
    if document["schema_version"] != 2 or document["phase"] not in {
        "prepared",
        "binding_committed",
    }:
        raise ContractError("v2 rename journal state is invalid")
    if fields == experiment_fields and (
        document["kind"] != "experiment"
        or document["authored_schema_version"] not in {1, 2}
    ):
        raise ContractError("v2 Experiment rename journal state is invalid")
    return document


def _load_variant_document(path: Path) -> dict[str, Any]:
    from ..config import load_yaml_file

    return load_yaml_file(path)


def _load_experiment_document(path: Path) -> dict[str, Any]:
    from ..config import load_yaml_file

    return load_yaml_file(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
