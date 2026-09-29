"""Migrate legacy YAML authoring into the name-free JSON authoring layout.

This migration is deliberately narrower than :mod:`hkdl.storage.graph.migration`.  The
latter imports a legacy workspace into the v2 graph; this module changes only
the human-authored surface of a workspace whose v2 graph already exists.  The
graph remains the authority for identity.  A plan therefore contains the
exact YAML bytes to read, the JSON bytes to publish, and the graph revision
objects that would be produced.  Planning never writes either authored files
or graph objects.

Reachable Run and Model bindings are replayed into the schema-2 identity
contract from their immutable snapshots.  Planning previews every replacement
object and binding operation without publication.  Apply remains fail-closed
when any historical snapshot, dependency, tracker default, or lease cannot be
resolved exactly.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hkdl.authoring import record_readers
from hkdl.authoring.authoring import (
    EXPERIMENT_AUXILIARY_DIRECTORIES,
    Authoring,
    _variant_name,
)
from hkdl.authoring.authoring_records import ExperimentRecord, VariantRecord
from hkdl.authoring.config import (
    NAME_PATTERN,
    load_yaml_file,
    validate_experiment,
    validate_variant,
)
from hkdl.authoring.research_json import (
    dump_json,
    load_json_file,
    split_legacy_experiment,
    split_legacy_variant,
    validate_code_json,
    validate_experiment_json,
    validate_options_json,
)
from hkdl.errors import ContractError
from hkdl.execution.run_contracts import (
    TERMINAL_STATUSES,
    validate_tracker,
)
from hkdl.storage.runs import RunStore
from hkdl.storage.settings import (
    WorkspaceSettingsStore,
    normalize_tracker,
    settings_document,
    settings_json,
)
from hkdl.storage.storage import (
    RepositoryPaths,
    atomic_replace,
    atomic_write_new,
    validate_locked_source_tree,
)

from .bindings import BindingOperation
from .cutover import CutoverJournal
from .graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    experiment_variant_scope,
    variant_model_scope,
    variant_run_scope,
    workspace_experiment_scope,
)
from .leases import lease_held
from .maintenance import workspace_access, workspace_operation
from .objects import ObjectRecord, canonical_json_bytes
from .projection import GraphProjection
from .provenance import committed_variant_revision
from .references import extract_references
from .replay import GraphReplay

CURRENT_AUTHORING_SCHEMA_VERSION = 2
AUTHORING_MARKER = ".hkdl/store/AUTHORING_CURRENT"
IMPORT_SCHEMA_VERSION = 1


class AuthoringMigrationConflict(RuntimeError):
    """The authoring migration is unsafe to apply at the current state."""


@dataclass(frozen=True)
class PlannedAuthoringFile:
    """One deterministic source-to-target conversion."""

    source: str
    target: str
    backup: str
    content: bytes
    source_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "backup": self.backup,
            "source_digest": self.source_digest,
            "bytes": len(self.content),
        }


@dataclass(frozen=True)
class AuthoringMigrationReport:
    """JSON-serializable deterministic migration diagnostics."""

    source_schema_version: int
    target_schema_version: int
    experiments: int
    variants: int
    runs: int
    models: int
    results: int
    artifacts: int
    input_bytes: int
    output_bytes: int
    estimated_additional_bytes: int
    transformations: list[dict[str, Any]]
    tracker_conflicts: list[dict[str, Any]]
    tracker_before: str | list[str]
    tracker_after: str | list[str]
    tracker_default: str | list[str] | None
    active_leases: list[str]
    stale_nonterminal: list[str]
    active_bindings: list[str]
    graph_replay_required: bool
    graph_replay_reason: str | None
    malformed_records: list[dict[str, str]]
    object_counts: dict[str, int]
    object_previews: list[dict[str, Any]]
    object_hashes: list[str]
    plan_digest: str
    cutover_ready: bool
    already_current: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_schema_version": self.source_schema_version,
            "target_schema_version": self.target_schema_version,
            "experiments": self.experiments,
            "variants": self.variants,
            "runs": self.runs,
            "models": self.models,
            "results": self.results,
            "artifacts": self.artifacts,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "estimated_additional_bytes": self.estimated_additional_bytes,
            "transformations": deepcopy(self.transformations),
            "tracker_conflicts": deepcopy(self.tracker_conflicts),
            "tracker_before": self.tracker_before,
            "tracker_after": self.tracker_after,
            "tracker_default": self.tracker_default,
            "active_leases": list(self.active_leases),
            "stale_nonterminal": list(self.stale_nonterminal),
            "active_bindings": list(self.active_bindings),
            "graph_replay_required": self.graph_replay_required,
            "graph_replay_reason": self.graph_replay_reason,
            "malformed_records": deepcopy(self.malformed_records),
            "object_counts": dict(self.object_counts),
            "object_previews": deepcopy(self.object_previews),
            "object_hashes": list(self.object_hashes),
            "plan_digest": self.plan_digest,
            "cutover_ready": self.cutover_ready,
            "already_current": self.already_current,
        }


@dataclass(frozen=True)
class AuthoringMigrationPlan:
    """Immutable-input plan plus in-memory graph and file candidates."""

    repository: RepositoryPaths
    files: tuple[PlannedAuthoringFile, ...]
    experiment_records: tuple[ExperimentRecord, ...]
    variant_records: tuple[VariantRecord, ...]
    graph_objects: tuple[ObjectRecord, ...]
    binding_operations: tuple[BindingOperation, ...]
    authority_digests: dict[str, str]
    tracker_target: tuple[str, ...] | None
    report: AuthoringMigrationReport
    blob_contents: dict[str, bytes]

    @property
    def plan_digest(self) -> str:
        return self.report.plan_digest

    @property
    def transformations(self) -> list[dict[str, Any]]:
        return self.report.transformations

    def as_dict(self) -> dict[str, Any]:
        return self.report.as_dict()


@dataclass(frozen=True)
class _LegacyExperiment:
    name: str
    path: Path
    record: ExperimentRecord
    authored: dict[str, Any]
    target: dict[str, Any]


@dataclass(frozen=True)
class _LegacyVariant:
    experiment: str
    name: str
    path: Path
    record: VariantRecord
    code: dict[str, Any]
    options: dict[str, Any]
    tracker: tuple[str, ...]


class AuthoringMigration:
    """Plan and apply schema-1 YAML to schema-2 JSON authoring."""

    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        failure_hook: Callable[[str], None] | None = None,
        tracker_default: str | None = None,
    ):
        self.repository = repository
        self.graph = V2Graph(repository)
        self.runs = RunStore(repository)
        self.settings = WorkspaceSettingsStore(repository.root)
        self._failure_hook = failure_hook
        self._tracker_default = (
            normalize_tracker(tracker_default) if tracker_default is not None else None
        )

    @workspace_operation(legacy=True)
    def plan(self) -> AuthoringMigrationPlan:
        """Build a deterministic, side-effect-free migration plan."""

        marker_state, marker_error = _read_marker(self.repository.root)
        malformed: list[dict[str, str]] = []
        if marker_error is not None:
            malformed.append({"path": AUTHORING_MARKER, "error": marker_error})

        try:
            graph_active = self.graph.is_active()
        except ContractError as error:
            graph_active = False
            malformed.append({"path": ".hkdl/store/CURRENT", "error": str(error)})
        if not graph_active:
            malformed.append(
                {
                    "path": ".hkdl/store/CURRENT",
                    "error": "HKDL v2 must be active before authored migration",
                }
            )

        try:
            tracker_before_tuple = self.settings.load().tracker_backends
            settings_exists = os.path.lexists(self.settings.path)
        except ContractError as error:
            tracker_before_tuple = ("local",)
            settings_exists = True
            malformed.append({"path": ".hkdl/settings.json", "error": str(error)})

        experiments, variants, files, transformations = self._scan_authored(
            malformed,
            current_schema=marker_state == "v2",
        )
        if not malformed:
            for variant in variants:
                try:
                    transformations.append(self._preview_code_parity(variant))
                except (ContractError, OSError) as error:
                    malformed.append(
                        {
                            "path": _relative(self.repository.root, variant.path),
                            "error": f"JSON authoring parity failed: {error}",
                        }
                    )
        active_bindings, run_count, model_count, result_count, artifact_count = (
            self._graph_counts(malformed)
        )
        active_leases, stale_nonterminal = self._leases(active_bindings, malformed)

        tracker_conflicts, tracker_target = _tracker_resolution(
            variants,
            tracker_before_tuple,
            settings_exists=settings_exists,
        )
        if self._tracker_default is not None:
            if marker_state == "v2" and self._tracker_default != tracker_before_tuple:
                tracker_conflicts = [
                    {
                        "path": ".hkdl/settings.json",
                        "error": "authoring is already schema 2; use settings tracker set",
                    }
                ]
                tracker_target = None
            else:
                tracker_conflicts = []
                tracker_target = self._tracker_default
        tracker_after = (
            tracker_target if tracker_target is not None else tracker_before_tuple
        )
        tracker_default = (
            _tracker_display(self._tracker_default)
            if self._tracker_default is not None
            else None
        )

        graph_objects: list[ObjectRecord] = []
        blob_contents: dict[str, bytes] = {}
        replay_operations: list[BindingOperation] = []
        replay = None
        if not malformed:
            try:
                replay = GraphReplay(self.repository)
                replay_operations = replay.preview()
                transformations.append(
                    {
                        "kind": "execution_graph_replay",
                        "runs": len(replay.records),
                        "models": len(replay.models),
                        "operations": len(replay_operations),
                        "status": "replacement_ready",
                    }
                )
                graph_objects = list(replay.objects.values())
                blob_contents = replay.blobs
                transformations.extend(
                    {
                        "kind": "graph_identity",
                        "old_hash": old,
                        "hash": new,
                        "changed": old != new,
                    }
                    for old, new in sorted(replay.mapped.items())
                )
            except (ContractError, KeyError, StopIteration, TypeError) as error:
                replay_operations.clear()
                malformed.append(
                    {"path": "v2/graph", "error": f"graph replay failed: {error}"}
                )
        if tracker_conflicts:
            malformed.extend(
                {
                    "path": str(item.get("path", ".hkdl/settings.json")),
                    "error": str(item["error"]),
                }
                for item in tracker_conflicts
            )

        source_schema = (
            CURRENT_AUTHORING_SCHEMA_VERSION
            if marker_state == "v2"
            else IMPORT_SCHEMA_VERSION
        )
        already_current = (
            marker_state == "v2"
            and not files
            and not malformed
            and not replay_operations
        )
        authority = _authority_digests(self.repository, malformed)
        input_bytes = sum(
            path.stat().st_size
            for item in files
            for path in [self.repository.root / item.source]
            if path.exists()
        )
        output_bytes = sum(len(item.content) for item in files)
        backup_bytes = sum(
            (self.repository.root / source).stat().st_size
            for source in {item.source for item in files}
            if (self.repository.root / source).exists()
        )
        graph_objects = _unique_objects(graph_objects)
        object_counts = dict(
            sorted(Counter(item.kind for item in graph_objects).items())
        )
        object_hashes = sorted(item.digest for item in graph_objects)
        object_bytes = sum(
            len(canonical_json_bytes(item.envelope)) + 1
            for item in graph_objects
            if not os.path.lexists(self.graph.store.object_path(item.digest))
        )
        blob_bytes = sum(
            int(item.payload["size"])
            for item in graph_objects
            if item.kind == "blob"
            and not os.path.lexists(
                self.graph.store.blob_path(str(item.payload["content_hash"]))
            )
        )
        additional = output_bytes + backup_bytes + object_bytes + blob_bytes

        graph_replay_required = bool(active_bindings and (replay is None or malformed))
        replay_reason = (
            "reachable Run/Model bindings require historical graph replay before "
            "authoring conversion can be applied"
            if graph_replay_required
            else None
        )
        # Leases are reported independently so callers can distinguish a live
        # process from an otherwise safe-but-replay-required graph.
        cutover_ready = (
            not malformed
            and not active_leases
            and not graph_replay_required
            and (marker_state in {None, "v1", "v2"})
        )
        digest_payload = {
            "source_schema_version": source_schema,
            "target_schema_version": CURRENT_AUTHORING_SCHEMA_VERSION,
            "files": [item.as_dict() for item in files],
            "transformations": transformations,
            "authority": authority,
            "objects": [
                {"hash": item.digest, "kind": item.kind, "payload": item.payload}
                for item in graph_objects
            ],
            "tracker_before": list(tracker_before_tuple),
            "tracker_after": list(tracker_after),
            "active_bindings": active_bindings,
            "active_leases": active_leases,
            "binding_operations": [item.as_dict() for item in replay_operations],
            "malformed": malformed,
        }
        if tracker_default is not None:
            digest_payload["tracker_default"] = tracker_default
        plan_digest = _sha256_digest(digest_payload)
        report = AuthoringMigrationReport(
            source_schema_version=source_schema,
            target_schema_version=CURRENT_AUTHORING_SCHEMA_VERSION,
            experiments=len(experiments),
            variants=len(variants),
            runs=run_count,
            models=model_count,
            results=result_count,
            artifacts=artifact_count,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            estimated_additional_bytes=additional,
            transformations=transformations,
            tracker_conflicts=tracker_conflicts,
            tracker_before=_tracker_display(tracker_before_tuple),
            tracker_after=_tracker_display(tracker_after),
            tracker_default=tracker_default,
            active_leases=active_leases,
            stale_nonterminal=stale_nonterminal,
            active_bindings=active_bindings,
            graph_replay_required=graph_replay_required,
            graph_replay_reason=replay_reason,
            malformed_records=malformed,
            object_counts=object_counts,
            object_previews=[
                {"hash": item.digest, "kind": item.kind, "payload": item.payload}
                for item in graph_objects
            ],
            object_hashes=object_hashes,
            plan_digest=plan_digest,
            cutover_ready=cutover_ready,
            already_current=already_current,
        )
        return AuthoringMigrationPlan(
            repository=self.repository,
            files=tuple(files),
            experiment_records=tuple(experiment.record for experiment in experiments),
            variant_records=tuple(variant.record for variant in variants),
            graph_objects=tuple(graph_objects),
            binding_operations=tuple(replay_operations),
            authority_digests=authority,
            tracker_target=tracker_target,
            report=report,
            blob_contents=blob_contents,
        )

    def apply(self, plan: AuthoringMigrationPlan) -> AuthoringMigrationReport:
        if plan.repository != self.repository:
            raise ContractError(
                "authoring migration plan belongs to another repository"
            )
        with workspace_access(
            self.repository, exclusive=True, recovery=True, legacy=True
        ):
            recovered = self.recover(expected_plan=plan.plan_digest)
            if recovered is not None and recovered["outcome"] == "completed":
                return plan.report
            return self._apply(plan)

    def recover(self, *, expected_plan: str | None = None):
        with workspace_access(
            self.repository, exclusive=True, recovery=True, legacy=True
        ):
            journal = CutoverJournal(self.repository)
            if journal.path.exists() and expected_plan is not None:
                if _load_json(journal.path).get("plan_digest") != expected_plan:
                    raise AuthoringMigrationConflict(
                        "unfinished migration does not match the approved plan"
                    )
            result = journal.recover()
            if result is not None:
                GraphProjection(self.graph).rebuild()
            return result

    def _apply(self, plan: AuthoringMigrationPlan) -> AuthoringMigrationReport:
        """Apply a previously planned conversion with marker-last semantics."""

        if plan.repository != self.repository:
            raise ContractError(
                "authoring migration plan belongs to another repository"
            )
        verified = self.plan()
        if plan.plan_digest != verified.plan_digest or any(
            getattr(plan, field) != getattr(verified, field)
            for field in (
                "files",
                "graph_objects",
                "binding_operations",
                "authority_digests",
                "tracker_target",
                "blob_contents",
            )
        ):
            raise AuthoringMigrationConflict(
                "migration authority or approved plan changed"
            )
        # Apply freshly reconstructed candidates, never caller-owned mutable payloads.
        plan = verified
        if not plan.report.cutover_ready:
            if plan.report.active_leases:
                raise AuthoringMigrationConflict(
                    "active Run lease blocks authoring migration"
                )
            if plan.report.graph_replay_required:
                raise AuthoringMigrationConflict(
                    plan.report.graph_replay_reason
                    or "historical graph replay is required before migration"
                )
            raise ContractError("malformed authoring records block migration")

        marker_state, marker_error = _read_marker(self.repository.root)
        if marker_error is not None:
            raise AuthoringMigrationConflict("authoring marker changed after planning")
        current = _authority_digests(self.repository, [])
        if current != plan.authority_digests:
            raise AuthoringMigrationConflict(
                "authoring authority changed after planning"
            )
        if plan.report.already_current:
            return plan.report
        leases, _ = self._leases(plan_active_bindings(plan), [])
        if leases:
            raise AuthoringMigrationConflict(
                "active Run lease blocks authoring migration"
            )

        backup_root = (
            self.repository.root
            / ".hkdl/backups/authoring-v1"
            / plan.report.plan_digest.removeprefix("sha256:")
        )
        staged_root = (
            self.repository.root
            / ".hkdl/store/.authoring-candidates"
            / plan.report.plan_digest.removeprefix("sha256:")
        )
        published: list[Path] = []
        removed_yaml: list[tuple[Path, bytes]] = []
        settings_after = (
            settings_json(
                settings_document(_tracker_display(plan.tracker_target))
            ).encode()
            if plan.tracker_target is not None
            else None
        )
        journal = CutoverJournal(self.repository)
        before_graph = _graph_parity(self.repository)
        journal.prepare(plan, settings_after)
        try:
            self._phase("after_journal")
            self._backup_sources(plan, backup_root)
            self._phase("after_backup")
            self._stage_candidates(plan, staged_root)
            self._phase("after_candidates")
            self._publish_candidates(plan, staged_root, published)
            self._phase("after_json")

            self._commit_graph(plan)
            self._phase("after_graph")
            GraphProjection(self.graph).rebuild()
            self._phase("after_projection")

            if plan.tracker_target is not None:
                self.settings.set_tracker(_tracker_display(plan.tracker_target))
            self._remove_yaml_sources(plan, removed_yaml)
            self._phase("after_yaml")
            _assert_outputs_unchanged(self.repository, plan.authority_digests)
            if _graph_parity(self.repository) != before_graph:
                raise AuthoringMigrationConflict(
                    "Run/Model/result parity changed during migration"
                )
            self._validate_authored_parity(plan)
            self._validate_code_parity(plan)
            self._phase("before_marker")
            self._publish_marker()
            self._phase("after_marker")
            journal.recover()
            return plan.report
        except BaseException:
            journal.recover()
            GraphProjection(self.graph).rebuild()
            raise
        finally:
            if staged_root.exists():
                shutil.rmtree(staged_root)

    def _scan_authored(
        self,
        malformed: list[dict[str, str]],
        *,
        current_schema: bool = False,
    ) -> tuple[
        list[_LegacyExperiment],
        list[_LegacyVariant],
        list[PlannedAuthoringFile],
        list[dict[str, Any]],
    ]:
        experiments: list[_LegacyExperiment] = []
        variants: list[_LegacyVariant] = []
        files: list[PlannedAuthoringFile] = []
        transformations: list[dict[str, Any]] = []
        root = self.repository.experiments
        if not os.path.lexists(root):
            return experiments, variants, files, transformations
        if _is_symlink(root) or not root.is_dir():
            malformed.append(
                {"path": "experiments", "error": "must be a real directory"}
            )
            return experiments, variants, files, transformations

        for entry in sorted(root.iterdir(), key=lambda item: item.name.encode("utf-8")):
            if entry.name.startswith("."):
                continue
            if _is_symlink(entry) or not entry.is_dir():
                malformed.append(
                    {
                        "path": _relative(self.repository.root, entry),
                        "error": "Experiment entry must be a real directory",
                    }
                )
                continue
            if not NAME_PATTERN.fullmatch(entry.name):
                malformed.append(
                    {
                        "path": _relative(self.repository.root, entry),
                        "error": "invalid Experiment name",
                    }
                )
                continue
            experiment = self._scan_experiment(
                entry,
                malformed,
                current_schema=current_schema,
            )
            if experiment is None:
                continue
            experiments.append(experiment)
            if not current_schema:
                files.append(
                    _planned_file(
                        self.repository.root,
                        entry / "experiment.yaml",
                        entry / "experiment.json",
                        _json_bytes(experiment.target),
                        root_suffix=f"experiments/{entry.name}",
                    )
                )
                transformations.append(
                    {
                        "kind": "experiment",
                        "source": _relative(
                            self.repository.root, entry / "experiment.yaml"
                        ),
                        "targets": [
                            _relative(self.repository.root, entry / "experiment.json")
                        ],
                        "status": "yaml_to_json",
                    }
                )
            for child in sorted(
                entry.iterdir(), key=lambda item: item.name.encode("utf-8")
            ):
                if child.name.startswith(".") or child.name in {
                    "experiment.yaml",
                    "experiment.json",
                    "notes",
                    *EXPERIMENT_AUXILIARY_DIRECTORIES,
                }:
                    continue
                if _is_symlink(child) or not child.is_dir():
                    malformed.append(
                        {
                            "path": _relative(self.repository.root, child),
                            "error": "Variant entry must be a real directory",
                        }
                    )
                    continue
                variant = self._scan_variant(
                    entry.name,
                    child,
                    experiment,
                    malformed,
                    current_schema=current_schema,
                )
                if variant is None:
                    continue
                variants.append(variant)
                if not current_schema:
                    files.extend(
                        [
                            _planned_file(
                                self.repository.root,
                                child / "variant.yaml",
                                child / "code.json",
                                _json_bytes(variant.code),
                                root_suffix=f"experiments/{entry.name}/{child.name}",
                            ),
                            _planned_file(
                                self.repository.root,
                                child / "variant.yaml",
                                child / "options.json",
                                _json_bytes(variant.options),
                                root_suffix=f"experiments/{entry.name}/{child.name}",
                            ),
                        ]
                    )
                    transformations.append(
                        {
                            "kind": "variant",
                            "source": _relative(
                                self.repository.root, child / "variant.yaml"
                            ),
                            "targets": [
                                _relative(self.repository.root, child / "code.json"),
                                _relative(self.repository.root, child / "options.json"),
                            ],
                            "status": "yaml_to_json",
                            "tracker": _tracker_display(variant.tracker),
                        }
                    )
        return experiments, variants, files, transformations

    def _scan_experiment(
        self,
        path: Path,
        malformed: list[dict[str, str]],
        *,
        current_schema: bool = False,
    ) -> _LegacyExperiment | None:
        for name in ("notes", *EXPERIMENT_AUXILIARY_DIRECTORIES):
            candidate = path / name
            if os.path.lexists(candidate) and (
                _is_symlink(candidate) or not candidate.is_dir()
            ):
                malformed.append(
                    {
                        "path": _relative(self.repository.root, candidate),
                        "error": "auxiliary entry must be a real directory",
                    }
                )
        yaml_path = path / "experiment.yaml"
        json_path = path / "experiment.json"
        if current_schema and os.path.lexists(yaml_path):
            malformed.append(
                {
                    "path": _relative(self.repository.root, yaml_path),
                    "error": "schema 2 marker cannot coexist with legacy YAML authoring",
                }
            )
            return None
        if os.path.lexists(json_path) and not os.path.lexists(yaml_path):
            if not current_schema:
                malformed.append(
                    {
                        "path": _relative(self.repository.root, json_path),
                        "error": "schema 2 Experiment is already present or layout is mixed",
                    }
                )
                return None
            try:
                authored = _load_json(json_path)
                validate_experiment_json(authored)
                document = _runtime_experiment_document(
                    path.name,
                    authored,
                    self._graph_created_at(path.name),
                )
                _validate_tree(path / "notes", required=True)
            except (ContractError, OSError) as error:
                malformed.append(
                    {
                        "path": _relative(self.repository.root, json_path),
                        "error": str(error),
                    }
                )
                return None
            return _LegacyExperiment(
                path.name, path, ExperimentRecord(path, document, 2), authored, authored
            )
        if os.path.lexists(json_path):
            malformed.append(
                {
                    "path": _relative(self.repository.root, json_path),
                    "error": "schema 2 Experiment is already present or layout is mixed",
                }
            )
        if not os.path.lexists(yaml_path):
            malformed.append(
                {
                    "path": _relative(self.repository.root, yaml_path),
                    "error": "Experiment experiment.yaml is missing",
                }
            )
            return None
        try:
            authored = _load_yaml(yaml_path)
            validate_experiment(authored, expected_name=path.name)
            target = split_legacy_experiment(authored)
            validate_experiment_json(target)
            _validate_tree(path / "notes", required=True)
        except (ContractError, OSError) as error:
            malformed.append(
                {
                    "path": _relative(self.repository.root, yaml_path),
                    "error": str(error),
                }
            )
            return None
        return _LegacyExperiment(
            path.name,
            path,
            ExperimentRecord(path, authored, 1),
            authored,
            target,
        )

    def _scan_variant(
        self,
        experiment_name: str,
        path: Path,
        experiment: _LegacyExperiment,
        malformed: list[dict[str, str]],
        *,
        current_schema: bool = False,
    ) -> _LegacyVariant | None:
        try:
            _variant_name(path.name)
        except ContractError as error:
            malformed.append(
                {"path": _relative(self.repository.root, path), "error": str(error)}
            )
            return None
        yaml_path = path / "variant.yaml"
        json_paths = [path / "code.json", path / "options.json"]
        has_json = any(os.path.lexists(item) for item in json_paths)
        has_yaml = os.path.lexists(yaml_path)
        if current_schema and has_yaml:
            malformed.append(
                {
                    "path": _relative(self.repository.root, yaml_path),
                    "error": "schema 2 marker cannot coexist with legacy YAML authoring",
                }
            )
            return None
        if has_json and not has_yaml and current_schema:
            if not all(os.path.lexists(item) for item in json_paths):
                malformed.append(
                    {
                        "path": _relative(self.repository.root, path),
                        "error": "Variant JSON authoring requires code.json and options.json",
                    }
                )
                return None
            try:
                code = _load_json(path / "code.json")
                options = _load_json(path / "options.json")
                validate_code_json(code)
                validate_options_json(options)
                if code["template"]["name"] != experiment.authored["template"]["name"]:
                    raise ContractError(
                        "Variant Template family does not match its Experiment"
                    )
                validate_locked_source_tree(path / "src")
                record = record_readers.variant_json_record(
                    self.repository,
                    path,
                    experiment_name=experiment.name,
                    expected_name=path.name,
                    code=code,
                    options=options,
                )
                tracker = self.settings.load().tracker_backends
            except (ContractError, OSError) as error:
                malformed.append(
                    {"path": _relative(self.repository.root, path), "error": str(error)}
                )
                return None
            return _LegacyVariant(
                experiment.name,
                path.name,
                path,
                record,
                code,
                options,
                tracker,
            )
        if has_json:
            malformed.append(
                {
                    "path": _relative(self.repository.root, path),
                    "error": "schema 2 Variant is already present or layout is mixed",
                }
            )
        if not has_yaml:
            malformed.append(
                {
                    "path": _relative(self.repository.root, yaml_path),
                    "error": "Variant variant.yaml is missing",
                }
            )
            return None
        try:
            authored = _load_yaml(yaml_path)
            validate_variant(authored, expected_name=path.name)
            split = split_legacy_variant(authored)
            validate_code_json(split.code)
            validate_options_json(split.options)
            if authored["template"]["name"] != experiment.authored["template"]["name"]:
                raise ContractError(
                    "Variant Template family does not match its Experiment"
                )
            validate_locked_source_tree(path / "src")
            tracker = normalize_tracker(authored["tracker"])
        except (ContractError, OSError) as error:
            malformed.append(
                {
                    "path": _relative(self.repository.root, yaml_path),
                    "error": str(error),
                }
            )
            return None
        record = VariantRecord(
            path,
            experiment_name,
            authored,
            2,
            deepcopy(split.code),
            deepcopy(split.options),
        )
        return _LegacyVariant(
            experiment_name,
            path.name,
            path,
            record,
            split.code,
            split.options,
            tracker,
        )

    def _graph_created_at(self, experiment_name: str) -> str:
        try:
            entity = self.graph.experiment_hash(experiment_name)
            return str(self.graph.store.load(entity).payload["created_at"])
        except (ContractError, KeyError, TypeError):
            return "1970-01-01T00:00:00+00:00"

    def _code_parity(self, record):
        current = committed_variant_revision(
            self.repository, record.experiment, record.document["name"]
        )
        semantic = {
            "template": record.document["template"],
            "components": record.document["components"],
            "source_tree": self.graph.capture_source_tree(
                record.path / "src", publish=False
            ).digest,
        }
        committed = {field: current[field] for field in semantic} if current else None
        return {
            "kind": "authoring_code_parity",
            "variant": f"{record.experiment}/{record.document['name']}",
            "code_hash": _sha256_digest(semantic),
            "changed": semantic != committed,
        }

    def _preview_code_parity(self, variant):
        expected = self._code_parity(variant.record)
        candidate = record_readers.variant_json_record(
            self.repository,
            variant.path,
            experiment_name=variant.experiment,
            expected_name=variant.name,
            code=variant.code,
            options=variant.options,
        )
        if self._code_parity(candidate) != expected:
            raise ContractError("JSON candidate changes Code meaning or dirty state")
        return expected

    def _validate_code_parity(self, plan):
        authoring = Authoring(self.repository)
        for item in plan.transformations:
            if item["kind"] != "authoring_code_parity":
                continue
            experiment, variant = item["variant"].split("/")
            actual = self._code_parity(authoring.load_variant(experiment, variant))
            if actual != item:
                raise AuthoringMigrationConflict(
                    f"published JSON changes Code meaning or dirty state: {item['variant']}"
                )

    def _graph_counts(
        self,
        malformed: list[dict[str, str]],
    ) -> tuple[list[str], int, int, int, int]:
        active: list[str] = []
        runs = models = results = artifacts = 0
        try:
            experiment_bindings = self.graph.bindings.names(
                workspace_experiment_scope()
            )
        except ContractError as error:
            malformed.append({"path": "v2/refs/bindings/HEAD", "error": str(error)})
            return active, runs, models, results, artifacts
        for experiment_name, experiment_hash in sorted(
            experiment_bindings.items(), key=lambda item: item[0].encode("utf-8")
        ):
            try:
                variants = self.graph.bindings.names(
                    experiment_variant_scope(experiment_hash)
                )
            except ContractError as error:
                malformed.append(
                    {"path": f"experiment/{experiment_name}", "error": str(error)}
                )
                continue
            for variant_name, variant_hash in sorted(
                variants.items(), key=lambda item: item[0].encode("utf-8")
            ):
                try:
                    run_bindings = self.graph.bindings.names(
                        variant_run_scope(variant_hash)
                    )
                    model_bindings = self.graph.bindings.names(
                        variant_model_scope(variant_hash)
                    )
                except ContractError as error:
                    malformed.append(
                        {
                            "path": f"{experiment_name}/{variant_name}",
                            "error": str(error),
                        }
                    )
                    continue
                runs += len(run_bindings)
                models += len(model_bindings)
                for run_name, attempt_hash in sorted(run_bindings.items()):
                    active.append(f"{experiment_name}/{variant_name}/{run_name}")
                    try:
                        attempt = self.graph.store.load(attempt_hash)
                        if attempt.kind != "attempt":
                            raise ContractError(
                                "Run binding does not reference an Attempt"
                            )
                        event_scope = (
                            f"attempt/{attempt_hash.removeprefix('sha256:')}/events"
                        )
                        event_hash = self.graph.bindings.resolve(
                            event_scope, CURRENT_REVISION_NAME
                        )
                        event = self.graph.store.load(event_hash)
                        status = event.payload.get("status")
                        if status in {"done"}:
                            result = event.payload.get("result_object")
                            if result:
                                results += 1
                            artifacts += len(event.payload.get("artifacts", []))
                        elif status in {"failed", "interrupted", "abandoned"}:
                            artifacts += len(event.payload.get("artifacts", []))
                    except (ContractError, KeyError, TypeError) as error:
                        malformed.append(
                            {
                                "path": f"{experiment_name}/{variant_name}/{run_name}",
                                "error": str(error),
                            }
                        )
                for model_name, model_hash in sorted(model_bindings.items()):
                    active.append(f"{experiment_name}/{variant_name}/{model_name}")
                    try:
                        model = self.graph.store.load(model_hash)
                        if model.kind != "model":
                            raise ContractError(
                                "Model binding does not reference a Model"
                            )
                    except ContractError as error:
                        malformed.append(
                            {
                                "path": f"{experiment_name}/{variant_name}/{model_name}",
                                "error": str(error),
                            }
                        )
        return sorted(set(active)), runs, models, results, artifacts

    def _leases(
        self,
        active_bindings: list[str],
        malformed: list[dict[str, str]],
    ) -> tuple[list[str], list[str]]:
        active_leases: list[str] = []
        stale: list[str] = []
        for address in active_bindings:
            parts = address.split("/")
            if len(parts) != 3 or not parts[2].startswith("run-"):
                continue
            experiment, variant, run_id = parts
            try:
                record = self.runs.load(experiment, variant, run_id)
            except (ContractError, OSError) as error:
                malformed.append({"path": f"outputs/{address}", "error": str(error)})
                continue
            path = record.path if record.path.exists() else None
            if lease_held(self.repository, record.graph_identity["attempt_hash"], path):
                active_leases.append(address)
            elif record.state["status"] not in TERMINAL_STATUSES:
                stale.append(address)
                if path is None:
                    malformed.append(
                        {
                            "path": f"outputs/{address}",
                            "error": "nonterminal Run working directory is unavailable",
                        }
                    )
        return sorted(active_leases), sorted(stale)

    def _backup_sources(self, plan: AuthoringMigrationPlan, root: Path) -> None:
        root.mkdir(mode=0o755, parents=True, exist_ok=True)
        if _is_symlink(root) or not root.is_dir():
            raise ContractError("authoring backup root must be a real directory")
        for item in plan.files:
            source = self.repository.root / item.source
            target = root / item.source
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            if os.path.lexists(target):
                if (
                    _is_symlink(target)
                    or not target.is_file()
                    or target.read_bytes() != source.read_bytes()
                ):
                    raise AuthoringMigrationConflict(
                        f"backup target disagrees: {target}"
                    )
                continue
            _copy_fsync(source, target)
        manifest = root / "manifest.json"
        data = {
            "schema_version": 1,
            "plan_digest": plan.plan_digest,
            "files": [item.as_dict() for item in plan.files],
        }
        if not os.path.lexists(manifest):
            atomic_write_new(manifest, _json_bytes(data))

    def _stage_candidates(self, plan: AuthoringMigrationPlan, root: Path) -> None:
        root.mkdir(mode=0o755, parents=True, exist_ok=True)
        for item in plan.files:
            target = root / item.target
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            if not os.path.lexists(target):
                atomic_write_new(target, item.content)
            elif target.read_bytes() != item.content:
                raise AuthoringMigrationConflict(f"candidate disagrees: {target}")

    def _publish_candidates(
        self,
        plan: AuthoringMigrationPlan,
        root: Path,
        published: list[Path],
    ) -> None:
        for item in plan.files:
            destination = self.repository.root / item.target
            if os.path.lexists(destination):
                raise AuthoringMigrationConflict(
                    f"JSON target already exists: {destination}"
                )
            candidate = root / item.target
            atomic_write_new(destination, candidate.read_bytes())
            published.append(destination)

    def _commit_graph(self, plan: AuthoringMigrationPlan) -> None:
        pending = {record.digest: record for record in plan.graph_objects}
        existing = {record.digest: record for record in self.graph.store.iter_records()}
        references = extract_references(list({**existing, **pending}.values()))
        dependencies = {digest: set() for digest in pending}
        for reference in references:
            if reference.source in dependencies:
                dependencies[reference.source].add(reference.target)
        ordered = []
        available = set(existing)
        while pending:
            ready = sorted(
                digest for digest in pending if dependencies[digest] <= available
            )
            if not ready:
                raise ContractError(
                    "migration object dependencies are cyclic or missing"
                )
            for digest in ready:
                ordered.append(pending.pop(digest))
                available.add(digest)
        for record in ordered:
            if record.kind == "blob":
                if record.digest in plan.blob_contents:
                    published = self.graph.store.put_blob_bytes(
                        plan.blob_contents[record.digest],
                        media_type=record.payload["media_type"],
                    )
                    if published.digest != record.digest:
                        raise ContractError("migration evidence hash changed")
                else:
                    self.graph.store.verify_blob(record.digest)
            else:
                self.graph.store.put(record.kind, record.payload)
        if plan.binding_operations:
            self.graph.bindings.commit(list(plan.binding_operations))

    def _validate_authored_parity(self, plan):
        expected = dict(plan.authority_digests)
        for item in plan.files:
            expected.pop(item.source, None)
            expected[item.target] = "sha256:" + hashlib.sha256(item.content).hexdigest()
        if plan.tracker_target is not None:
            expected[".hkdl/settings.json"] = _file_digest(self.settings.path)
        head = self.graph.bindings.head_path
        if head.exists():
            expected[".hkdl/store/v2/refs/bindings/HEAD"] = _file_digest(head)
        malformed = []
        actual = _authority_digests(self.repository, malformed)
        if malformed or actual != expected:
            raise AuthoringMigrationConflict(
                "research/source bytes changed during migration"
            )

    def _remove_yaml_sources(
        self,
        plan: AuthoringMigrationPlan,
        removed: list[tuple[Path, bytes]],
    ) -> None:
        seen: set[Path] = set()
        for item in plan.files:
            source = self.repository.root / item.source
            if source in seen:
                continue
            seen.add(source)
            content = source.read_bytes()
            removed.append((source, content))
            source.unlink()
            _fsync_directory(source.parent)

    def _publish_marker(self) -> None:
        marker = self.repository.root / AUTHORING_MARKER
        marker.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if os.path.lexists(marker):
            atomic_replace(marker, "v2\n")
        else:
            atomic_write_new(marker, "v2\n")

    def _phase(self, name: str) -> None:
        if self._failure_hook is not None:
            self._failure_hook(name)


# Compatibility name for callers that describe the operation as a migration
# service rather than an authoring migration.
WorkspaceAuthoringMigration = AuthoringMigration
MigrationConflict = AuthoringMigrationConflict


def plan_active_bindings(plan: AuthoringMigrationPlan) -> list[str]:
    """Return the active binding addresses captured by a plan."""

    return list(plan.report.active_bindings)


def _graph_parity(repository):
    from hkdl.storage.status import Status

    from .reader import GraphReader

    reader = GraphReader(repository)
    records = {}
    for run in reader.runs():
        snapshot = deepcopy(run.snapshot)
        snapshot["variant"]["tracker"] = list(
            validate_tracker(snapshot["variant"]["tracker"])
        )
        records[run.address] = {
            "snapshot": snapshot,
            "request": run.request,
            "state": run.state,
        }
    return {
        "status": Status(
            repository, now=lambda: datetime(2000, 1, 1, tzinfo=timezone.utc)
        ).query(),
        "records": records,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    return load_json_file(path)


def _runtime_experiment_document(
    name: str,
    authored: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    document = {
        "schema_version": 1,
        "name": name,
        "type": deepcopy(authored["type"]),
        "question": deepcopy(authored["question"]),
        "template": deepcopy(authored["template"]),
        "created_at": created_at,
    }
    validate_experiment(document, expected_name=name)
    return document


def _json_bytes(document: dict[str, Any]) -> bytes:
    return dump_json(document).encode("utf-8")


def _planned_file(
    root: Path,
    source: Path,
    target: Path,
    content: bytes,
    *,
    root_suffix: str,
) -> PlannedAuthoringFile:
    del root_suffix
    source_digest = _file_digest(source)
    return PlannedAuthoringFile(
        source=_relative(root, source),
        target=_relative(root, target),
        backup=f".hkdl/backups/authoring-v1/<plan-digest>/{_relative(root, source)}",
        content=content,
        source_digest=source_digest,
    )


def _tracker_resolution(
    variants: list[_LegacyVariant],
    workspace: tuple[str, ...],
    *,
    settings_exists: bool,
) -> tuple[list[dict[str, Any]], tuple[str, ...] | None]:
    values = sorted({variant.tracker for variant in variants})
    conflicts: list[dict[str, Any]] = []
    if len(values) > 1:
        conflicts.append(
            {
                "path": "experiments",
                "error": "legacy Variants disagree on tracker backend",
                "values": [_tracker_display(item) for item in values],
            }
        )
        return conflicts, None
    if not values:
        return [], None
    selected = values[0]
    if settings_exists and selected != workspace:
        conflicts.append(
            {
                "path": ".hkdl/settings.json",
                "error": "legacy tracker conflicts with workspace default",
                "legacy": _tracker_display(selected),
                "workspace": _tracker_display(workspace),
            }
        )
        return conflicts, None
    return [], selected


def _tracker_display(value: tuple[str, ...]) -> str | list[str]:
    if not value:
        return "none"
    if len(value) == 1:
        return value[0]
    return list(value)


def _unique_objects(objects: list[ObjectRecord]) -> list[ObjectRecord]:
    by_digest: dict[str, ObjectRecord] = {}
    for record in objects:
        existing = by_digest.get(record.digest)
        if existing is not None and (
            existing.kind != record.kind or existing.payload != record.payload
        ):
            raise ContractError("authoring migration object hash collision")
        by_digest[record.digest] = record
    return [by_digest[digest] for digest in sorted(by_digest)]


def _read_marker(root: Path) -> tuple[str | None, str | None]:
    path = root / AUTHORING_MARKER
    if not os.path.lexists(path):
        return None, None
    try:
        metadata = path.lstat()
        if _is_symlink(path) or not stat.S_ISREG(metadata.st_mode):
            return None, "authoring marker must be a regular non-symlink file"
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        return None, f"authoring marker is unavailable: {error}"
    if content == "v1\n":
        return "v1", None
    if content == "v2\n":
        return "v2", None
    return None, "authoring marker format is unsupported"


def _validate_tree(path: Path, *, required: bool) -> None:
    if not os.path.lexists(path):
        if required:
            raise ContractError(f"required directory is unavailable: {path}")
        return
    if _is_symlink(path) or not path.is_dir():
        raise ContractError(f"directory is invalid: {path}")
    for entry in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if _is_symlink(entry):
            raise ContractError(f"symlinks are not allowed: {entry}")
        if not entry.is_dir() and not entry.is_file():
            raise ContractError(f"non-regular entry: {entry}")


def _authority_digests(
    repository: RepositoryPaths,
    malformed: list[dict[str, str]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    roots = [repository.experiments, repository.outputs]
    for relative in (".hkdl/store/CURRENT", ".hkdl/store/v2/refs/bindings/HEAD"):
        if os.path.lexists(repository.root / relative):
            roots.append(repository.root / relative)
    marker = repository.root / AUTHORING_MARKER
    if os.path.lexists(marker):
        roots.append(marker)
    settings = repository.root / ".hkdl/settings.json"
    if os.path.lexists(settings):
        roots.append(settings)
    for root in roots:
        if not os.path.lexists(root):
            continue
        if _is_symlink(root):
            malformed.append(
                {
                    "path": _relative(repository.root, root),
                    "error": "symlink is unsupported",
                }
            )
            continue
        if root.is_file():
            result[_relative(repository.root, root)] = _file_digest(root)
            continue
        if not root.is_dir():
            malformed.append(
                {
                    "path": _relative(repository.root, root),
                    "error": "authority root is not a directory",
                }
            )
            continue
        for path in sorted(
            root.rglob("*"), key=lambda item: _relative(repository.root, item)
        ):
            relative = _relative(repository.root, path)
            try:
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    malformed.append(
                        {"path": relative, "error": "symlink is unsupported"}
                    )
                elif stat.S_ISREG(metadata.st_mode):
                    result[relative] = _file_digest(path)
                elif not stat.S_ISDIR(metadata.st_mode):
                    malformed.append(
                        {"path": relative, "error": "non-regular authority entry"}
                    )
            except OSError as error:
                malformed.append({"path": relative, "error": str(error)})
    return result


def _assert_outputs_unchanged(
    repository: RepositoryPaths, before: dict[str, str]
) -> None:
    after: dict[str, str] = {}
    root = repository.outputs
    if os.path.lexists(root) and root.is_dir() and not _is_symlink(root):
        for path in sorted(
            root.rglob("*"), key=lambda item: _relative(repository.root, item)
        ):
            if path.is_file() and not _is_symlink(path):
                after[_relative(repository.root, path)] = _file_digest(path)
    previous = {
        key: value
        for key, value in before.items()
        if key == "outputs" or key.startswith("outputs/")
    }
    if after != previous:
        raise AuthoringMigrationConflict("outputs changed during authoring migration")


def _file_digest(path: Path) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _sha256_digest(document: dict[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(document)).hexdigest()}"


def _migration_nonce(address: str) -> str:
    return hashlib.sha256(f"hkdl-authoring-v2:{address}".encode("utf-8")).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.absolute().relative_to(root.absolute()).as_posix()


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(path.lstat().st_mode)
    except OSError:
        return False


def _copy_fsync(source: Path, target: Path) -> None:
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(
            f"backup source must be a regular non-symlink file: {source}"
        )
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    directory_descriptor: int | None = None
    try:
        candidate = target.parent / f".{target.name}.candidate-{os.getpid()}"
        output = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with (
                os.fdopen(descriptor, "rb") as reader,
                os.fdopen(output, "wb") as writer,
            ):
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            os.link(candidate, target)
            directory_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            os.fsync(directory_descriptor)
        finally:
            candidate.unlink(missing_ok=True)
    except BaseException:
        # The descriptor is owned by fdopen on the successful path; close it
        # here only when opening the output or linking failed before that.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AUTHORING_MARKER",
    "AuthoringMigration",
    "AuthoringMigrationConflict",
    "AuthoringMigrationPlan",
    "AuthoringMigrationReport",
    "MigrationConflict",
    "PlannedAuthoringFile",
    "WorkspaceAuthoringMigration",
]
