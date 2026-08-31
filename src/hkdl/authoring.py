"""Template, Experiment, and Variant authoring operations."""

from __future__ import annotations

import copy
import os
import secrets
import shutil
import stat
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import (
    ContractError,
    NAME_PATTERN,
    dump_yaml,
    load_yaml_file,
    validate_experiment,
    validate_variant,
)
from .v2.maintenance import BOOTSTRAP_JOURNAL, workspace_access, workspace_operation
from .v2.leases import authoring_write
from .research_json import (
    dump_json,
    load_json_file,
    split_legacy_variant,
    validate_code_json,
    validate_experiment_json,
    validate_options_json,
)
from .storage import (
    AlreadyExistsError,
    NotFoundError,
    OwnershipError,
    RepositoryPaths,
    ResolvedTemplate,
    TemplateResolver,
    atomic_write_new,
    compute_bundle_digest,
    compute_source_digest,
    publish_directory,
    validate_locked_source_tree,
    validate_repository_root,
)

EXPERIMENT_AUXILIARY_DIRECTORIES = ("docs", "tools")
RESERVED_VARIANT_NAMES = frozenset(
    {"notes", "src", "variants", *EXPERIMENT_AUXILIARY_DIRECTORIES}
)


@dataclass(frozen=True)
class ExperimentRecord:
    path: Path
    document: dict[str, object]
    authored_schema_version: int = 1


@dataclass(frozen=True)
class VariantRecord:
    path: Path
    experiment: str
    document: dict[str, object]
    authored_schema_version: int = 1
    code_document: dict[str, object] | None = None
    options_document: dict[str, object] | None = None


class Authoring:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
    ):
        self.repository = repository
        self.templates = TemplateResolver(repository)
        self._now = now or (lambda: datetime.now(timezone.utc))

    def list_templates(self) -> list[ResolvedTemplate]:
        return self.templates.list()

    def show_template(self, reference: str) -> ResolvedTemplate:
        return self.templates.resolve(reference)

    def create_experiment(self, name: str, template_name: str) -> ExperimentRecord:
        from .v2.bootstrap import recover_bootstrap

        root = self.repository.root
        pending = os.path.lexists(root / BOOTSTRAP_JOURNAL)
        with workspace_access(
            self.repository,
            exclusive=pending or not os.path.lexists(root / ".hkdl/store/CURRENT"),
            bootstrap=True,
        ):
            recovered = recover_bootstrap(self, name, template_name)
            if recovered is not None:
                return recovered
            return self._create_experiment(name, template_name)

    def _create_experiment(self, name: str, template_name: str) -> ExperimentRecord:
        _name(name, "Experiment")
        _name(template_name, "Template")
        experiments = self._ensure_experiment_catalog()
        target = experiments / name
        if os.path.lexists(target):
            raise AlreadyExistsError(f"Experiment already exists: {name}")
        template = self.templates.latest(template_name)
        document = {
            "schema_version": 1,
            "name": name,
            "type": template.experiment_seed["type"],
            "question": template.experiment_seed["question"],
            "template": {"name": template.manifest["name"]},
            "created_at": _timestamp(self._now()),
        }
        use_json = self._use_json_authoring(template)
        if use_json:
            from .v2.bootstrap import bootstrap_experiment
            from .v2.graph import V2Graph

            if not V2Graph(self.repository).is_active():
                return bootstrap_experiment(
                    self, ExperimentRecord(target, document, 2), target
                )

        candidate = Path(
            tempfile.mkdtemp(prefix=f".{name}.candidate-", dir=experiments)
        )
        try:
            (candidate / "notes").mkdir()

            if use_json:
                authored = {
                    "schema_version": 2,
                    "type": document["type"],
                    "question": document["question"],
                    "template": document["template"],
                }
                validate_experiment_json(authored)
                atomic_write_new(candidate / "experiment.json", dump_json(authored))
            else:
                atomic_write_new(candidate / "experiment.yaml", dump_yaml(document))

            record = (
                ExperimentRecord(candidate, document, 2)
                if use_json
                else _load_experiment(candidate, expected_name=name)
            )
            if record.document["type"] != template.manifest["type"]:
                raise ContractError("copied Experiment type changed")
            if record.document["template"] != {"name": template.manifest["name"]}:
                raise ContractError("copied Experiment provenance changed")

            publish_directory(candidate, target)
            created = ExperimentRecord(
                target, record.document, record.authored_schema_version
            )
            if use_json:
                self._activate_json_authoring()
            _commit_created_experiment(self, created)
            return created
        finally:
            if candidate.exists():
                shutil.rmtree(candidate)

    @workspace_operation
    def list_experiments(self) -> list[ExperimentRecord]:
        if not os.path.lexists(self.repository.experiments):
            return []
        records = [
            self.load_experiment(entry.name)
            for entry in _catalog_entries(self.repository.experiments)
        ]
        return sorted(
            records,
            key=lambda record: record.document["name"].encode("utf-8"),
        )

    @workspace_operation
    def load_experiment(self, name: str) -> ExperimentRecord:
        _name(name, "Experiment")
        if os.path.lexists(self.repository.experiments):
            _require_directory(self.repository.experiments)
        path = self.repository.experiments / name
        if not os.path.lexists(path):
            raise NotFoundError(f"Experiment not found: {name}")
        return _load_experiment(path, expected_name=name)

    @workspace_operation
    @authoring_write
    def create_variant(
        self,
        experiment_name: str,
        variant_name: str,
        *,
        template_version: str | None = None,
    ) -> VariantRecord:
        _variant_name(variant_name)
        experiment = self.load_experiment(experiment_name)
        _check_variant_creation(self, experiment, variant_name)
        target = experiment.path / variant_name
        if os.path.lexists(target):
            raise AlreadyExistsError(
                f"Variant already exists: {experiment_name}/{variant_name}"
            )

        template_name = str(experiment.document["template"]["name"])
        if template_version is None:
            template = self.templates.latest(template_name)
        else:
            template = self.templates.resolve(f"{template_name}@{template_version}")
        if template.manifest["type"] != experiment.document["type"]:
            raise ContractError("Template type does not match Experiment type")

        candidate = Path(
            tempfile.mkdtemp(prefix=f".{variant_name}.candidate-", dir=experiment.path)
        )
        try:
            shutil.copytree(template.path / "src", candidate / "src")
            use_json = experiment.authored_schema_version == 2
            if use_json:
                if template.code_seed is not None and template.options_seed is not None:
                    code = copy.deepcopy(template.code_seed)
                    options = copy.deepcopy(template.options_seed)
                else:
                    seed = copy.deepcopy(template.variant_seed)
                    legacy = {
                        "schema_version": seed.pop("schema_version"),
                        "name": variant_name,
                        "template": {
                            "name": template.manifest["name"],
                            "version": template.manifest["version"],
                            "digest": template.bundle_digest,
                        },
                        **seed,
                    }
                    split = split_legacy_variant(legacy)
                    code, options = split.code, split.options
                validate_code_json(code)
                validate_options_json(options, template.options_schema)
                atomic_write_new(candidate / "code.json", dump_json(code))
                atomic_write_new(candidate / "options.json", dump_json(options))
                document = _runtime_variant_document(
                    variant_name, code, options, template.bundle_digest
                )
            else:
                seed = copy.deepcopy(template.variant_seed)
                document = {
                    "schema_version": seed.pop("schema_version"),
                    "name": variant_name,
                    "template": {
                        "name": template.manifest["name"],
                        "version": template.manifest["version"],
                        "digest": template.bundle_digest,
                    },
                    **seed,
                }
                atomic_write_new(candidate / "variant.yaml", dump_yaml(document))
            record = _load_variant_directory(
                candidate,
                experiment_name=experiment_name,
                expected_name=variant_name,
            )
            if compute_source_digest(candidate / "src") != compute_source_digest(
                template.path / "src"
            ):
                raise ContractError("copied Variant source digest changed")
            if compute_bundle_digest(template.path) != template.bundle_digest:
                raise ContractError("Template Bundle changed during Variant creation")
            return _publish_created_variant(self, experiment, record, target)
        finally:
            if candidate.exists():
                shutil.rmtree(candidate)

    @workspace_operation
    @authoring_write
    def clone_variant(
        self,
        experiment_name: str,
        variant_name: str,
        *,
        source_variant: str,
        source_experiment: str | None = None,
    ) -> VariantRecord:
        _variant_name(variant_name)
        _variant_name(source_variant)
        target_experiment = self.load_experiment(experiment_name)
        _check_variant_creation(self, target_experiment, variant_name)
        source_experiment = source_experiment or experiment_name
        source = self.load_variant(source_experiment, source_variant)
        source_owner = self.load_experiment(source_experiment)
        if source_owner.document["type"] != target_experiment.document["type"]:
            raise ContractError("source and target Experiment types do not match")
        if (
            source_owner.document["template"]["name"]
            != target_experiment.document["template"]["name"]
        ):
            raise ContractError(
                "source and target Experiment Template families do not match"
            )

        target = target_experiment.path / variant_name
        if os.path.lexists(target):
            raise AlreadyExistsError(
                f"Variant already exists: {experiment_name}/{variant_name}"
            )
        candidate = Path(
            tempfile.mkdtemp(
                prefix=f".{variant_name}.candidate-", dir=target_experiment.path
            )
        )
        try:
            source_digest = compute_source_digest(source.path / "src")
            shutil.copytree(source.path / "src", candidate / "src")
            if target_experiment.authored_schema_version == 2:
                if source.authored_schema_version == 2:
                    assert source.code_document is not None
                    assert source.options_document is not None
                    code = copy.deepcopy(source.code_document)
                    options = copy.deepcopy(source.options_document)
                else:
                    split = split_legacy_variant(source.document)
                    code, options = split.code, split.options
                validate_code_json(code)
                template = self.templates.resolve(
                    f"{code['template']['name']}@{code['template']['version']}"
                )
                validate_options_json(options, template.options_schema)
                atomic_write_new(candidate / "code.json", dump_json(code))
                atomic_write_new(candidate / "options.json", dump_json(options))
            elif source.authored_schema_version == 2:
                raise ContractError(
                    "cannot clone a schema 2 Variant into a schema 1 Experiment"
                )
            else:
                document = copy.deepcopy(source.document)
                document["name"] = variant_name
                validate_variant(document, expected_name=variant_name)
                atomic_write_new(candidate / "variant.yaml", dump_yaml(document))
            record = _load_variant_directory(
                candidate,
                experiment_name=experiment_name,
                expected_name=variant_name,
                initial_provenance=source.document["template"],
            )
            if (
                compute_source_digest(candidate / "src") != source_digest
                or compute_source_digest(source.path / "src") != source_digest
            ):
                raise ContractError("source Variant changed during clone")
            return _publish_created_variant(
                self,
                target_experiment,
                record,
                target,
                source_experiment=source_experiment,
                source_variant=source_variant,
            )
        finally:
            if candidate.exists():
                shutil.rmtree(candidate)

    @workspace_operation
    def list_variants(self, experiment_name: str) -> list[VariantRecord]:
        experiment = self.load_experiment(experiment_name)
        records: list[VariantRecord] = []
        for entry in _variant_catalog_entries(experiment.path):
            _variant_name(entry.name)
            records.append(self.load_variant(experiment_name, entry.name))
        return sorted(
            records,
            key=lambda record: record.document["name"].encode("utf-8"),
        )

    def check_variant(self, experiment_name: str, variant_name: str) -> VariantRecord:
        return self.load_variant(experiment_name, variant_name)

    @workspace_operation
    def load_variant(self, experiment_name: str, variant_name: str) -> VariantRecord:
        _name(experiment_name, "Experiment")
        _variant_name(variant_name)
        experiment = self.load_experiment(experiment_name)
        path = experiment.path / variant_name
        if not os.path.lexists(path):
            raise NotFoundError(f"Variant not found: {experiment_name}/{variant_name}")
        record = _load_variant_directory(
            path,
            experiment_name=experiment_name,
            expected_name=variant_name,
        )
        if (
            record.document["template"]["name"]
            != experiment.document["template"]["name"]
        ):
            raise OwnershipError(
                f"Variant Template family does not match its Experiment: "
                f"{experiment_name}/{variant_name}"
            )
        return record

    def _ensure_experiment_catalog(self) -> Path:
        path = self.repository.experiments
        try:
            path.mkdir(mode=0o755)
        except FileExistsError:
            pass
        _require_directory(path)
        return path

    def _use_json_authoring(self, template: ResolvedTemplate) -> bool:
        marker = self.repository.root / ".hkdl/store/AUTHORING_CURRENT"
        if os.path.lexists(marker):
            try:
                return marker.read_text(encoding="ascii") == "v2\n"
            except (OSError, UnicodeError) as error:
                raise ContractError("authoring format marker is unavailable") from error
        existing = (
            _catalog_entries(self.repository.experiments)
            if os.path.lexists(self.repository.experiments)
            else []
        )
        return not existing and template.authoring_schema_version == 2

    def _activate_json_authoring(self) -> None:
        marker = self.repository.root / ".hkdl/store/AUTHORING_CURRENT"
        marker.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if not os.path.lexists(marker):
            atomic_write_new(marker, "v2\n")


def _load_experiment(path: Path, *, expected_name: str) -> ExperimentRecord:
    _require_directory(path)
    if any(os.path.lexists(path / name) for name in ("src", "variants", ".venv")):
        raise ContractError(
            "legacy Experiment layout is unsupported; migration is deferred"
        )
    _require_directory(path / "notes")
    for name in EXPERIMENT_AUXILIARY_DIRECTORIES:
        auxiliary = path / name
        if os.path.lexists(auxiliary):
            _require_directory(auxiliary)
    yaml_path = path / "experiment.yaml"
    json_path = path / "experiment.json"
    if os.path.lexists(yaml_path) and os.path.lexists(json_path):
        raise ContractError("mixed Experiment YAML/JSON authoring is unsupported")
    if os.path.lexists(json_path):
        authored = load_json_file(json_path)
        validate_experiment_json(authored)
        created_at = _experiment_created_at(path, expected_name)
        document = {
            "schema_version": 1,
            "name": expected_name,
            "type": authored["type"],
            "question": authored["question"],
            "template": authored["template"],
            "created_at": created_at,
        }
        validate_experiment(document, expected_name=expected_name)
        return ExperimentRecord(path, document, 2)
    document = load_yaml_file(yaml_path)
    validate_experiment(document)
    if document["name"] != expected_name:
        raise OwnershipError(
            f"Experiment name does not match its directory: {expected_name}"
        )
    return ExperimentRecord(path, document, 1)


def _load_variant_directory(
    path: Path,
    *,
    experiment_name: str,
    expected_name: str,
    initial_provenance: dict[str, object] | None = None,
) -> VariantRecord:
    _require_directory(path)
    validate_locked_source_tree(path / "src")
    yaml_path = path / "variant.yaml"
    code_path = path / "code.json"
    options_path = path / "options.json"
    json_present = os.path.lexists(code_path) or os.path.lexists(options_path)
    if os.path.lexists(yaml_path) and json_present:
        raise ContractError("mixed Variant YAML/JSON authoring is unsupported")
    if json_present:
        if not os.path.lexists(code_path) or not os.path.lexists(options_path):
            raise ContractError(
                "Variant JSON authoring requires code.json and options.json"
            )
        code = load_json_file(code_path)
        options = load_json_file(options_path)
        return _variant_json_record(
            validate_repository_root(path.parents[2]),
            path,
            experiment_name=experiment_name,
            expected_name=expected_name,
            code=code,
            options=options,
            initial_provenance=initial_provenance,
        )
    document = load_yaml_file(yaml_path)
    validate_variant(document)
    if document["name"] != expected_name:
        raise OwnershipError(
            f"Variant name does not match its path: {experiment_name}/{expected_name}"
        )
    return VariantRecord(path, experiment_name, document, 1)


def _variant_json_record(
    repository,
    path,
    *,
    experiment_name,
    expected_name,
    code,
    options,
    initial_provenance=None,
):
    """Common JSON interpretation for file reads and migration candidates."""
    from .v2.provenance import committed_variant_revision, validate_provenance

    validate_code_json(code)
    template = TemplateResolver(repository).resolve(
        f"{code['template']['name']}@{code['template']['version']}"
    )
    validate_options_json(options, template.options_schema)
    revision = committed_variant_revision(repository, experiment_name, expected_name)
    provenance = revision["template"] if revision is not None else initial_provenance
    digest = template.bundle_digest
    if provenance is not None:
        provenance = validate_provenance(provenance)
        if all(
            provenance[field] == code["template"][field]
            for field in ("name", "version")
        ):
            digest = provenance["digest"]
    # Source and components still enter Code identity independently. Retaining
    # its origin must not make a genuine research edit clean.
    document = _runtime_variant_document(expected_name, code, options, digest)
    return VariantRecord(path, experiment_name, document, 2, code, options)


def _catalog_entries(path: Path) -> list[Path]:
    _require_directory(path)
    entries: list[Path] = []
    for entry in os.scandir(path):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            raise ContractError(f"symlinked catalog entry: {entry.path}")
        entries.append(Path(entry.path))
    return sorted(entries, key=lambda entry: entry.name.encode("utf-8"))


def _variant_catalog_entries(path: Path) -> list[Path]:
    entries: list[Path] = []
    for entry in os.scandir(path):
        if entry.name.startswith(".") or entry.name in {
            "experiment.yaml",
            "experiment.json",
            "notes",
            *EXPERIMENT_AUXILIARY_DIRECTORIES,
        }:
            continue
        if entry.name in RESERVED_VARIANT_NAMES:
            raise ContractError(
                "legacy Experiment layout is unsupported; migration is deferred"
            )
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ContractError(f"invalid Variant catalog entry: {entry.name}")
        entries.append(Path(entry.path))
    return sorted(entries, key=lambda entry: entry.name.encode("utf-8"))


def _require_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"required directory is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise ContractError(f"required directory is invalid: {path}")


def _commit_created_experiment(authoring: Authoring, record: ExperimentRecord) -> None:
    from .v2.graph import V2Graph

    graph = V2Graph(authoring.repository)
    if graph.is_active():
        graph.commit_experiment(record)


def _check_variant_creation(
    authoring: Authoring, experiment: ExperimentRecord, name: str
) -> None:
    from .v2.graph import V2Graph, experiment_variant_scope

    graph = V2Graph(authoring.repository)
    if not graph.is_active():
        return
    identity = graph.assert_experiment_clean(
        str(experiment.document["name"]), experiment
    )
    scope = experiment_variant_scope(identity.experiment_hash)
    if name in graph.bindings.names(scope):
        raise AlreadyExistsError(f"Variant already exists: {name}")
    if graph.bindings.historical_target(scope, name) is not None:
        raise AlreadyExistsError(f"Variant name is retained in binding history: {name}")


def _creation_fingerprint(path: Path) -> tuple:
    # Include directories and inode ownership, not just file bytes. A concurrent
    # replacement or even an empty user directory must prevent rollback deletion.
    digest = compute_source_digest(path)
    entries = []
    for child in sorted([path, *path.rglob("*")]):
        metadata = child.lstat()
        entries.append(
            (
                str(child.relative_to(path)),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_mtime_ns,
            )
        )
    return digest, tuple(entries)


def _rollback_created_variant(
    authoring: Authoring, experiment: ExperimentRecord, target: Path, expected: tuple
) -> bool:
    from .v2.graph import V2Graph, experiment_variant_scope

    if not os.path.lexists(target):
        return True
    graph = V2Graph(authoring.repository)
    if graph.is_active():
        scope = experiment_variant_scope(
            graph.experiment_hash(str(experiment.document["name"]))
        )
        if target.name in graph.bindings.names(scope):
            return False
    if _creation_fingerprint(target) != expected:
        return False
    quarantine = target.with_name(f".{target.name}.rollback-{secrets.token_hex(16)}")
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Move out of the public address before deletion, then recheck ownership.
        # A newly created/replaced public path must never be a deletion target.
        publish_directory(target, quarantine)
        if _creation_fingerprint(quarantine) != expected:
            publish_directory(quarantine, target)
            return False
        shutil.rmtree(quarantine)
        os.fsync(descriptor)
    except BaseException as error:
        raise ContractError(
            f"Variant rollback incomplete; inspect {target} and {quarantine}"
        ) from error
    finally:
        os.close(descriptor)
    return True


def _publish_created_variant(
    authoring: Authoring,
    experiment: ExperimentRecord,
    record: VariantRecord,
    target: Path,
    *,
    source_experiment: str | None = None,
    source_variant: str | None = None,
) -> VariantRecord:
    expected = _creation_fingerprint(record.path)
    created = replace(record, path=target)
    try:
        publish_directory(record.path, target)
        _commit_created_variant(
            authoring,
            experiment,
            created,
            source_experiment=source_experiment,
            source_variant=source_variant,
        )
    except BaseException as error:
        # A no-replace collision did not publish our candidate. Preserve the
        # original error class and never treat another creator's target as ours.
        if os.path.lexists(record.path):
            raise
        try:
            restored = _rollback_created_variant(
                authoring, experiment, target, expected
            )
        except (OSError, ContractError) as rollback_error:
            raise ContractError(
                f"Variant creation failed; rollback incomplete: {rollback_error}"
            ) from error
        if not restored:
            raise ContractError(
                f"Variant creation failed; published draft preserved at {target}; inspect bindings and draft before retry"
            ) from error
        raise
    return created


def _commit_created_variant(
    authoring: Authoring,
    experiment: ExperimentRecord,
    record: VariantRecord,
    *,
    source_experiment: str | None = None,
    source_variant: str | None = None,
) -> None:
    from .v2.graph import V2Graph

    graph = V2Graph(authoring.repository)
    if not graph.is_active():
        return
    derivation_parent = None
    if source_experiment is not None and source_variant is not None:
        source_experiment_hash = graph.experiment_hash(source_experiment)
        source_variant_hash = graph.variant_hash(
            source_experiment_hash,
            source_variant,
        )
        derivation_parent = graph.current_revision(source_variant_hash)
    graph.commit_variant(
        experiment,
        record,
        derivation_parent=derivation_parent,
        commit_experiment=False,
    )


def _name(value: object, kind: str) -> None:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ContractError(f"{kind} has an invalid name")


def _variant_name(value: object) -> None:
    _name(value, "Variant")
    if value in RESERVED_VARIANT_NAMES:
        raise ContractError("Variant uses a reserved name")


def _runtime_variant_document(
    name: str,
    code: dict[str, object],
    options: dict[str, object],
    bundle_digest: str,
) -> dict[str, object]:
    """Synthesize the legacy-shaped runtime document from authored JSON."""

    code_template = code["template"]
    assert isinstance(code_template, dict)
    document = {
        "schema_version": 1,
        "name": name,
        "template": {
            "name": code_template["name"],
            "version": code_template["version"],
            "digest": bundle_digest,
        },
        "dataset": copy.deepcopy(options["dataset"]),
        "metrics": copy.deepcopy(options["metrics"]),
        "tracker": {"backend": "local"},
        "components": copy.deepcopy(code["components"]),
        "train": copy.deepcopy(options["train"]),
        "eval": copy.deepcopy(options["eval"]),
        "infer": copy.deepcopy(options["infer"]),
    }
    validate_variant(document, expected_name=name)
    return document


def _experiment_created_at(path: Path, name: str) -> str:
    """Provide compatibility metadata without storing it in experiment.json."""

    try:
        repository = validate_repository_root(path.parents[1])
        from .v2.graph import V2Graph

        graph = V2Graph(repository)
        if graph.is_active():
            experiment = graph.store.load(graph.experiment_hash(name))
            created_at = experiment.payload.get("created_at")
            if isinstance(created_at, str):
                return created_at
    except (ContractError, NotFoundError, OSError):
        pass
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError as error:
        raise ContractError(f"cannot inspect Experiment directory: {path}") from error
    return _timestamp(modified)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractError("creation clock must include a timezone")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
