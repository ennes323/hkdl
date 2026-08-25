"""Repository, Template, digest, and atomic publication primitives."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import (
    ContractError,
    NAME_PATTERN,
    VERSION_PATTERN,
    load_yaml_file,
    validate_template_experiment_seed,
    validate_template_manifest,
    validate_template_variant_seed,
)
from .filesystem import lock_directory as _raw_lock_directory
from .filesystem import unlock_directory as _raw_unlock_directory


class AlreadyExistsError(ContractError):
    """Atomic publication refused to replace an existing target."""


class NotFoundError(ContractError):
    """An addressed HKDL object does not exist."""


class OwnershipError(ContractError):
    """An authored object's embedded identity disagrees with its path."""


class LockUnavailableError(RuntimeError):
    """A non-blocking advisory lock is already held."""


STATUS_INDEX_FILENAME = ".hkdl-index.sqlite3"


@dataclass(frozen=True)
class RepositoryPaths:
    root: Path
    template_catalog: Path
    experiments: Path
    outputs: Path


@dataclass(frozen=True)
class ResolvedTemplate:
    path: Path
    manifest: dict[str, object]
    experiment_seed: dict[str, object]
    variant_seed: dict[str, object]
    bundle_digest: str


def storage_usage(repository: RepositoryPaths) -> dict[str, int]:
    """Return logical bytes for repository-owned authored and generated state."""

    authored, legacy_environments = _tree_sizes(
        repository.experiments,
        split_environments=True,
    )
    shared_environments, _ = _tree_sizes(repository.root / ".hkdl/environments")
    environments = legacy_environments + shared_environments
    outputs, _ = _tree_sizes(
        repository.outputs,
        ignored_root_prefixes=(STATUS_INDEX_FILENAME,),
    )
    return {
        "authored": authored,
        "environments": environments,
        "outputs": outputs,
        "total": authored + environments + outputs,
    }


def _tree_sizes(
    root: Path,
    *,
    split_environments: bool = False,
    ignored_root_prefixes: tuple[str, ...] = (),
) -> tuple[int, int]:
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return 0, 0
    except OSError as error:
        raise ContractError(f"cannot inspect storage tree {root}: {error}") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        return root_stat.st_size, 0

    sizes = [0, 0]
    pending = [(root, (), False)]
    while pending:
        directory, relative, in_environment = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ContractError(
                f"cannot scan storage tree {directory}: {error}"
            ) from error
        for entry in entries:
            entry_relative = (*relative, entry.name)
            if not relative and entry.name.startswith(ignored_root_prefixes):
                continue
            entry_in_environment = in_environment or (
                split_environments
                and len(entry_relative) == 3
                and entry.name == ".venv"
            )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ContractError(
                    f"cannot inspect storage entry {entry.path}: {error}"
                ) from error
            if stat.S_ISDIR(metadata.st_mode):
                pending.append((Path(entry.path), entry_relative, entry_in_environment))
            else:
                sizes[int(entry_in_environment)] += metadata.st_size
    return sizes[0], sizes[1]


def validate_repository_root(root: Path | None = None) -> RepositoryPaths:
    """Validate exactly one repository root without parent discovery."""

    root = Path.cwd() if root is None else Path(root)
    root = root.absolute()
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise ContractError(f"invalid repository root: {root}")

    for relative in (".python-version", "pyproject.toml", "uv.lock"):
        _require_regular_file(root / relative)
    for relative in ("src/hkdl", "src/templates"):
        _require_directory(root / relative)

    return RepositoryPaths(
        root=root,
        template_catalog=root / "src/templates",
        experiments=root / "experiments",
        outputs=root / "outputs",
    )


class TemplateResolver:
    def __init__(self, repository: RepositoryPaths):
        self.repository = repository

    def resolve(self, reference: str) -> ResolvedTemplate:
        name, version = _parse_template_reference(reference)
        name_root = self.repository.template_catalog / name
        bundle = name_root / version
        if not os.path.lexists(name_root) or not os.path.lexists(bundle):
            raise NotFoundError(f"Template not found: {reference}")
        _require_directory(name_root)
        _require_directory(bundle)
        _require_directory(bundle / "src")

        manifest = load_yaml_file(bundle / "template.yaml")
        validate_template_manifest(
            manifest,
            expected_name=name,
            expected_version=version,
        )
        validate_locked_source_tree(bundle / "src")
        experiment_seed = load_yaml_file(bundle / "experiment.yaml")
        validate_template_experiment_seed(experiment_seed, manifest)
        variant_seed = load_yaml_file(bundle / "variant.yaml")
        validate_template_variant_seed(variant_seed)
        bundle_digest = compute_bundle_digest(bundle)
        return ResolvedTemplate(
            bundle,
            manifest,
            experiment_seed,
            variant_seed,
            bundle_digest,
        )

    def latest(self, name: str) -> ResolvedTemplate:
        if not NAME_PATTERN.fullmatch(name):
            raise ContractError("invalid Template name")
        name_root = self.repository.template_catalog / name
        if not os.path.lexists(name_root):
            raise NotFoundError(f"Template not found: {name}")
        _require_directory(name_root)
        versions = _catalog_directories(name_root)
        if not versions:
            raise NotFoundError(f"Template has no versions: {name}")
        for version_entry in versions:
            if not VERSION_PATTERN.fullmatch(version_entry.name):
                raise ContractError(
                    f"invalid Template catalog version: {version_entry.name}"
                )
        selected = max(versions, key=lambda entry: parse_version(entry.name))
        return self.resolve(f"{name}@{selected.name}")

    def list(self) -> list[ResolvedTemplate]:
        resolved: list[ResolvedTemplate] = []
        for name_entry in _catalog_directories(self.repository.template_catalog):
            if not NAME_PATTERN.fullmatch(name_entry.name):
                raise ContractError(f"invalid Template catalog name: {name_entry.name}")
            for version_entry in _catalog_directories(name_entry):
                if not VERSION_PATTERN.fullmatch(version_entry.name):
                    raise ContractError(
                        f"invalid Template catalog version: {version_entry.name}"
                    )
                resolved.append(self.resolve(f"{name_entry.name}@{version_entry.name}"))
        return sorted(
            resolved,
            key=lambda item: (
                str(item.manifest["name"]).encode("utf-8"),
                parse_version(str(item.manifest["version"])),
            ),
        )


def validate_locked_source_tree(src: Path) -> str:
    _require_directory(src)
    for name in ("pyproject.toml", "uv.lock", "registry.py"):
        _require_regular_file(src / name)
    if os.path.lexists(src / ".venv"):
        raise ContractError(
            f"generated environment must be outside src: {src / '.venv'}"
        )
    return compute_source_digest(src)


def compute_source_digest(src: Path) -> str:
    """Hash every regular file below src with deterministic path framing."""

    _require_directory(src)
    return _hash_files(_collect_regular_files(src), src)


def compute_bundle_digest(bundle: Path) -> str:
    """Hash the complete fixed-shape Template Bundle."""

    _require_directory(bundle)
    expected = {"template.yaml", "experiment.yaml", "variant.yaml", "src"}
    actual: set[str] = set()
    for entry in os.scandir(bundle):
        if entry.is_symlink():
            raise ContractError(f"symlinks are not allowed in Template: {entry.name}")
        actual.add(entry.name)
    if actual != expected:
        raise ContractError("Template Bundle entries are invalid")
    for name in ("template.yaml", "experiment.yaml", "variant.yaml"):
        _require_regular_file(bundle / name)
    _require_directory(bundle / "src")
    files = [
        (Path(name).as_posix().encode("utf-8"), bundle / name)
        for name in ("template.yaml", "experiment.yaml", "variant.yaml")
    ]
    for relative_bytes, path in _collect_regular_files(bundle / "src"):
        files.append((b"src/" + relative_bytes, path))
    return _hash_files(files, bundle)


def parse_version(version: str) -> tuple[int, int, int]:
    if not VERSION_PATTERN.fullmatch(version):
        raise ContractError("invalid Template version")
    major, minor, patch = version.split(".")
    try:
        return int(major), int(minor), int(patch)
    except ValueError as error:
        raise ContractError("invalid Template version") from error


def _collect_regular_files(root: Path) -> list[tuple[bytes, Path]]:
    files: list[tuple[bytes, Path]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ContractError(
                f"cannot scan file tree {directory}: {error}"
            ) from error
        for entry in entries:
            relative = Path(entry.path).relative_to(root).as_posix()
            relative_bytes = relative.encode("utf-8")
            if entry.is_symlink():
                raise ContractError(f"symlinks are not allowed: {relative}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                files.append((relative_bytes, Path(entry.path)))
            else:
                raise ContractError(f"non-regular file entry: {relative}")
    return files


def _hash_files(files: list[tuple[bytes, Path]], root: Path) -> str:
    digest = hashlib.sha256()
    for relative_bytes, path in sorted(files, key=lambda item: item[0]):
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ContractError(f"cannot read file below {root}: {path}") from error
        digest.update(struct.pack(">Q", len(relative_bytes)))
        digest.update(relative_bytes)
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def publish_directory(candidate: Path, target: Path) -> None:
    """Atomically publish a sibling candidate without replacing a target."""

    candidate = Path(candidate).absolute()
    target = Path(target).absolute()
    if candidate.parent != target.parent:
        raise ContractError("candidate and target must share a parent directory")
    _require_directory(candidate)
    parent_fd = _lock_directory(target.parent)
    try:
        if os.path.lexists(target):
            raise AlreadyExistsError(f"target already exists: {target}")
        os.rename(candidate, target)
        os.fsync(parent_fd)
    finally:
        _unlock_directory(parent_fd)


def publish_file(candidate: Path, target: Path) -> None:
    """Atomically publish a sibling regular file without replacing a target."""

    candidate = Path(candidate).absolute()
    target = Path(target).absolute()
    if candidate.parent != target.parent:
        raise ContractError("candidate and target must share a parent directory")
    _require_regular_file(candidate)
    parent_fd = _lock_directory(target.parent)
    try:
        if os.path.lexists(target):
            raise AlreadyExistsError(f"target already exists: {target}")
        try:
            os.link(candidate, target)
        except FileExistsError as error:
            raise AlreadyExistsError(f"target already exists: {target}") from error
        os.fsync(parent_fd)
        candidate.unlink()
        os.fsync(parent_fd)
    finally:
        _unlock_directory(parent_fd)


def atomic_write_new(
    path: Path,
    content: str | bytes,
    *,
    directory_descriptor: int | None = None,
) -> None:
    """Publish one complete new file without replacing an existing path."""

    path = Path(path).absolute()
    _require_directory(path.parent)
    payload = content.encode("utf-8") if isinstance(content, str) else content
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.candidate-",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o644)
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        parent_fd = (
            directory_descriptor
            if directory_descriptor is not None
            else _lock_directory(path.parent)
        )
        try:
            if os.path.lexists(path):
                raise AlreadyExistsError(f"target already exists: {path}")
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise AlreadyExistsError(f"target already exists: {path}") from error
            os.fsync(parent_fd)
        finally:
            if directory_descriptor is None:
                _unlock_directory(parent_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_replace(path: Path, content: str | bytes) -> None:
    """Atomically replace one regular file with complete flushed content."""

    path = Path(path).absolute()
    _require_directory(path.parent)
    if not os.path.lexists(path):
        raise ContractError(f"replace target does not exist: {path}")
    _require_regular_file(path)
    payload = content.encode("utf-8") if isinstance(content, str) else content
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.replacement-",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o644)
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def directory_lock(path: Path) -> Iterator[int]:
    """Hold an advisory lock on one validated directory."""

    descriptor = _lock_directory(Path(path).absolute())
    try:
        yield descriptor
    finally:
        _unlock_directory(descriptor)


@contextmanager
def try_directory_lock(path: Path) -> Iterator[int]:
    """Hold an advisory lock or fail immediately when it is already held."""

    descriptor = _lock_directory(Path(path).absolute(), blocking=False)
    try:
        yield descriptor
    finally:
        _unlock_directory(descriptor)


def _parse_template_reference(reference: str) -> tuple[str, str]:
    if reference.count("@") != 1:
        raise ContractError("Template reference must be <name>@<version>")
    name, version = reference.split("@", 1)
    if not NAME_PATTERN.fullmatch(name) or not VERSION_PATTERN.fullmatch(version):
        raise ContractError("invalid Template reference")
    return name, version


def _catalog_directories(path: Path) -> list[Path]:
    _require_directory(path)
    directories: list[Path] = []
    for entry in os.scandir(path):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            raise ContractError(f"symlinked catalog entry: {entry.path}")
        if not entry.is_dir(follow_symlinks=False):
            raise ContractError(f"non-directory catalog entry: {entry.path}")
        directories.append(Path(entry.path))
    return sorted(directories, key=lambda entry: entry.name.encode("utf-8"))


def _require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"required regular file is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ContractError(f"required regular file is invalid: {path}")


def _require_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"required directory is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise ContractError(f"required directory is invalid: {path}")


def _lock_directory(path: Path, *, blocking: bool = True) -> int:
    _require_directory(path)
    try:
        return _raw_lock_directory(path, blocking=blocking)
    except BlockingIOError as error:
        raise LockUnavailableError(f"directory is locked: {path}") from error


def _unlock_directory(descriptor: int) -> None:
    _raw_unlock_directory(descriptor)
