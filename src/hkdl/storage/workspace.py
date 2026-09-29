"""Explicit workspace initialization without research or schema migration."""

from __future__ import annotations

import importlib.metadata
import json
import os
import tomllib
from dataclasses import replace
from pathlib import Path

from hkdl.errors import ContractError
from hkdl.installer import guidance
from hkdl.installer.common import InstallFailure

from .storage import (
    WORKSPACE_DOCUMENT,
    WORKSPACE_MARKER,
    AlreadyExistsError,
    RepositoryPaths,
    TemplateResolver,
    atomic_write_new,
    packaged_template_catalog,
    validate_repository_root,
)


def initialize_workspace(
    root: Path,
    *,
    choose_guidance: guidance.Choice | None = None,
    report_guidance: guidance.Report = lambda message: None,
) -> tuple[RepositoryPaths, bool]:
    """Mark a new workspace or explicitly adopt an intact source checkout."""

    root = root.absolute()
    if root.resolve() != root or (root.exists() and not root.is_dir()):
        raise ContractError(f"invalid workspace root: {root}")
    packaged_template_catalog()
    from hkdl.installer.common import directory_lease

    root.mkdir(parents=True, exist_ok=True)
    with directory_lease(root, exclusive=True):
        return _initialize_with_guidance(root, choose_guidance, report_guidance)


def _initialize_with_guidance(
    root: Path, choose: guidance.Choice | None, report: guidance.Report
) -> tuple[RepositoryPaths, bool]:
    repository, created = _initialize_locked(root)
    try:
        guidance.initialize(
            root,
            version=importlib.metadata.version("hkdl"),
            choose=choose,
            report=report,
        )
    except OSError as error:
        raise InstallFailure(
            f"workspace initialized but guidance is incomplete: {error}; "
            f"repeat hkdl workspace init {root} to review/recover it"
        ) from error
    return repository, created


def _initialize_locked(root: Path) -> tuple[RepositoryPaths, bool]:
    marker = root / WORKSPACE_MARKER
    if os.path.lexists(marker):
        return validate_repository_root(root), False
    if root.exists() and any(
        os.path.lexists(root / name) for name in ("experiments", "outputs", ".hkdl")
    ):
        # Existing research is accepted only through the existing checkout
        # contract. A missing marker is not evidence of an empty workspace.
        repository = validate_repository_root(root)
        _check_catalog_adoption(repository)
    elif all(
        (root / name).exists()
        for name in ("pyproject.toml", "src/hkdl", "src/templates")
    ):
        _check_catalog_adoption(validate_repository_root(root))
    metadata = marker.parent
    if metadata.is_symlink() or (metadata.exists() and not metadata.is_dir()):
        raise ContractError(f"invalid workspace metadata directory: {metadata}")
    metadata.mkdir(exist_ok=True)
    payload = json.dumps(WORKSPACE_DOCUMENT, sort_keys=True, indent=2) + "\n"
    try:
        atomic_write_new(marker, payload)
    except AlreadyExistsError:
        return validate_repository_root(root), False
    return validate_repository_root(root), True


def _check_catalog_adoption(repository: RepositoryPaths) -> None:
    packaged = TemplateResolver(
        replace(repository, template_catalog=packaged_template_catalog())
    )
    for template in TemplateResolver(repository).list():
        reference = f"{template.manifest['name']}@{template.manifest['version']}"
        try:
            installed = packaged.resolve(reference)
        except ContractError as error:
            raise ContractError(
                f"source-only Template {reference} is not in the installed catalog; "
                "keep using the source checkout until this Template is explicitly reconciled"
            ) from error
        if installed.bundle_digest != template.bundle_digest:
            raise ContractError(
                f"locally modified Template {reference} differs from the installed catalog; "
                "keep using the source checkout until its local changes are reconciled"
            )


def initialize_managed_workspace(
    root: Path,
    installation,
    *,
    choose_guidance: guidance.Choice | None = None,
    report_guidance: guidance.Report = lambda message: None,
) -> tuple[RepositoryPaths, bool]:
    """Validate explicit adoption before publishing the marker and registration."""

    from hkdl.installation_probe import check_workspace
    from hkdl.installer.bundle import require_transition
    from hkdl.installer.common import directory_lease

    root = root.absolute()
    current = installation.current()
    if current is None:
        raise ContractError("no active HKDL release")
    if not root.exists():
        repository, created = initialize_workspace(
            root, choose_guidance=choose_guidance, report_guidance=report_guidance
        )
        installation.register(root)
        return repository, created
    with directory_lease(root, exclusive=True) as descriptor:
        marker = root / WORKSPACE_MARKER
        if os.path.lexists(marker):
            check_workspace(root, current.manifest, descriptor=descriptor)
        elif all(
            (root / name).exists()
            for name in ("pyproject.toml", "src/hkdl", "src/templates")
        ):
            validate_repository_root(root)
            try:
                project = tomllib.loads((root / "pyproject.toml").read_text())[
                    "project"
                ]
                source_version = project["version"]
            except (ValueError, KeyError, TypeError) as error:
                raise ContractError(
                    "cannot identify the source checkout version"
                ) from error
            require_transition(source_version, current.manifest)
            check_workspace(root, current.manifest, descriptor=descriptor)
        repository, created = _initialize_with_guidance(
            root, choose_guidance, report_guidance
        )
        installation.register(root)
        return repository, created
