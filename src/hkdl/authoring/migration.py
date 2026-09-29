"""Explicit authored-schema migration boundary."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hkdl.errors import ContractError
from hkdl.storage.storage import NotFoundError, RepositoryPaths

from .config import NAME_PATTERN, load_yaml_file, validate_experiment, validate_variant

CURRENT_AUTHORED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MigrationResult:
    path: str
    schema_version: int
    changed: bool


class Migration:
    def __init__(self, repository: RepositoryPaths):
        self.repository = repository

    def migrate(self, path: str | Path) -> MigrationResult:
        target, relative, kind, expected_name = self._resolve_target(path)
        document = load_yaml_file(target)
        version = document.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ContractError(
                f"migration target schema_version must be an integer: {relative}"
            )
        if version != CURRENT_AUTHORED_SCHEMA_VERSION:
            raise ContractError(
                "no migration path available for "
                f"schema {version} (current: {CURRENT_AUTHORED_SCHEMA_VERSION}): "
                f"{relative}"
            )

        if kind == "experiment":
            validate_experiment(document, expected_name=expected_name)
        else:
            validate_variant(document, expected_name=expected_name)

        return MigrationResult(
            path=relative,
            schema_version=version,
            changed=False,
        )

    def _resolve_target(
        self,
        path: str | Path,
    ) -> tuple[Path, str, str, str]:
        raw = Path(path)
        target = raw.absolute() if raw.is_absolute() else (self.repository.root / raw)
        target = target.absolute()
        try:
            relative_path = target.relative_to(self.repository.root)
        except ValueError as error:
            raise ContractError(
                f"migration target must be inside the repository: {target}"
            ) from error

        relative = PurePosixPath(*relative_path.parts).as_posix()
        parts = relative_path.parts
        if parts and parts[0] == "outputs":
            raise ContractError(
                f"generated records are immutable and cannot be migrated: {relative}"
            )

        if (
            len(parts) == 3
            and parts[0] == "experiments"
            and parts[2] == "experiment.yaml"
            and NAME_PATTERN.fullmatch(parts[1])
        ):
            kind = "experiment"
            expected_name = parts[1]
        elif (
            len(parts) == 4
            and parts[0] == "experiments"
            and parts[3] == "variant.yaml"
            and NAME_PATTERN.fullmatch(parts[1])
            and NAME_PATTERN.fullmatch(parts[2])
        ):
            kind = "variant"
            expected_name = parts[2]
        else:
            raise ContractError(
                "migration target must be an authored experiment.yaml or "
                f"variant.yaml: {relative}"
            )

        current = self.repository.root
        for index, part in enumerate(parts):
            current = current / part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError as error:
                raise NotFoundError(
                    f"migration target not found: {relative}"
                ) from error
            if stat.S_ISLNK(mode):
                raise ContractError(f"migration target contains a symlink: {relative}")
            if index < len(parts) - 1 and not stat.S_ISDIR(mode):
                raise ContractError(
                    f"migration target has an invalid path component: {relative}"
                )
            if index == len(parts) - 1 and not stat.S_ISREG(mode):
                raise ContractError(
                    f"migration target is not a regular file: {relative}"
                )

        return target, relative, kind, expected_name
