"""Read-only shell completion for the HKDL CLI."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import argcomplete
from argcomplete.completers import FilesCompleter, SuppressCompleter

from hkdl.authoring.authoring import RESERVED_VARIANT_NAMES
from hkdl.authoring.config import NAME_PATTERN, VERSION_PATTERN, load_yaml_file
from hkdl.errors import ContractError
from hkdl.execution.run_contracts import MAX_SEED, MODEL_ID_PATTERN, RUN_ID_PATTERN
from hkdl.storage.graph.graph import (
    V2Graph,
    experiment_variant_scope,
    variant_model_scope,
    variant_run_scope,
    workspace_experiment_scope,
)
from hkdl.storage.graph.maintenance import WorkspaceBusy, workspace_access
from hkdl.storage.graph.reader import GraphReader
from hkdl.storage.storage import (
    NotFoundError,
    RepositoryPaths,
    validate_repository_root,
)

SHELLS = ("bash", "zsh")
DEVICES = ("auto", "cpu", "mps", "cuda")
file_completer = FilesCompleter()


def activate(parser: argparse.ArgumentParser) -> None:
    argcomplete.autocomplete(parser, default_completer=SuppressCompleter())


def shellcode(shell: str) -> str:
    script = argcomplete.shellcode(
        ["hkdl"],
        shell=shell,
        complete_arguments=["-o", "nospace"],
    )
    return script.rstrip() + "\n"


def complete_templates(*, prefix: str, **_: Any) -> list[str]:
    return _complete(
        prefix,
        lambda repository: _names(
            repository.template_catalog, owner=repository.template_catalog
        ),
    )


def complete_devices(*, prefix: str, **_: Any) -> list[str]:
    return [device for device in DEVICES if device.startswith(prefix)]


def complete_template_references(*, prefix: str, **_: Any) -> list[str]:
    def candidates(repository: RepositoryPaths) -> Iterable[str]:
        for name in _names(
            repository.template_catalog, owner=repository.template_catalog
        ):
            for version in _names(
                repository.template_catalog / name,
                pattern=VERSION_PATTERN.fullmatch,
                owner=repository.template_catalog,
            ):
                yield f"{name}@{version}"

    return _complete(prefix, candidates)


def complete_template_versions(
    *, prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    experiment = getattr(parsed_args, "experiment", None)
    if not _name(experiment):
        return []

    def candidates(repository: RepositoryPaths) -> Iterable[str]:
        document = _authored(
            repository.experiments / experiment, "experiment.yaml", "experiment.json"
        )
        template = document.get("template") if document else None
        family = template.get("name") if isinstance(template, dict) else None
        if not _name(family):
            return ()
        return _names(
            repository.template_catalog / family,
            pattern=VERSION_PATTERN.fullmatch,
            owner=repository.template_catalog,
        )

    return _complete(prefix, candidates)


def complete_experiments(*, prefix: str, **_: Any) -> list[str]:
    def candidates(repository):
        reader = _graph(repository)
        return (
            reader.names(workspace_experiment_scope())
            if reader
            else _names(repository.experiments)
        )

    return _complete(prefix, candidates)


def complete_variants(
    *, prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    return _variants(prefix, getattr(parsed_args, "experiment", None))


def complete_source_variants(
    *, prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    experiment = getattr(parsed_args, "source_experiment", None) or getattr(
        parsed_args, "experiment", None
    )
    return _variants(prefix, experiment)


def complete_runs(
    *, prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    return _owned_names(prefix, parsed_args, "runs", RUN_ID_PATTERN.fullmatch)


def complete_models(
    *, prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    return _owned_names(prefix, parsed_args, "models", MODEL_ID_PATTERN.fullmatch)


def complete_training_groups(
    *, prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    experiment = getattr(parsed_args, "experiment", None)
    variant = getattr(parsed_args, "variant", None)
    if not _name(experiment) or not _name(variant):
        return []

    def candidates(repository: RepositoryPaths) -> Iterable[str]:
        reader = _graph(repository)
        if reader is not None:
            groups = {
                record.request["target"].get("training_group")
                for record in reader.runs(experiment=experiment, variant=variant)
            }
            groups.update(
                model.document["training_group"]
                for model in reader.models(experiment, variant)
            )
            return {group for group in groups if _name(group)}
        root = repository.outputs / experiment / variant
        groups: set[str] = set()
        for run_id in _names(root / "runs", pattern=RUN_ID_PATTERN.fullmatch):
            request = _json(root / "runs" / run_id / "request.json")
            target = request.get("target") if request else None
            group = target.get("training_group") if isinstance(target, dict) else None
            if _name(group):
                groups.add(group)
        for model_id in _names(root / "models", pattern=MODEL_ID_PATTERN.fullmatch):
            model = _json(root / "models" / model_id / "model.json")
            group = model.get("training_group") if model else None
            if _name(group):
                groups.add(group)
        return groups

    return _complete(prefix, candidates)


def complete_evaluation_cases(
    *, prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    experiment = getattr(parsed_args, "experiment", None)
    variant = getattr(parsed_args, "variant", None)
    if not _name(experiment) or not _name(variant):
        return []

    def candidates(repository: RepositoryPaths) -> Iterable[str]:
        document = _authored(
            repository.experiments / experiment / variant,
            "variant.yaml",
            "options.json",
        )
        evaluation = document.get("eval") if document else None
        if not isinstance(evaluation, dict):
            return ()
        cases = evaluation.get("cases")
        if cases is None:
            return ("default",)
        if not isinstance(cases, dict):
            return ()
        return (name for name in cases if _name(name))

    return _complete(prefix, candidates)


def complete_eval_seeds(
    *, prefix: str, parsed_args: argparse.Namespace, **_: Any
) -> list[str]:
    experiment = getattr(parsed_args, "experiment", None)
    variant = getattr(parsed_args, "variant", None)
    group = getattr(parsed_args, "training_group", None)
    if not _name(experiment) or not _name(variant) or not _name(group):
        return []

    def candidates(repository: RepositoryPaths) -> Iterable[str]:
        reader = _graph(repository)
        if reader is not None:
            return {"all"} | {
                str(model.document["seed"])
                for model in reader.models(experiment, variant)
                if model.document["training_group"] == group
            }
        models = repository.outputs / experiment / variant / "models"
        seeds = {"all"}
        for model_id in _names(models, pattern=MODEL_ID_PATTERN.fullmatch):
            model = _json(models / model_id / "model.json")
            if not model or model.get("training_group") != group:
                continue
            seed = model.get("seed")
            if (
                isinstance(seed, int)
                and not isinstance(seed, bool)
                and 0 <= seed <= MAX_SEED
            ):
                seeds.add(str(seed))
        return seeds

    return _complete(prefix, candidates)


def _variants(prefix: str, experiment: object) -> list[str]:
    if not _name(experiment):
        return []

    def candidates(repository):
        reader = _graph(repository)
        if reader is not None:
            entity = reader.resolve(workspace_experiment_scope(), experiment)
            return reader.names(experiment_variant_scope(entity))
        return _names(
            repository.experiments / experiment,
            excluded=RESERVED_VARIANT_NAMES | {"notes"},
        )

    return _complete(prefix, candidates)


def _owned_names(
    prefix: str,
    parsed_args: argparse.Namespace,
    catalog: str,
    pattern: Callable[[str], object],
) -> list[str]:
    experiment = getattr(parsed_args, "experiment", None)
    variant = getattr(parsed_args, "variant", None)
    if not _name(experiment) or not _name(variant):
        return []

    def candidates(repository):
        reader = _graph(repository)
        if reader is not None:
            _, entity = reader.entities(experiment, variant)
            scope = (
                variant_run_scope(entity)
                if catalog == "runs"
                else variant_model_scope(entity)
            )
            return reader.names(scope)
        return _names(
            repository.outputs / experiment / variant / catalog,
            pattern=pattern,
        )

    return _complete(prefix, candidates)


def _graph(repository: RepositoryPaths) -> GraphReader | None:
    return GraphReader(repository) if V2Graph(repository).is_active() else None


def _authored(root: Path, legacy: str, current: str) -> dict[str, Any] | None:
    if os.path.lexists(root / legacy) and os.path.lexists(root / current):
        return None
    if os.path.lexists(root / current):
        return _json(root / current)
    return _yaml(root / legacy)


def _complete(
    prefix: str,
    candidates: Callable[[RepositoryPaths], Iterable[str]],
) -> list[str]:
    try:
        repository = validate_repository_root()
        with workspace_access(repository):
            values = candidates(repository)
            return sorted(
                {value for value in values if value.startswith(prefix)},
                key=lambda value: value.encode("utf-8"),
            )
    except (
        WorkspaceBusy,
        ContractError,
        NotFoundError,
        OSError,
        TypeError,
        ValueError,
    ):
        return []


def _names(
    path: Path,
    *,
    pattern: Callable[[str], object] = NAME_PATTERN.fullmatch,
    excluded: frozenset[str] | set[str] = frozenset(),
    owner: Path | None = None,
) -> list[str]:
    mode = _owned_mode(path, owner=owner)
    if mode is None or not stat.S_ISDIR(mode):
        return []
    names: list[str] = []
    with os.scandir(path) as entries:
        for entry in entries:
            if (
                entry.name.startswith(".")
                or entry.name in excluded
                or entry.is_symlink()
                or not entry.is_dir(follow_symlinks=False)
                or not pattern(entry.name)
            ):
                continue
            names.append(entry.name)
    return sorted(names, key=lambda value: value.encode("utf-8"))


def _yaml(path: Path) -> dict[str, Any] | None:
    mode = _owned_mode(path)
    if mode is None or not stat.S_ISREG(mode):
        return None
    document = load_yaml_file(path)
    return document if isinstance(document, dict) else None


def _json(path: Path) -> dict[str, Any] | None:
    mode = _owned_mode(path)
    if mode is None or not stat.S_ISREG(mode):
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _name(value: object) -> bool:
    return isinstance(value, str) and NAME_PATTERN.fullmatch(value) is not None


def _owned_mode(path: Path, *, owner: Path | None = None) -> int | None:
    root = Path.cwd().absolute() if owner is None else owner.absolute()
    if root.resolve() != root or root.is_symlink():
        return None
    try:
        relative = path.absolute().relative_to(root)
    except ValueError:
        return None
    current = root
    try:
        for part in relative.parts:
            current /= part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                return None
        return current.lstat().st_mode
    except OSError:
        return None


__all__ = [
    "DEVICES",
    "SHELLS",
    "activate",
    "complete_devices",
    "complete_eval_seeds",
    "complete_evaluation_cases",
    "complete_experiments",
    "complete_models",
    "complete_runs",
    "complete_source_variants",
    "complete_template_references",
    "complete_template_versions",
    "complete_templates",
    "complete_training_groups",
    "complete_variants",
    "file_completer",
    "shellcode",
]
