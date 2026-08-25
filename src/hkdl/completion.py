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

from .authoring import RESERVED_VARIANT_NAMES
from .config import NAME_PATTERN, VERSION_PATTERN, ContractError, load_yaml_file
from .run_contracts import MAX_SEED, MODEL_ID_PATTERN, RUN_ID_PATTERN
from .storage import RepositoryPaths, validate_repository_root

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
    return _complete(prefix, lambda repository: _names(repository.template_catalog))


def complete_devices(*, prefix: str, **_: Any) -> list[str]:
    return [device for device in DEVICES if device.startswith(prefix)]


def complete_template_references(*, prefix: str, **_: Any) -> list[str]:
    def candidates(repository: RepositoryPaths) -> Iterable[str]:
        for name in _names(repository.template_catalog):
            for version in _names(
                repository.template_catalog / name,
                pattern=VERSION_PATTERN.fullmatch,
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
        document = _yaml(repository.experiments / experiment / "experiment.yaml")
        template = document.get("template") if document else None
        family = template.get("name") if isinstance(template, dict) else None
        if not _name(family):
            return ()
        return _names(
            repository.template_catalog / family,
            pattern=VERSION_PATTERN.fullmatch,
        )

    return _complete(prefix, candidates)


def complete_experiments(*, prefix: str, **_: Any) -> list[str]:
    return _complete(prefix, lambda repository: _names(repository.experiments))


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
        document = _yaml(repository.experiments / experiment / variant / "variant.yaml")
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
    return _complete(
        prefix,
        lambda repository: _names(
            repository.experiments / experiment,
            excluded=RESERVED_VARIANT_NAMES | {"notes"},
        ),
    )


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
    return _complete(
        prefix,
        lambda repository: _names(
            repository.outputs / experiment / variant / catalog,
            pattern=pattern,
        ),
    )


def _complete(
    prefix: str,
    candidates: Callable[[RepositoryPaths], Iterable[str]],
) -> list[str]:
    try:
        repository = validate_repository_root()
        values = candidates(repository)
        return sorted(
            {value for value in values if value.startswith(prefix)},
            key=lambda value: value.encode("utf-8"),
        )
    except (ContractError, OSError, TypeError, ValueError):
        return []


def _names(
    path: Path,
    *,
    pattern: Callable[[str], object] = NAME_PATTERN.fullmatch,
    excluded: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    mode = _owned_mode(path)
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


def _owned_mode(path: Path) -> int | None:
    root = Path.cwd().absolute()
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
