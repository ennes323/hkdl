"""Identity-scoped operation locks, independent of compatibility projections."""

from __future__ import annotations

import os
from inspect import signature
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps

from ..config import ContractError, DIGEST_PATTERN
from ..storage import (
    LockUnavailableError,
    NotFoundError,
    directory_lock,
    try_directory_lock,
)


_INHERITED: ContextVar[tuple[int, ...]] = ContextVar("hkdl_identity_leases", default=())
_ENTITIES: ContextVar[frozenset[str]] = ContextVar(
    "hkdl_entity_guards", default=frozenset()
)


def inherited_leases() -> tuple[int, ...]:
    return _INHERITED.get()


def lease_path(repository, kind, digest, *, create):
    if not isinstance(digest, str) or not DIGEST_PATTERN.fullmatch(digest):
        raise ContractError("identity lease requires a full hash")
    if kind not in {"attempt", "variant", "experiment"}:
        raise ContractError("identity lease kind is invalid")
    root = repository.root
    for component in (
        ".hkdl",
        "store",
        "v2",
        "leases",
        kind,
        digest.removeprefix("sha256:"),
    ):
        root = root / component
        if create:
            root.mkdir(exist_ok=True)
        if os.path.lexists(root) and (root.is_symlink() or not root.is_dir()):
            raise ContractError("identity lease path must be a real directory")
    return root


@contextmanager
def attempt_lease(repository, digest, path, *, create=True):
    """Keep old physical leases effective and inherit both lease descriptors."""
    anchor = lease_path(repository, "attempt", digest, create=create)
    with ExitStack() as stack:
        descriptors = []
        if anchor.exists():
            descriptors.append(stack.enter_context(try_directory_lock(anchor)))
        if path is not None and os.path.lexists(path):
            descriptors.append(stack.enter_context(try_directory_lock(path)))
        token = _INHERITED.set((*_INHERITED.get(), *descriptors))
        try:
            yield descriptors[-1] if descriptors else None
        finally:
            _INHERITED.reset(token)


def lease_held(repository, digest, path):
    try:
        with attempt_lease(repository, digest, path, create=False):
            return False
    except LockUnavailableError:
        return True


@contextmanager
def entity_guard(repository, entities):
    held = _ENTITIES.get()
    with ExitStack() as stack:
        keys = set(held)
        for kind, digest in sorted(set(entities)):
            path = lease_path(repository, kind, digest, create=True)
            key = str(path)
            if key not in held:
                stack.enter_context(directory_lock(path))
                keys.add(key)
        token = _ENTITIES.set(frozenset(keys))
        try:
            yield
        finally:
            _ENTITIES.reset(token)


def authoring_write(function):
    """Serialize mutation by stable Experiment/Variant, not mutable directory."""

    parameters = signature(function)

    @wraps(function)
    def write(owner, *args, **kwargs):
        from .graph import V2Graph

        repository = getattr(owner, "repository", None) or owner.authoring.repository
        graph = V2Graph(repository)
        if not graph.is_active() or kwargs.get("dry_run", False):
            return function(owner, *args, **kwargs)
        arguments = parameters.bind(owner, *args, **kwargs).arguments
        values = list(arguments.values())[1:]
        experiment = values[0]
        name = (
            experiment.document["name"]
            if hasattr(experiment, "document")
            else experiment
        )
        entities = []
        try:
            entity = graph.experiment_hash(name)
            entities.append(("experiment", entity))
            if (
                function.__name__ not in {"commit_experiment", "rename_experiment"}
                and len(values) > 1
            ):
                variant = values[1]
                variant_name = (
                    variant.document["name"]
                    if hasattr(variant, "document")
                    else variant
                )
                entities.append(("variant", graph.variant_hash(entity, variant_name)))
        except NotFoundError:
            pass
        if function.__name__ == "clone_variant":
            source_experiment = arguments.get("source_experiment") or name
            source_entity = graph.experiment_hash(source_experiment)
            entities.extend(
                [
                    ("experiment", source_entity),
                    (
                        "variant",
                        graph.variant_hash(source_entity, arguments["source_variant"]),
                    ),
                ]
            )
        with entity_guard(repository, entities):
            return function(owner, *args, **kwargs)

    return write
