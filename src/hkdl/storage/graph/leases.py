"""Identity-scoped operation locks, independent of compatibility projections."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

from hkdl.authoring.config import DIGEST_PATTERN
from hkdl.errors import ContractError
from hkdl.storage.storage import (
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


@contextmanager
def authoring_guard(
    repository, experiment, variant=None, *, source=None, dry_run=False
):
    """Hold existing target identities and, for cloning, both source identities.

    Resolve the complete set before acquisition so nested authoring retains the
    existing deterministic ordering and reentrant entity guards.
    """
    from .graph import V2Graph

    graph = V2Graph(repository)
    if not graph.is_active() or dry_run:
        yield
        return
    entities = []
    try:
        entity = graph.experiment_hash(experiment)
        entities.append(("experiment", entity))
        if variant is not None:
            entities.append(("variant", graph.variant_hash(entity, variant)))
    except NotFoundError:
        # New targets have no identity yet; existing owners still stay guarded.
        pass
    if source is not None:
        source_experiment, source_variant = source
        source_entity = graph.experiment_hash(source_experiment)
        entities.extend(
            [
                ("experiment", source_entity),
                ("variant", graph.variant_hash(source_entity, source_variant)),
            ]
        )
    with entity_guard(repository, entities):
        yield
