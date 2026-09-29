"""Read committed authoring provenance without publishing or repairing it."""

from copy import deepcopy

from hkdl.authoring.config import DIGEST_PATTERN, NAME_PATTERN, VERSION_PATTERN
from hkdl.errors import ContractError
from hkdl.storage.storage import NotFoundError

from .graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    entity_revision_scope,
    experiment_variant_scope,
    workspace_experiment_scope,
)
from .reader import current_reader


def validate_provenance(template):
    if not isinstance(template, dict) or set(template) != {"name", "version", "digest"}:
        raise ContractError("committed Template provenance is malformed")
    for field, pattern in (
        ("name", NAME_PATTERN),
        ("version", VERSION_PATTERN),
        ("digest", DIGEST_PATTERN),
    ):
        if not isinstance(template[field], str) or not pattern.fullmatch(
            template[field]
        ):
            raise ContractError("committed Template provenance is malformed")
    return deepcopy(template)


def committed_variant_revision(repository, experiment, variant):
    """Absent name bindings are drafts; broken bound authority is never a fallback."""
    if not V2Graph(repository).is_active():
        return None
    reader = current_reader(repository)
    experiments = reader.names(workspace_experiment_scope())
    if experiment not in experiments:
        return None
    experiment_hash = experiments[experiment]
    reader.object(experiment_hash, "experiment")
    variants = reader.names(experiment_variant_scope(experiment_hash))
    if variant not in variants:
        return None
    try:
        _, entity = reader.entities(experiment, variant)
        revision = reader.object(
            reader.resolve(entity_revision_scope(entity), CURRENT_REVISION_NAME),
            "variant_revision",
        )
        if revision.get("variant") != entity:
            raise ContractError("committed Variant revision ownership mismatch")
        validate_provenance(revision.get("template"))
        if not isinstance(revision.get("components"), dict):
            raise ContractError("committed Variant components are malformed")
        reader.object(revision["source_tree"], "source_tree")
        return revision
    except (NotFoundError, KeyError, TypeError) as error:
        raise ContractError(
            "committed Variant authority is missing or malformed"
        ) from error
