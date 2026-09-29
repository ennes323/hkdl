"""Interpret authored JSON records for ordinary reads and migration parity."""

from __future__ import annotations

import copy
from pathlib import Path

from hkdl.storage.storage import RepositoryPaths, TemplateResolver

from .authoring_records import VariantRecord
from .config import validate_variant
from .research_json import validate_code_json, validate_options_json


def variant_json_record(
    repository: RepositoryPaths,
    path: Path,
    *,
    experiment_name: str,
    expected_name: str,
    code: dict[str, object],
    options: dict[str, object],
    initial_provenance: dict[str, object] | None = None,
) -> VariantRecord:
    """Interpret one JSON-authored Variant without publishing or repairing it."""

    # Keep graph loading deferred so this leaf can be imported while storage
    # readers are initializing their authoring record dependencies.
    from hkdl.storage.graph.provenance import (
        committed_variant_revision,
        validate_provenance,
    )

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
    document = runtime_variant_document(expected_name, code, options, digest)
    return VariantRecord(path, experiment_name, document, 2, code, options)


def runtime_variant_document(
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
