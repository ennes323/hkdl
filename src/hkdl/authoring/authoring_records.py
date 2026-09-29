"""Authored Experiment and Variant values, independent of authoring services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


# Preserve existing serialized Python record references.
ExperimentRecord.__module__ = "hkdl.authoring.authoring"
VariantRecord.__module__ = "hkdl.authoring.authoring"
