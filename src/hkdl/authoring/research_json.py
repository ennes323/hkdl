"""Strict JSON authoring contracts for the research-oriented HKDL layout.

The original authoring format is YAML and keeps the Variant's code, options,
and operational settings in one document.  The v2 authoring format deliberately
keeps those concerns in separate JSON documents.  This module is intentionally
independent from the authoring and storage implementations: callers can use it
to validate or convert documents while the rest of the repository is being
migrated.

There are two small but important properties here:

* JSON files are read from a regular, non-symlink file and are decoded as
  strict UTF-8.  Duplicate object keys and the non-standard JSON constants
  ``NaN``/``Infinity`` are rejected instead of silently normalised.
* Validation errors use :class:`hkdl.errors.ContractError`, so existing CLI
  and authoring callers can present them using the same error class as the
  YAML contracts.

``jsonschema`` is an optional import at module import time.  The application
  declares it as a runtime dependency in the v2 package; the small fallback
  validator below keeps this low-level module usable by source checkouts before
  dependencies have been installed and covers the Draft 2020-12 vocabulary
  used by the bundled option schemas.
"""

from __future__ import annotations

import copy
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hkdl.errors import ContractError

from .config import (
    COMPONENT_KINDS,
    NAME_PATTERN,
    VERSION_PATTERN,
    load_yaml_file,
    validate_experiment,
    validate_variant,
)

SCHEMA_VERSION = 2
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
EXPERIMENT_FIELDS = frozenset({"schema_version", "type", "question", "template"})
CODE_FIELDS = frozenset({"schema_version", "template", "components"})
OPTION_FIELDS = frozenset(
    {"schema_version", "dataset", "metrics", "train", "eval", "infer"}
)


class ResearchJSONError(ContractError):
    """A strict authored JSON file or schema violates the v2 contract."""


def _duplicate_key_error(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResearchJSONError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ResearchJSONError(f"non-finite JSON number is not allowed: {value}")


def _regular_file(path: Path) -> None:
    """Require *path* to be a regular, non-symlink file.

    ``Path.is_file`` follows links, which is precisely what an authored input
    must not do.  Use ``lstat`` and the mode from the same operation so a
    symlink cannot be accepted merely because its target is a regular file.
    """

    try:
        metadata = path.lstat()
    except OSError as error:
        raise ResearchJSONError(f"cannot read JSON file {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ResearchJSONError(f"expected a regular non-symlink JSON file: {path}")


def _json_compatible(
    value: Any,
    location: str = "document",
    ancestors: set[int] | None = None,
) -> None:
    """Reject values that Python's JSON encoder would otherwise coerce."""

    ancestors = set() if ancestors is None else ancestors
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResearchJSONError(f"{location} contains a non-finite number")
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            raise ResearchJSONError(f"{location} contains a recursive value")
        ancestors.add(identity)
        try:
            for index, item in enumerate(value):
                _json_compatible(item, f"{location}[{index}]", ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in ancestors:
            raise ResearchJSONError(f"{location} contains a recursive value")
        ancestors.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ResearchJSONError(
                        f"{location} contains a non-string mapping key"
                    )
                _json_compatible(item, f"{location}.{key}", ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise ResearchJSONError(
        f"{location} contains a non-JSON value: {type(value).__name__}"
    )


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a strict, top-level object from a UTF-8 JSON file.

    The parser intentionally does not accept JSON extensions.  In particular,
    duplicate keys are an error (rather than ``json.loads``'s usual
    last-value-wins behaviour), and ``parse_constant`` turns non-standard
    numeric constants into a contract error.
    """

    path = Path(path)
    _regular_file(path)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        document = json.loads(
            text,
            object_pairs_hook=_duplicate_key_error,
            parse_constant=_reject_constant,
        )
    except ResearchJSONError:
        raise
    except (OSError, ValueError) as error:
        # Decoder limits (such as oversized integers) are authored-input errors too.
        raise ResearchJSONError(f"invalid JSON in {path}: {error}") from error

    if not isinstance(document, dict):
        raise ResearchJSONError(f"JSON document must be an object: {path}")
    _json_compatible(document, str(path))
    return document


def dump_json(document: Mapping[str, Any]) -> str:
    """Serialize an authored object deterministically as UTF-8 JSON text.

    Human-authored files use two-space indentation and a final newline.  Keys
    are sorted so a conversion or rewrite does not create meaningless diffs;
    object identity code should use :func:`canonical_json_bytes` instead of
    relying on this pretty representation.
    """

    if not isinstance(document, dict):
        raise ResearchJSONError("JSON document must be an object")
    _json_compatible(document)
    try:
        text = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ResearchJSONError(f"cannot serialize JSON document: {error}") from error
    return f"{text}\n"


def write_json_file(path: Path, document: Mapping[str, Any]) -> None:
    """Write one validated authored document as UTF-8 text.

    This helper intentionally refuses to follow an existing symlink.  Atomic
    publication belongs to the storage layer; callers that need no-replace or
    fsync semantics should publish the returned :func:`dump_json` text through
    that layer.
    """

    path = Path(path)
    if os.path.lexists(path):
        _regular_file(path)
    text = dump_json(document)
    try:
        path.write_bytes(text.encode("utf-8"))
    except OSError as error:
        raise ResearchJSONError(f"cannot write JSON file {path}: {error}") from error


def canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON bytes suitable for content hashing."""

    if not isinstance(document, dict):
        raise ResearchJSONError("JSON document must be an object")
    _json_compatible(document)
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ResearchJSONError(f"cannot serialize canonical JSON: {error}") from error


# These aliases make the low-level contract readable at call sites that use
# ``read``/``encode`` terminology rather than ``load``/``dump``.
load_authored_json = load_json_file
dump_authored_json = dump_json
read_json_file = load_json_file
canonical_json = canonical_json_bytes


def _exact_fields(
    document: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    actual = set(document)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ResearchJSONError(
            f"{location} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ResearchJSONError(
            f"{location} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _schema_version(value: Any, location: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise ResearchJSONError(f"{location} must be integer {SCHEMA_VERSION}")


def _name(value: Any, location: str) -> None:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ResearchJSONError(f"{location} has an invalid name")


def _version(value: Any, location: str) -> None:
    if not isinstance(value, str) or not VERSION_PATTERN.fullmatch(value):
        raise ResearchJSONError(f"{location} has an invalid version")


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResearchJSONError(f"{location} must be an object")
    return value


def validate_experiment_json(document: Mapping[str, Any]) -> None:
    """Validate the name-free v2 ``experiment.json`` contract."""

    if not isinstance(document, dict):
        raise ResearchJSONError("experiment.json must be an object")
    _exact_fields(document, EXPERIMENT_FIELDS, "experiment.json")
    _schema_version(document["schema_version"], "experiment.schema_version")
    _name(document["type"], "experiment.type")
    if not isinstance(document["question"], str) or not document["question"].strip():
        raise ResearchJSONError("experiment.question must be a non-empty string")
    template = _mapping(document["template"], "experiment.template")
    _exact_fields(template, frozenset({"name"}), "experiment.template")
    _name(template["name"], "experiment.template.name")


def validate_code_json(document: Mapping[str, Any]) -> None:
    """Validate the structural, name-free v2 ``code.json`` contract."""

    if not isinstance(document, dict):
        raise ResearchJSONError("code.json must be an object")
    _exact_fields(document, CODE_FIELDS, "code.json")
    _schema_version(document["schema_version"], "code.schema_version")
    template = _mapping(document["template"], "code.template")
    _exact_fields(template, frozenset({"name", "version"}), "code.template")
    _name(template["name"], "code.template.name")
    _version(template["version"], "code.template.version")
    components = _mapping(document["components"], "code.components")
    unknown = set(components) - COMPONENT_KINDS
    if unknown:
        raise ResearchJSONError(
            f"unknown code component kinds: {', '.join(sorted(unknown))}"
        )
    for kind, component in components.items():
        _name(component, f"code.components.{kind}")


def _validate_option_sections(document: Mapping[str, Any]) -> None:
    for name in ("dataset", "metrics", "train", "eval", "infer"):
        _mapping(document[name], f"options.{name}")


def validate_options_json(
    document: Mapping[str, Any],
    schema: Mapping[str, Any] | Path | None = None,
) -> None:
    """Validate ``options.json`` and, when supplied, its Draft 2020-12 schema.

    ``schema`` may be a parsed schema mapping or a regular JSON schema file.
    The structural checks are always performed before schema validation, which
    makes the error useful even in a source checkout without ``jsonschema``.
    """

    if not isinstance(document, dict):
        raise ResearchJSONError("options.json must be an object")
    _exact_fields(document, OPTION_FIELDS, "options.json")
    _schema_version(document["schema_version"], "options.schema_version")
    _validate_option_sections(document)

    if schema is None:
        return
    schema_document = (
        load_json_file(Path(schema)) if isinstance(schema, (str, Path)) else schema
    )
    if not isinstance(schema_document, dict):
        raise ResearchJSONError("options schema must be an object")
    _validate_options_schema_document(schema_document)
    _validate_against_schema(document, schema_document)


def validate_options_file(
    path: Path, schema: Mapping[str, Any] | Path | None = None
) -> dict[str, Any]:
    """Load and validate an ``options.json`` file, returning its document."""

    document = load_json_file(path)
    validate_options_json(document, schema)
    return document


def load_options_schema(path: Path) -> dict[str, Any]:
    """Load one Draft 2020-12 options schema from a regular JSON file."""

    schema = load_json_file(path)
    _validate_options_schema_document(schema)
    return schema


def _validate_options_schema_document(schema: Mapping[str, Any]) -> None:
    declared = schema.get("$schema")
    if declared != DRAFT_2020_12:
        raise ResearchJSONError(
            "options schema must declare Draft 2020-12: " + DRAFT_2020_12
        )
    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        # The fallback validator checks the subset used below.  It still
        # validates the actual shape and does not silently skip a malformed
        # schema document.
        _validate_fallback_schema_shape(schema, "options schema")
    else:
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as error:  # jsonschema.SchemaError without hard import
            raise ResearchJSONError(f"invalid options schema: {error}") from error


def _validate_against_schema(
    document: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        _fallback_validate(document, schema, "options.json")
        return

    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(
            (type(component).__name__, str(component)) for component in error.path
        ),
    )
    if errors:
        error = errors[0]
        location = "options.json"
        for component in error.path:
            location += (
                f"[{component!r}]" if isinstance(component, int) else f".{component}"
            )
        raise ResearchJSONError(f"{location}: {error.message}")


def _validate_fallback_schema_shape(schema: Mapping[str, Any], location: str) -> None:
    """Check enough schema structure to reject malformed bundled schemas."""

    schema_type = schema.get("type")
    supported_types = {
        "object",
        "array",
        "string",
        "number",
        "integer",
        "boolean",
        "null",
    }
    if isinstance(schema_type, str):
        if schema_type not in supported_types:
            raise ResearchJSONError(f"{location}.type is not a supported JSON type")
    elif schema_type is not None and (
        not isinstance(schema_type, list)
        or not schema_type
        or not all(
            isinstance(item, str) and item in supported_types for item in schema_type
        )
    ):
        raise ResearchJSONError(f"{location}.type is not a supported JSON type")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise ResearchJSONError(f"{location}.required must be an array of strings")
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict):
            raise ResearchJSONError(f"{location}.properties must be an object")
        for name, child in properties.items():
            if not isinstance(name, str) or not isinstance(child, dict):
                raise ResearchJSONError(
                    f"{location}.properties must map strings to schemas"
                )
            _validate_fallback_schema_shape(child, f"{location}.properties.{name}")
    if "prefixItems" in schema:
        prefix_items = schema["prefixItems"]
        if not isinstance(prefix_items, list) or not all(
            isinstance(item, dict) for item in prefix_items
        ):
            raise ResearchJSONError(
                f"{location}.prefixItems must be an array of schemas"
            )
        for index, item in enumerate(prefix_items):
            _validate_fallback_schema_shape(item, f"{location}.prefixItems[{index}]")
    if "items" in schema:
        if not isinstance(schema["items"], (bool, dict)):
            raise ResearchJSONError(
                f"{location}.items must be a boolean or object schema"
            )
        if isinstance(schema["items"], dict):
            _validate_fallback_schema_shape(schema["items"], f"{location}.items")
    if "additionalProperties" in schema and not isinstance(
        schema["additionalProperties"], (bool, dict)
    ):
        raise ResearchJSONError(
            f"{location}.additionalProperties must be a boolean or schema"
        )


def _fallback_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _fallback_validate(value: Any, schema: Mapping[str, Any], location: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_fallback_type_matches(value, item) for item in expected):
            raise ResearchJSONError(f"{location} must match one of types {expected}")
    elif isinstance(expected, str) and not _fallback_type_matches(value, expected):
        raise ResearchJSONError(f"{location} must be a {expected}")

    if "const" in schema and value != schema["const"]:
        raise ResearchJSONError(f"{location} must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ResearchJSONError(f"{location} must be one of {schema['enum']!r}")

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ResearchJSONError(f"{location} is shorter than minLength {minimum}")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ResearchJSONError(f"{location} is longer than maxLength {maximum}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise ResearchJSONError(f"{location} is below minimum {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise ResearchJSONError(f"{location} is above maximum {maximum}")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            raise ResearchJSONError(
                f"{location} is not above exclusiveMinimum {exclusive_minimum}"
            )
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            raise ResearchJSONError(
                f"{location} is not below exclusiveMaximum {exclusive_maximum}"
            )

    if isinstance(value, dict):
        minimum = schema.get("minProperties")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ResearchJSONError(f"{location} has fewer than {minimum} properties")
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ResearchJSONError(f"{location} is missing required field {name}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for name, child in properties.items():
                if name in value and isinstance(child, dict):
                    _fallback_validate(value[name], child, f"{location}.{name}")
            additional = schema.get("additionalProperties", True)
            if additional is False:
                unknown = set(value) - set(properties)
                if unknown:
                    raise ResearchJSONError(
                        f"{location} has unknown fields: {', '.join(sorted(unknown))}"
                    )
            elif isinstance(additional, dict):
                for name in set(value) - set(properties):
                    _fallback_validate(value[name], additional, f"{location}.{name}")
    elif isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ResearchJSONError(f"{location} has fewer than {minimum} items")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ResearchJSONError(f"{location} has more than {maximum} items")
        prefix_items = schema.get("prefixItems", [])
        if isinstance(prefix_items, list):
            for index, item_schema in enumerate(prefix_items):
                if index < len(value) and isinstance(item_schema, dict):
                    _fallback_validate(
                        value[index], item_schema, f"{location}[{index}]"
                    )
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            start = len(prefix_items) if isinstance(prefix_items, list) else 0
            for index, item in enumerate(value[start:], start=start):
                _fallback_validate(item, items_schema, f"{location}[{index}]")
        elif items_schema is False:
            start = len(prefix_items) if isinstance(prefix_items, list) else 0
            if len(value) > start:
                raise ResearchJSONError(
                    f"{location} does not allow items after its prefixItems"
                )


@dataclass(frozen=True)
class LegacyVariantSplit:
    """Deterministic v2 documents derived from one v1 Variant document."""

    code: dict[str, Any]
    options: dict[str, Any]
    tracker: dict[str, Any]

    @property
    def code_json(self) -> dict[str, Any]:
        return self.code

    @property
    def options_json(self) -> dict[str, Any]:
        return self.options


@dataclass(frozen=True)
class LegacyAuthoringSplit:
    """Deterministic v2 documents derived from one v1 Experiment and Variant."""

    experiment: dict[str, Any]
    variant: LegacyVariantSplit

    @property
    def experiment_json(self) -> dict[str, Any]:
        return self.experiment

    @property
    def code(self) -> dict[str, Any]:
        return self.variant.code

    @property
    def options(self) -> dict[str, Any]:
        return self.variant.options


def split_legacy_experiment(document: Mapping[str, Any]) -> dict[str, Any]:
    """Strip v1 identity/metadata and return deterministic ``experiment.json``."""

    if not isinstance(document, dict):
        raise ResearchJSONError("legacy experiment must be an object")
    try:
        validate_experiment(document)
    except ContractError as error:
        raise ResearchJSONError(f"invalid legacy experiment: {error}") from error
    result = {
        "schema_version": SCHEMA_VERSION,
        "type": copy.deepcopy(document["type"]),
        "question": copy.deepcopy(document["question"]),
        "template": {"name": copy.deepcopy(document["template"]["name"])},
    }
    validate_experiment_json(result)
    return result


def split_legacy_variant(document: Mapping[str, Any]) -> LegacyVariantSplit:
    """Split one v1 Variant into code, options, and operational tracker data."""

    if not isinstance(document, dict):
        raise ResearchJSONError("legacy variant must be an object")
    try:
        validate_variant(document)
    except ContractError as error:
        raise ResearchJSONError(f"invalid legacy variant: {error}") from error

    provenance = document["template"]
    code = {
        "schema_version": SCHEMA_VERSION,
        "template": {
            "name": copy.deepcopy(provenance["name"]),
            "version": copy.deepcopy(provenance["version"]),
        },
        "components": copy.deepcopy(document["components"]),
    }
    options = {
        "schema_version": SCHEMA_VERSION,
        "dataset": copy.deepcopy(document["dataset"]),
        "metrics": copy.deepcopy(document["metrics"]),
        "train": copy.deepcopy(document["train"]),
        "eval": copy.deepcopy(document["eval"]),
        "infer": copy.deepcopy(document["infer"]),
    }
    tracker = copy.deepcopy(document["tracker"])
    validate_code_json(code)
    validate_options_json(options)
    if not isinstance(tracker, dict):  # guarded by validate_variant, for type narrowing
        raise ResearchJSONError("legacy variant tracker must be an object")
    return LegacyVariantSplit(code=code, options=options, tracker=tracker)


def split_legacy_documents(
    experiment: Mapping[str, Any], variant: Mapping[str, Any]
) -> LegacyAuthoringSplit:
    """Convert a legacy Experiment/Variant pair without mutating either input."""

    return LegacyAuthoringSplit(
        experiment=split_legacy_experiment(experiment),
        variant=split_legacy_variant(variant),
    )


def split_legacy_files(
    experiment_path: Path, variant_path: Path
) -> LegacyAuthoringSplit:
    """Load v1 YAML files and apply :func:`split_legacy_documents`."""

    try:
        experiment = load_yaml_file(Path(experiment_path))
        variant = load_yaml_file(Path(variant_path))
    except ContractError as error:
        raise ResearchJSONError(f"invalid legacy authoring files: {error}") from error
    return split_legacy_documents(experiment, variant)


# Descriptive aliases for callers that call this operation an adapter.
adapt_legacy_experiment = split_legacy_experiment
adapt_legacy_variant = split_legacy_variant
adapt_legacy_documents = split_legacy_documents
adapt_legacy_files = split_legacy_files


__all__ = [
    "CODE_FIELDS",
    "DRAFT_2020_12",
    "EXPERIMENT_FIELDS",
    "OPTION_FIELDS",
    "SCHEMA_VERSION",
    "LegacyAuthoringSplit",
    "LegacyVariantSplit",
    "ResearchJSONError",
    "adapt_legacy_documents",
    "adapt_legacy_experiment",
    "adapt_legacy_files",
    "adapt_legacy_variant",
    "canonical_json",
    "canonical_json_bytes",
    "dump_authored_json",
    "dump_json",
    "load_authored_json",
    "load_json_file",
    "load_options_schema",
    "read_json_file",
    "split_legacy_documents",
    "split_legacy_experiment",
    "split_legacy_files",
    "split_legacy_variant",
    "validate_code_json",
    "validate_experiment_json",
    "validate_options_file",
    "validate_options_json",
    "write_json_file",
]
