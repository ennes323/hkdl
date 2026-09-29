"""Safe YAML loading and structural schema validation."""

from __future__ import annotations

import math
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from hkdl.errors import ContractError as ContractError

NAME_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
VERSION_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
RFC3339_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)

COMPONENT_KINDS = frozenset(
    {
        "model",
        "loss",
        "optimizer",
        "dataloader",
        "trainer",
        "evaluator",
        "exporter",
    }
)


_SafeLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class _JsonSafeLoader(_SafeLoader):
    pass


class _JsonSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


_JsonSafeLoader.yaml_implicit_resolvers = {
    key: [
        resolver
        for resolver in resolvers
        if resolver[0] != "tag:yaml.org,2002:timestamp"
    ]
    for key, resolvers in _SafeLoader.yaml_implicit_resolvers.items()
}


def _construct_unique_mapping(
    loader: _JsonSafeLoader, node: MappingNode, deep: bool = False
) -> dict[str, Any]:
    if not isinstance(node, MappingNode):
        raise ConstructorError(None, None, "expected a mapping", node.start_mark)

    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be strings",
                key_node.start_mark,
            )
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_JsonSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load one regular UTF-8 YAML mapping without YAML-specific object types."""

    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"cannot read {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ContractError(f"expected a regular file: {path}")

    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_JsonSafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ContractError(f"invalid YAML in {path}: {error}") from error

    _ensure_json_compatible(document, str(path))
    if not isinstance(document, dict):
        raise ContractError(f"expected a top-level mapping: {path}")
    return document


def dump_yaml(document: dict[str, Any]) -> str:
    """Serialize one JSON-compatible mapping without YAML aliases."""

    _ensure_json_compatible(document, "document")
    text = yaml.dump(
        document,
        Dumper=_JsonSafeDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    return text if text.endswith("\n") else f"{text}\n"


def validate_template_manifest(
    document: dict[str, Any],
    *,
    expected_name: str | None = None,
    expected_version: str | None = None,
) -> None:
    _exact_fields(document, {"schema_version", "name", "version", "type"}, "template")
    _schema_version(document["schema_version"], "template.schema_version")
    _name(document["name"], "template.name")
    _version(document["version"], "template.version")
    _name(document["type"], "template.type")
    if expected_name is not None and document["name"] != expected_name:
        raise ContractError("template.name does not match its catalog identity")
    if expected_version is not None and document["version"] != expected_version:
        raise ContractError("template.version does not match its catalog identity")


def validate_experiment(
    document: dict[str, Any], *, expected_name: str | None = None
) -> None:
    _exact_fields(
        document,
        {
            "schema_version",
            "name",
            "type",
            "question",
            "template",
            "created_at",
        },
        "experiment",
    )
    _schema_version(document["schema_version"], "experiment.schema_version")
    _name(document["name"], "experiment.name")
    _name(document["type"], "experiment.type")
    if expected_name is not None and document["name"] != expected_name:
        raise ContractError("experiment.name does not match its directory")
    if not isinstance(document["question"], str) or not document["question"].strip():
        raise ContractError("experiment.question must be a non-empty string")

    provenance = _mapping(document["template"], "experiment.template")
    _exact_fields(provenance, {"name"}, "experiment.template")
    _name(provenance["name"], "experiment.template.name")
    _rfc3339(document["created_at"], "experiment.created_at")


def validate_template_experiment_seed(
    document: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    _exact_fields(document, {"schema_version", "type", "question"}, "experiment seed")
    _schema_version(document["schema_version"], "experiment seed.schema_version")
    _name(document["type"], "experiment seed.type")
    if not isinstance(document["question"], str) or not document["question"].strip():
        raise ContractError("experiment seed.question must be a non-empty string")
    if document["type"] != manifest["type"]:
        raise ContractError("Template seed type does not match template.yaml")


def validate_template_variant_seed(document: dict[str, Any]) -> None:
    _exact_fields(
        document,
        {
            "schema_version",
            "dataset",
            "metrics",
            "tracker",
            "components",
            "train",
            "eval",
            "infer",
        },
        "variant seed",
    )
    _schema_version(document["schema_version"], "variant seed.schema_version")
    _validate_variant_body(document, "variant seed")


def validate_variant(
    document: dict[str, Any], *, expected_name: str | None = None
) -> None:
    _exact_fields(
        document,
        {
            "schema_version",
            "name",
            "template",
            "dataset",
            "metrics",
            "tracker",
            "components",
            "train",
            "eval",
            "infer",
        },
        "variant",
    )
    _schema_version(document["schema_version"], "variant.schema_version")
    _name(document["name"], "variant.name")
    if expected_name is not None and document["name"] != expected_name:
        raise ContractError("variant.name does not match its filename")

    provenance = _mapping(document["template"], "variant.template")
    _exact_fields(provenance, {"name", "version", "digest"}, "variant.template")
    _name(provenance["name"], "variant.template.name")
    _version(provenance["version"], "variant.template.version")
    if not isinstance(provenance["digest"], str) or not DIGEST_PATTERN.fullmatch(
        provenance["digest"]
    ):
        raise ContractError("variant.template.digest is invalid")

    _validate_variant_body(document, "variant")


def _validate_variant_body(document: dict[str, Any], location: str) -> None:
    _mapping(document["dataset"], f"{location}.dataset")
    _mapping(document["metrics"], f"{location}.metrics")
    _mapping(document["tracker"], f"{location}.tracker")

    components = _mapping(document["components"], f"{location}.components")
    unknown = set(components) - COMPONENT_KINDS
    if unknown:
        raise ContractError(f"unknown component kinds: {', '.join(sorted(unknown))}")
    for kind, registry_name in components.items():
        _name(registry_name, f"{location}.components.{kind}")

    _mapping(document["train"], f"{location}.train")
    _mapping(document["eval"], f"{location}.eval")
    _mapping(document["infer"], f"{location}.infer")


def _ensure_json_compatible(
    value: Any,
    location: str,
    ancestors: set[int] | None = None,
) -> None:
    ancestors = set() if ancestors is None else ancestors
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{location} contains a non-finite number")
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            raise ContractError(f"{location} contains a recursive YAML alias")
        ancestors.add(identity)
        try:
            for index, item in enumerate(value):
                _ensure_json_compatible(item, f"{location}[{index}]", ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in ancestors:
            raise ContractError(f"{location} contains a recursive YAML alias")
        ancestors.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ContractError(f"{location} contains a non-string mapping key")
                _ensure_json_compatible(item, f"{location}.{key}", ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise ContractError(f"{location} contains a non-JSON value: {type(value).__name__}")


def _exact_fields(document: dict[str, Any], expected: set[str], location: str) -> None:
    actual = set(document)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ContractError(
            f"{location} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ContractError(
            f"{location} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{location} must be a mapping")
    return value


def _schema_version(value: Any, location: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ContractError(f"{location} must be integer 1")


def _name(value: Any, location: str) -> None:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ContractError(f"{location} has an invalid name")


def _version(value: Any, location: str) -> None:
    if not isinstance(value, str) or not VERSION_PATTERN.fullmatch(value):
        raise ContractError(f"{location} has an invalid version")


def _rfc3339(value: Any, location: str) -> None:
    if not isinstance(value, str) or not RFC3339_PATTERN.fullmatch(value):
        raise ContractError(f"{location} must be an RFC 3339 string with an offset")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{location} is not a valid timestamp") from error
    if parsed.utcoffset() is None:
        raise ContractError(f"{location} must include an offset")
