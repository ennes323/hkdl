"""Read-only authoring change projection for the local HKDL web view."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .authoring import Authoring, ExperimentRecord, VariantRecord
from .research_json import split_legacy_variant
from .storage import NotFoundError, RepositoryPaths
from .v2.graph import V2Graph, variant_model_scope, variant_run_scope

MAX_AUTHORING_CHANGES = 200


def build_authoring_state(
    repository: RepositoryPaths,
    experiment_name: str,
    *,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Describe authored changes without publishing or exposing graph hashes."""

    observed_at = (now or (lambda: datetime.now(timezone.utc)))()
    authoring = Authoring(repository)
    experiment = authoring.load_experiment(experiment_name)
    variants = authoring.list_variants(experiment_name)
    graph = V2Graph(repository)
    active = graph.is_active()
    return {
        "observed_at": observed_at.isoformat(),
        "available": active,
        "experiment": _experiment_authoring_view(
            graph,
            experiment,
            variants,
            active=active,
        ),
        "variants": [
            _variant_authoring_view(
                graph,
                experiment_name,
                variant,
                active=active,
            )
            for variant in variants
        ],
    }


def _experiment_authoring_view(
    graph: V2Graph,
    experiment: ExperimentRecord,
    variants: Sequence[VariantRecord],
    *,
    active: bool,
) -> dict[str, Any]:
    name = str(experiment.document["name"])
    draft = {
        "type": deepcopy(experiment.document["type"]),
        "question": deepcopy(experiment.document["question"]),
        "template": deepcopy(experiment.document["template"]),
    }
    if not active:
        return _authoring_item(
            name,
            "unavailable",
            False,
            "Commit management requires an active v2 workspace.",
            [],
        )
    try:
        experiment_hash = graph.experiment_hash(name)
        current_hash = graph.current_revision(experiment_hash)
        current_payload = graph.store.load(current_hash).payload
        committed = {
            "type": deepcopy(current_payload["type"]),
            "question": deepcopy(current_payload["question"]),
            "template": deepcopy(current_payload["template"]),
        }
    except NotFoundError:
        committed = {}
    changes = _json_changes(committed, draft)
    if not changes:
        return _authoring_item(name, "clean", False, None, changes)
    structural_changed = any(
        change["path"] == "type" or str(change["path"]).startswith("template")
        for change in changes
    )
    if structural_changed and variants:
        return _authoring_item(
            name,
            "locked",
            False,
            "Type and Template family cannot change while Variants exist.",
            changes,
        )
    return _authoring_item(name, "uncommitted", True, None, changes)


def _variant_authoring_view(
    graph: V2Graph,
    experiment_name: str,
    variant: VariantRecord,
    *,
    active: bool,
) -> dict[str, Any]:
    name = str(variant.document["name"])
    options = (
        dict(variant.options_document)
        if variant.options_document is not None
        else dict(split_legacy_variant(variant.document).options)
    )
    options_view = {
        "mode": "next_run",
        "message": "Current Options will be validated and captured by the next Run.",
        "summary": options,
    }
    empty_records = {"runs": [], "models": []}
    if not active:
        return {
            "name": name,
            "code": _authoring_item(
                name,
                "unavailable",
                False,
                "Commit management requires an active v2 workspace.",
                [],
            ),
            "options": options_view,
            "active_records": empty_records,
        }
    try:
        experiment_hash = graph.experiment_hash(experiment_name)
        variant_hash = graph.variant_hash(experiment_hash, name)
        current_hash = graph.current_revision(variant_hash)
        current = graph.store.load(current_hash).payload
    except NotFoundError:
        return {
            "name": name,
            "code": _authoring_item(name, "uncommitted", True, None, []),
            "options": options_view,
            "active_records": empty_records,
        }

    draft_source_record = graph.capture_source_tree(variant.path / "src", publish=False)
    draft_revision = graph.variant_revision_payload(
        variant_hash,
        variant.document,
        draft_source_record.digest,
        parent=current_hash,
        derivation_parent=None,
    )
    changes = _json_changes(
        {
            "template": deepcopy(current["template"]),
            "components": deepcopy(current["components"]),
        },
        {
            "template": deepcopy(draft_revision["template"]),
            "components": deepcopy(draft_revision["components"]),
        },
    )
    committed_source = graph.store.load(str(current["source_tree"])).payload
    changes.extend(_source_changes(committed_source, draft_source_record.payload))
    changes.sort(key=lambda item: str(item["path"]).encode("utf-8"))
    active_records = {
        "runs": _binding_names(graph, variant_run_scope(variant_hash)),
        "models": _binding_names(graph, variant_model_scope(variant_hash)),
    }
    if not changes:
        state = "clean"
        committable = False
        reason = None
    elif active_records["runs"] or active_records["models"]:
        state = "locked"
        committable = False
        reason = "Code cannot be committed while active Runs or Models exist."
    else:
        state = "uncommitted"
        committable = True
        reason = None
    return {
        "name": name,
        "code": _authoring_item(name, state, committable, reason, changes),
        "options": options_view,
        "active_records": active_records,
    }


def _binding_names(graph: V2Graph, scope: str) -> list[str]:
    return sorted(
        graph.bindings.names(scope),
        key=lambda value: value.encode("utf-8"),
    )


def _authoring_item(
    name: str,
    state: str,
    committable: bool,
    reason: str | None,
    changes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    visible = list(changes[:MAX_AUTHORING_CHANGES])
    return {
        "name": name,
        "state": state,
        "committable": committable,
        "reason": reason,
        "validation": {"status": "valid", "errors": []},
        "changes_total": len(changes),
        "changes_truncated": len(changes) > len(visible),
        "changes": visible,
    }


def _json_changes(
    before: Any,
    after: Any,
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes: list[dict[str, Any]] = []
        keys = sorted(
            set(before) | set(after),
            key=lambda value: str(value).encode("utf-8"),
        )
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                changes.append(_change(path, "added", False, None, True, after[key]))
            elif key not in after:
                changes.append(_change(path, "removed", True, before[key], False, None))
            else:
                changes.extend(_json_changes(before[key], after[key], prefix=path))
        return changes
    if before == after:
        return []
    return [_change(prefix, "modified", True, before, True, after)]


def _source_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    before_files = _source_files(before)
    after_files = _source_files(after)
    result: list[dict[str, Any]] = []
    for path in sorted(
        set(before_files) | set(after_files), key=lambda value: value.encode("utf-8")
    ):
        key = f"src/{path}"
        if path not in before_files:
            item = after_files[path]
            result.append(
                _change(
                    key, "added", False, None, True, f"New file ({item['size']} bytes)"
                )
            )
        elif path not in after_files:
            item = before_files[path]
            result.append(
                _change(
                    key,
                    "removed",
                    True,
                    f"Committed file ({item['size']} bytes)",
                    False,
                    None,
                )
            )
        elif before_files[path].get("blob") != after_files[path].get("blob"):
            old = before_files[path]
            new = after_files[path]
            result.append(
                _change(
                    key,
                    "modified",
                    True,
                    f"Committed file ({old['size']} bytes)",
                    True,
                    f"Draft file ({new['size']} bytes)",
                )
            )
    return result


def _source_files(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["path"]): item
        for item in document.get("files", [])
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }


def _change(
    path: str,
    kind: str,
    before_present: bool,
    before_value: Any,
    after_present: bool,
    after_value: Any,
) -> dict[str, Any]:
    leaf = path.rsplit(".", 1)[-1].replace("-", "_").lower()
    identity_value = leaf in {
        "hash",
        "digest",
        "id",
        "identity",
        "revision",
    } or leaf.endswith(("_hash", "_digest", "_id", "_identity", "_revision"))
    if identity_value:
        parent = path.rsplit(".", 1)[0] if "." in path else "authority"
        label = "provenance" if "digest" in leaf or "hash" in leaf else "identity"
        path = f"{parent}.{label}"
        before_value = "Committed value" if before_present else None
        after_value = "Draft value" if after_present else None
    return {
        "path": path,
        "change": kind,
        "before": {"present": before_present, "value": deepcopy(before_value)},
        "after": {"present": after_present, "value": deepcopy(after_value)},
    }


__all__ = ["build_authoring_state"]
