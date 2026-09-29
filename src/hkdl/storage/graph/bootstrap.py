"""Recoverable first JSON Experiment publication, never an implicit migration."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from hkdl.authoring.authoring_records import ExperimentRecord
from hkdl.authoring.config import validate_experiment
from hkdl.authoring.research_json import dump_json, load_json_file
from hkdl.errors import ContractError
from hkdl.storage.storage import atomic_write_new, publish_directory

from .graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    entity_revision_scope,
    workspace_experiment_scope,
)
from .maintenance import BOOTSTRAP_JOURNAL, WorkspaceBusy
from .objects import _ensure_real_directory, _fsync_directory
from .projection import GraphProjection


def _store_directory(root):
    for path in (root / ".hkdl", root / ".hkdl/store"):
        _ensure_real_directory(path)
        _fsync_directory(path.parent)
    return root / ".hkdl/store"


def _empty(path, *, allowed=()):
    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_dir() or set(path.iterdir()) - set(allowed):
        raise ContractError(
            f"first JSON initialization requires an empty workspace; "
            f"existing history at {path} requires explicit migration"
        )


def bootstrap_experiment(authoring, record, target):
    root = authoring.repository.root
    _empty(root / "outputs")
    _empty(root / "experiments")
    # Even an orphan graph or a lone authoring marker is existing state, not
    # proof of an empty workspace. Only our persisted intent permits recovery.
    _empty(root / ".hkdl/store")
    store = _store_directory(root)
    validate_experiment(record.document)
    journal = {
        "schema_version": 1,
        "candidate": f".{target.name}.candidate-{secrets.token_hex(16)}",
        "experiment": record.document,
    }
    try:
        atomic_write_new(store / "bootstrap.json", dump_json(journal))
        return recover_bootstrap(
            authoring, target.name, str(record.document["template"]["name"])
        )
    except OSError as error:
        raise ContractError(
            "workspace initialization interrupted; repeat the original "
            "hkdl experiment create command"
        ) from error


def _draft(path, expected, *, complete=False):
    if complete and not os.path.lexists(path):
        path.mkdir()
        _fsync_directory(path.parent)
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"bootstrap draft must be a real directory: {path}")
    names = {item.name for item in path.iterdir()}
    allowed = {"experiment.json", "notes"}
    if names - allowed or (not complete and names != allowed):
        raise ContractError(f"bootstrap draft changed; preserved at {path}")
    _empty(path / "notes")
    if (
        "experiment.json" in names
        and load_json_file(path / "experiment.json") != expected
    ):
        raise ContractError(f"bootstrap draft changed; preserved at {path}")
    if complete:
        if "notes" not in names:
            (path / "notes").mkdir()
        if "experiment.json" not in names:
            atomic_write_new(path / "experiment.json", dump_json(expected))
        _fsync_directory(path / "notes")
        _fsync_directory(path)


def _marker(path, *, publish=True):
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != b"v2\n":
            raise ContractError(
                f"bootstrap marker conflicts with existing state: {path}"
            )
    elif publish:
        atomic_write_new(path, "v2\n")


def recover_bootstrap(authoring, name, template_name):
    try:
        return _recover_bootstrap(authoring, name, template_name)
    except OSError as error:
        raise ContractError(
            "workspace initialization interrupted; repeat the original "
            "hkdl experiment create command"
        ) from error


def _recover_bootstrap(authoring, name, template_name):
    root = authoring.repository.root
    path = root / BOOTSTRAP_JOURNAL
    if not os.path.lexists(path):
        return None
    _store_directory(root)
    document = load_json_file(path)
    if (
        set(document) != {"schema_version", "candidate", "experiment"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["candidate"], str)
    ):
        raise ContractError("bootstrap journal is invalid")
    experiment = document["experiment"]
    validate_experiment(experiment)
    if experiment["name"] != name or experiment["template"]["name"] != template_name:
        raise WorkspaceBusy(
            "workspace initialization pending; repeat hkdl experiment create "
            f"{experiment['name']} --template {experiment['template']['name']}"
        )
    candidate_name = document["candidate"]
    if (
        Path(candidate_name).name != candidate_name
        or not candidate_name.startswith(f".{name}.candidate-")
        or not candidate_name.removeprefix(f".{name}.candidate-")
    ):
        raise ContractError("bootstrap candidate path is invalid")
    candidate = root / "experiments" / candidate_name
    target = root / "experiments" / name
    _empty(root / "outputs")
    _empty(root / "experiments", allowed=(candidate, target))
    expected = {
        "schema_version": 2,
        **{key: experiment[key] for key in ("type", "question", "template")},
    }
    has_candidate = os.path.lexists(candidate)
    has_target = os.path.lexists(target)
    if has_candidate and has_target:
        raise ContractError(
            "bootstrap requires exactly one owned draft; files preserved"
        )
    graph = V2Graph(authoring.repository)
    _marker(root / ".hkdl/store/AUTHORING_CURRENT", publish=False)
    _marker(graph.current_path, publish=False)
    nonce = f"bootstrap:{candidate_name}"
    entity = graph.store.preview(
        "experiment", {"created_at": experiment["created_at"], "creation_nonce": nonce}
    )
    revision = graph.store.preview(
        "experiment_revision",
        {
            "experiment": entity.digest,
            "parent": None,
            **{key: experiment[key] for key in ("type", "question", "template")},
        },
    )
    expected_bindings = {
        (workspace_experiment_scope(), name): entity.digest,
        (entity_revision_scope(entity.digest), CURRENT_REVISION_NAME): revision.digest,
    }
    bindings = graph.bindings.bindings()
    if bindings and bindings != expected_bindings:
        raise ContractError("bootstrap graph conflicts with existing bindings")
    if not bindings and graph.bindings.head() is not None:
        raise ContractError("bootstrap graph contains unexpected binding history")
    if (
        bindings
        and graph.store.load(graph.bindings.head()).payload["previous"] is not None
    ):
        raise ContractError("bootstrap graph contains unexpected binding history")
    for existing in graph.store.iter_records():
        if existing.digest in {entity.digest, revision.digest}:
            continue
        operations = existing.payload.get("operations", [])
        expected_operations = [
            {"action": "bind", "scope": scope, "name": key, "target": value}
            for (scope, key), value in expected_bindings.items()
        ]
        if (
            existing.kind != "binding_transaction"
            or existing.payload.get("previous") is not None
            or operations != expected_operations
        ):
            raise ContractError("bootstrap store contains unrelated objects; preserved")
    _empty(graph.store.blobs)
    if any(
        os.path.lexists(p)
        for p in (graph.current_path, root / ".hkdl/store/AUTHORING_CURRENT")
    ):
        if not has_target or bindings != expected_bindings:
            raise ContractError("bootstrap marker appeared before verified publication")
    _draft(target if has_target else candidate, expected, complete=not has_target)
    record = ExperimentRecord(target, experiment, 2)
    graph.commit_experiment(record, creation_nonce=nonce)
    if graph.bindings.bindings() != expected_bindings:
        raise ContractError("bootstrap graph verification failed")
    GraphProjection(graph).rebuild()
    if not has_target:
        publish_directory(candidate, target)
    _marker(root / ".hkdl/store/AUTHORING_CURRENT")
    _marker(graph.current_path)
    if not graph.is_active():
        raise ContractError("bootstrap graph activation failed")
    path.unlink()
    _fsync_directory(path.parent)
    return record
