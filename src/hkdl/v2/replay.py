"""Deterministic graph normalization; never commits mutable authored drafts."""

from __future__ import annotations

import hashlib
from copy import deepcopy

from ..config import ContractError
from ..research_json import split_legacy_variant
from ..run_contracts import evaluation_case, metric_spec, validate_tracker
from .bindings import BindingOperation
from .graph import (
    CURRENT_REVISION_NAME,
    comparison_digest,
    entity_revision_scope,
    experiment_variant_scope,
    workspace_experiment_scope,
)
from .objects import canonical_json_bytes
from .reader import GraphReader
from .projection import _extract_references


class GraphReplay:
    def __init__(self, repository):
        self.reader = GraphReader(repository)
        self.store = self.reader.graph.store
        self.objects = {}
        self.blobs = {}
        self.mapped = {}
        self.visiting = set()
        self.records = {r.graph_identity["attempt_hash"]: r for r in self.reader.runs()}
        self.models = {}
        for exp, exp_hash in self.reader.names(workspace_experiment_scope()).items():
            for variant in self.reader.names(experiment_variant_scope(exp_hash)):
                for model in self.reader.models(exp, variant):
                    self.models[model.graph_hash] = model

    def add(self, kind, payload):
        record = self.store.preview(kind, payload)
        self.objects[record.digest] = record
        return record.digest

    def payload(self, digest, kind):
        if digest in self.objects:
            record = self.objects[digest]
            if record.kind != kind:
                raise ContractError("replay reference kind mismatch")
            return deepcopy(record.payload)
        return self.reader.object(digest, kind)

    def evidence(self, record):
        request = deepcopy(record.request)
        snapshot = deepcopy(record.snapshot)
        physical = record.path.relative_to(self.reader.repository.outputs).parts
        if len(physical) != 4 or physical[2] != "runs":
            raise ContractError("legacy Run locator is invalid")
        request.update(experiment=physical[0], variant=physical[1], run_id=physical[3])
        snapshot["experiment"]["name"] = physical[0]
        snapshot["variant"]["name"] = physical[1]
        content = (
            canonical_json_bytes(
                {"schema_version": 1, "snapshot": snapshot, "request": request}
            )
            + b"\n"
        )
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        blob = self.add(
            "blob",
            {
                "content_hash": digest,
                "size": len(content),
                "media_type": "application/json",
            },
        )
        self.blobs[blob] = content
        return blob

    def code(self, digest):
        if digest is None:
            return None
        if digest in self.mapped:
            return self.mapped[digest]
        if digest in self.visiting:
            raise ContractError("Code revision cycle in migration")
        self.visiting.add(digest)
        old = self.reader.object(digest, "variant_revision")
        parent = self.code(old.get("parent"))
        derivation = self.code(old.get("derivation_parent"))
        merge = self.code(old.get("merge_parent"))
        source = self.reader.object(old["source_tree"], "source_tree")
        for entry in source.get("files", []):
            self.store.verify_blob(entry["blob"])
        payload = {
            key: deepcopy(old[key])
            for key in ("variant", "template", "source_tree", "components")
        }
        payload.update(parent=parent, derivation_parent=derivation)
        if merge is not None:
            payload["merge_parent"] = merge

        def semantic(value):
            return {
                key: item
                for key, item in value.items()
                if key not in {"parent", "derivation_parent", "merge_parent"}
            }

        if (
            parent is not None
            and derivation is None
            and merge is None
            and semantic(payload) == semantic(self.payload(parent, "variant_revision"))
        ):
            mapped = parent
        else:
            mapped = self.add("variant_revision", payload)
        self.mapped[digest] = mapped
        self.visiting.remove(digest)
        return mapped

    def attempt(self, digest):
        if digest in self.mapped:
            return self.mapped[digest]
        if digest in self.visiting:
            raise ContractError("execution dependency cycle in migration")
        self.visiting.add(digest)
        record = self.records[digest]
        old = self.reader.object(digest, "attempt")
        old_spec = self.reader.object(old["run_spec"], "run_spec")
        action = old_spec["action"]
        revision_field = {
            "train": "variant_revision",
            "eval": "evaluator_revision",
            "export": "exporter_revision",
        }[action]
        code = self.code(old_spec[revision_field])
        experiment_revision = old_spec.get("experiment_revision")
        if experiment_revision is None:
            experiment_revision = self.add(
                "experiment_revision",
                {
                    "experiment": record.graph_identity["experiment_hash"],
                    "parent": None,
                    **{
                        key: deepcopy(record.snapshot["experiment"][key])
                        for key in ("type", "question", "template")
                    },
                },
            )
        options = split_legacy_variant(record.snapshot["variant"]).options
        full_options = self.add("option_set", {"scope": "full", "document": options})
        spec = {
            "action": action,
            "experiment_revision": experiment_revision,
            revision_field: code,
            "device": record.request["exec"]["device"],
        }
        if action == "train":
            train_options = self.add(
                "option_set",
                {
                    "scope": "train",
                    **{
                        key: deepcopy(options[key])
                        for key in ("dataset", "metrics", "train")
                    },
                },
            )
            meaning = {
                "variant_revision": code,
                "train_options": train_options,
                "backend_identity": record.request["identity_fingerprint"],
                "device": spec["device"],
            }
            comparison = comparison_digest(meaning)
            spec.update(
                train_options=train_options,
                seed=record.request["target"]["seed"],
                backend_identity=meaning["backend_identity"],
                comparison_hash=comparison,
            )
            prior = self.mapped.get(old_spec["comparison_hash"])
            if prior is not None and prior != comparison:
                raise ContractError(
                    "legacy comparison identity has conflicting execution meanings"
                )
            self.mapped[old_spec["comparison_hash"]] = comparison
        else:
            spec["model"] = self.model(old_spec["model"])
            if action == "eval":
                case = record.request["target"]["evaluation_case"]
                spec["evaluation_case"] = self.add(
                    "evaluation_case",
                    {
                        "definition": evaluation_case(record.snapshot["variant"], case),
                        "metrics": metric_spec(record.snapshot["variant"], case),
                    },
                )
            else:
                spec["export_profile"] = self.add(
                    "export_profile", {"definition": deepcopy(options["infer"])}
                )
        run_spec = self.add("run_spec", spec)
        retry = old.get("retry_parent")
        retry = self.attempt(retry) if retry else None
        if retry is not None:
            parent = self.payload(retry, "attempt")
            if parent["run_spec"] != run_spec or parent["option_set"] != full_options:
                raise ContractError(
                    "retry evidence disagrees with parent RunSpec or Options"
                )
        evidence = old.get("record_evidence")
        if evidence is None:
            evidence = self.evidence(record)
        else:
            self.store.verify_blob(evidence)
        payload = deepcopy(old)
        payload.update(
            run_spec=run_spec,
            option_set=full_options,
            record_evidence=evidence,
            tracker_backends=list(
                validate_tracker(record.snapshot["variant"]["tracker"])
            ),
            retry_parent=retry,
        )
        result = self.add("attempt", payload)
        self.mapped[digest] = result
        self.mapped[old["run_spec"]] = run_spec
        self.visiting.remove(digest)
        return result

    def model(self, digest):
        if digest in self.mapped:
            return self.mapped[digest]
        old = self.reader.object(digest, "model")
        producing = self.attempt(old["producing_attempt"])
        attempt = self.payload(producing, "attempt")
        spec = self.payload(attempt["run_spec"], "run_spec")
        self.store.verify_blob(old["checkpoint_blob"])
        blob = self.reader.object(old["checkpoint_blob"], "blob")
        if blob["content_hash"] != old["checkpoint_content_hash"]:
            raise ContractError("migration Model checkpoint identity mismatch")
        document = self.models[digest].document
        prefix = f"runs/{document['producer_run']}/"
        path = document["checkpoint"]["path"]
        if not path.startswith(prefix):
            raise ContractError("migration Model checkpoint path ownership mismatch")
        payload = deepcopy(old)
        payload.update(
            producing_attempt=producing,
            experiment_revision=spec["experiment_revision"],
            variant_revision=spec["variant_revision"],
            train_options=spec["train_options"],
            comparison_hash=spec["comparison_hash"],
            checkpoint_relative_path=path[len(prefix) :],
        )
        result = self.add("model", payload)
        self.mapped[digest] = result
        return result

    def result(self, digest):
        record = self.store.load(digest)
        if record.kind == "model":
            return self.model(digest)
        if record.kind not in {"eval_result", "export_result"}:
            raise ContractError("migration result kind is invalid")
        payload = deepcopy(record.payload)
        payload["attempt"] = self.attempt(payload["attempt"])
        payload["model"] = self.model(payload["model"])
        spec = self.payload(
            self.payload(payload["attempt"], "attempt")["run_spec"], "run_spec"
        )
        field = "evaluation_case" if record.kind == "eval_result" else "export_profile"
        payload[field] = spec[field]
        for reference in payload.get("artifacts", []):
            self.store.verify_blob(reference["blob"])
        result = self.add(record.kind, payload)
        self.mapped[digest] = result
        return result

    def event(self, digest, attempt):
        chain = []
        seen = set()
        cursor = digest
        while cursor is not None:
            if cursor in seen:
                raise ContractError("migration Attempt event cycle")
            seen.add(cursor)
            payload = self.reader.object(cursor, "attempt_event")
            if payload["attempt"] != attempt:
                raise ContractError("migration Attempt event ownership mismatch")
            chain.append((cursor, payload))
            cursor = payload.get("parent")
        parent = None
        for old_hash, payload in reversed(chain):
            payload.update(attempt=self.attempt(attempt), parent=parent)
            if payload.get("result_object") is not None:
                payload["result_object"] = self.result(payload["result_object"])
            for ref in payload.get("artifacts", []):
                blob = self.reader.object(ref["blob"], "blob")
                if any(
                    ref.get(field) != blob[field] for field in ("content_hash", "size")
                ):
                    raise ContractError("migration artifact descriptor mismatch")
                self.store.verify_blob(ref["blob"])
            parent = self.add("attempt_event", payload)
            self.mapped[old_hash] = parent
        return parent

    def preview(self):
        for exp_hash in self.reader.names(workspace_experiment_scope()).values():
            for entity in self.reader.names(
                experiment_variant_scope(exp_hash)
            ).values():
                self.code(
                    self.reader.resolve(
                        entity_revision_scope(entity), CURRENT_REVISION_NAME
                    )
                )
        for digest, record in self.records.items():
            self.attempt(digest)
            self.event(record.event_hash, digest)
        for digest in self.models:
            self.model(digest)
        desired = {}
        for (scope, name), target in self.reader.bindings.items():
            pieces = scope.split("/")
            if len(pieces) == 3 and pieces[0] in {"attempt", "variant-revision"}:
                old_hash = "sha256:" + pieces[1]
                mapped = self.mapped.get(old_hash, old_hash)
                scope = f"{pieces[0]}/{mapped.removeprefix('sha256:')}/{pieces[2]}"
            key = scope, name
            value = self.mapped.get(target, target)
            if key in desired and desired[key] != value:
                raise ContractError(
                    "migration labels collapse onto conflicting meanings"
                )
            desired[key] = value
        operations = []
        for (scope, name), target in sorted(self.reader.bindings.items()):
            if desired.get((scope, name)) != target:
                operations.append(BindingOperation("unbind", scope, name, target))
        for (scope, name), target in sorted(desired.items()):
            if self.reader.bindings.get((scope, name)) != target:
                operations.append(BindingOperation("bind", scope, name, target))
        records = {record.digest: record for record in self.store.iter_records()}
        records.update(self.objects)
        _extract_references(list(records.values()))
        return operations
