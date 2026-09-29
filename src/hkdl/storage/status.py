"""Read-only Variant-managed status projections."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from hkdl.errors import ContractError
from hkdl.execution.run_contracts import validate_tracker
from hkdl.execution.run_records import ModelRecord, RunRecord
from hkdl.interfaces.status_rendering import render_status_tree as render_status_tree
from hkdl.storage.graph.graph import (
    CURRENT_REVISION_NAME,
    V2Graph,
    entity_revision_scope,
    workspace_experiment_scope,
)
from hkdl.storage.graph.maintenance import workspace_access
from hkdl.storage.graph.projection import (
    SCHEMA_VERSION,
    GraphProjection,
    ProjectionReport,
)
from hkdl.storage.graph.reader import graph_observation

from .runs import RunStore
from .status_index import IndexReport, StatusIndex, VariantInventory
from .storage import NotFoundError, RepositoryPaths


class Status:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
    ):
        self.store = RunStore(repository)
        self.graph = V2Graph(repository)
        self._now = now or (lambda: datetime.now(timezone.utc))

    @graph_observation
    def query(
        self,
        *,
        experiment: str | None = None,
        variant: str | None = None,
        run_id: str | None = None,
        include_identities: bool = False,
    ) -> dict[str, Any]:
        if variant is not None and experiment is None:
            raise ContractError("Variant status filter requires Experiment")
        observed_at = self._observed_at()
        graph_active = self.graph.is_active()
        if run_id is not None:
            if experiment is None or variant is None:
                raise ContractError("Run status filter requires Experiment and Variant")
            record = self.store.load(experiment, variant, run_id)
            records = [record]
            result = self._tree(
                records,
                include_all_models=False,
                observed_at=observed_at,
            )
            return result if include_identities else _without_identities(result)
        if graph_active:
            records = self.store.scan(experiment=experiment, variant=variant)
            result = self._tree(
                records,
                include_all_models=True,
                observed_at=observed_at,
            )
            return result if include_identities else _without_identities(result)
        result = StatusIndex(self.store.repository).query(
            experiment=experiment,
            variant=variant,
            observed_at=observed_at,
            load=lambda inventory: self._project_variant(inventory, observed_at),
        )
        return result if include_identities else _without_identities(result)

    def index_status(self) -> IndexReport:
        """Inspect the disposable index under the same admission as API reads."""
        with workspace_access(self.store.repository):
            if self.graph.is_active():
                return self._v2_index_report(GraphProjection(self.graph).inspect())
            return StatusIndex(self.store.repository).inspect()

    def rebuild_index(self) -> IndexReport:
        """Rebuild derived state while excluding workspace cutover/recovery."""
        with workspace_access(self.store.repository):
            if self.graph.is_active():
                return self._v2_index_report(GraphProjection(self.graph).rebuild())
            observed_at = self._observed_at()
            return StatusIndex(self.store.repository).rebuild(
                lambda inventory: self._project_variant(inventory, observed_at)
            )

    def _v2_index_report(self, report: ProjectionReport) -> IndexReport:
        counts = defaultdict(int)
        for record in self.graph.store.iter_records():
            counts[record.kind] += 1
        return IndexReport(
            state=report.state,
            path=str(
                GraphProjection(self.graph).path.relative_to(self.store.repository.root)
            ),
            schema_version=SCHEMA_VERSION,
            variants=counts["variant"],
            runs=counts["attempt"],
            models=counts["model"],
            detail=report.detail,
        )

    def _observed_at(self) -> datetime:
        observed_at = self._now()
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ContractError("Status clock must be timezone-aware")
        return observed_at

    def _project_variant(
        self,
        inventory: VariantInventory,
        observed_at: datetime,
    ) -> dict[str, Any] | None:
        records = self.store.scan(
            experiment=inventory.experiment,
            variant=inventory.variant,
        )
        document = self._tree(
            records,
            include_all_models=True,
            observed_at=observed_at,
        )
        if not document["experiments"]:
            return None
        experiments = document["experiments"]
        if len(experiments) != 1 or len(experiments[0]["variants"]) != 1:
            raise ContractError("status Variant projection scope changed")
        return experiments[0]["variants"][0]

    def _tree(
        self,
        records: list[RunRecord],
        *,
        include_all_models: bool,
        observed_at: datetime,
    ) -> dict[str, Any]:
        graph_active = self.graph.is_active()
        by_variant: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
        for record in records:
            by_variant[
                (
                    record.request["experiment"],
                    record.request["variant"],
                )
            ].append(record)
        experiments: dict[str, dict[str, Any]] = {}
        for (experiment, variant), variant_runs in sorted(
            by_variant.items(),
            key=lambda item: (
                item[0][0].encode("utf-8"),
                item[0][1].encode("utf-8"),
            ),
        ):
            experiment_node = experiments.setdefault(
                experiment,
                {"name": experiment, "variants": []},
            )
            if graph_active:
                experiment_node["experiment_hash"] = self.store.graph_reader().resolve(
                    workspace_experiment_scope(), experiment
                )
            variant_hash = None
            variant_revision_hash = None
            if graph_active:
                reader = self.store.graph_reader()
                _, variant_hash = reader.entities(experiment, variant)
                variant_revision_hash = reader.resolve(
                    entity_revision_scope(variant_hash), CURRENT_REVISION_NAME
                )
            all_models = (
                self.store.scan_model_manifests(experiment=experiment, variant=variant)
                if include_all_models
                else self._selected_models(experiment, variant, variant_runs)
            )
            model_by_id = {model.document["model_id"]: model for model in all_models}
            groups: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
            for model in all_models:
                self._seed_node(groups, model.document["training_group"], model)[
                    "model"
                ] = self._model_summary(model)
            for record in variant_runs:
                group, seed = _run_group_seed(record, model_by_id)
                seed_node = self._seed_node(
                    groups,
                    group,
                    model_by_id.get(record.request["target"].get("model_id")),
                    seed=seed,
                )
                seed_node["runs"].append(self._run_summary(record, observed_at))
            group_documents: list[dict[str, Any]] = []
            for group_name in sorted(groups, key=lambda value: value.encode("utf-8")):
                seeds = [
                    groups[group_name][seed] for seed in sorted(groups[group_name])
                ]
                for seed_node in seeds:
                    seed_node["runs"].sort(key=_run_number)
                group_documents.append(
                    {
                        "name": group_name,
                        "seeds": seeds,
                        "aggregates": _aggregates(seeds),
                    }
                )
            variant_node = {
                "name": variant,
                "training_groups": group_documents,
            }
            if graph_active:
                variant_node.update(
                    variant_hash=variant_hash,
                    variant_revision_hash=variant_revision_hash,
                )
            experiment_node["variants"].append(variant_node)
        return {"experiments": list(experiments.values())}

    def _selected_models(
        self, experiment: str, variant: str, records: list[RunRecord]
    ) -> list[ModelRecord]:
        models: dict[str, ModelRecord] = {}
        for record in records:
            action = record.request["action"]
            if action == "train":
                if record.state["status"] != "done":
                    # A receipt may publish a Model before terminal Run state.
                    for model in self.store.scan_model_manifests(
                        experiment=experiment, variant=variant
                    ):
                        if model.document["producer_run"] == record.request["run_id"]:
                            models[model.document["model_id"]] = model
                    continue
                model_id = record.state["result"]["model_id"]
            else:
                model_id = record.request["target"]["model_id"]
            if model_id not in models:
                try:
                    models[model_id] = self.store.load_model_manifest(
                        experiment, variant, model_id
                    )
                except NotFoundError:
                    # Legacy Train/Eval status permits an absent Model manifest.
                    # An active graph has already validated the Run's Model
                    # binding; lookup failures there are broken evidence.
                    if self.graph.is_active():
                        raise
                    continue
            if action == "train" and (
                models[model_id].document["producer_run"] != record.request["run_id"]
            ):
                raise ContractError(
                    f"Run Model producer ownership mismatch: {model_id}"
                )
        return sorted(
            models.values(),
            key=lambda model: (
                model.document["created_at"],
                model.document["model_id"].encode("utf-8"),
            ),
        )

    @staticmethod
    def _seed_node(
        groups: dict[str, dict[int, dict[str, Any]]],
        group: str,
        model: ModelRecord | None,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if model is not None:
            seed = model.document["seed"]
        if seed is None:
            raise ContractError("Run has no Training Group seed")
        return groups[group].setdefault(
            seed,
            {
                "seed": seed,
                "model": None,
                "runs": [],
            },
        )

    def _run_summary(
        self,
        record: RunRecord,
        observed_at: datetime,
    ) -> dict[str, Any]:
        primary = None
        values: dict[str, float] = {}
        artifacts: list[str] = []
        if record.request["action"] == "eval" and record.state["status"] == "done":
            evaluation = self.store.load_evaluation(record)
            primary = dict(evaluation["primary"])
            values = dict(evaluation["values"])
            artifacts = list(evaluation["artifacts"])
        train = record.snapshot["variant"].get("train", {})
        is_train = record.request["action"] == "train"

        def configured_integer(name: str) -> int | None:
            value = train.get(name) if is_train else None
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
                else None
            )

        created_at = record.request["created_at"]
        end = (
            observed_at
            if record.state["status"] in {"allocated", "running"}
            else _parse_timestamp(record.state["updated_at"])
        )
        elapsed_seconds = max(
            0,
            int((end - _parse_timestamp(created_at)).total_seconds()),
        )
        summary = {
            "run_id": record.request["run_id"],
            "action": record.request["action"],
            "status": record.state["status"],
            "retry_of": record.request["retry_of"],
            "model_id": record.request["target"].get("model_id"),
            "evaluation_case": record.request["target"].get("evaluation_case"),
            "primary": primary,
            "values": values,
            "artifacts": artifacts,
            "reason": record.state["reason"],
            "tracker_run_id": record.state["tracker_run_id"],
            "tracker_backends": list(
                validate_tracker(record.snapshot["variant"]["tracker"])
            ),
            "metric_summary": self.store.load_training_metric_summary(record),
            "created_at": created_at,
            "updated_at": record.state["updated_at"],
            "elapsed_seconds": elapsed_seconds,
            "device": record.request["exec"].get("device"),
            "configured_batch_size": configured_integer("batch_size"),
            "configured_steps": configured_integer("steps"),
            "configured_epochs": configured_integer("epochs"),
            "best_checkpoint": record.state["best_checkpoint"],
            "last_checkpoint": record.state["last_checkpoint"],
        }
        if self.graph.is_active():
            summary.update(self.store.graph_reader().identity(record).as_dict())
        return summary

    def _model_summary(self, model: ModelRecord) -> dict[str, Any]:
        summary = _model_summary(model)
        if self.graph.is_active():
            summary["model_hash"] = (
                model.graph_hash
                or self.store.graph_reader().model_hash(
                    str(model.document["experiment"]),
                    str(model.document["variant"]),
                    str(model.document["model_id"]),
                )
            )
        return summary


def _without_identities(value: Any) -> Any:
    """Remove content-addressed implementation identities from normal views."""

    if isinstance(value, dict):
        return {
            key: _without_identities(item)
            for key, item in value.items()
            if not key.endswith("_hash")
        }
    if isinstance(value, list):
        return [_without_identities(item) for item in value]
    return value


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _model_summary(model: ModelRecord) -> dict[str, Any]:
    return {
        "model_id": model.document["model_id"],
        "producer_run": model.document["producer_run"],
        "device": model.document["device"],
        "training_fingerprint": model.document["training_fingerprint"],
        "created_at": model.document["created_at"],
    }


def _run_group_seed(
    record: RunRecord,
    models: dict[str, ModelRecord],
) -> tuple[str, int]:
    target = record.request["target"]
    if "training_group" in target and "seed" in target:
        return target["training_group"], target["seed"]
    model_id = target["model_id"]
    model = models.get(model_id)
    if model is None:
        raise ContractError(f"Run references a missing Model: {model_id}")
    return model.document["training_group"], model.document["seed"]


def _run_number(run: dict[str, Any]) -> int:
    return int(run["run_id"].removeprefix("run-"))


def _aggregates(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    eligible = sum(seed["model"] is not None for seed in seeds)
    for seed in seeds:
        for run in seed["runs"]:
            if (
                run["action"] == "eval"
                and run["status"] == "done"
                and run["evaluation_case"] is not None
            ):
                for metric, value in run["values"].items():
                    values[(run["evaluation_case"], metric)].append(float(value))
    result: list[dict[str, Any]] = []
    for (case, metric), samples in sorted(
        values.items(),
        key=lambda item: (
            item[0][0].encode("utf-8"),
            item[0][1].encode("utf-8"),
        ),
    ):
        result.append(
            {
                "evaluation_case": case,
                "metric": metric,
                "eligible": eligible,
                "count": len(samples),
                "mean": statistics.fmean(samples),
                "sample_std": statistics.stdev(samples) if len(samples) >= 2 else None,
            }
        )
    return result


__all__ = ["Status", "render_status_tree"]
