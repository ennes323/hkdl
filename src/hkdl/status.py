"""Read-only Variant-managed status projections."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from .config import ContractError
from .run_contracts import validate_tracker
from .runs import ModelRecord, RunRecord, RunStore
from .status_index import IndexReport, StatusIndex, VariantInventory
from .storage import RepositoryPaths
from .v2.execution import GraphRecorder
from .v2.projection import SCHEMA_VERSION, GraphProjection, ProjectionReport
from .v2.reader import graph_observation
from .v2.graph import (
    CURRENT_REVISION_NAME,
    entity_revision_scope,
    workspace_experiment_scope,
)


class Status:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
    ):
        self.store = RunStore(repository)
        self.graph = GraphRecorder(repository)
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
        graph_active = self.graph.active()
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
        if self.graph.active():
            return self._v2_index_report(GraphProjection(self.graph.graph).inspect())
        return StatusIndex(self.store.repository).inspect()

    def rebuild_index(self) -> IndexReport:
        if self.graph.active():
            return self._v2_index_report(GraphProjection(self.graph.graph).rebuild())
        observed_at = self._observed_at()
        return StatusIndex(self.store.repository).rebuild(
            lambda inventory: self._project_variant(inventory, observed_at)
        )

    def _v2_index_report(self, report: ProjectionReport) -> IndexReport:
        counts = defaultdict(int)
        for record in self.graph.graph.store.iter_records():
            counts[record.kind] += 1
        return IndexReport(
            state=report.state,
            path=str(
                GraphProjection(self.graph.graph).path.relative_to(
                    self.store.repository.root
                )
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
        graph_active = self.graph.active()
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
            all_models = self.store.scan_model_manifests(
                experiment=experiment,
                variant=variant,
            )
            if not include_all_models:
                referenced = {
                    record.request["target"].get("model_id") for record in variant_runs
                }
                producers = {
                    record.request["run_id"]
                    for record in variant_runs
                    if record.request["action"] == "train"
                }
                all_models = [
                    model
                    for model in all_models
                    if model.document["model_id"] in referenced
                    or model.document["producer_run"] in producers
                ]
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
        if self.graph.active():
            summary.update(self.graph.identity(record).as_dict())
        return summary

    def _model_summary(self, model: ModelRecord) -> dict[str, Any]:
        summary = _model_summary(model)
        if self.graph.active():
            summary["model_hash"] = model.graph_hash or self.graph.model_hash(
                str(model.document["experiment"]),
                str(model.document["variant"]),
                str(model.document["model_id"]),
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


def render_status_tree(document: dict[str, Any], *, full: bool = False) -> str:
    experiments = document["experiments"]
    if not experiments:
        return "no runs\n"
    lines: list[str] = []
    for experiment in experiments:
        lines.append(experiment["name"])
        variants = experiment["variants"]
        for variant_index, variant in enumerate(variants):
            variant_last = variant_index == len(variants) - 1
            lines.append(f"{'`--' if variant_last else '|--'} {variant['name']}")
            variant_prefix = "    " if variant_last else "|   "
            groups = variant["training_groups"]
            for group_index, group in enumerate(groups):
                group_last = group_index == len(groups) - 1
                group_branch = "`--" if group_last else "|--"
                lines.append(f"{variant_prefix}{group_branch} {group['name']}")
                group_prefix = variant_prefix + ("    " if group_last else "|   ")
                seeds = group["seeds"]
                for seed_index, seed in enumerate(seeds):
                    seed_last = seed_index == len(seeds) - 1
                    model = (
                        f"  model={seed['model']['model_id']}"
                        if seed["model"] is not None
                        else ""
                    )
                    seed_branch = "`--" if seed_last else "|--"
                    lines.append(
                        f"{group_prefix}{seed_branch} seed={seed['seed']}{model}"
                    )
                    seed_prefix = group_prefix + ("    " if seed_last else "|   ")
                    runs = seed["runs"]
                    for run_index, run in enumerate(runs):
                        run_last = run_index == len(runs) - 1
                        branch = "`--" if run_last else "|--"
                        fields = [
                            run["action"],
                            run["run_id"],
                            run["status"],
                            f"elapsed={_format_elapsed(run['elapsed_seconds'])}",
                        ]
                        if run["configured_batch_size"] is not None:
                            fields.append(f"batch_size={run['configured_batch_size']}")
                        if run["configured_steps"] is not None:
                            fields.append(f"steps={run['configured_steps']}")
                        if run["configured_epochs"] is not None:
                            fields.append(f"epochs={run['configured_epochs']}")
                        if run["retry_of"] is not None:
                            fields.append(f"retry_of={run['retry_of']}")
                        if run["evaluation_case"] is not None:
                            fields.append(f"case={run['evaluation_case']}")
                        if run["primary"] is not None:
                            fields.append(
                                f"primary={run['primary']['name']}:"
                                f"{run['primary']['value']}"
                            )
                        if run["reason"] is not None:
                            fields.append(f"reason={run['reason']}")
                        if run["tracker_run_id"] is not None:
                            fields.append(f"tracker={run['tracker_run_id']}")
                        if run["metric_summary"]:
                            summary = ",".join(
                                f"{name}:{metric['last_value']}@"
                                f"{metric['last_step']}({metric['count']})"
                                for name, metric in sorted(
                                    run["metric_summary"].items(),
                                    key=lambda item: item[0].encode("utf-8"),
                                )
                            )
                            fields.append(f"metrics={summary}")
                        fields.append(f"updated={run['updated_at']}")
                        lines.append(f"{seed_prefix}{branch} {'  '.join(fields)}")
                        if full:
                            detail_prefix = seed_prefix + (
                                "    " if run_last else "|   "
                            )
                            lines.append(
                                f"{detail_prefix}timing: "
                                f"created_at={run['created_at']} "
                                f"updated_at={run['updated_at']} "
                                f"elapsed_seconds={run['elapsed_seconds']}"
                            )
                            execution = f"device={run['device']}"
                            for name in (
                                "configured_batch_size",
                                "configured_steps",
                                "configured_epochs",
                            ):
                                if run[name] is not None:
                                    execution += f" {name}={run[name]}"
                            lines.append(f"{detail_prefix}execution: {execution}")
                            lines.append(
                                f"{detail_prefix}checkpoints: "
                                f"best={run['best_checkpoint']} "
                                f"last={run['last_checkpoint']}"
                            )
                            backends = ",".join(run["tracker_backends"]) or "none"
                            lines.append(
                                f"{detail_prefix}tracker: backends={backends} "
                                f"external_run_id={run['tracker_run_id']}"
                            )
                            if run["metric_summary"]:
                                for name, metric in sorted(
                                    run["metric_summary"].items(),
                                    key=lambda item: item[0].encode("utf-8"),
                                ):
                                    lines.append(
                                        f"{detail_prefix}metric: {name} "
                                        f"count={metric['count']} "
                                        f"last_step={metric['last_step']} "
                                        f"last_value={metric['last_value']}"
                                    )
                            else:
                                lines.append(f"{detail_prefix}metrics: none")
                            if run["values"]:
                                values = ",".join(
                                    f"{name}={value}"
                                    for name, value in sorted(
                                        run["values"].items(),
                                        key=lambda item: item[0].encode("utf-8"),
                                    )
                                )
                                lines.append(f"{detail_prefix}evaluation: {values}")
                            if run["artifacts"]:
                                lines.append(
                                    f"{detail_prefix}artifacts: "
                                    f"{','.join(run['artifacts'])}"
                                )
    return "\n".join(lines) + "\n"


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _format_elapsed(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


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
