"""Render structured status snapshots without owning their storage query."""

from __future__ import annotations

from typing import Any


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


def _format_elapsed(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


__all__ = ["render_status_tree"]
