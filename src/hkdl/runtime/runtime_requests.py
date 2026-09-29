"""Exact parent-side JSON payload builders for the isolated runtime worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hkdl.authoring.authoring_records import VariantRecord
from hkdl.execution.run_contracts import validate_tracker


def build_validate_request(
    variant: VariantRecord,
    *,
    action: str,
    cfg: dict[str, Any],
    selected: dict[str, str],
    seed: int,
    device: str,
    identity_fallback: dict[str, Any] | None,
    runtime_target: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime_cfg = _runtime_cfg(cfg, action, runtime_target)
    return {
        "operation": "validate",
        "action": action,
        "source": str(variant.path / "src"),
        "cfg": runtime_cfg,
        "selected": selected,
        "exec": {"seed": seed, "device": device},
        "identity_fallback": identity_fallback or {},
        "repository_root": str(variant.path.parents[2]),
        "tracker_backends": _tracker_backends(runtime_cfg),
    }


def build_train_request(
    variant: VariantRecord,
    *,
    cfg: dict[str, Any],
    selected: dict[str, str],
    exec_info: dict[str, Any],
    run_dir: Path,
    resume_from: Path | None,
    tracker_run_id: str | None,
    attempt_path: Path | None,
    runtime_target: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime_cfg = _runtime_cfg(cfg, "train", runtime_target)
    return {
        "operation": "train",
        "action": "train",
        "source": str(variant.path / "src"),
        "cfg": runtime_cfg,
        "selected": selected,
        "exec": exec_info,
        "run_dir": str(run_dir),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "tracker_run_id": tracker_run_id,
        "attempt_path": str(attempt_path) if attempt_path is not None else None,
        "repository_root": str(variant.path.parents[2]),
        "tracker_backends": _tracker_backends(runtime_cfg),
    }


def build_evaluate_request(
    variant: VariantRecord,
    *,
    cfg: dict[str, Any],
    selected: dict[str, str],
    exec_info: dict[str, Any],
    run_dir: Path,
    checkpoint: Path,
    results_dir: Path | None,
    tracker_run_id: str | None,
    attempt_path: Path | None,
    runtime_target: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime_cfg = _runtime_cfg(cfg, "eval", runtime_target)
    return {
        "operation": "evaluate",
        "action": "eval",
        "source": str(variant.path / "src"),
        "cfg": runtime_cfg,
        "selected": selected,
        "exec": exec_info,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "results_dir": str(
            results_dir if results_dir is not None else run_dir / "artifacts/results"
        ),
        "tracker_run_id": tracker_run_id,
        "attempt_path": str(attempt_path) if attempt_path is not None else None,
        "tracker_backends": _tracker_backends(runtime_cfg),
    }


def build_export_request(
    variant: VariantRecord,
    *,
    cfg: dict[str, Any],
    selected: dict[str, str],
    exec_info: dict[str, Any],
    run_dir: Path,
    export_dir: Path,
    checkpoint: Path,
    tracker_run_id: str | None,
    attempt_path: Path | None,
    runtime_target: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime_cfg = _runtime_cfg(cfg, "export", runtime_target)
    return {
        "operation": "export",
        "action": "export",
        "source": str(variant.path / "src"),
        "cfg": runtime_cfg,
        "selected": selected,
        "exec": exec_info,
        "run_dir": str(run_dir),
        "export_dir": str(export_dir),
        "checkpoint": str(checkpoint),
        "tracker_run_id": tracker_run_id,
        "attempt_path": str(attempt_path) if attempt_path is not None else None,
        "tracker_backends": _tracker_backends(runtime_cfg),
    }


def build_tracker_request(
    variant: VariantRecord,
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    current_tracker_run_id: str | None,
    metadata: dict[str, Any],
    tracker_backends: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "operation": "tracker",
        "source": str(variant.path / "src"),
        "cfg": cfg,
        "run_dir": str(run_dir),
        "repository_root": str(variant.path.parents[2]),
        "current_tracker_run_id": current_tracker_run_id,
        "metadata": metadata,
        "tracker_backends": list(tracker_backends),
    }


def build_tracker_metrics_request(
    variant: VariantRecord,
    *,
    tracker_run_id: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    return {
        "operation": "tracker_metrics",
        "tracker_run_id": tracker_run_id,
        "values": values,
        "repository_root": str(variant.path.parents[2]),
    }


def build_tracker_finish_request(
    variant: VariantRecord,
    *,
    tracker_run_id: str,
    status: str,
) -> dict[str, Any]:
    return {
        "operation": "tracker_finish",
        "tracker_run_id": tracker_run_id,
        "run_status": status,
        "repository_root": str(variant.path.parents[2]),
    }


def _runtime_cfg(
    cfg: dict[str, Any],
    action: str,
    runtime_target: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime_cfg = dict(cfg)
    runtime_cfg["runtime"] = {
        "action": action,
        "target": runtime_target or {},
    }
    return runtime_cfg


def _tracker_backends(cfg: dict[str, Any]) -> list[str]:
    return list(validate_tracker(cfg["variant"]["tracker"]))
