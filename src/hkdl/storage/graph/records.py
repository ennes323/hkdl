"""Values shared by graph readers and recorders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionIdentity:
    experiment_hash: str
    experiment_revision_hash: str
    variant_hash: str
    variant_revision_hash: str
    option_set_hash: str | None
    run_spec_hash: str
    attempt_hash: str
    comparison_hash: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "experiment_hash": self.experiment_hash,
            "experiment_revision_hash": self.experiment_revision_hash,
            "variant_hash": self.variant_hash,
            "variant_revision_hash": self.variant_revision_hash,
            "option_set_hash": self.option_set_hash,
            "run_spec_hash": self.run_spec_hash,
            "attempt_hash": self.attempt_hash,
            "comparison_hash": self.comparison_hash,
        }


ExecutionIdentity.__module__ = "hkdl.storage.graph.execution"


@dataclass(frozen=True)
class TrainingEvaluation:
    """An Eval observation joined to the selected producing Train Run."""

    variant: str
    train_run_id: str
    eval_run_id: str
    status: str
    case_hash: str
    document: dict[str, Any] | None
