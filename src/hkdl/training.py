"""Variant-managed training facade."""

from __future__ import annotations

from .execution import ExecutionFailure, ExecutionInterrupted, RunExecution
from .runs import RunRecord, RunStore
from .runtime import VariantRuntime
from .storage import RepositoryPaths


class TrainingFailure(RuntimeError):
    def __init__(self, message: str, *, address: str | None = None):
        super().__init__(message)
        self.address = address


class TrainingInterrupted(KeyboardInterrupt):
    def __init__(self, address: str):
        super().__init__(address)
        self.address = address


class Training:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        runtime: VariantRuntime | None = None,
        store: RunStore | None = None,
    ):
        self.execution = RunExecution(
            repository,
            runtime=runtime,
            store=store,
        )

    def train(
        self,
        experiment_name: str,
        variant_name: str,
        training_group: str,
        *,
        seed: int = 0,
        device: str = "auto",
        tracker: str | None = None,
    ) -> RunRecord:
        try:
            return self.execution.train(
                experiment_name,
                variant_name,
                training_group,
                seed=seed,
                device=device,
                tracker=tracker,
            )
        except ExecutionInterrupted as error:
            raise TrainingInterrupted(error.address) from error
        except ExecutionFailure as error:
            raise TrainingFailure(str(error), address=error.address) from error
