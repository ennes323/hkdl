"""Variant-managed evaluation facade."""

from __future__ import annotations

from hkdl.runtime.runtime import VariantRuntime
from hkdl.storage.runs import RunStore
from hkdl.storage.storage import RepositoryPaths

from .execution import (
    ExecutionFailure,
    ExecutionInterrupted,
    LifecycleConflict,
    RunExecution,
)
from .run_records import RunRecord


class EvaluationFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        action: str = "evaluation",
        address: str | None = None,
    ):
        super().__init__(message)
        self.action = action
        self.address = address


class EvaluationInterrupted(KeyboardInterrupt):
    def __init__(self, address: str):
        super().__init__(address)
        self.address = address


class Evaluation:
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

    def evaluate(
        self,
        experiment: str,
        variant: str,
        training_group: str,
        evaluation_case: str,
        *,
        seed: int | str | None = None,
        device: str = "auto",
    ) -> list[RunRecord]:
        return self._call(
            lambda: self.execution.evaluate(
                experiment,
                variant,
                training_group,
                evaluation_case,
                seed=seed,
                device=device,
            )
        )

    @staticmethod
    def _call(operation):
        try:
            return operation()
        except ExecutionInterrupted as error:
            raise EvaluationInterrupted(error.address) from error
        except ExecutionFailure as error:
            raise EvaluationFailure(
                str(error),
                action=error.action,
                address=error.address,
            ) from error


__all__ = [
    "Evaluation",
    "EvaluationFailure",
    "EvaluationInterrupted",
    "LifecycleConflict",
]
