"""New-Run retry facade."""

from __future__ import annotations

from .execution import (
    ExecutionFailure,
    ExecutionInterrupted,
    LifecycleConflict,
    RunExecution,
)
from .runs import RunRecord, RunStore
from .runtime import VariantRuntime
from .storage import RepositoryPaths


class RecoveryFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        action: str,
        address: str | None = None,
    ):
        super().__init__(message)
        self.action = action
        self.address = address


class RecoveryInterrupted(KeyboardInterrupt):
    def __init__(self, address: str):
        super().__init__(address)
        self.address = address


class Recovery:
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

    def retry(
        self,
        experiment: str,
        variant: str,
        run_id: str,
        *,
        tracker: str | None = None,
    ) -> RunRecord:
        try:
            return self.execution.retry(experiment, variant, run_id, tracker=tracker)
        except ExecutionInterrupted as error:
            raise RecoveryInterrupted(error.address) from error
        except ExecutionFailure as error:
            raise RecoveryFailure(
                str(error),
                action=error.action,
                address=error.address,
            ) from error


__all__ = [
    "LifecycleConflict",
    "Recovery",
    "RecoveryFailure",
    "RecoveryInterrupted",
]
