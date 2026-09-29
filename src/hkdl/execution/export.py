"""Variant-managed export facade."""

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


class ExportFailure(RuntimeError):
    def __init__(self, message: str, *, address: str | None = None):
        super().__init__(message)
        self.address = address


class ExportInterrupted(KeyboardInterrupt):
    def __init__(self, address: str):
        super().__init__(address)
        self.address = address


class Export:
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

    def export(
        self,
        experiment: str,
        variant: str,
        model_id: str,
        *,
        device: str = "auto",
    ) -> RunRecord:
        try:
            return self.execution.export(
                experiment,
                variant,
                model_id,
                device=device,
            )
        except ExecutionInterrupted as error:
            raise ExportInterrupted(error.address) from error
        except ExecutionFailure as error:
            raise ExportFailure(str(error), address=error.address) from error


__all__ = [
    "Export",
    "ExportFailure",
    "ExportInterrupted",
    "LifecycleConflict",
]
