"""Private durable receipt validation and terminal publication service."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hkdl.authoring.authoring import Authoring
from hkdl.errors import ContractError
from hkdl.runtime.runtime import RuntimeFailure
from hkdl.storage.runs import RunStore
from hkdl.storage.storage import (
    RepositoryPaths,
    atomic_write_new,
    compute_source_digest,
    publish_directory,
    publish_file,
)

from .attempts import (
    load_attempt,
    remove_attempt,
    validate_attempt_owner,
    write_attempt,
)
from .run_records import ModelRecord, RunRecord


class ReceiptCommitter:
    def __init__(
        self,
        repository: RepositoryPaths,
        store: RunStore,
        authoring: Authoring,
    ):
        self.repository = repository
        self.store = store
        self.authoring = authoring

    # Resume publication from the durable journal phase: validate worker output,
    # record or verify a ready receipt, reconcile any completed publication, and
    # remove the journal only after the Run is done.
    def complete_from_journal(self, record: RunRecord, attempt_path: Path) -> RunRecord:
        """Complete a Run from its durable worker journal.

        Verify journal ownership before any publication or candidate cleanup. The
        action publisher then resumes prior publication before removing the journal.
        """
        journal = load_attempt(attempt_path)
        if journal is None or journal["phase"] not in {"worker_done", "ready"}:
            raise ContractError("attempt journal has no durable worker result")
        validate_attempt_owner(journal, record.request["action"])
        if journal["action"] == "train":
            return self._complete_train(record, journal, attempt_path)
        if journal["action"] == "eval":
            return self._complete_eval(record, journal, attempt_path)
        return self._complete_export(record, journal, attempt_path)

    def commit_receipt(self, record: RunRecord) -> RunRecord:
        current_variant = self.authoring.check_variant(
            record.request["experiment"],
            record.request["variant"],
        )
        if (
            compute_source_digest(current_variant.path / "src")
            != record.request["source_digest"]
        ):
            raise ContractError("Variant source changed since the original Run")
        return self.complete_from_journal(record, record.path / ".attempt.json")

    def _complete_train(
        self,
        record: RunRecord,
        journal: dict[str, Any],
        attempt_path: Path,
    ) -> RunRecord:
        best, last, digests, metric_files, metric_digests = self._train_receipt(
            record,
            journal,
        )
        model = self._train_model(record, journal, best, digests[best])
        if journal["phase"] == "ready":
            if journal["result"].get("model_id") != model.document["model_id"]:
                raise ContractError("Train receipt Model disagrees with publication")
        else:
            journal["phase"] = "ready"
            journal["checkpoint"] = {"best": best, "last": last}
            journal["result"] = {
                "model_id": model.document["model_id"],
                "best_checkpoint": best,
                "last_checkpoint": last,
                "digests": digests,
                "metrics": metric_files,
                "metric_digests": metric_digests,
            }
            write_attempt(attempt_path, journal)
        record = self.store.update_state(
            record,
            status="done",
            result={"model_id": model.document["model_id"]},
            best_checkpoint=best,
            last_checkpoint=last,
        )
        remove_attempt(attempt_path)
        return record

    def _train_receipt(
        self,
        record: RunRecord,
        journal: dict[str, Any],
    ) -> tuple[str, str, dict[str, str], Any, dict[str, str]]:
        result = journal["result"]
        if journal["phase"] == "worker_done":
            best = _checkpoint_result(record, result.get("best_checkpoint"))
            last = _checkpoint_result(record, result.get("last_checkpoint"))
            digests = {
                best: _file_digest(record.path / best),
                last: _file_digest(record.path / last),
            }
            metric_files = result.get("metrics")
            metric_digests = self.store.validate_completed_training_metrics(
                record,
                metric_files,
            )
            return best, last, digests, metric_files, metric_digests
        return self._ready_train_receipt(record, journal)

    def _ready_train_receipt(
        self,
        record: RunRecord,
        journal: dict[str, Any],
    ) -> tuple[str, str, dict[str, str], Any, dict[str, str]]:
        result = journal["result"]
        best = result.get("best_checkpoint")
        last = result.get("last_checkpoint")
        digests = result.get("digests")
        metric_files = result.get("metrics")
        metric_digests = result.get("metric_digests")
        if (
            not isinstance(best, str)
            or not isinstance(last, str)
            or journal["checkpoint"] != {"best": best, "last": last}
            or not isinstance(digests, dict)
            or set(digests) != {best, last}
            or not isinstance(metric_digests, dict)
        ):
            raise ContractError("Train receipt is invalid")
        self._verify_train_checkpoints(record, best, last, digests)
        current_metric_digests = self.store.validate_completed_training_metrics(
            record,
            metric_files,
        )
        if metric_digests != current_metric_digests:
            raise ContractError("Train receipt metric files changed")
        return best, last, digests, metric_files, metric_digests

    @staticmethod
    def _verify_train_checkpoints(
        record: RunRecord,
        best: str,
        last: str,
        digests: dict[str, str],
    ) -> None:
        for relative in {best, last}:
            try:
                validated = _checkpoint_result(record, str(record.path / relative))
            except RuntimeFailure as error:
                raise ContractError("Train receipt checkpoint is invalid") from error
            if validated != relative or digests[relative] != _file_digest(
                record.path / relative
            ):
                raise ContractError("Train receipt checkpoint changed")

    def _train_model(
        self,
        record: RunRecord,
        journal: dict[str, Any],
        best: str,
        best_digest: str,
    ) -> ModelRecord:
        models = [
            model
            for model in self.store.scan_models(
                experiment=record.request["experiment"],
                variant=record.request["variant"],
            )
            if model.document["producer_run"] == record.request["run_id"]
        ]
        if len(models) > 1:
            raise ContractError("Train Run produced multiple Models")
        if not models:
            if journal["phase"] == "ready":
                raise ContractError("ready Train receipt has no Model")
            return self.store.allocate_model(
                record,
                checkpoint=best,
                checkpoint_digest=best_digest,
            )
        model = models[0]
        if (
            model.document["checkpoint"]["path"]
            != f"runs/{record.request['run_id']}/{best}"
            or model.document["checkpoint"]["digest"] != best_digest
        ):
            raise ContractError("published Model disagrees with Train receipt")
        return model

    def _complete_eval(
        self,
        record: RunRecord,
        journal: dict[str, Any],
        attempt_path: Path,
    ) -> RunRecord:
        candidate_value = journal["candidate"]
        if not isinstance(candidate_value, str):
            raise ContractError("Eval receipt candidate is invalid")
        candidate = record.path / candidate_value
        final_results = record.path / "artifacts/results"
        final_metrics = record.path / "metrics/eval.json"
        if journal["phase"] == "worker_done":
            files = _validate_candidate_files(
                candidate,
                journal["result"].get("files"),
            )
            artifacts = [f"artifacts/results/{relative}" for relative in files]
            values = journal["result"].get("values")
            if not isinstance(values, dict):
                raise RuntimeFailure("Evaluator values must be a mapping")
            document = self.store.evaluation_document(
                record,
                values=values,
                artifacts=artifacts,
            )
            metrics_candidate = record.path / "metrics/.eval.candidate.json"
            atomic_write_new(metrics_candidate, self.store.json_text(document))
            journal["phase"] = "ready"
            journal["result"] = {
                "document": document,
                "metrics_digest": _file_digest(metrics_candidate),
                "files": [
                    {"path": relative, "digest": _file_digest(candidate / relative)}
                    for relative in files
                ],
            }
            write_attempt(attempt_path, journal)
        document = journal["result"].get("document")
        files = journal["result"].get("files")
        if not isinstance(document, dict) or not isinstance(files, list):
            raise ContractError("Eval receipt is invalid")
        metrics_candidate = record.path / "metrics/.eval.candidate.json"
        if files:
            _publish_or_verify_directory(candidate, final_results, files)
        elif candidate.exists():
            candidate.rmdir()
        _publish_or_verify_file(
            metrics_candidate,
            final_metrics,
            self.store.json_text(document).encode("utf-8"),
        )
        record = self.store.update_state(
            record,
            status="done",
            result={"metrics": "metrics/eval.json"},
        )
        remove_attempt(attempt_path)
        return record

    def _complete_export(
        self,
        record: RunRecord,
        journal: dict[str, Any],
        attempt_path: Path,
    ) -> RunRecord:
        candidate_value = journal["candidate"]
        if not isinstance(candidate_value, str):
            raise ContractError("Export receipt candidate is invalid")
        candidate = record.path / candidate_value
        final = record.path / "artifacts/export"
        if journal["phase"] == "worker_done":
            paths = _validate_candidate_files(
                candidate,
                journal["result"].get("files"),
                require_nonempty=True,
            )
            journal["phase"] = "ready"
            journal["result"] = {
                "files": [
                    {"path": relative, "digest": _file_digest(candidate / relative)}
                    for relative in paths
                ]
            }
            write_attempt(attempt_path, journal)
        files = journal["result"].get("files")
        if not isinstance(files, list) or not files:
            raise ContractError("Export receipt is invalid")
        _publish_or_verify_directory(candidate, final, files)
        record = self.store.update_state(
            record,
            status="done",
            result={"export": "artifacts/export"},
        )
        remove_attempt(attempt_path)
        return record


def _checkpoint_result(record: RunRecord, value: Any) -> str:
    if not isinstance(value, str):
        raise RuntimeFailure("Trainer checkpoint result must be a path")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeFailure("Trainer checkpoint result must be absolute")
    try:
        relative = path.relative_to(record.path)
        resolved = path.resolve(strict=True)
        resolved.relative_to((record.path / "artifacts/checkpoints").resolve())
    except (OSError, ValueError) as error:
        raise RuntimeFailure("Trainer checkpoint is outside its root") from error
    if path.is_symlink() or not path.is_file():
        raise RuntimeFailure("Trainer checkpoint must be a regular file")
    return relative.as_posix()


def _validate_candidate_files(
    candidate: Path,
    returned: Any,
    *,
    require_nonempty: bool = False,
) -> list[str]:
    if not isinstance(returned, list) or any(
        not isinstance(value, str) for value in returned
    ):
        raise RuntimeFailure("worker file result must be a list of paths")
    if require_nonempty and not returned:
        raise RuntimeFailure("worker file result must not be empty")
    if len(returned) != len(set(returned)):
        raise RuntimeFailure("worker file result contains duplicates")
    expected = {_candidate_relative(candidate, raw) for raw in returned}
    actual = set(_candidate_files(candidate))
    if actual != expected:
        raise RuntimeFailure("worker file result does not match candidate files")
    return sorted(actual, key=lambda value: value.encode("utf-8"))


def _candidate_relative(candidate: Path, raw: str) -> str:
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeFailure("worker file result must be absolute")
    try:
        relative = path.relative_to(candidate)
        resolved = path.resolve(strict=True)
        resolved.relative_to(candidate.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise RuntimeFailure("worker file result is outside candidate") from error
    if path.is_symlink() or not path.is_file():
        raise RuntimeFailure("worker file result must be a regular file")
    return relative.as_posix()


def _publish_or_verify_file(candidate: Path, final: Path, expected: bytes) -> None:
    if os.path.lexists(final):
        if final.is_symlink() or not final.is_file() or final.read_bytes() != expected:
            raise ContractError("published file disagrees with receipt")
        candidate.unlink(missing_ok=True)
        return
    if not candidate.is_file() or candidate.is_symlink():
        raise ContractError("receipt candidate file is unavailable")
    if candidate.read_bytes() != expected:
        raise ContractError("receipt candidate file changed")
    publish_file(candidate, final)


def _publish_or_verify_directory(
    candidate: Path,
    final: Path,
    files: Sequence[Mapping[str, Any]],
) -> None:
    expected = {
        item["path"]: item["digest"]
        for item in files
        if isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("digest"), str)
    }
    if len(expected) != len(files):
        raise ContractError("directory receipt is invalid")
    if os.path.lexists(final):
        _verify_published_directory(candidate, final, expected)
        return
    actual = {
        relative: _file_digest(candidate / relative)
        for relative in _candidate_files(candidate)
    }
    if actual != expected:
        raise ContractError("candidate directory disagrees with receipt")
    publish_directory(candidate, final)


def _verify_published_directory(
    candidate: Path,
    final: Path,
    expected: Mapping[str, str],
) -> None:
    if final.is_symlink() or not final.is_dir():
        raise ContractError("published directory is invalid")
    actual = {
        relative: _file_digest(final / relative) for relative in _candidate_files(final)
    }
    if actual != expected:
        raise ContractError("published directory disagrees with receipt")
    if candidate.exists():
        shutil.rmtree(candidate)


def _candidate_files(root: Path) -> list[str]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError("candidate directory is invalid")
    files: list[str] = []
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            if (directory_path / name).is_symlink():
                raise ContractError("candidate contains a symlink")
        for name in filenames:
            path = directory_path / name
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                raise ContractError("candidate file is unavailable") from error
            if path.is_symlink() or not stat.S_ISREG(mode):
                raise ContractError("candidate contains a non-regular file")
            files.append(path.relative_to(root).as_posix())
    return sorted(files, key=lambda value: value.encode("utf-8"))


def _file_digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ContractError("artifact must be a regular file")
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
