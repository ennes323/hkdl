"""Private identity, synchronization, and validation service for environments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hkdl.authoring.authoring_records import VariantRecord
from hkdl.execution.run_contracts import validate_tracker
from hkdl.installation import installation_descriptors
from hkdl.storage.storage import RepositoryPaths

from ._diagnostics import (
    report_exception,
    report_message,
    report_process_failure,
)
from .environment_types import EnvironmentFailure, EnvironmentIdentity

ENVIRONMENT_SCHEMA_VERSION = 1


class EnvironmentBuilder:
    def __init__(self, repository: RepositoryPaths):
        self.repository = repository
        self.root = repository.root / ".hkdl/environments"
        self.store = self.root / f"v{ENVIRONMENT_SCHEMA_VERSION}"
        self.locks = self.root / "locks"
        self._uv_runtime: tuple[Path, str] | None = None
        self._interpreters: dict[Path, dict[str, str]] = {}

    def identity(
        self,
        variant: VariantRecord,
        *,
        runtime_identity: (
            Callable[[Path], tuple[Path, str, Path, dict[str, str]]] | None
        ) = None,
    ) -> EnvironmentIdentity:
        source = variant.path / "src"
        project = _read_regular(source / "pyproject.toml")
        lock = _read_regular(source / "uv.lock")
        extras = (
            ("mlflow",)
            if "mlflow" in validate_tracker(variant.document.get("tracker"))
            else ()
        )
        identify_runtime = runtime_identity or self.runtime_identity
        uv, uv_version, python, interpreter = identify_runtime(source)
        document: dict[str, Any] = {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "project_digest": _digest(project),
            "lock_digest": _digest(lock),
            "extras": list(extras),
            "interpreter": interpreter,
            "uv_version": uv_version,
        }
        encoded = json.dumps(
            document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        return EnvironmentIdentity(key, document, uv, python, extras)

    def runtime_identity(
        self,
        source: Path,
    ) -> tuple[Path, str, Path, dict[str, str]]:
        if self._uv_runtime is None:
            uv_name = shutil.which("uv")
            if uv_name is None:
                raise EnvironmentFailure("uv is unavailable")
            uv = Path(uv_name).absolute()
            uv_version = _run_text([str(uv), "--version"], "uv version discovery")
            self._uv_runtime = (uv, uv_version)
        uv, uv_version = self._uv_runtime
        python = Path(
            _run_text(
                [str(uv), "python", "find", "--project", str(source)],
                "Variant Python discovery",
            )
        ).absolute()
        if python in self._interpreters:
            return uv, uv_version, python, self._interpreters[python]
        probe = (
            "import json,platform,sys,sysconfig;"
            "print(json.dumps({"
            "'implementation':platform.python_implementation(),"
            "'version':platform.python_version(),"
            "'cache_tag':sys.implementation.cache_tag,"
            "'abi':sysconfig.get_config_var('SOABI') or '',"
            "'system':platform.system(),"
            "'machine':platform.machine()"
            "},sort_keys=True,separators=(',',':')))"
        )
        raw = _run_text([str(python), "-c", probe], "Variant Python inspection")
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EnvironmentFailure(
                "Variant Python returned invalid identity"
            ) from error
        fields = ("implementation", "version", "cache_tag", "abi", "system", "machine")
        if not isinstance(document, dict) or any(
            not isinstance(document.get(name), str) for name in fields
        ):
            raise EnvironmentFailure("Variant Python returned invalid identity")
        interpreter = {name: str(document[name]) for name in sorted(document)}
        self._interpreters[python] = interpreter
        return uv, uv_version, python, interpreter

    def synchronize(
        self,
        environment: Path,
        variant: VariantRecord,
        identity: EnvironmentIdentity,
    ) -> None:
        command = self.sync_command(environment, variant, identity)
        result = _run_sync(command, environment, "environment.create")
        if result.returncode != 0:
            raise EnvironmentFailure("Variant environment synchronization failed")

    def check(
        self,
        environment: Path,
        variant: VariantRecord,
        identity: EnvironmentIdentity,
    ) -> None:
        command = [*self.sync_command(environment, variant, identity), "--check"]
        result = _run_sync(command, environment, "environment.check")
        if result.returncode != 0:
            raise EnvironmentFailure("shared Variant environment is inconsistent")

    @staticmethod
    def sync_command(
        environment: Path,
        variant: VariantRecord,
        identity: EnvironmentIdentity,
    ) -> list[str]:
        command = [
            str(identity.uv),
            "sync",
            "--project",
            str(variant.path / "src"),
            "--python",
            str(identity.python),
            "--locked",
            "--no-dev",
            "--no-install-project",
        ]
        for extra in identity.extras:
            command.extend(["--extra", extra])
        return command

    def ensure_layout(self) -> None:
        current = self.repository.root
        for part in (".hkdl", "environments", f"v{ENVIRONMENT_SCHEMA_VERSION}"):
            current /= part
            ensure_real_directory(current)
        ensure_real_directory(self.locks)

    def write_manifest(self, path: Path, identity: EnvironmentIdentity) -> None:
        payload = (
            json.dumps(
                {"key": identity.key, "identity": identity.document},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        try:
            path.write_text(payload, encoding="utf-8")
        except OSError as error:
            raise EnvironmentFailure(
                f"could not write environment manifest: {path}"
            ) from error

    def validate(self, path: Path, identity: EnvironmentIdentity) -> None:
        require_real_directory(path, "shared Variant environment")
        manifest = path / "environment.json"
        try:
            mode = manifest.lstat().st_mode
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EnvironmentFailure(
                f"invalid environment manifest: {manifest}"
            ) from error
        if manifest.is_symlink() or not stat.S_ISREG(mode):
            raise EnvironmentFailure(f"invalid environment manifest: {manifest}")
        expected = {"key": identity.key, "identity": identity.document}
        if document != expected:
            raise EnvironmentFailure(f"environment identity mismatch: {path}")
        python = environment_python(path)
        if (
            not python.exists()
            or not python.is_file()
            or not os.access(python, os.X_OK)
        ):
            raise EnvironmentFailure(f"shared Variant Python is unavailable: {python}")


def _read_regular(path: Path) -> bytes:
    try:
        mode = path.lstat().st_mode
        payload = path.read_bytes()
    except OSError as error:
        raise EnvironmentFailure(f"environment input is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise EnvironmentFailure(f"environment input is invalid: {path}")
    return payload


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _run_sync(
    command: list[str], environment: Path, phase: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)},
            pass_fds=installation_descriptors(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        report_exception(phase, error)
        raise
    if result.returncode != 0:
        report_process_failure(
            phase, result.returncode, stdout=result.stdout, stderr=result.stderr
        )
    return result


def _run_text(command: list[str], purpose: str) -> str:
    try:
        result = subprocess.run(
            command,
            pass_fds=installation_descriptors(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        report_message("environment.discovery", purpose)
        report_exception("environment.discovery", error)
        raise EnvironmentFailure(f"{purpose} failed") from error
    if result.returncode != 0 or not result.stdout.strip():
        report_message("environment.discovery", purpose)
        report_process_failure(
            "environment.discovery",
            result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        raise EnvironmentFailure(f"{purpose} failed")
    return result.stdout.strip()


def environment_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_real_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o755)
    except FileExistsError:
        pass
    except OSError as error:
        raise EnvironmentFailure(
            f"could not create environment directory: {path}"
        ) from error
    require_real_directory(path, "environment directory")


def require_real_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise EnvironmentFailure(f"{label} is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise EnvironmentFailure(f"{label} is invalid: {path}")
