"""Prepare complete releases, validate workspaces, then switch one pointer."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from . import guidance
from .bundle import INSTALLER_VERSION, PROTOCOL, Bundle, require_transition
from .common import (
    InstallError,
    InstallFailure,
    digest,
    directory_lease,
    json_bytes,
    read_json,
    real_directory,
    regular_bytes,
    sync_directory,
    write_file,
)
from .paths import LEASE_ENV, InstalledRelease, InstallPaths

_INSTALL_LEASE: ContextVar[int | None] = ContextVar(
    "hkdl_installer_lease", default=None
)


def _child_descriptors(descriptors: tuple[int, ...] = ()) -> tuple[int, ...]:
    inherited = _INSTALL_LEASE.get()
    return tuple(
        dict.fromkeys((*descriptors, *((inherited,) if inherited is not None else ())))
    )


@contextmanager
def _installation_write(paths: InstallPaths):
    with directory_lease(paths.root, exclusive=True) as descriptor:
        token = _INSTALL_LEASE.set(descriptor)
        try:
            yield descriptor
        finally:
            _INSTALL_LEASE.reset(token)


def _run(command: list[str], *, log: Path, pass_fds: tuple[int, ...] = ()) -> str:
    environment = os.environ.copy()
    environment.pop(LEASE_ENV, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            pass_fds=_child_descriptors(pass_fds),
            env=environment,
        )
    except OSError as error:
        raise InstallFailure(
            f"could not start installation step: {error}; log: {log}"
        ) from error
    with log.open("ab") as stream:
        stream.write((result.stdout + result.stderr).encode())
        stream.flush()
        os.fsync(stream.fileno())
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        reason = detail[-1] if detail else f"exit {result.returncode}"
        raise InstallFailure(f"installation step failed: {reason}; details: {log}")
    return result.stdout


def _probe(
    release: Path, *, workspace: Path | None = None, descriptor: int | None = None
) -> dict:
    command = [
        str(release / ".venv/bin/python"),
        "-I",
        "-m",
        "hkdl.installation_probe",
        str(release / "release.json"),
    ]
    if workspace is not None:
        command.extend(["--workspace", str(workspace)])
    if descriptor is not None:
        command.extend(["--workspace-fd", str(descriptor)])
    output = _run(
        command,
        log=release / "install.log",
        pass_fds=() if descriptor is None else (descriptor,),
    )
    return read_json(output.encode())


def prepare_release(paths: InstallPaths, bundle: Bundle) -> InstalledRelease:
    uv_name = shutil.which("uv")
    if uv_name is None:
        raise InstallError("uv is required to prepare the isolated HKDL environment")
    uv = str(Path(uv_name).absolute())
    name = f"{bundle.manifest['version']}-{bundle.identity[:16]}-{uuid.uuid4().hex}"
    candidate = paths.root / "staging" / name
    candidate.mkdir()
    bundle.unpack(candidate)
    log = candidate / "install.log"
    python = candidate / ".venv/bin/python"
    _run(
        [
            uv,
            "--no-config",
            "venv",
            "--python",
            bundle.manifest["python"],
            "--relocatable",
            str(candidate / ".venv"),
        ],
        log=log,
    )
    _run(
        [
            uv,
            "--no-config",
            "pip",
            "sync",
            "--python",
            str(python),
            "--require-hashes",
            str(candidate / "requirements.txt"),
        ],
        log=log,
    )
    _run(
        [
            uv,
            "--no-config",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--no-index",
            str(candidate / bundle.manifest["wheel"]),
        ],
        log=log,
    )
    _run([uv, "--no-config", "pip", "check", "--python", str(python)], log=log)
    _probe(candidate)
    target = paths.releases / name
    os.rename(candidate, target)
    sync_directory(paths.root / "staging")
    sync_directory(paths.releases)
    # A relocatable environment must also work at its permanent path.
    _probe(target)
    write_file(
        target / "installed.json",
        json_bytes({"schema_version": 1, "manifest_sha256": bundle.identity}),
    )
    return InstalledRelease(target, bundle.manifest, bundle.identity)


def _manager_config(paths: InstallPaths) -> dict[str, Any] | None:
    path = paths.root / "manager/config.json"
    if not os.path.lexists(path):
        return None
    value = read_json(regular_bytes(path))
    if (
        set(value) != {"schema_version", "python", "bin_dir", "manager_sha256"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or any(
            not isinstance(value[key], str)
            for key in ("python", "bin_dir", "manager_sha256")
        )
    ):
        raise InstallError("invalid installation manager configuration")
    if (
        digest(regular_bytes(paths.root / "manager/hkdl-install.pyz"))
        != value["manager_sha256"]
    ):
        raise InstallError(
            "installation manager checksum differs from its configuration"
        )
    return value


def _launcher(paths: InstallPaths, python: str) -> bytes:
    arguments = [
        python,
        str(paths.root / "manager/hkdl-install.pyz"),
        "--root",
        str(paths.root),
        "run",
        "--",
    ]
    command = " ".join(shlex.quote(value) for value in arguments)
    return f'#!/bin/sh\n# HKDL managed launcher v1\nexec {command} "$@"\n'.encode()


def ensure_manager(
    paths: InstallPaths, *, archive: Path | None, bin_dir: Path | None
) -> Path:
    config = _manager_config(paths)
    if config is None:
        if archive is None or archive.suffix != ".pyz":
            raise InstallError(
                "bootstrap with the standalone hkdl-install.pyz tool first"
            )
        payload = regular_bytes(archive)
        directory = (bin_dir or Path.home() / ".local/bin").absolute()
        real_directory(directory, create=True)
        python = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
        _verify_manager_tool(archive, python)
        command = directory / "hkdl"
        if os.path.lexists(command):
            raise InstallError(
                f"launcher already exists; choose another --bin-dir: {command}"
            )
        manager_path = paths.root / "manager/hkdl-install.pyz"
        if os.path.lexists(manager_path):
            if regular_bytes(manager_path) != payload:
                raise InstallError(
                    "incomplete manager bootstrap has different tool bytes"
                )
        else:
            write_file(manager_path, payload)
        config = {
            "schema_version": 1,
            "python": python,
            "bin_dir": str(directory),
            "manager_sha256": digest(payload),
        }
        write_file(paths.root / "manager/config.json", json_bytes(config))
    directory = real_directory(Path(config["bin_dir"]))
    if bin_dir is not None and bin_dir.absolute() != directory:
        raise InstallError(
            "this installation already uses a different launcher directory"
        )
    launcher = directory / "hkdl"
    expected = _launcher(paths, config["python"])
    if os.path.lexists(launcher):
        if regular_bytes(launcher) != expected:
            raise InstallError(f"refusing to replace a user-owned launcher: {launcher}")
    else:
        write_file(launcher, expected)
    launcher.chmod(0o755)
    sync_directory(directory)
    return launcher


def _verify_manager_tool(archive: Path, python: str) -> None:
    try:
        result = subprocess.run(
            [python, "-I", str(archive), "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
            pass_fds=_child_descriptors(),
        )
    except subprocess.TimeoutExpired as error:
        raise InstallFailure("standalone manager verification timed out") from error
    expected = f"hkdl-install {INSTALLER_VERSION} (bundle protocol {PROTOCOL})\n"
    if result.returncode or result.stdout != expected:
        raise InstallError(
            "the standalone manager could not verify its version and protocol"
        )


# Plan and prepare under the installation lease, then lock and probe every
# workspace before activation. Refresh owned guidance only after activation.
def install(
    paths: InstallPaths,
    bundle: Bundle,
    *,
    assume_yes: bool = False,
    repair: bool = False,
    archive: Path | None = None,
    bin_dir: Path | None = None,
) -> bool:
    """Prepare and activate a bundle after every workspace passes its probe.

    The installation lease serializes changes; guidance refresh is best-effort.
    """

    paths.initialize()
    with _installation_write(paths):
        current = paths.current()
        same_bundle = current is not None and current.identity == bundle.identity
        if current is not None:
            require_transition(current.manifest["version"], bundle.manifest)
        roots = paths.registered_workspaces()
        payload = guidance.bundle_guidance(bundle.payloads[bundle.manifest["wheel"]])
        guidance_plans = [guidance.plan_refresh(root, payload) for root in roots]
        _show_plan(current, bundle, roots, reuse=same_bundle and not repair)
        for plan in guidance_plans:
            print(f"  guidance {plan.action}: {plan.root}", file=sys.stderr)
        print(
            f"  HKDL guide: {guidance.reference(bundle.manifest['version'])}",
            file=sys.stderr,
        )
        if (
            same_bundle
            and not repair
            and not any(plan.action == "refresh" for plan in guidance_plans)
        ):
            print(f"HKDL already uses this bundle: {bundle.manifest['version']}")
            return False
        if not assume_yes:
            print(
                "Install and activate this bundle? [y/N] ",
                end="",
                file=sys.stderr,
                flush=True,
            )
            if sys.stdin.readline().strip().lower() not in {"y", "yes"}:
                print(
                    "Installation cancelled. Active version and research were preserved."
                )
                return False
        release = (
            current if same_bundle and not repair else prepare_release(paths, bundle)
        )
        assert release is not None
        with ExitStack() as stack:
            # Lock every registered workspace through the last check and switch.
            # This also excludes legacy processes using the same research root.
            admissions = [
                (root, stack.enter_context(directory_lease(root, exclusive=True)))
                for root in roots
            ]
            for root, descriptor in admissions:
                _probe(release.path, workspace=root, descriptor=descriptor)
            launcher = ensure_manager(paths, archive=archive, bin_dir=bin_dir)
            if not same_bundle or repair:
                _activate(paths, release)
            # Guidance failure must not masquerade as a failed/rolled-back activation.
            if payload is not None:
                for plan in guidance_plans:
                    try:
                        print(
                            guidance.refresh(plan, payload, bundle.manifest["version"])
                        )
                    except (OSError, InstallError) as error:
                        print(
                            f"Warning: HKDL is active; guidance refresh incomplete: {error}. "
                            "Repeat workspace init to review/recover the guide.",
                            file=sys.stderr,
                        )
        print(
            f"HKDL {bundle.manifest['version']} active ({bundle.manifest['kind']})\n"
            f"  bundle: {bundle.identity}\n"
            f"  command: {launcher}\n"
            "Research data and previous installations were preserved."
        )
        return True


def _show_plan(
    current: InstalledRelease | None,
    bundle: Bundle,
    roots: list[Path],
    *,
    reuse: bool = False,
) -> None:
    previous = current.manifest["version"] if current else "not installed"
    preparation = (
        "The active bundle will be reverified; only eligible guidance may refresh."
        if reuse
        else "A separate environment will be installed and verified before activation."
    )
    print(
        f"HKDL installation\n  current: {previous}\n"
        f"  target: {bundle.manifest['version']} ({bundle.manifest['kind']})\n"
        f"  bundle: {bundle.identity}\n  workspaces to verify: {len(roots)}\n"
        f"{preparation}\n"
        "Research files, user guidance and old installations are preserved.\n"
        "Only unchanged HKDL-owned AGENTS.md files listed below may refresh after activation.\n"
        "Automatic migration: no",
        file=sys.stderr,
    )


def _activate(paths: InstallPaths, release: InstalledRelease) -> None:
    temporary = paths.root / f".current-{uuid.uuid4().hex}"
    temporary.symlink_to(f"releases/{release.path.name}")
    try:
        os.replace(temporary, paths.root / "current")
        try:
            sync_directory(paths.root)
        except OSError as error:
            raise InstallFailure(
                "the active pointer was switched, but durability confirmation failed; "
                "inspect installer status before retrying"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
