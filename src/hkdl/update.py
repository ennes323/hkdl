"""Safe updates for a public HKDL source checkout."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path

from hkdl.authoring.config import VERSION_PATTERN
from hkdl.storage.storage import RepositoryPaths, parse_version

from .errors import ContractError


class UpdateConflict(RuntimeError):
    """The checkout cannot be updated without reconciling local Git state."""


class UpdateFailure(RuntimeError):
    """Git or environment setup failed."""


# Flow: validate the checkout, fetch and check the target, confirm, then
# fast-forward and reinstall. An unchanged checkout may only need reinstallation.
def update(repository: RepositoryPaths, *, assume_yes: bool = False) -> None:
    """Update a clean public main checkout or repair its installed environment.

    Reject divergent or incompatible releases before changing source. If setup
    fails after fast-forwarding, keep the updated source for a setup retry.
    """
    if repository.standalone:
        raise ContractError(
            "source update is unavailable in an independent workspace; "
            "use the managed hkdl launcher and select a release bundle"
        )
    root = repository.root
    _require_public_checkout(root)
    branch_result = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch_result.returncode or branch_result.stdout.strip() != "main":
        raise ContractError("hkdl update requires the public main branch")
    branch = branch_result.stdout.strip()
    if _git(root, "remote", "get-url", "origin").returncode:
        raise ContractError("hkdl update requires an origin remote")
    _require_clean(root)

    head = _git_output(root, "rev-parse", "--verify", "HEAD")
    source_version = _project_version((root / "pyproject.toml").read_text())
    installed_version = _installed_version()

    print("Checking origin/main for updates...", file=sys.stderr)
    fetched = _git(root, "fetch", "--quiet", "origin", "main")
    if fetched.returncode:
        raise UpdateFailure("could not fetch origin/main")

    target = _git_output(root, "rev-parse", "--verify", "FETCH_HEAD")
    # FETCH_HEAD can change in another terminal; validate the commit we will merge.
    _require_public_target(root, target)
    target_version = _project_version(
        _git_output(root, "show", f"{target}:pyproject.toml")
    )
    ancestry = _git(root, "merge-base", "--is-ancestor", head, target)
    if ancestry.returncode == 1:
        raise UpdateConflict("main has diverged from origin/main")
    if ancestry.returncode:
        raise UpdateFailure("could not verify origin/main ancestry")

    if head == target:
        if installed_version == source_version:
            print(
                "HKDL is already current\n"
                f"  version:  {source_version}\n"
                f"  checkout: {branch}"
            )
            return
        _show_repair(installed_version, source_version)
        if not _confirm("Reinstall from the current source?", assume_yes):
            print("Update cancelled. The checkout was not changed.")
            return
        _require_reviewed_checkout(root, branch=branch, head=head)
        print("Reinstalling HKDL...", file=sys.stderr)
        _run_setup(root, source_updated=False, version=source_version)
        print(
            "HKDL installation repaired\n"
            f"  version:  {source_version}\n"
            f"  checkout: {branch}"
        )
        return

    current = parse_version(source_version)
    available = parse_version(target_version)
    if available <= current:
        raise ContractError(
            "origin/main must contain a newer HKDL version "
            f"(current: {source_version}, available: {target_version})"
        )
    if available[0] != current[0]:
        raise ContractError(
            "major-version updates require manual release instructions "
            f"({source_version} -> {target_version})"
        )

    _show_update(
        branch=branch,
        installed=installed_version,
        current=source_version,
        current_commit=head,
        available=target_version,
        target_commit=target,
    )
    if not _confirm("Continue with update?", assume_yes):
        print("Update cancelled. The checkout was not changed.")
        return

    _require_reviewed_checkout(root, branch=branch, head=head)

    print("Updating source...", file=sys.stderr)
    merged = _git(root, "merge", "--ff-only", "--quiet", target)
    if merged.returncode:
        raise UpdateFailure("source fast-forward failed")
    print("Reinstalling HKDL...", file=sys.stderr)
    _run_setup(root, source_updated=True, version=target_version)
    print(
        "HKDL updated successfully\n"
        f"  previous: {source_version}\n"
        f"  current:  {target_version}\n"
        "  source:   origin/main\n\n"
        "Experiments, outputs, and existing Variants were preserved."
    )


def _require_public_checkout(root: Path) -> None:
    """Require the Git root and tracked files of a public HKDL source checkout.

    Raise ContractError for a non-root path, the internal release manifest,
    missing setup metadata, or tracked user-owned AGENTS.user.md. This checks
    checkout layout; branch, remote, and cleanliness are checked by the caller.
    """
    top = _git(root, "rev-parse", "--show-toplevel")
    if top.returncode or Path(top.stdout.strip()).resolve() != root:
        raise ContractError("hkdl update requires a public Git checkout root")
    if (
        _git(
            root, "ls-files", "--error-unmatch", "public-release-manifest.txt"
        ).returncode
        == 0
    ):
        raise ContractError("hkdl update requires a public HKDL source checkout")
    public_files = _git(
        root,
        "ls-files",
        "--error-unmatch",
        "setup.sh",
        "pyproject.toml",
        "uv.lock",
    )
    if public_files.returncode:
        raise ContractError("hkdl update requires a public HKDL source checkout")
    if _git(root, "ls-files", "--error-unmatch", "AGENTS.user.md").returncode == 0:
        raise ContractError("public HKDL must not track user-owned AGENTS.user.md")


def _require_public_target(root: Path, target: str) -> None:
    if not _git(
        root, "cat-file", "-e", f"{target}:public-release-manifest.txt"
    ).returncode:
        raise ContractError("origin/main is not a public HKDL release")
    if not _git(root, "cat-file", "-e", f"{target}:AGENTS.user.md").returncode:
        raise ContractError("origin/main must not track user-owned AGENTS.user.md")
    for path in (
        "setup.sh",
        "pyproject.toml",
        "uv.lock",
        "src/hkdl",
        "src/templates",
    ):
        if _git(root, "cat-file", "-e", f"{target}:{path}").returncode:
            raise ContractError("origin/main is not a public HKDL release")
    setup = _git_output(root, "ls-tree", target, "--", "setup.sh").split()
    if not setup or setup[0] != "100755":
        raise ContractError("origin/main setup.sh is not executable")


def _require_clean(root: Path) -> None:
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status.returncode:
        raise UpdateFailure("could not inspect the Git checkout")
    if status.stdout:
        raise UpdateConflict("public checkout has local changes")


def _require_reviewed_checkout(root: Path, *, branch: str, head: str) -> None:
    """Recheck the approved source before either fast-forward or setup repair.

    A branch switch can preserve HEAD, so commit equality alone is insufficient.
    """
    _require_clean(root)
    current_branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if current_branch.returncode or current_branch.stdout.strip() != branch:
        raise UpdateConflict("checkout branch changed while confirmation was pending")
    if _git_output(root, "rev-parse", "--verify", "HEAD") != head:
        raise UpdateConflict("main changed while update confirmation was pending")


def _show_update(
    *,
    branch: str,
    installed: str,
    current: str,
    current_commit: str,
    available: str,
    target_commit: str,
) -> None:
    print(
        "\nHKDL update\n"
        f"  checkout:  {branch} (clean)\n"
        f"  installed: {installed}\n"
        f"  current:   {current} ({current_commit[:12]})\n"
        f"  available: {available} ({target_commit[:12]})\n"
        "  source:    origin/main\n\n"
        "Will update:\n"
        "  - HKDL Core and dependency lock\n"
        "  - bundled Templates\n"
        "  - public documentation and setup files\n\n"
        "Will preserve:\n"
        "  - AGENTS.user.md\n"
        "  - experiments/\n"
        "  - outputs/\n"
        "  - existing Variant source and environments\n\n"
        "Automatic schema migration: no\n"
        "If environment setup fails, the source remains updated and can be\n"
        "repaired by running ./setup.sh.\n",
        file=sys.stderr,
    )


def _show_repair(installed: str, source: str) -> None:
    print(
        "\nHKDL installation is out of date\n"
        f"  installed: {installed}\n"
        f"  source:    {source}\n",
        file=sys.stderr,
    )


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        print("Confirmation: accepted (--yes)", file=sys.stderr)
        return True
    print(f"{question} [y/N] ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in {"y", "yes"}


def _run_setup(root: Path, *, source_updated: bool, version: str) -> None:
    setup = root / "setup.sh"
    if setup.is_symlink() or not setup.is_file():
        raise UpdateFailure("public setup.sh is unavailable")
    try:
        result = subprocess.run(
            [str(setup)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise UpdateFailure(f"could not run ./setup.sh: {error}") from error
    if result.returncode == 0:
        return
    if result.stdout:
        sys.stderr.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if source_updated:
        raise UpdateFailure(
            f"source was updated to {version}, but environment setup failed; "
            "run ./setup.sh"
        )
    raise UpdateFailure("environment setup failed; run ./setup.sh")


def _installed_version() -> str:
    versions = [
        distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"] == "hkdl"
        and isinstance(distribution.version, str)
        and VERSION_PATTERN.fullmatch(distribution.version)
    ]
    return max(versions, key=parse_version, default="unknown")


def _project_version(text: str) -> str:
    try:
        version = tomllib.loads(text)["project"]["version"]
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ContractError("HKDL project version is invalid") from error
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ContractError("HKDL project version is invalid")
    return version


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as error:
        raise UpdateFailure("git is required for hkdl update") from error


def _git_output(root: Path, *arguments: str) -> str:
    result = _git(root, *arguments)
    if result.returncode:
        raise ContractError("public Git checkout state is invalid")
    return result.stdout.strip()
