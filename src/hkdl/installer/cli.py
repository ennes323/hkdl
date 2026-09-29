"""Command line for the independent installer and its stable hkdl launcher."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .bundle import INSTALLER_VERSION, MAX_BUNDLE_BYTES, PROTOCOL, SHA256, load_bundle
from .common import (
    InstallError,
    InstallFailure,
    directory_lease,
    regular_bytes,
    sync_directory,
)
from .documentation import procedure_help
from .manager import install
from .paths import LEASE_ENV, InstallPaths, default_root, registration_name


@contextmanager
def bundle_path(source: str, checksum: str | None):
    url = urllib.parse.urlsplit(source)
    if url.scheme not in {"http", "https"}:
        yield Path(source).absolute()
        return
    if url.scheme != "https" or not checksum or not SHA256.fullmatch(checksum):
        raise InstallError(
            "HTTPS downloads require the release's explicit --sha256 value"
        )
    with tempfile.TemporaryDirectory(prefix="hkdl-download-") as temporary:
        target = Path(temporary) / "bundle.zip"
        try:
            with (
                urllib.request.urlopen(source, timeout=30) as response,
                target.open("wb") as stream,
            ):
                if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                    raise InstallError("release download redirected outside HTTPS")
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_BUNDLE_BYTES:
                        raise InstallError(
                            "release download exceeds the supported size"
                        )
                    stream.write(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise InstallFailure(
                f"could not download release bundle: {error}"
            ) from error
        yield target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hkdl-install",
        epilog=procedure_help("updating"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hkdl-install {INSTALLER_VERSION} (bundle protocol {PROTOCOL})",
    )
    parser.add_argument(
        "--root", type=Path, default=default_root(), help="HKDL installation root"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    installation = commands.add_parser(
        "install",
        help="Verify and activate a selected bundle",
        description="Review the preview before confirming. Repair preserves research and is not migration.",
        epilog=procedure_help("updating"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    installation.add_argument("bundle", help="Release bundle path or HTTPS URL")
    installation.add_argument(
        "--sha256", help="Release archive checksum (required for HTTPS)"
    )
    installation.add_argument("--yes", "-y", action="store_true")
    installation.add_argument("--repair", action="store_true")
    installation.add_argument("--bin-dir", type=Path, help="Initial launcher directory")
    commands.add_parser(
        "status", help="Inspect installation state without importing HKDL"
    )
    forget = commands.add_parser(
        "forget-workspace", help="Remove only a workspace registration"
    )
    forget.add_argument("path", type=Path)
    forget.add_argument("--yes", "-y", action="store_true")
    run = commands.add_parser("run", add_help=False)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = InstallPaths(args.root.absolute())
    try:
        if args.command == "install":
            with bundle_path(args.bundle, args.sha256) as source:
                bundle = load_bundle(source, expected_sha256=args.sha256)
                executable = Path(sys.argv[0]).absolute()
                archive = executable if executable.suffix == ".pyz" else None
                install(
                    paths,
                    bundle,
                    assume_yes=args.yes,
                    repair=args.repair,
                    archive=archive,
                    bin_dir=args.bin_dir,
                )
        elif args.command == "status":
            _status(paths)
        elif args.command == "forget-workspace":
            _forget(paths, args.path.absolute(), args.yes)
        else:
            arguments = args.arguments
            if arguments[:1] == ["--"]:
                arguments = arguments[1:]
            return _run(paths, arguments)
        return 0
    except InstallError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError) as error:
        print(f"error: installation failed: {error}", file=sys.stderr)
        return 6
    except KeyboardInterrupt:
        print(
            "Installation interrupted. Inspect status before retrying the same bundle.",
            file=sys.stderr,
        )
        return 130


def _status(paths: InstallPaths) -> None:
    paths.validate()
    with directory_lease(paths.root, exclusive=False):
        current = paths.current()
        print(
            json.dumps(
                {
                    "root": str(paths.root),
                    "version": current.manifest["version"] if current else None,
                    "kind": current.manifest["kind"] if current else None,
                    "bundle": current.identity if current else None,
                    "release": str(current.path) if current else None,
                    "workspaces": [str(root) for root in paths.registered_workspaces()],
                },
                indent=2,
            )
        )


def _run(paths: InstallPaths, arguments: list[str]) -> int:
    if arguments[:1] == ["update"] and not any(
        arg in {"-h", "--help"} for arg in arguments
    ):
        return main(["--root", str(paths.root), "install", *arguments[1:]])
    paths.validate()
    with directory_lease(paths.root, exclusive=False) as descriptor:
        release = paths.current()
        if release is None:
            raise InstallError("no active HKDL release; install a bundle first")
        command = release.path / ".venv/bin/hkdl"
        regular_bytes(command)
        os.set_inheritable(descriptor, True)
        environment = os.environ.copy()
        environment[LEASE_ENV] = str(descriptor)
        os.execve(command, [str(command), *arguments], environment)
    return 0


def _forget(paths: InstallPaths, root: Path, assume_yes: bool) -> None:
    paths.validate()
    with directory_lease(paths.root, exclusive=True):
        target = paths.workspaces / registration_name(root)
        regular_bytes(target)
        if not assume_yes:
            print(
                f"Forget only the registration for {root}? [y/N] ",
                end="",
                file=sys.stderr,
                flush=True,
            )
            if sys.stdin.readline().strip().lower() not in {"y", "yes"}:
                print("Registration preserved.")
                return
        target.unlink()
        sync_directory(paths.workspaces)
        print(f"Registration removed. Workspace files were preserved: {root}")


if __name__ == "__main__":
    raise SystemExit(main())
