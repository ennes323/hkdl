"""Read-only checks run with the candidate interpreter before activation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from contextlib import nullcontext
from pathlib import Path

from hkdl.authoring.authoring import Authoring
from hkdl.authoring.config import NAME_PATTERN, VERSION_PATTERN
from hkdl.installer.bundle import validate_manifest
from hkdl.installer.common import read_json, regular_bytes
from hkdl.interfaces.cli.parser import build_parser
from hkdl.interfaces.completion import shellcode
from hkdl.storage.graph.maintenance import borrowed_workspace_access, workspace_access
from hkdl.storage.runs import RunStore
from hkdl.storage.storage import (
    RepositoryPaths,
    TemplateResolver,
    packaged_template_catalog,
    validate_repository_root,
)
from hkdl.storage.workspace_modes import require_current_workspace

from .errors import ContractError


def check_installation(manifest: dict) -> None:
    expected = manifest["version"]
    if importlib.metadata.version("hkdl") != expected:
        raise ContractError(
            "installed package version differs from the release manifest"
        )
    package = Path(__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise ContractError("candidate probe imported HKDL outside its environment")
    for relative in (
        "runtime/_runtime_worker.py",
        "installer/research-agents.md",
        "interfaces/web_ui/app.js",
        "interfaces/web_ui/app.css",
        "interfaces/web_ui/index.html",
    ):
        regular_bytes(package / relative)
    if not build_parser().format_help() or not shellcode("zsh"):
        raise ContractError("candidate CLI or completion could not load")
    catalog = packaged_template_catalog()
    actual = {
        (family.name, version.name)
        for family in catalog.iterdir()
        if family.is_dir() and NAME_PATTERN.fullmatch(family.name)
        for version in family.iterdir()
        if version.is_dir() and VERSION_PATTERN.fullmatch(version.name)
    }
    expected_templates = {
        (item["name"], item["version"]) for item in manifest["templates"]
    }
    if actual != expected_templates:
        raise ContractError(
            "installed Template inventory differs from the release manifest"
        )
    root = Path.cwd()
    repository = RepositoryPaths(root, catalog, root / "experiments", root / "outputs")
    resolver = TemplateResolver(repository)
    for item in manifest["templates"]:
        template = resolver.resolve(f"{item['name']}@{item['version']}")
        if template.bundle_digest != item["digest"]:
            raise ContractError(
                "installed Template bytes differ from the release manifest"
            )


def check_workspace(
    root: Path, manifest: dict, *, descriptor: int | None = None
) -> dict:
    """Read research under the installer's borrowed lease before activation.

    Traverse authoritative records without rebuilding disposable indexes or
    executing Variant code. Any unreadable record prevents candidate activation.
    """
    repository = validate_repository_root(root)
    admission = (
        borrowed_workspace_access(repository, descriptor)
        if descriptor is not None
        else nullcontext()
    )
    with admission, workspace_access(repository):
        require_current_workspace(repository)
        # Pristine workspaces are admitted for their first graph/JSON bootstrap.
        mode = "v2-json"
        if mode not in manifest["workspace_modes"]:
            raise ContractError(f"candidate does not support workspace mode {mode}")
        authoring = Authoring(repository)
        experiments = authoring.list_experiments()
        store = RunStore(repository)
        variants = 0
        models = 0
        for item in experiments:
            name = str(item.document["name"])
            records = authoring.list_variants(name)
            variants += len(records)
            for variant in records:
                models += len(
                    store.scan_model_manifests(
                        experiment=name, variant=str(variant.document["name"])
                    )
                )
        runs = store.scan()
        return {
            "path": str(root),
            "mode": mode,
            "experiments": len(experiments),
            "variants": variants,
            "runs": len(runs),
            "models": models,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--workspace-fd", type=int)
    args = parser.parse_args()
    try:
        manifest = validate_manifest(read_json(regular_bytes(args.manifest)))
        if args.workspace is None:
            check_installation(manifest)
            result = {"version": manifest["version"], "installation": "verified"}
        else:
            result = check_workspace(
                args.workspace, manifest, descriptor=args.workspace_fd
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"candidate validation failed: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
