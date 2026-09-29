"""Render stored Model manifests for CLI list and detail commands.

``RunStore`` owns manifest discovery and validation. This module limits itself
to selecting public fields and choosing JSON, table, or text presentation.
"""

from __future__ import annotations

import argparse

from hkdl.authoring.authoring import Authoring
from hkdl.interfaces.cli import io
from hkdl.storage.runs import RunStore
from hkdl.storage.storage import RepositoryPaths


def run(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Read Model manifests through ``RunStore`` and render the requested view."""
    authoring = Authoring(repository)

    if (args.noun, args.verb) == ("model", "list"):
        models = RunStore(authoring.repository).scan_model_manifests(
            experiment=args.experiment,
            variant=args.variant,
        )
        payload = [
            {
                "model_id": item.document["model_id"],
                "training_group": item.document["training_group"],
                "seed": item.document["seed"],
                "producer_run": item.document["producer_run"],
                "created_at": item.document["created_at"],
            }
            for item in models
        ]
        if args.output == "json":
            io.print_json(
                {
                    "experiment": args.experiment,
                    "variant": args.variant,
                    "models": payload,
                }
            )
        else:
            io.table(
                ("MODEL", "GROUP", "SEED", "RUN", "CREATED"),
                (
                    (
                        item["model_id"],
                        item["training_group"],
                        item["seed"],
                        item["producer_run"],
                        item["created_at"],
                    )
                    for item in payload
                ),
            )
        return 0

    if (args.noun, args.verb) == ("model", "show"):
        model = RunStore(authoring.repository).load_model_manifest(
            args.experiment,
            args.variant,
            args.model_id,
        )
        if args.output == "json":
            io.print_json({"model": dict(model.document)})
        else:
            for key in (
                "model_id",
                "training_group",
                "seed",
                "device",
                "producer_run",
                "created_at",
            ):
                print(f"{key}: {model.document[key]}")
        return 0

    raise AssertionError("unreachable command")
