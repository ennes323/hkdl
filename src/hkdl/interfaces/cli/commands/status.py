"""Present the read-only status projection produced by storage services.

``Status`` owns snapshot construction across Runs, Models, and evaluation
aggregates. This module selects the public rendering without changing that
projection or its observation scope.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from hkdl.authoring.authoring import Authoring
from hkdl.interfaces.cli import io
from hkdl.interfaces.status_rendering import render_status_tree
from hkdl.storage.status import Status
from hkdl.storage.storage import RepositoryPaths


def run(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Query one status snapshot and render it as JSON, a table, or a tree."""
    authoring = Authoring(repository)

    if args.noun == "status":
        document = Status(authoring.repository).query(
            experiment=args.experiment,
            variant=args.variant,
            run_id=args.run_id,
        )
        if args.output == "json":
            io.print_json(document)
        elif getattr(args, "table", False):
            _status_table(document)
        else:
            sys.stdout.write(render_status_tree(document, full=args.full))
        return 0

    raise AssertionError("unreachable command")


def _status_table(document: dict[str, Any]) -> None:
    """Flatten evaluation aggregates from a status tree into table rows."""
    io.table(
        (
            "EXPERIMENT",
            "VARIANT",
            "GROUP",
            "CASE",
            "METRIC",
            "COUNT",
            "ELIGIBLE",
            "MEAN",
            "SAMPLE_STD",
        ),
        (
            (
                experiment["name"],
                variant["name"],
                group["name"],
                aggregate["evaluation_case"],
                aggregate["metric"],
                aggregate["count"],
                aggregate["eligible"],
                aggregate["mean"],
                aggregate["sample_std"] if aggregate["sample_std"] is not None else "-",
            )
            for experiment in document["experiments"]
            for variant in experiment["variants"]
            for group in variant["training_groups"]
            for aggregate in group["aggregates"]
        ),
    )
