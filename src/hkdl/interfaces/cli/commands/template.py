"""Present Template catalog queries at the CLI boundary.

Dispatch supplies workspace access. Authoring and its Template resolver own
catalog lookup, reference resolution, and validation; this module only selects
the query and shapes its text or JSON representation.
"""

from __future__ import annotations

import argparse

from hkdl.authoring.authoring import Authoring
from hkdl.interfaces.cli import io
from hkdl.storage.storage import RepositoryPaths


def run(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Execute a parsed Template query and render its public fields."""
    authoring = Authoring(repository)

    if (args.noun, args.verb) == ("template", "list"):
        templates = authoring.list_templates()
        if args.output == "json":
            io.print_json(
                {
                    "templates": [
                        {
                            "name": item.manifest["name"],
                            "version": item.manifest["version"],
                            "type": item.manifest["type"],
                        }
                        for item in templates
                    ]
                }
            )
        else:
            io.table(
                ("NAME", "VERSION", "TYPE"),
                (
                    (
                        item.manifest["name"],
                        item.manifest["version"],
                        item.manifest["type"],
                    )
                    for item in templates
                ),
            )
        return 0

    if (args.noun, args.verb) == ("template", "show"):
        template = authoring.show_template(args.reference)
        payload = {
            "name": template.manifest["name"],
            "version": template.manifest["version"],
            "type": template.manifest["type"],
            "digest": template.bundle_digest,
        }
        if args.output == "json":
            io.print_json({"template": payload})
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    raise AssertionError("unreachable command")
