"""Versioned procedure links, available even in the standalone installer."""

from importlib import metadata, resources


def procedure_help(document: str) -> str:
    release = resources.files(__package__).joinpath("documentation-version.txt")
    version = (
        release.read_text(encoding="utf-8").strip()
        if release.is_file()
        else metadata.version("hkdl")
    )
    return (
        f"Procedure (HKDL {version}; requires the published release and network access):\n"
        f"  https://github.com/hukuhaka/hkdl/blob/v{version}/docs/user/{document}.md\n"
        "Read the procedure before applying changes. If unavailable, stop and obtain\n"
        "the matching release documentation; do not substitute another version."
    )
