"""Errors shared across HKDL structural contracts."""


class ContractError(ValueError):
    """An HKDL value or persisted record violates a structural contract."""


# Preserve existing serialized Python exception references.
ContractError.__module__ = "hkdl.authoring.config"
