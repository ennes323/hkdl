"""Append-only name bindings backed by immutable v2 transactions."""

from __future__ import annotations

import json
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from ..config import DIGEST_PATTERN, NAME_PATTERN, ContractError
from ..filesystem import lock_directory, unlock_directory
from ..storage import (
    AlreadyExistsError,
    NotFoundError,
    atomic_replace,
    atomic_write_new,
)
from .objects import ObjectStore

BindingAction = Literal["bind", "unbind"]
_ANY_HEAD = object()


class BindingHeadConflict(RuntimeError):
    """The binding HEAD changed before a conditional transaction committed."""


@dataclass(frozen=True)
class BindingOperation:
    action: BindingAction
    scope: str
    name: str
    target: str

    def as_dict(self) -> dict[str, str]:
        return {
            "action": self.action,
            "scope": self.scope,
            "name": self.name,
            "target": self.target,
        }


class BindingLog:
    """One atomic transaction chain for all scoped active bindings."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        now: Callable[[], datetime] | None = None,
        nonce: Callable[[], str] | None = None,
    ):
        self.store = store
        self.root = store.root / "refs/bindings"
        self.head_path = self.root / "HEAD"
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._nonce = nonce or (lambda: secrets.token_hex(16))

    def head(self) -> str | None:
        if not os.path.lexists(self.head_path):
            return None
        try:
            metadata = self.head_path.lstat()
            content = self.head_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise ContractError("v2 binding HEAD is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError("v2 binding HEAD must be a regular non-symlink file")
        digest = content.rstrip("\n")
        if content != f"{digest}\n" or not DIGEST_PATTERN.fullmatch(digest):
            raise ContractError("v2 binding HEAD is invalid")
        record = self.store.load(digest)
        if record.kind != "binding_transaction":
            raise ContractError("v2 binding HEAD does not reference a transaction")
        return digest

    def bindings(self, head: str | None = None) -> dict[tuple[str, str], str]:
        current = self.head() if head is None else head
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ContractError("v2 binding transaction chain contains a cycle")
            seen.add(current)
            record = self.store.load(current)
            if record.kind != "binding_transaction":
                raise ContractError("v2 binding chain contains a non-transaction")
            payload = _validate_transaction(record.payload)
            chain.append(payload)
            current = payload["previous"]
        result: dict[tuple[str, str], str] = {}
        for payload in reversed(chain):
            for raw in payload["operations"]:
                operation = _operation(raw)
                key = (operation.scope, operation.name)
                if operation.action == "bind":
                    result[key] = operation.target
                else:
                    if result.get(key) != operation.target:
                        raise ContractError("v2 unbind does not match active binding")
                    del result[key]
        return result

    def resolve(self, scope: str, name: str) -> str:
        _scope(scope)
        _name(name)
        try:
            return self.bindings()[(scope, name)]
        except KeyError as error:
            raise NotFoundError(f"name binding not found: {scope}/{name}") from error

    def names(self, scope: str) -> dict[str, str]:
        _scope(scope)
        return {
            name: target
            for (bound_scope, name), target in self.bindings().items()
            if bound_scope == scope
        }

    def historical_names(self, scope: str, target: str) -> tuple[str, ...]:
        _scope(scope)
        names: set[str] = set()
        current = self.head()
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ContractError("v2 binding transaction chain contains a cycle")
            seen.add(current)
            payload = _validate_transaction(self.store.load(current).payload)
            for raw in payload["operations"]:
                operation = _operation(raw)
                if (
                    operation.action == "bind"
                    and operation.scope == scope
                    and operation.target == target
                ):
                    names.add(operation.name)
            current = payload["previous"]
        return tuple(sorted(names, key=lambda name: name.encode("utf-8")))

    def historical_target(self, scope: str, name: str) -> str | None:
        """Return the most recently bound target for one scoped name."""

        targets = self.historical_targets(scope, name)
        return targets[0] if targets else None

    def historical_targets(self, scope: str, name: str) -> tuple[str, ...]:
        """Return distinct targets for one name, newest binding first."""

        _scope(scope)
        _name(name)
        targets: list[str] = []
        observed: set[str] = set()
        current = self.head()
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ContractError("v2 binding transaction chain contains a cycle")
            seen.add(current)
            payload = _validate_transaction(self.store.load(current).payload)
            for raw in reversed(payload["operations"]):
                operation = _operation(raw)
                if (
                    operation.action == "bind"
                    and operation.scope == scope
                    and operation.name == name
                    and operation.target not in observed
                ):
                    observed.add(operation.target)
                    targets.append(operation.target)
            current = payload["previous"]
        return tuple(targets)

    def can_bind_name(
        self, scope: str, name: str, *, target: str | None = None
    ) -> bool:
        """Whether a name is free from every other still-active entity."""

        current = self.names(scope)
        active = current.get(name)
        if active is not None:
            return target is not None and active == target
        active_targets = set(current.values())
        return not any(
            historical != target and historical in active_targets
            for historical in self.historical_targets(scope, name)
        )

    def historical_scope_names(self, scope: str) -> tuple[str, ...]:
        """Return every name ever bound or unbound in one scope."""

        _scope(scope)
        names: set[str] = set()
        current = self.head()
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ContractError("v2 binding transaction chain contains a cycle")
            seen.add(current)
            payload = _validate_transaction(self.store.load(current).payload)
            for raw in payload["operations"]:
                operation = _operation(raw)
                if operation.scope == scope:
                    names.add(operation.name)
            current = payload["previous"]
        return tuple(sorted(names, key=lambda item: item.encode("utf-8")))

    def commit(
        self,
        operations: list[BindingOperation],
        *,
        expected_head: str | None | object = _ANY_HEAD,
    ) -> str:
        if not operations:
            raise ContractError("binding transaction requires an operation")
        self.store._ensure_layout()
        self.root.mkdir(mode=0o755, parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ContractError("v2 binding refs must be a real directory")
        descriptor = lock_directory(self.root)
        try:
            previous = self.head()
            if expected_head is not _ANY_HEAD and previous != expected_head:
                raise BindingHeadConflict("binding HEAD changed before commit")
            current = self.bindings(previous)
            _apply_operations(current, operations)
            timestamp = self._now()
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ContractError("binding clock must be timezone-aware")
            payload = {
                "previous": previous,
                "operations": [operation.as_dict() for operation in operations],
                "created_at": timestamp.isoformat(),
                "nonce": self._nonce(),
            }
            transaction = self.store.put("binding_transaction", payload)
            if os.path.lexists(self.head_path):
                atomic_replace(self.head_path, f"{transaction.digest}\n")
            else:
                atomic_write_new(
                    self.head_path,
                    f"{transaction.digest}\n",
                    directory_descriptor=descriptor,
                )
            if self.head() != transaction.digest:
                raise ContractError("v2 binding HEAD publication failed")
            return transaction.digest
        finally:
            unlock_directory(descriptor)

    def bind(self, scope: str, name: str, target: str) -> str:
        return self.commit([BindingOperation("bind", scope, name, target)])

    def rename(self, scope: str, old: str, new: str) -> str:
        _scope(scope)
        _name(old)
        _name(new)
        current = self.names(scope)
        if old not in current:
            raise NotFoundError(f"name binding not found: {scope}/{old}")
        if new in current:
            raise AlreadyExistsError(f"name binding already exists: {scope}/{new}")
        target = current[old]
        return self.commit(
            [
                BindingOperation("unbind", scope, old, target),
                BindingOperation("bind", scope, new, target),
            ]
        )

    def dump(self) -> str:
        payload = [
            {"scope": scope, "name": name, "target": target}
            for (scope, name), target in sorted(
                self.bindings().items(),
                key=lambda item: (
                    item[0][0].encode("utf-8"),
                    item[0][1].encode("utf-8"),
                ),
            )
        ]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _apply_operations(
    current: dict[tuple[str, str], str],
    operations: list[BindingOperation],
) -> None:
    previous_actions: dict[tuple[str, str], str] = {}
    for operation in operations:
        _validate_operation(operation)
        key = (operation.scope, operation.name)
        previous_action = previous_actions.get(key)
        if previous_action is not None and not (
            previous_action == "unbind" and operation.action == "bind"
        ):
            raise ContractError("binding transaction touches one name more than once")
        previous_actions[key] = operation.action
        if operation.action == "bind":
            if key in current:
                raise AlreadyExistsError(
                    f"name binding already exists: {operation.scope}/{operation.name}"
                )
            current[key] = operation.target
        else:
            if current.get(key) != operation.target:
                raise NotFoundError(
                    f"name binding not found: {operation.scope}/{operation.name}"
                )
            del current[key]


def _validate_transaction(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"previous", "operations", "created_at", "nonce"}:
        raise ContractError("v2 binding transaction fields are invalid")
    previous = payload["previous"]
    if previous is not None and (
        not isinstance(previous, str) or not DIGEST_PATTERN.fullmatch(previous)
    ):
        raise ContractError("v2 binding transaction parent is invalid")
    if not isinstance(payload["operations"], list) or not payload["operations"]:
        raise ContractError("v2 binding transaction operations are invalid")
    for raw in payload["operations"]:
        _operation(raw)
    if not isinstance(payload["created_at"], str) or not payload["created_at"]:
        raise ContractError("v2 binding transaction timestamp is invalid")
    if not isinstance(payload["nonce"], str) or not payload["nonce"]:
        raise ContractError("v2 binding transaction nonce is invalid")
    return payload


def _operation(raw: Any) -> BindingOperation:
    if not isinstance(raw, dict) or set(raw) != {"action", "scope", "name", "target"}:
        raise ContractError("v2 binding operation fields are invalid")
    operation = BindingOperation(
        raw["action"], raw["scope"], raw["name"], raw["target"]
    )
    _validate_operation(operation)
    return operation


def _validate_operation(operation: BindingOperation) -> None:
    if operation.action not in {"bind", "unbind"}:
        raise ContractError("v2 binding action is invalid")
    _scope(operation.scope)
    _name(operation.name)
    if not isinstance(operation.target, str) or not DIGEST_PATTERN.fullmatch(
        operation.target
    ):
        raise ContractError("v2 binding target is invalid")


def _scope(value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 1024
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ContractError("v2 binding scope is invalid")


def _name(value: Any) -> None:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ContractError("v2 binding name is invalid")
