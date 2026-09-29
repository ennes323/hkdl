"""Disposable SQLite projection rebuilt from v2 objects and binding refs."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import stat
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hkdl.authoring.config import DIGEST_PATTERN
from hkdl.errors import ContractError

from .graph import V2Graph
from .objects import canonical_json_bytes
from .references import ObjectReference, extract_references

APPLICATION_ID = 0x484B5632
SCHEMA_VERSION = 4


@dataclass(frozen=True)
class ProjectionReport:
    state: str
    objects: int
    bindings: int
    head: str | None
    detail: str | None = None
    object_refs: int = 0
    active_objects: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "schema_version": SCHEMA_VERSION,
            "objects": self.objects,
            "bindings": self.bindings,
            "head": self.head,
            "detail": self.detail,
            "object_refs": self.object_refs,
            "active_objects": self.active_objects,
        }


class GraphProjection:
    def __init__(self, graph: V2Graph):
        self.graph = graph
        self.path = graph.store.root / "index.sqlite3"

    def rebuild(self) -> ProjectionReport:
        # Capture bindings before scanning immutable objects: publication writes
        # their complete dependencies before HEAD. None must stay an empty view;
        # BindingLog.bindings(None) would instead observe the latest HEAD.
        head = self.graph.bindings.head()
        bindings = self.graph.bindings.bindings(head) if head is not None else {}
        records = self.graph.store.iter_records()
        references = extract_references(records)
        active_reachability = _active_reachability(bindings, references)
        indexed_digests = {record.digest for record in records}
        active_objects = {
            target for _, target, _ in active_reachability if target in indexed_digests
        }
        self.graph.store._ensure_layout()
        candidate = (
            self.graph.store.candidates / f"index-{secrets.token_hex(16)}.sqlite3"
        )
        try:
            connection = sqlite3.connect(candidate)
            try:
                _initialize(connection)
                connection.executemany(
                    "INSERT INTO objects (digest, kind, payload_json) VALUES (?, ?, ?)",
                    [
                        (
                            record.digest,
                            record.kind,
                            canonical_json_bytes(record.payload).decode("utf-8"),
                        )
                        for record in records
                    ],
                )
                connection.executemany(
                    "INSERT INTO bindings (scope, name, target) VALUES (?, ?, ?)",
                    [
                        (scope, name, target)
                        for (scope, name), target in sorted(bindings.items())
                    ],
                )
                connection.executemany(
                    "INSERT INTO object_refs "
                    "(source, source_kind, target, target_kind, relation) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            reference.source,
                            reference.source_kind,
                            reference.target,
                            reference.target_kind,
                            reference.relation,
                        )
                        for reference in references
                    ],
                )
                connection.executemany(
                    "INSERT INTO active_reachability (root, target, depth) "
                    "VALUES (?, ?, ?)",
                    active_reachability,
                )
                connection.execute(
                    "INSERT INTO metadata (key, value) VALUES ('head', ?)",
                    (head or "",),
                )
                connection.commit()
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                connection.close()
            descriptor = os.open(candidate, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _publish_database(candidate, self.path)
        finally:
            candidate.unlink(missing_ok=True)
        report = self.inspect()
        if (
            report.state != "ready"
            or report.objects != len(records)
            or report.bindings != len(bindings)
            or report.head != head
            or report.object_refs != len(references)
            or report.active_objects != len(active_objects)
        ):
            raise ContractError("v2 projection verification failed")
        return report

    def inspect(self) -> ProjectionReport:
        head = self.graph.bindings.head()
        if not os.path.lexists(self.path):
            return ProjectionReport("absent", 0, 0, head)
        try:
            metadata = self.path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ContractError("v2 projection must be a regular non-symlink file")
            uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True)
            try:
                application_id = connection.execute("PRAGMA application_id").fetchone()[
                    0
                ]
                user_version = connection.execute("PRAGMA user_version").fetchone()[0]
                if application_id != APPLICATION_ID or user_version != SCHEMA_VERSION:
                    return ProjectionReport(
                        "incompatible", 0, 0, head, "schema metadata disagrees"
                    )
                integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
                if integrity != "ok":
                    return ProjectionReport("corrupt", 0, 0, head, str(integrity))
                objects = connection.execute("SELECT COUNT(*) FROM objects").fetchone()[
                    0
                ]
                bindings = connection.execute(
                    "SELECT COUNT(*) FROM bindings"
                ).fetchone()[0]
                object_refs = connection.execute(
                    "SELECT COUNT(*) FROM object_refs"
                ).fetchone()[0]
                active_objects = connection.execute(
                    "SELECT COUNT(DISTINCT reach.target) "
                    "FROM active_reachability AS reach "
                    "JOIN objects ON objects.digest = reach.target"
                ).fetchone()[0]
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'head'"
                ).fetchone()
                projected_head = row[0] if row and row[0] else None
            finally:
                connection.close()
        except (OSError, sqlite3.Error, ContractError) as error:
            return ProjectionReport("corrupt", 0, 0, head, str(error))
        state = "ready" if projected_head == head else "stale"
        return ProjectionReport(
            state,
            objects,
            bindings,
            projected_head,
            object_refs=object_refs,
            active_objects=active_objects,
        )

    def object_payload(self, digest: str) -> dict[str, object]:
        if not DIGEST_PATTERN.fullmatch(digest):
            raise ContractError("invalid v2 projection object digest")
        with self._read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM objects WHERE digest = ?", (digest,)
            ).fetchone()
        if row is None:
            raise ContractError(f"v2 projection object not found: {digest}")
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise ContractError("v2 projection payload is invalid")
        return value

    def resolve(self, scope: str, name: str) -> str | None:
        with self._read() as connection:
            row = connection.execute(
                "SELECT target FROM bindings WHERE scope = ? AND name = ?",
                (scope, name),
            ).fetchone()
        return None if row is None else str(row[0])

    def object_refs(
        self,
        source: str | None = None,
        target: str | None = None,
        *,
        active_only: bool = False,
    ) -> tuple[ObjectReference, ...]:
        """Return typed object edges, optionally scoped by source/target.

        ``active_only`` filters on the source object.  This makes
        ``references_to`` useful for deletion planning: an orphan object that
        happens to point at an active object is not an active dependency.
        """

        if source is not None:
            _validate_digest(source, "projection source digest")
        if target is not None:
            _validate_digest(target, "projection target digest")
        clauses: list[str] = []
        parameters: list[str] = []
        if source is not None:
            clauses.append("refs.source = ?")
            parameters.append(source)
        if target is not None:
            clauses.append("refs.target = ?")
            parameters.append(target)
        if active_only:
            clauses.append(
                "EXISTS (SELECT 1 FROM active_reachability AS reach "
                "WHERE reach.target = refs.source)"
            )
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._read() as connection:
            rows = connection.execute(
                "SELECT source, source_kind, target, target_kind, relation "
                f"FROM object_refs AS refs{where} "
                "ORDER BY source, relation, target",
                parameters,
            ).fetchall()
        return tuple(ObjectReference(*map(str, row)) for row in rows)

    def references_from(
        self, source: str, *, active_only: bool = False
    ) -> tuple[ObjectReference, ...]:
        """Return outgoing typed references for one object."""

        return self.object_refs(source, active_only=active_only)

    def references_to(
        self, target: str, *, active_only: bool = False
    ) -> tuple[ObjectReference, ...]:
        """Return incoming typed references for one object."""

        return self.object_refs(target=target, active_only=active_only)

    def is_active(self, digest: str) -> bool:
        """Return whether a digest is reachable from a current binding.

        A current binding may point at a digest that has no object (for
        example, a comparison-group label).  Such a digest is still an active
        target; callers that require an indexed object can additionally use
        ``object_payload`` or ``active_targets(include_missing=False)``.
        """

        _validate_digest(digest, "projection digest")
        with self._read() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM active_reachability WHERE target = ? LIMIT 1",
                    (digest,),
                ).fetchone()
                is not None
            )

    def active_targets(
        self,
        *,
        kind: str | None = None,
        include_missing: bool = True,
    ) -> tuple[str, ...]:
        """Return unique digests in the active binding closure."""

        if kind is not None and (not isinstance(kind, str) or not kind):
            raise ContractError("projection object kind is invalid")
        joins = ""
        clauses: list[str] = []
        parameters: list[str] = []
        if not include_missing or kind is not None:
            joins = " JOIN objects ON objects.digest = reach.target"
        if kind is not None:
            clauses.append("objects.kind = ?")
            parameters.append(kind)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._read() as connection:
            rows = connection.execute(
                "SELECT DISTINCT reach.target FROM active_reachability AS reach"
                f"{joins}{where} ORDER BY reach.target",
                parameters,
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def active_reachable(
        self,
        roots: str | Iterable[str] | None = None,
        *,
        include_missing: bool = True,
    ) -> tuple[str, ...]:
        """Return targets reachable from current binding roots.

        With no roots this is the union of all active binding closures.  With
        roots supplied, only rows whose direct root matches are returned.
        """

        normalized = _normalize_digests(roots, "projection root digest")
        if roots is not None and not normalized:
            return ()
        clauses: list[str] = []
        parameters: list[str] = []
        joins = ""
        if normalized:
            placeholders = ", ".join("?" for _ in normalized)
            clauses.append(f"reach.root IN ({placeholders})")
            parameters.extend(normalized)
        if not include_missing:
            joins = " JOIN objects ON objects.digest = reach.target"
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._read() as connection:
            rows = connection.execute(
                "SELECT DISTINCT reach.target FROM active_reachability AS reach"
                f"{joins}{where} ORDER BY reach.target",
                parameters,
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def reachable(
        self,
        roots: str | Iterable[str] | None = None,
        *,
        include_missing: bool = True,
    ) -> tuple[str, ...]:
        """Compatibility alias for ``active_reachable``."""

        return self.active_reachable(roots, include_missing=include_missing)

    def dependency_closure(
        self,
        roots: str | Iterable[str],
        *,
        include_roots: bool = True,
        active_only: bool = True,
        include_missing: bool = True,
    ) -> tuple[str, ...]:
        """Return a forward dependency closure for one or more roots.

        The default uses the materialized active reachability table.  Setting
        ``active_only=False`` traverses the indexed object references and is
        useful for diagnostics involving orphan objects.
        """

        normalized = _normalize_digests(roots, "projection root digest")
        if not normalized:
            return ()
        if active_only:
            values = self.active_reachable(normalized, include_missing=include_missing)
            if not include_roots:
                values = tuple(value for value in values if value not in normalized)
            return values
        joins = ""
        if not include_missing:
            joins = " JOIN objects ON objects.digest = closure.target"
        with self._read() as connection:
            rows = connection.execute(
                "WITH RECURSIVE closure(target) AS ("
                f"{' UNION ALL '.join('SELECT ?' for _ in normalized)} "
                "UNION "
                "SELECT refs.target FROM closure "
                "JOIN object_refs AS refs ON refs.source = closure.target"
                ") SELECT DISTINCT closure.target FROM closure"
                f"{joins} ORDER BY closure.target",
                normalized,
            ).fetchall()
        values = tuple(str(row[0]) for row in rows)
        if not include_roots:
            values = tuple(value for value in values if value not in normalized)
        return values

    def _read(self) -> sqlite3.Connection:
        report = self.inspect()
        if report.state != "ready":
            self.rebuild()
        return sqlite3.connect(f"{self.path.as_uri()}?mode=ro&immutable=1", uri=True)


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute(
        "CREATE TABLE objects ("
        "digest TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE bindings ("
        "scope TEXT NOT NULL, name TEXT NOT NULL, target TEXT NOT NULL, "
        "PRIMARY KEY (scope, name))"
    )
    connection.execute(
        "CREATE TABLE object_refs ("
        "source TEXT NOT NULL, source_kind TEXT NOT NULL, "
        "target TEXT NOT NULL, target_kind TEXT NOT NULL, relation TEXT NOT NULL, "
        "PRIMARY KEY (source, target, relation), "
        "FOREIGN KEY (source) REFERENCES objects(digest))"
    )
    connection.execute(
        "CREATE TABLE active_reachability ("
        "root TEXT NOT NULL, target TEXT NOT NULL, depth INTEGER NOT NULL, "
        "PRIMARY KEY (root, target), CHECK (depth >= 0))"
    )
    connection.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute("CREATE INDEX objects_kind ON objects(kind)")
    connection.execute("CREATE INDEX bindings_target ON bindings(target)")
    connection.execute("CREATE INDEX object_refs_source ON object_refs(source)")
    connection.execute("CREATE INDEX object_refs_target ON object_refs(target)")
    connection.execute(
        "CREATE INDEX active_reachability_target ON active_reachability(target)"
    )
    connection.execute(
        "CREATE INDEX active_reachability_root ON active_reachability(root)"
    )


def _active_reachability(
    bindings: Mapping[tuple[str, str], str],
    references: tuple[ObjectReference, ...],
) -> tuple[tuple[str, str, int], ...]:
    """Build ``(binding root, target, shortest depth)`` rows deterministically."""

    adjacency: dict[str, set[str]] = defaultdict(set)
    for reference in references:
        adjacency[reference.source].add(reference.target)
    rows: set[tuple[str, str, int]] = set()
    roots = sorted(set(bindings.values()))
    for root in roots:
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        seen: set[str] = set()
        while queue:
            current, depth = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            rows.add((root, current, depth))
            for target in sorted(adjacency.get(current, ())):
                if target not in seen:
                    queue.append((target, depth + 1))
    return tuple(sorted(rows, key=lambda row: (row[0], row[2], row[1])))


def _normalize_digests(
    values: str | Iterable[str] | None,
    location: str,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    try:
        normalized = {_validate_digest(value, location) for value in values}
    except TypeError as error:
        raise ContractError(f"{location} values are invalid") from error
    return tuple(sorted(normalized))


def _validate_digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
        raise ContractError(f"{location} is invalid")
    return value


def _publish_database(candidate: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        metadata = destination.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError("v2 projection target is invalid")
        os.replace(candidate, destination)
    else:
        try:
            os.link(candidate, destination, follow_symlinks=False)
        except FileExistsError:
            os.replace(candidate, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
