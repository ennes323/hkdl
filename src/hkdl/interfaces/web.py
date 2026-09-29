"""HTTP application and server lifetime for one local Experiment view."""

from __future__ import annotations

import io
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from hkdl.authoring.authoring import Authoring
from hkdl.authoring.config import NAME_PATTERN
from hkdl.errors import ContractError
from hkdl.installation import installation_lease, installation_operation
from hkdl.storage.graph.authoring import V2Authoring
from hkdl.storage.graph.graph import (
    DirtyDraftError,
    StalePreviewError,
)
from hkdl.storage.graph.maintenance import WorkspaceBusy, workspace_access
from hkdl.storage.graph.reader import graph_observation
from hkdl.storage.status_index import IndexFailure
from hkdl.storage.storage import NotFoundError, OwnershipError, RepositoryPaths

from .web_authoring import (
    AuthoringConfirmation,
    PreviewSigner,
)
from .web_authoring import (
    _json_changes as _authoring_json_changes,
)
from .web_authoring import (
    build_authoring_state as _build_authoring_state,
)
from .web_views import (
    MAX_COMPARISON_POINTS as MAX_COMPARISON_POINTS,
)
from .web_views import (
    MAX_COMPARISON_RUNS as MAX_COMPARISON_RUNS,
)
from .web_views import (
    MIN_COMPARISON_RUNS as MIN_COMPARISON_RUNS,
)
from .web_views import (
    WebRequestError as WebRequestError,
)
from .web_views import (
    build_comparison as build_comparison,
)
from .web_views import (
    build_overview as build_overview,
)
from .web_views import (
    build_run_detail as build_run_detail,
)

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
MAX_MUTATION_BODY_BYTES = 4_096
REQUEST_READ_TIMEOUT_SECONDS = 10.0
RESPONSE_WRITE_TIMEOUT_SECONDS = 10.0
MUTATION_HEADER = "X-HKDL-Action"
MUTATION_HEADER_VALUE = "commit"


class WebFailure(RuntimeError):
    """Local web server setup or serving failure."""


@graph_observation
def build_authoring_state(
    repository: RepositoryPaths,
    experiment_name: str,
    *,
    now: Callable[[], datetime] | None = None,
    sign: PreviewSigner | None = None,
) -> dict[str, Any]:
    return _build_authoring_state(repository, experiment_name, now=now, sign=sign)


def _json_changes(
    before: Any,
    after: Any,
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    return _authoring_json_changes(before, after, prefix=prefix)


class _WebApplication:
    def __init__(
        self,
        repository: RepositoryPaths,
        experiment_name: str,
        *,
        now: Callable[[], datetime] | None,
    ):
        self.repository = repository
        self.experiment_name = experiment_name
        self.now = now
        self.confirmation = AuthoringConfirmation(experiment_name)
        try:
            root = resources.files("hkdl.interfaces.web_ui")
            self.assets = {
                "/": (
                    "text/html; charset=utf-8",
                    root.joinpath("index.html").read_bytes(),
                ),
                "/assets/app.css": (
                    "text/css; charset=utf-8",
                    root.joinpath("app.css").read_bytes(),
                ),
                "/assets/metric-chart.js": (
                    "text/javascript; charset=utf-8",
                    root.joinpath("metric-chart.js").read_bytes(),
                ),
                "/assets/app.js": (
                    "text/javascript; charset=utf-8",
                    root.joinpath("app.js").read_bytes(),
                ),
                "/assets/favicon.svg": (
                    "image/svg+xml",
                    root.joinpath("favicon.svg").read_bytes(),
                ),
            }
        except (FileNotFoundError, OSError) as error:
            raise WebFailure(f"web UI assets are unavailable: {error}") from error

    def overview(self) -> dict[str, Any]:
        return build_overview(
            self.repository,
            self.experiment_name,
            now=self.now,
        )

    def comparison(
        self,
        run_addresses: Sequence[str],
        *,
        metric_name: str | None,
    ) -> dict[str, Any]:
        return build_comparison(
            self.repository,
            self.experiment_name,
            run_addresses,
            metric_name=metric_name,
            now=self.now,
        )

    def authoring(self) -> dict[str, Any]:
        return build_authoring_state(
            self.repository,
            self.experiment_name,
            now=self.now,
            sign=self.confirmation.token,
        )

    def run_detail(self, address: str, metric_name: str | None) -> dict[str, Any]:
        return build_run_detail(
            self.repository,
            self.experiment_name,
            address,
            metric_name=metric_name,
            now=self.now,
        )

    def commit_experiment(self, confirmation_token: str) -> dict[str, Any]:
        identity = V2Authoring(Authoring(self.repository)).commit_experiment(
            self.experiment_name,
            validate_revision=self.confirmation.validator(
                "experiment", self.experiment_name, confirmation_token
            ),
        )
        return {
            "kind": "experiment",
            "name": self.experiment_name,
            "changed": identity.changed,
            "authoring": self._authoring_after_commit(),
        }

    def commit_variant(self, variant: str, confirmation_token: str) -> dict[str, Any]:
        identity = V2Authoring(Authoring(self.repository)).commit_variant(
            self.experiment_name,
            variant,
            validate_revision=self.confirmation.validator(
                "variant", variant, confirmation_token
            ),
        )
        return {
            "kind": "variant",
            "name": variant,
            "changed": identity.changed,
            "authoring": self._authoring_after_commit(),
        }

    def _authoring_after_commit(self) -> dict[str, Any] | None:
        # A later read failure must not misreport a published commit as rejected.
        try:
            return self.authoring()
        except (StalePreviewError, ContractError, OSError):
            return None


class _WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: _WebApplication,
    ):
        self.application = application
        self._installation = ExitStack()
        self._installation.enter_context(installation_lease())
        try:
            super().__init__(server_address, _RequestHandler)
        except BaseException:
            self._installation.close()
            raise

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self._installation.close()


class _DeadlineReader(io.RawIOBase):
    """Bound all socket reads for one request, including slow partial lines."""

    def __init__(self, connection):
        self.connection = connection
        self.stream = connection.makefile("rb", buffering=0)
        self.deadline = 0.0

    def readable(self) -> bool:
        return True

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request read deadline exceeded")
        return remaining

    def readinto(self, buffer):
        self.connection.settimeout(self.remaining())
        return self.stream.readinto(buffer)

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            super().close()


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = RESPONSE_WRITE_TIMEOUT_SECONDS

    def setup(self) -> None:
        self._installation = ExitStack()
        self._installation.enter_context(installation_lease())
        try:
            super().setup()
            self.rfile.close()
            self._request_reader = _DeadlineReader(self.connection)
            self.rfile = io.BufferedReader(self._request_reader)
        except BaseException:
            self._installation.close()
            raise

    def handle_one_request(self) -> None:
        self._request_reader.deadline = time.monotonic() + REQUEST_READ_TIMEOUT_SECONDS
        super().handle_one_request()

    def parse_request(self) -> bool:
        try:
            parsed = super().parse_request()
            self._request_reader.remaining()
            if parsed and not self._valid_request_origin():
                self.close_connection = True
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "invalid_origin",
                    "request must address this local server",
                    head_only=self.command == "HEAD",
                )
                return False
            return parsed
        finally:
            self.connection.settimeout(self.timeout)

    def _valid_request_origin(self) -> bool:
        """Bind virtual hosts to the loopback endpoint before reading authority."""
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:
            return False
        host = hosts[0].lower()
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update({"127.0.0.1", "localhost"})
        if host not in allowed:
            return False
        origins = self.headers.get_all("Origin", [])
        # Non-browser API clients need not send Origin. Browsers that do send
        # it must match their request authority, including the port.
        return self.command != "POST" or not origins or origins == [f"http://{host}"]

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self._installation.close()

    @property
    def application(self) -> _WebApplication:
        return cast(_WebServer, self.server).application

    def do_GET(self) -> None:  # noqa: N802
        self._handle(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle(head_only=True)

    def do_POST(self) -> None:  # noqa: N802
        try:
            with workspace_access(self.application.repository):
                self._handle_post()
        except WorkspaceBusy as error:
            self.close_connection = True
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "workspace_maintenance",
                str(error),
                head_only=False,
            )
        except ContractError as error:
            self.close_connection = True
            self._error(
                HTTPStatus.CONFLICT,
                "workspace_format",
                str(error),
                head_only=False,
            )

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _handle(self, *, head_only: bool) -> None:
        try:
            with workspace_access(self.application.repository):
                self._handle_admitted(head_only=head_only)
        except WorkspaceBusy as error:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "workspace_maintenance",
                str(error),
                head_only=head_only,
            )
        except ContractError as error:
            self._error(
                HTTPStatus.CONFLICT,
                "workspace_format",
                str(error),
                head_only=head_only,
            )

    def _handle_admitted(self, *, head_only: bool) -> None:
        target = urlsplit(self.path)
        path = target.path
        if path == "/api/v1/overview":
            self._overview(head_only=head_only)
            return
        if path == "/api/v1/comparison":
            self._comparison(target.query, head_only=head_only)
            return
        if path == "/api/v1/run":
            self._run_detail(target.query, head_only=head_only)
            return
        if path == "/api/v1/authoring":
            self._authoring(head_only=head_only)
            return
        if path in {
            "/api/v1/authoring/experiment/commit",
            "/api/v1/authoring/variant/commit",
        }:
            self._method_not_allowed(allow="POST", head_only=head_only)
            return
        asset = self.application.assets.get(path)
        if asset is not None:
            content_type, body = asset
            self._respond(
                HTTPStatus.OK,
                body,
                content_type=content_type,
                head_only=head_only,
            )
            return
        self._error(
            HTTPStatus.NOT_FOUND,
            "not_found",
            "resource not found",
            head_only=head_only,
        )

    def _overview(self, *, head_only: bool) -> None:
        try:
            document = self.application.overview()
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                str(error),
                head_only=head_only,
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=head_only,
            )
            return
        except IndexFailure as error:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "status_unavailable",
                str(error),
                head_only=head_only,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _comparison(self, raw_query: str, *, head_only: bool) -> None:
        try:
            query = parse_qs(raw_query, keep_blank_values=True, max_num_fields=8)
        except ValueError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=head_only,
            )
            return
        try:
            unexpected = set(query) - {"run", "metric"}
            if unexpected or len(query.get("metric", [])) > 1:
                raise WebRequestError("comparison query fields are invalid")
            document = self.application.comparison(
                query.get("run", []),
                metric_name=(query.get("metric") or [None])[0],
            )
        except WebRequestError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=head_only,
            )
            return
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                str(error),
                head_only=head_only,
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=head_only,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _run_detail(self, raw_query: str, *, head_only: bool) -> None:
        try:
            query = parse_qs(raw_query, keep_blank_values=True, max_num_fields=2)
        except ValueError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=head_only,
            )
            return
        try:
            if (
                set(query) - {"run", "metric"}
                or len(query.get("run", [])) != 1
                or len(query.get("metric", [])) > 1
            ):
                raise WebRequestError("Run detail query fields are invalid")
            document = self.application.run_detail(
                query["run"][0], (query.get("metric") or [None])[0]
            )
        except WebRequestError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=head_only,
            )
            return
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND, "not_found", str(error), head_only=head_only
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=head_only,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _authoring(self, *, head_only: bool) -> None:
        try:
            document = self.application.authoring()
        except StalePreviewError as error:
            self._error(
                HTTPStatus.CONFLICT, "stale_preview", str(error), head_only=head_only
            )
            return
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                str(error),
                head_only=head_only,
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=head_only,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _handle_post(self) -> None:
        path = urlsplit(self.path).path
        if path not in {
            "/api/v1/authoring/experiment/commit",
            "/api/v1/authoring/variant/commit",
        }:
            self._method_not_allowed()
            return
        try:
            payload = self._mutation_document()
            token = payload.get("confirmation_token")
            if not isinstance(token, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{43}", token
            ):
                raise WebRequestError("Commit body requires a valid confirmation token")
            if path == "/api/v1/authoring/experiment/commit":
                if set(payload) != {"confirmation_token"}:
                    raise WebRequestError(
                        "Experiment commit body must contain only a confirmation token"
                    )
                document = self.application.commit_experiment(token)
            else:
                if set(payload) != {"variant", "confirmation_token"} or not isinstance(
                    payload.get("variant"), str
                ):
                    raise WebRequestError(
                        "Variant commit body must contain a Variant name and confirmation token"
                    )
                variant = str(payload["variant"])
                if not NAME_PATTERN.fullmatch(variant):
                    raise WebRequestError("Variant commit name is invalid")
                document = self.application.commit_variant(variant, token)
        except WebRequestError as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(error),
                head_only=False,
            )
            return
        except NotFoundError as error:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                str(error),
                head_only=False,
            )
            return
        except StalePreviewError as error:
            self._error(
                HTTPStatus.CONFLICT,
                "stale_preview",
                str(error),
                head_only=False,
            )
            return
        except DirtyDraftError as error:
            self._error(
                HTTPStatus.CONFLICT,
                "conflict",
                str(error),
                head_only=False,
            )
            return
        except (ContractError, OwnershipError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "contract_error",
                str(error),
                head_only=False,
            )
            return
        self._respond(
            HTTPStatus.OK,
            _json_bytes(document),
            content_type="application/json; charset=utf-8",
            head_only=False,
        )

    def _mutation_document(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            raise WebRequestError("Commit request transfer encoding is unsupported")
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError as error:
            self.close_connection = True
            raise WebRequestError("Commit request Content-Length is invalid") from error
        if length < 0 or length > MAX_MUTATION_BODY_BYTES:
            self.close_connection = True
            raise WebRequestError("Commit request body is too large")
        try:
            self._request_reader.remaining()
            encoded = self.rfile.read(length)
            self._request_reader.remaining()
        finally:
            self.connection.settimeout(self.timeout)
        if len(encoded) != length:
            self.close_connection = True
            raise WebRequestError("Commit request body is incomplete")
        if self.headers.get(MUTATION_HEADER) != MUTATION_HEADER_VALUE:
            raise WebRequestError(
                "Commit request is missing its same-origin action header"
            )
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise WebRequestError("Commit request must use application/json")
        try:
            decoded = encoded.decode("utf-8")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WebRequestError("Commit request body is not valid JSON") from error
        if not isinstance(payload, dict):
            raise WebRequestError("Commit request body must be an object")
        return cast(dict[str, Any], payload)

    def _method_not_allowed(
        self,
        *,
        allow: str = "GET, HEAD",
        head_only: bool = False,
    ) -> None:
        if (
            self.headers.get("Content-Length") not in (None, "0")
            or self.headers.get("Transfer-Encoding") is not None
        ):
            self.close_connection = True
        body = _json_bytes(
            {
                "error": {
                    "code": "method_not_allowed",
                    "message": f"method not allowed; use {allow}",
                }
            }
        )
        self._respond(
            HTTPStatus.METHOD_NOT_ALLOWED,
            body,
            content_type="application/json; charset=utf-8",
            extra_headers={"Allow": allow},
            head_only=head_only,
        )

    def _error(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        *,
        head_only: bool,
    ) -> None:
        self._respond(
            status,
            _json_bytes({"error": {"code": code, "message": message}}),
            content_type="application/json; charset=utf-8",
            head_only=head_only,
        )

    def _respond(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str,
        head_only: bool,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; base-uri 'none'; "
            "form-action 'none'",
        )
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)


@installation_operation
def create_server(
    repository: RepositoryPaths,
    experiment_name: str,
    *,
    port: int = DEFAULT_WEB_PORT,
    now: Callable[[], datetime] | None = None,
) -> _WebServer:
    """Validate the selected Experiment, then bind a loopback-only server."""
    build_overview(repository, experiment_name, now=now)
    application = _WebApplication(repository, experiment_name, now=now)
    try:
        return _WebServer((LOOPBACK_HOST, port), application)
    except OSError as error:
        raise WebFailure(
            f"could not bind http://{LOOPBACK_HOST}:{port}: {error}"
        ) from error


@installation_operation
def serve(
    repository: RepositoryPaths,
    experiment_name: str,
    *,
    port: int = DEFAULT_WEB_PORT,
) -> None:
    """Serve one selected Experiment until interrupted."""
    server = create_server(repository, experiment_name, port=port)
    actual_port = server.server_port
    print(
        f"serving {experiment_name} at http://{LOOPBACK_HOST}:{actual_port}/",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
