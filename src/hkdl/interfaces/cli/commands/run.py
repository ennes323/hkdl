"""Present Run operations while execution services own lifecycle changes.

Training, evaluation, export, retry, and deletion services validate and mutate
Run state. This module selects an operation, renders its result, and implements
the CLI-only streaming view over metrics stored by ``RunStore``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

from hkdl.authoring.authoring import Authoring
from hkdl.errors import ContractError
from hkdl.execution.evaluation import Evaluation
from hkdl.execution.export import Export
from hkdl.execution.recovery import Recovery
from hkdl.execution.run_contracts import TERMINAL_STATUSES
from hkdl.execution.run_records import RunRecord
from hkdl.execution.training import Training, TrainingFailure
from hkdl.interfaces.cli import io
from hkdl.storage.graph.deletion import RunDeletionService
from hkdl.storage.runs import RunStore
from hkdl.storage.storage import RepositoryPaths


# Flow: select an operation -> delegate state work -> render its public result.
def run(repository: RepositoryPaths, args: argparse.Namespace) -> int:
    """Invoke the selected Run service and render its CLI result.

    Multi-seed training reports each ``TrainingFailure`` and continues with the
    remaining seeds. Other failures propagate to the process boundary, where
    ``main`` maps them to stable exit codes after access scopes unwind.
    """
    assert args.noun == "run"
    authoring = Authoring(repository)

    if args.verb == "train":
        failed = False
        training = Training(authoring.repository)
        for seed in args.seeds:
            try:
                record = training.train(
                    args.experiment,
                    args.variant,
                    args.training_group,
                    seed=seed,
                    device=args.device,
                    tracker=args.tracker,
                )
            except TrainingFailure as error:
                address = f" for {error.address}" if error.address else ""
                print(
                    f"error: training failed{address}: {error}",
                    file=sys.stderr,
                )
                failed = True
                continue
            print(
                f"trained {record.address} "
                f"model={record.state['result']['model_id']} "
                f"group={record.request['target']['training_group']} "
                f"seed={record.request['target']['seed']}"
            )
        return 6 if failed else 0

    if args.verb == "eval":
        records = Evaluation(authoring.repository).evaluate(
            args.experiment,
            args.variant,
            args.training_group,
            args.evaluation_case,
            seed=args.seed,
            device=args.device,
        )
        if not records:
            print(
                f"no evaluations needed "
                f"{args.experiment}/{args.variant}/{args.training_group}/"
                f"{args.evaluation_case}"
            )
        for record in records:
            print(
                f"evaluated {record.address} "
                f"model={record.request['target']['model_id']} "
                f"case={record.request['target']['evaluation_case']}"
            )
        return 0

    if args.verb == "export":
        record = Export(authoring.repository).export(
            args.experiment,
            args.variant,
            args.model_id,
            device=args.device,
        )
        print(f"exported {record.address} model={record.request['target']['model_id']}")
        return 0

    if args.verb == "retry":
        record = Recovery(authoring.repository).retry(
            args.experiment,
            args.variant,
            args.run_id,
            tracker=args.tracker,
        )
        action = record.request["action"]
        if action == "train":
            print(
                f"trained {record.address} "
                f"model={record.state['result']['model_id']} "
                f"group={record.request['target']['training_group']} "
                f"seed={record.request['target']['seed']}"
            )
        elif action == "eval":
            print(
                f"evaluated {record.address}"
                f" model={record.request['target']['model_id']}"
                f" case={record.request['target']['evaluation_case']}"
            )
        else:
            print(
                f"exported {record.address} "
                f"model={record.request['target']['model_id']}"
            )
        return 0

    if args.verb == "delete":
        service = RunDeletionService(authoring.repository)
        result = service.delete(
            args.experiment,
            args.variant,
            args.run_id,
            cascade=args.cascade,
            force_stale=args.force_stale,
            yes=args.yes,
            dry_run=args.dry_run,
        )
        document = io.public_payload(result.as_dict())
        if args.output == "json":
            io.print_json({"deletion": document, "dry_run": args.dry_run})
        elif args.dry_run:
            readiness = "ready" if document["ready"] else "blocked"
            print(
                f"deletion {readiness} {args.experiment}/{args.variant}/{args.run_id} "
                f"runs={len(document['runs'])} models={len(document['models'])}"
            )
            for conflict in document["conflicts"]:
                print(f"conflict: {conflict}")
        else:
            print(
                f"deleted run {args.experiment}/{args.variant}/{args.run_id} "
                f"transaction={document['transaction']}"
            )
        return 5 if args.dry_run and not document["ready"] else 0

    if args.verb == "metrics":
        store = RunStore(authoring.repository)
        record = store.load(args.experiment, args.variant, args.run_id)
        if args.follow:
            _follow_training_metrics(store, record)
            return 0
        metrics = store.load_training_metrics(record)
        if args.output == "json":
            io.print_json(
                {
                    "experiment": args.experiment,
                    "variant": args.variant,
                    "run_id": args.run_id,
                    "status": record.state["status"],
                    "partial": metrics["partial"],
                    "metrics": metrics["events"],
                }
            )
        else:
            io.table(
                ("STEP", "METRIC", "VALUE"),
                (
                    (event["step"], event["name"], event["value"])
                    for event in metrics["events"]
                ),
            )
        return 0

    if args.verb == "logs":
        store = RunStore(authoring.repository)
        record = store.load(args.experiment, args.variant, args.run_id)
        log_path = store.resolve_worker_log(record)
        with log_path.open("rb") as source:
            while chunk := source.read(64 * 1024):
                sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        return 0

    raise AssertionError("unreachable command")


# Flow: poll appended events -> reject duplicate positions -> reconcile at exit.
def _follow_training_metrics(
    store: RunStore,
    record: RunRecord,
    *,
    wait: Callable[[float], None] = time.sleep,
) -> None:
    """Stream append-only training metrics and reconcile them at termination.

    Polling intentionally performs live reads rather than using dispatch's
    single-HEAD observation. The final full history detects truncation, rewrite,
    or duplicate steps before the command reports a terminal Run.
    """
    experiment = record.request["experiment"]
    variant = record.request["variant"]
    run_id = record.request["run_id"]
    offset = 0
    events: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    header_written = False
    while True:
        current = store.load(experiment, variant, run_id)
        chunk = store.load_training_metric_chunk(current, offset=offset)
        if not header_written:
            print("STEP  METRIC  VALUE", flush=True)
            header_written = True
        for event in chunk["events"]:
            key = (event["name"], event["step"])
            if key in seen:
                raise ContractError("Train metric history contains a duplicate step")
            seen.add(key)
            events.append(event)
            print(
                f"{event['step']}  {event['name']}  {event['value']}",
                flush=True,
            )
        offset = chunk["offset"]
        if current.state["status"] in TERMINAL_STATUSES:
            final = store.load_training_metrics(current)
            if final["events"] != events:
                raise ContractError("Train metric history changed while following")
            partial = str(final["partial"]).lower()
            print(
                f"run {current.address} status={current.state['status']} "
                f"partial={partial}",
                file=sys.stderr,
                flush=True,
            )
            return
        wait(1.0)
