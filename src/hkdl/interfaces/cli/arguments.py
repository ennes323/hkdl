"""Register options whose syntax and defaults are shared by command families.

parser composes these helpers into individual commands. Converters reject
malformed values during argparse processing; completion callbacks only suggest
candidates. Workspace and execution validity remain with the called services.
"""

from __future__ import annotations

import argparse

from hkdl.execution.run_contracts import MAX_SEED
from hkdl.interfaces import completion


def parse_web_port(value: str) -> int:
    """Convert a port token and report invalid values through argparse."""
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def add_experiment_argument(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
    help_text: str = "Experiment containing the Variant",
) -> None:
    action = parser.add_argument(
        "experiment",
        nargs=None if required else "?",
        metavar="EXPERIMENT",
        help=help_text,
    )
    action.completer = completion.complete_experiments


def add_variant_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
    required: bool = True,
    existing: bool = True,
) -> None:
    """Register a Variant name, omitting existing-name completion for creation."""
    action = parser.add_argument(
        "variant",
        nargs=None if required else "?",
        metavar="VARIANT",
        help=help_text,
    )
    if existing:
        action.completer = completion.complete_variants


def add_run_argument(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
) -> None:
    action = parser.add_argument(
        "run_id",
        nargs=None if required else "?",
        metavar="RUN",
        help="Run ID",
    )
    action.completer = completion.complete_runs


def add_train_seed_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-s",
        "--seed",
        dest="seeds",
        type=_parse_seed_list,
        default=(0,),
        metavar="SEED[,SEED...]",
        help="One or more random seeds (default: 0)",
    )


def add_eval_seed_argument(parser: argparse.ArgumentParser) -> None:
    action = parser.add_argument(
        "-s",
        "--seed",
        type=_parse_eval_seed,
        default=None,
        metavar="SEED|all",
        help="One Model seed or all; inferred when the group has one Model",
    )
    action.completer = completion.complete_eval_seeds


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    action = parser.add_argument(
        "model_id",
        metavar="MODEL",
        help="Model ID",
    )
    action.completer = completion.complete_models


def _parse_seed_list(value: str) -> tuple[int, ...]:
    """Preserve requested seed order while rejecting duplicates and invalid values."""
    parts = value.split(",")
    if not parts or any(
        not part or not part.isascii() or not part.isdigit() for part in parts
    ):
        raise argparse.ArgumentTypeError("seed list must contain decimal integers")
    seeds = tuple(int(part) for part in parts)
    if any(seed > MAX_SEED for seed in seeds):
        raise argparse.ArgumentTypeError(f"seed must not exceed {MAX_SEED}")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seed list must not contain duplicates")
    return seeds


def _parse_eval_seed(value: str) -> int | str:
    """Accept one seed or the all selector; model selection happens in execution."""
    if value == "all":
        return value
    return _parse_seed_list(value)[0] if "," not in value else _invalid_eval_seed()


def _invalid_eval_seed():
    raise argparse.ArgumentTypeError("evaluation seed must be one integer or all")


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    action = parser.add_argument(
        "-d",
        "--device",
        default="auto",
        metavar="DEVICE",
        help="auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    action.completer = completion.complete_devices


def add_tracker_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracker",
        metavar="BACKEND",
        help="Override workspace tracker: none, local, mlflow, or local+mlflow",
    )


def add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
