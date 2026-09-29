"""Define command syntax and the namespace consumed by main and dispatch.

Shared option registration lives in arguments; command-specific options, help
and completion bindings are composed here. main validates option combinations,
while command handlers and services validate workspace state at execution time.
"""

from __future__ import annotations

import argparse
import importlib.metadata

from hkdl.installer.documentation import procedure_help
from hkdl.interfaces import completion, web

from . import arguments


# Flow: root options -> command families and their verbs -> complete parser.
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI grammar without executing a command.

    noun and verb select dispatch branches; commands without a subcommand use
    verb=None. Argument destinations and defaults are the handlers' input
    contract. Completion callbacks offer candidates, not validity guarantees.
    """
    parser = argparse.ArgumentParser(
        prog="hkdl",
        description="Author and run reproducible ML experiments.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('hkdl')}",
    )
    nouns = parser.add_subparsers(title="commands", dest="noun", required=True)

    workspace = nouns.add_parser(
        "workspace", help="Initialize an independent research workspace"
    )
    workspace_verbs = workspace.add_subparsers(dest="verb", required=True)
    workspace_init = workspace_verbs.add_parser(
        "init",
        help="Initialize a workspace or adopt an existing source checkout",
    )
    workspace_path = workspace_init.add_argument(
        "path", nargs="?", default=".", metavar="PATH"
    )
    workspace_path.completer = completion.file_completer

    template = nouns.add_parser(
        "template",
        help="Inspect available Templates",
        description="Inspect Template bundles available in this checkout.",
    )
    template_verbs = template.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    template_list = template_verbs.add_parser(
        "list",
        help="List available Template versions",
        description="List available Template versions.",
        epilog=_examples("hkdl template list", "hkdl template list -o json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_output_argument(template_list)
    template_show = template_verbs.add_parser(
        "show",
        help="Show one Template version",
        description="Show one Template version and its bundle digest.",
        epilog=_examples("hkdl template show resnet18@1.0.0"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    template_reference = template_show.add_argument(
        "reference",
        metavar="TEMPLATE@VERSION",
        help="Template reference",
    )
    template_reference.completer = completion.complete_template_references
    arguments.add_output_argument(template_show)

    experiment = nouns.add_parser(
        "experiment",
        help="Create and inspect Experiments",
        description="Create and inspect authored Experiments.",
    )
    experiment_verbs = experiment.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    experiment_create = experiment_verbs.add_parser(
        "create",
        help="Create an Experiment",
        description="Create an Experiment for a Template family.",
        epilog=_examples("hkdl experiment create smoke -t resnet18"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    experiment_create.add_argument(
        "experiment",
        metavar="EXPERIMENT",
        help="Experiment name",
    )
    template_name = experiment_create.add_argument(
        "-t",
        "--template",
        required=True,
        metavar="TEMPLATE",
        help="Template family name",
    )
    template_name.completer = completion.complete_templates
    experiment_list = experiment_verbs.add_parser(
        "list",
        help="List authored Experiments",
        description="List authored Experiments.",
        epilog=_examples("hkdl experiment list", "hkdl experiment list -o json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_output_argument(experiment_list)
    experiment_commit = experiment_verbs.add_parser(
        "commit",
        help="Commit an Experiment draft revision",
        description="Publish the current Experiment draft as an immutable v2 revision.",
        epilog=_examples("hkdl experiment commit smoke"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(experiment_commit)
    arguments.add_output_argument(experiment_commit)
    experiment_rename = experiment_verbs.add_parser(
        "rename",
        help="Rename one Experiment binding and authored directory",
    )
    old_experiment = experiment_rename.add_argument(
        "old_name", metavar="OLD", help="Existing Experiment name"
    )
    old_experiment.completer = completion.complete_experiments
    experiment_rename.add_argument("new_name", metavar="NEW", help="New name")
    experiment_rename.add_argument(
        "--dry-run", action="store_true", help="Preview without changing authority"
    )
    arguments.add_output_argument(experiment_rename)
    experiment_delete = experiment_verbs.add_parser(
        "delete",
        help="Delete an Experiment and its complete active closure",
        description=(
            "Plan or confirm recoverable deletion of one Experiment and every "
            "owned Variant, Run, Model, result, and local projection."
        ),
        epilog=_examples(
            "hkdl experiment delete smoke --dry-run",
            "hkdl experiment delete smoke",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(
        experiment_delete,
        help_text="Experiment whose complete owned closure will be deleted",
    )
    experiment_delete.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the exact deletion plan without prompting or changing state",
    )
    arguments.add_output_argument(experiment_delete)

    variant = nouns.add_parser(
        "variant",
        help="Create, inspect, promote, rename, and delete Variants",
        description=(
            "Create, clone, inspect, validate, promote, rename, and recoverably "
            "delete authored Variants."
        ),
    )
    variant_verbs = variant.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    variant_create = variant_verbs.add_parser(
        "create",
        help="Create a Variant",
        description="Create a Variant from a Template version.",
        epilog=_examples("hkdl variant create smoke baseline"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(variant_create)
    arguments.add_variant_argument(
        variant_create, help_text="Variant name", existing=False
    )
    template_version = variant_create.add_argument(
        "--template-version",
        metavar="VERSION",
        help="Exact Template version; defaults to the latest numeric version",
    )
    template_version.completer = completion.complete_template_versions
    variant_clone = variant_verbs.add_parser(
        "clone",
        help="Clone an existing Variant",
        description="Clone an existing Variant and its source.",
        epilog=_examples(
            "hkdl variant clone smoke tuned -f baseline",
            ("hkdl variant clone target tuned -f baseline --from-experiment source"),
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(variant_clone)
    arguments.add_variant_argument(
        variant_clone, help_text="Target Variant name", existing=False
    )
    source_variant = variant_clone.add_argument(
        "-f",
        "--from",
        dest="source",
        required=True,
        metavar="VARIANT",
        help="Source Variant name",
    )
    source_variant.completer = completion.complete_source_variants
    source_experiment = variant_clone.add_argument(
        "--from-experiment",
        dest="source_experiment",
        metavar="EXPERIMENT",
        help="Source Experiment; defaults to the target Experiment",
    )
    source_experiment.completer = completion.complete_experiments
    variant_promote = variant_verbs.add_parser(
        "promote",
        help="Promote committed Source Code into an unchanged Target Variant",
        description=(
            "Plan or confirm one-sided promotion of committed schema-2 Variant "
            "Code while preserving Target Options and deleting the Source Variant."
        ),
        epilog=_examples(
            "hkdl variant promote smoke tuned --to baseline --dry-run",
            "hkdl variant promote smoke tuned --to baseline",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(variant_promote)
    promote_source = variant_promote.add_argument(
        "source",
        metavar="SOURCE",
        help="Source Variant whose committed Code is promoted",
    )
    promote_source.completer = completion.complete_variants
    promote_target = variant_promote.add_argument(
        "--to",
        dest="target",
        required=True,
        metavar="TARGET",
        help="Unchanged Target Variant that keeps its identity and Options",
    )
    promote_target.completer = completion.complete_variants
    variant_promote.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the exact promotion plan without prompting or changing state",
    )
    arguments.add_output_argument(variant_promote)
    variant_list = variant_verbs.add_parser(
        "list",
        help="List Variants",
        description="List Variants owned by an Experiment.",
        epilog=_examples("hkdl variant list smoke", "hkdl variant list smoke -o json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(variant_list)
    arguments.add_output_argument(variant_list)
    variant_check = variant_verbs.add_parser(
        "check",
        help="Validate a Variant",
        description="Validate one authored Variant.",
        epilog=_examples("hkdl variant check smoke baseline"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(variant_check)
    arguments.add_variant_argument(variant_check, help_text="Variant name")
    arguments.add_output_argument(variant_check)
    variant_commit = variant_verbs.add_parser(
        "commit",
        help="Commit a Variant draft revision",
        description="Publish the current Variant recipe and source as a v2 revision.",
        epilog=_examples("hkdl variant commit smoke baseline"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(variant_commit)
    arguments.add_variant_argument(variant_commit, help_text="Variant draft to commit")
    arguments.add_output_argument(variant_commit)
    variant_rename = variant_verbs.add_parser(
        "rename",
        help="Rename one Variant binding",
        description="Atomically move one active Variant name binding without an alias.",
        epilog=_examples(
            "hkdl variant rename smoke old-name new-name --dry-run",
            "hkdl variant rename smoke old-name new-name",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(variant_rename)
    old_name = variant_rename.add_argument("old_name", metavar="OLD")
    old_name.completer = completion.complete_variants
    variant_rename.add_argument("new_name", metavar="NEW")
    variant_rename.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the binding change without writing it",
    )
    arguments.add_output_argument(variant_rename)
    variant_delete = variant_verbs.add_parser(
        "delete",
        help="Delete one Variant and its complete active closure",
        description=(
            "Plan or confirm recoverable deletion of one Variant and every owned "
            "Run, Model, result, and local projection."
        ),
        epilog=_examples(
            "hkdl variant delete smoke baseline --dry-run",
            "hkdl variant delete smoke baseline",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(variant_delete)
    arguments.add_variant_argument(
        variant_delete,
        help_text="Variant whose complete owned closure will be deleted",
    )
    variant_delete.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the exact deletion and lineage plan without prompting",
    )
    arguments.add_output_argument(variant_delete)

    model = nouns.add_parser(
        "model",
        help="Inspect trained Models",
        description="Inspect immutable Models produced by Train Runs.",
    )
    model_verbs = model.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    model_list = model_verbs.add_parser(
        "list",
        help="List Models",
        description="List Models owned by one Variant.",
        epilog=_examples("hkdl model list smoke baseline"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(model_list)
    arguments.add_variant_argument(model_list, help_text="Variant owning the Models")
    arguments.add_output_argument(model_list)
    model_show = model_verbs.add_parser(
        "show",
        help="Show one Model",
        description="Show one immutable Model manifest.",
        epilog=_examples(
            "hkdl model show smoke baseline model-0123456789abcdef0123456789abcdef"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(model_show)
    arguments.add_variant_argument(model_show, help_text="Variant owning the Model")
    arguments.add_model_argument(model_show)
    arguments.add_output_argument(model_show)

    run = nouns.add_parser(
        "run",
        help="Execute Variants",
        description="Execute authored Variants.",
    )
    run_verbs = run.add_subparsers(title="commands", dest="verb", required=True)
    run_train = run_verbs.add_parser(
        "train",
        help="Train a Variant",
        description="Train a Variant and create a persistent Run.",
        epilog=_examples(
            "hkdl run train smoke baseline stability",
            "hkdl run train smoke baseline stability -s 0,1,2 -d cpu",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(run_train)
    arguments.add_variant_argument(run_train, help_text="Variant to train")
    training_group = run_train.add_argument(
        "training_group",
        metavar="GROUP",
        help="Training Group name",
    )
    training_group.completer = completion.complete_training_groups
    arguments.add_train_seed_argument(run_train)
    arguments.add_device_argument(run_train)
    arguments.add_tracker_argument(run_train)

    run_eval = run_verbs.add_parser(
        "eval",
        help="Evaluate Models in a Training Group",
        description="Create Eval Runs for selected Models and one Evaluation Case.",
        epilog=_examples(
            "hkdl run eval smoke baseline stability clean",
            "hkdl run eval smoke baseline stability clean -s all",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(run_eval)
    arguments.add_variant_argument(run_eval, help_text="Variant owning the Models")
    evaluation_group = run_eval.add_argument("training_group", metavar="GROUP")
    evaluation_group.completer = completion.complete_training_groups
    evaluation_case = run_eval.add_argument("evaluation_case", metavar="CASE")
    evaluation_case.completer = completion.complete_evaluation_cases
    arguments.add_eval_seed_argument(run_eval)
    arguments.add_device_argument(run_eval)

    run_export = run_verbs.add_parser(
        "export",
        help="Export one Model",
        description="Create one Export Run for an immutable Model.",
        epilog=_examples(
            "hkdl run export smoke baseline model-0123456789abcdef0123456789abcdef"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(run_export)
    arguments.add_variant_argument(run_export, help_text="Variant owning the Model")
    arguments.add_model_argument(run_export)
    arguments.add_device_argument(run_export)

    run_retry = run_verbs.add_parser(
        "retry",
        help="Retry a stopped action as a new Run",
        description="Create one new Run from a failed, interrupted, or abandoned Run.",
        epilog=_examples("hkdl run retry smoke baseline run-001"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(run_retry)
    arguments.add_variant_argument(run_retry, help_text="Variant owning the Run")
    arguments.add_run_argument(run_retry)
    arguments.add_tracker_argument(run_retry)

    run_delete = run_verbs.add_parser(
        "delete",
        help="Delete an active Run lineage from the v2 graph",
        description="Plan or apply a recoverable graph-aware Run deletion.",
    )
    arguments.add_experiment_argument(run_delete)
    arguments.add_variant_argument(run_delete, help_text="Variant owning the Run")
    arguments.add_run_argument(run_delete)
    run_delete.add_argument(
        "--cascade",
        action="store_true",
        help="Include retry children, Models, Eval/Export Runs, and results",
    )
    run_delete.add_argument(
        "--force-stale",
        action="store_true",
        help="Permit deletion of unlocked nonterminal Runs",
    )
    run_delete.add_argument(
        "--dry-run", action="store_true", help="Show the exact deletion closure"
    )
    run_delete.add_argument(
        "--yes", action="store_true", help="Confirm and apply the deletion"
    )
    arguments.add_output_argument(run_delete)

    run_metrics = run_verbs.add_parser(
        "metrics",
        help="Inspect local Train metric history",
        description="Show local scalar history for one Train Run.",
        epilog=_examples(
            "hkdl run metrics smoke baseline run-001",
            "hkdl run metrics smoke baseline run-001 -o json",
            "hkdl run metrics smoke baseline run-001 --follow",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(run_metrics)
    arguments.add_variant_argument(
        run_metrics, help_text="Variant owning the Train Run"
    )
    arguments.add_run_argument(run_metrics)
    arguments.add_output_argument(run_metrics)
    run_metrics.add_argument(
        "--follow",
        action="store_true",
        help="Follow new local metrics until the Run becomes terminal",
    )

    run_logs = run_verbs.add_parser(
        "logs",
        help="Inspect one Run's action-worker output",
        description="Show merged stdout and stderr captured from one action worker.",
        epilog=_examples("hkdl run logs smoke baseline run-001"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arguments.add_experiment_argument(run_logs)
    arguments.add_variant_argument(run_logs, help_text="Variant owning the Run")
    arguments.add_run_argument(run_logs)

    status = nouns.add_parser(
        "status",
        help="Inspect authoritative Run state",
        description="Show authoritative Runs grouped by Variant, Training Group, and seed.",
        epilog=_examples(
            "hkdl status",
            "hkdl status smoke baseline",
            "hkdl status smoke --table",
            "hkdl status smoke baseline run-001 -o json",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    status.set_defaults(verb=None)
    arguments.add_experiment_argument(
        status,
        required=False,
        help_text="Experiment filter",
    )
    arguments.add_variant_argument(
        status,
        required=False,
        help_text="Variant filter",
    )
    arguments.add_run_argument(status, required=False)
    arguments.add_output_argument(status)
    status_views = status.add_mutually_exclusive_group()
    status_views.add_argument(
        "--full",
        action="store_true",
        help="Show expanded Run details in text output",
    )
    status_views.add_argument(
        "--table",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show aggregate evaluation results as a table",
    )

    web_parser = nouns.add_parser(
        "web",
        help="Open a local read-only Experiment view",
        description="Serve one Experiment through a loopback-only read-only web view.",
        epilog=_examples("hkdl web smoke", "hkdl web smoke --port 8765"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    web_parser.set_defaults(verb=None)
    arguments.add_experiment_argument(web_parser, help_text="Experiment to inspect")
    web_parser.add_argument(
        "--port",
        type=arguments.parse_web_port,
        default=web.DEFAULT_WEB_PORT,
        metavar="PORT",
        help=f"Loopback TCP port (default: {web.DEFAULT_WEB_PORT})",
    )

    storage = nouns.add_parser(
        "storage",
        help="Inspect repository-owned storage usage",
        description="Show logical bytes used by authored and generated HKDL state.",
        epilog=_examples("hkdl storage", "hkdl storage -o json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    storage.set_defaults(verb=None)
    arguments.add_output_argument(storage)

    settings = nouns.add_parser(
        "settings",
        help="Inspect and change workspace execution settings",
        description="Manage operational defaults outside research authoring files.",
    )
    settings_verbs = settings.add_subparsers(
        title="commands", dest="verb", required=True
    )
    settings_show = settings_verbs.add_parser("show", help="Show workspace settings")
    arguments.add_output_argument(settings_show)
    tracker = settings_verbs.add_parser("tracker", help="Manage tracker defaults")
    tracker_verbs = tracker.add_subparsers(
        title="commands", dest="settings_tracker_verb", required=True
    )
    tracker_set = tracker_verbs.add_parser(
        "set", help="Set the workspace default tracker"
    )
    tracker_set.add_argument(
        "backend",
        metavar="BACKEND",
        help="none, local, mlflow, or local+mlflow",
    )
    arguments.add_output_argument(tracker_set)

    identity = nouns.add_parser(
        "identity",
        help="Inspect content-addressed identity details",
        description="Show full v2 hashes only when explicitly requested.",
    )
    identity_verbs = identity.add_subparsers(
        title="commands", dest="verb", required=True
    )
    identity_show = identity_verbs.add_parser("show", help="Show one identity")
    identity_show.add_argument(
        "kind", choices=("experiment", "variant", "run", "model")
    )
    identity_show.add_argument(
        "address",
        nargs="+",
        metavar="ADDRESS",
        help="Experiment [Variant [Run or Model]]",
    )
    arguments.add_output_argument(identity_show)

    index = nouns.add_parser(
        "index",
        help="Inspect and rebuild the status projection",
        description="Inspect and rebuild the disposable SQLite status projection.",
    )
    index_verbs = index.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    index_status = index_verbs.add_parser(
        "status",
        help="Inspect projection health without changing it",
        description="Inspect the disposable status projection without changing it.",
        epilog=_examples("hkdl index status"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    index_status.set_defaults(verb="status")
    index_rebuild = index_verbs.add_parser(
        "rebuild",
        help="Rebuild the projection from authoritative files",
        description="Validate authoritative files and atomically rebuild the projection.",
        epilog=_examples("hkdl index rebuild"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    index_rebuild.set_defaults(verb="rebuild")

    environment = nouns.add_parser(
        "environment",
        help="Manage generated Variant environments",
        description="Manage repository-local generated Variant environments.",
    )
    environment_verbs = environment.add_subparsers(
        title="commands",
        dest="verb",
        required=True,
    )
    environment_prune = environment_verbs.add_parser(
        "prune",
        help="Remove unused generated environments",
        description="Preview and remove safely reproducible Variant environments.",
        epilog=_examples(
            "hkdl environment prune --dry-run",
            "hkdl environment prune",
            "hkdl environment prune --all --yes",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    environment_prune.add_argument(
        "--all",
        action="store_true",
        help="Also remove inactive environments referenced by current Variants",
    )
    environment_confirmation = environment_prune.add_mutually_exclusive_group()
    environment_confirmation.add_argument(
        "--dry-run",
        action="store_true",
        help="Show removable environments without prompting or modifying state",
    )
    environment_confirmation.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm the shown prune without prompting",
    )

    completion_parser = nouns.add_parser(
        "completion",
        help="Generate shell completion registration",
        description="Generate shell code that registers HKDL completion.",
        epilog=_examples(
            'eval "$(hkdl completion zsh)"',
            'eval "$(hkdl completion bash)"',
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    completion_parser.set_defaults(verb=None)
    completion_parser.add_argument(
        "shell",
        choices=completion.SHELLS,
        metavar="SHELL",
        help="Shell to register: bash or zsh",
    )

    migrate = nouns.add_parser(
        "migrate",
        help="Inspect one schema or migrate the full workspace",
        description=(
            "Inspect or migrate one authored Experiment or Variant file. "
            "Use --all for storage import or --authoring for authoring conversion. "
            "Preview and approve the plan before applying; migration is not error repair."
        ),
        epilog=_examples(
            "hkdl migrate experiments/smoke/experiment.yaml",
            "hkdl migrate experiments/smoke/baseline/variant.yaml",
            "hkdl migrate --all --dry-run -o json",
            "hkdl migrate --authoring --dry-run -o json",
        )
        + "\n\n"
        + procedure_help("migration"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    migrate.set_defaults(verb=None)
    migrate.add_argument(
        "--expect-plan",
        default=argparse.SUPPRESS,
        metavar="SHA256",
        help="Require the approved authoring migration plan digest",
    )
    migrate.add_argument(
        "--tracker-default",
        choices=("none", "local", "mlflow", "local,mlflow"),
        default=argparse.SUPPRESS,
        help="Explicit future workspace tracker for --authoring; preserve Run history",
    )
    migrate_path = migrate.add_argument(
        "path",
        nargs="?",
        metavar="PATH",
        help="Repository-owned experiment.yaml or variant.yaml",
    )
    migrate_path.completer = completion.file_completer
    migrate.add_argument(
        "--all",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Plan or execute a complete v1-to-v2 workspace import",
    )
    migrate.add_argument(
        "--authoring",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Migrate active-v2 authored YAML and execution identity to schema 2",
    )
    migrate.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Compute and report the full import without writing authority",
    )
    migrate.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Confirm v2 cutover without prompting",
    )
    migrate.add_argument(
        "-o",
        "--output",
        choices=("text", "json"),
        default=argparse.SUPPRESS,
        help="Output format for --all (default: text)",
    )

    update_parser = nouns.add_parser(
        "update",
        help="Update HKDL from a source checkout or release bundle",
        description=(
            "Inspect and update a clean public HKDL source checkout. "
            "Managed installations instead accept an explicitly selected release bundle. "
            "Review the update preview before confirming. Updating does not migrate research."
        ),
        epilog=_examples("hkdl update BUNDLE.zip", "hkdl update")
        + "\n\n"
        + procedure_help("updating"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    update_parser.set_defaults(verb=None)
    bundle_path = update_parser.add_argument(
        "bundle",
        nargs="?",
        metavar="BUNDLE",
        help="Managed release bundle path or HTTPS URL",
    )
    bundle_path.completer = completion.file_completer
    update_parser.add_argument("--sha256", help="Expected managed bundle SHA-256")
    update_parser.add_argument(
        "--repair",
        action="store_true",
        help="Rebuild a managed installation separately",
    )
    update_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm the shown update without prompting",
    )

    return parser


def _examples(*commands: str) -> str:
    return "examples:\n" + "\n".join(f"  {command}" for command in commands)
