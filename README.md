# HKDL

HKDL authors self-contained ML Variants and records immutable execution
history. Version 1.1.5 supports authoring, multi-seed training, immutable
Models, named evaluation cases with optional result artifacts, Variant-owned
export, new-Run retry, brief, full, and aggregate-table status projections,
disposable SQLite acceleration for multi-Variant status reads,
live local metric following, Run-owned worker logs, shared locked Variant
environments, explicit environment pruning, read-only storage reporting, shell
completion, and opt-in MLflow tracking.

An Experiment may also contain optional `docs/` and `tools/` directories for
authored documentation and utilities. HKDL excludes these two real,
non-symlink directories from Variant discovery and shell completion and
reserves both names from Variant creation. Every other visible Experiment
directory remains a Variant catalog entry and is validated as such. An
existing Variant named `docs` or `tools` must be renamed before updating;
HKDL does not migrate it automatically.

For optional research-decision history, copy the
[Experiment Decision Record starter](docs/examples/experiment-decision-record.yaml)
into `experiments/<experiment>/docs/records/<RECORD-ID>.yaml` and follow the
[authoring guidance](docs/experiment-decision-records.md). These notes are not
discovered, validated, updated, or required by HKDL and do not affect execution
or cleanup behavior.

## Setup

From the cloned repository root:

```text
./setup.sh
source ./activate.sh
hkdl --version
hkdl --help
```

`setup.sh` is the maintained environment entrypoint. It creates the root
`.venv` as a relocatable environment when needed, normalizes an existing root
environment after a checkout move, and recreates only a recognized incomplete
virtual environment. It then performs a locked, non-editable full reinstall
and verifies the installed CLI. A symlink, file, or non-venv directory at
`.venv` is rejected without deletion. Setup changes only the generated root
environment; authored Experiments, Variant source, Runs, Models, outputs, and
other generated state are preserved. `activate.sh` activates that root
environment and registers completion in the current Bash or Zsh session.

## Shell completion

The recommended `source ./activate.sh` flow registers completion automatically.
To register it manually after activating the environment, run only the line for
your shell:

```text
eval "$(hkdl completion zsh)"
eval "$(hkdl completion bash)"
```

HKDL completes commands, options, Templates, Experiments, Variants, Runs,
Models, Training Groups, Evaluation Cases, seeds, and devices from the current
repository. Static command and option completion also works outside a
repository. Neither setup nor activation modifies shell startup files or
installs completion globally; source `activate.sh` once in each new shell.

## Update

From a public source checkout root:

```text
hkdl update
```

The command shows the installed and available versions, lists what will change
and what will be preserved, and asks before updating. It fast-forwards the
public source and reinstalls HKDL. Experiments, outputs, existing Variants, and
authored schemas are not changed. The optional ignored `AGENTS.user.md` is
user-owned and is also preserved. Use `hkdl update --yes` only when confirmation
has already been provided.

The public `AGENTS.md` contains HKDL-managed coding-agent defaults. Put local
execution policy, resource limits, and reporting preferences in an optional
root `AGENTS.user.md`; HKDL does not distribute, track, replace, migrate, or
delete that file.

## End-to-end example

Create an Experiment and copy the latest Template into a Variant:

```text
hkdl experiment create demo --template resnet18
hkdl variant create demo baseline
```

Train two seeds in one Training Group:

```text
hkdl run train demo baseline stability --seed 0,1
hkdl model list demo baseline
```

Evaluate all Models with a named Evaluation Case:

```text
hkdl run eval demo baseline stability default --seed all
hkdl run eval demo baseline stability daisy-only --seed 0
```

Export one exact Model:

```text
hkdl run export demo baseline model-0123456789abcdef0123456789abcdef
```

Inspect the reconstructed hierarchy:

```text
hkdl status demo baseline
hkdl status demo baseline --full
hkdl status demo --table
hkdl status demo baseline --output json
```

`--table` compares aggregate Eval results across Variants and Training Groups,
with one row per Evaluation Case and metric. It is a text-only alternative to
the brief and full hierarchy views.

Range status queries may maintain `outputs/.hkdl-index.sqlite3` as a disposable
projection. Authored Experiment/Variant files and generated Run/Model files
remain authoritative; exact Run status continues to read its files directly.
Missing, stale, incompatible, corrupt, or unavailable projection state falls
back to validated files.

```text
hkdl index status
hkdl index rebuild
```

`index status` inspects projection health without writing. `index rebuild`
validates current authority, builds a sibling candidate database, and
atomically replaces only the projection. Neither command migrates or rewrites
authored or generated records.

Inspect repository-owned local storage without changing it:

```text
hkdl storage
hkdl storage --output json
```

The report separates authored Experiment content, Variant environments, and
outputs. Disposable status-index files and sidecars are excluded. Environment
bytes count legacy per-Variant `.venv` directories plus
the repository-local shared store once. Sizes are logical file bytes; symlinks
are not followed. The command does not delete or prune anything.

Variant actions reuse an immutable environment when their lock files, optional
extras, exact Python runtime, platform, and `uv` version match. HKDL keeps the
shared store under `.hkdl/environments/`; existing per-Variant `.venv`
directories remain untouched until an explicit prune.

Preview or confirm safe cleanup with:

```text
hkdl environment prune --dry-run
hkdl environment prune
hkdl environment prune --yes
```

The default prune removes legacy Variant environments, incomplete cache
entries, and shared environments no longer referenced by an authored Variant.
`hkdl environment prune --all` also selects referenced but inactive shared
environments. Environments with an active execution lease are always skipped.

Each train, evaluation, or export command above creates one immutable action
Run. A successful Train Run also creates an immutable Model. Evaluation and
export target Models; they do not advance a shared pipeline Run.

Inspect the merged stdout and stderr captured from one action worker:

```text
hkdl run logs demo baseline run-001
```

The log also captures ordinary child processes that inherit the worker's output
descriptors. It is raw local output with no redaction or size limit, so Variant
code must not print secrets. Direct terminal writes and detached daemons are not
captured. Existing Runs created before this feature may have no log.

## Retry

A failed, interrupted, or abandoned action is retried as a new Run:

```text
hkdl run retry demo baseline run-003
```

The parent stays sealed and the child records `retry_of`. A valid last
checkpoint may be used to continue training.

## Schema compatibility

Authored Experiment and Variant files currently use schema version 1. Inspect
one authored file through the stable migration boundary:

```text
hkdl migrate experiments/demo/experiment.yaml
hkdl migrate experiments/demo/baseline/variant.yaml
```

Version 1 is already current, so these commands validate the file and do not
rewrite it. No migration path is registered yet. Unsupported schema versions
fail without changing the target. Generated Run and Model records are immutable
and cannot be migrated in place.

## ResNet18 fixtures

`resnet18@1.0.1` includes the attributed small TF-Flowers JPEG fixture, two
evaluation cases (`default` and `daisy-only`), prediction result output, ONNX
export support, checkpoint continuation, and optional MLflow dependencies.
It records loss and per-batch wall time and performs no dataset or
pretrained-weight download at runtime. The immutable `1.0.0` remains available.

## YOLO26n object detection

`yolo26n@1.0.1` includes a deterministic synthetic shapes dataset with eight
training images, four validation images, and two classes (`circle` and
`rectangle`). It trains the architecture from scratch for ten epochs and
records loss and per-epoch wall time. It performs no dataset or
pretrained-weight download at runtime. The immutable `1.0.0` remains available.

```text
hkdl experiment create detection --template yolo26n
hkdl variant create detection baseline
hkdl run train detection baseline smoke --seed 0 -d cpu
hkdl run eval detection baseline smoke default --seed 0
hkdl model list detection baseline
```

The Template reports finite detection metrics, writes deterministic prediction
JSON, and exports a fixed-shape ONNX model together with its AGPL license.
Ultralytics network checks, automatic dependency installation, and third-party
tracking integrations are disabled. HKDL remains the only tracking owner.

## Tracking

New bundled `1.0.1` Variants default to `tracker.backend: local`. Training
scalars, including `train.batch_seconds` for ResNet18 and
`train.epoch_seconds` for YOLO26n, are stored with the Run and can be inspected
with:

```text
.venv/bin/hkdl run metrics <experiment> <variant> <run-id>
```

Follow an active local-tracked Train Run from another terminal:

```text
.venv/bin/hkdl run metrics <experiment> <variant> <run-id> --follow
```

Follow prints existing metric rows, then newly completed rows until the Run
becomes terminal. It remains a read-only local view: JSON streaming, progress
percentages, ETA, and MLflow history polling are not provided.

Use `tracker.backend: none` to disable tracking, `mlflow` for MLflow only, or
`[local, mlflow]` for both. MLflow requires an external `MLFLOW_TRACKING_URI`.
Every local execution Run gets a distinct external Run. Retry uses a new
external identity with parent relation tags. HKDL does not start or manage an
MLflow server.

## License

HKDL Core and files without a more specific notice are released under the MIT
License. The bundled TF-Flowers fixture retains its own attribution and CC BY
4.0 terms in its `ATTRIBUTION.md`.

The `yolo26n` Template, Variants derived from it, and model artifacts produced
with Ultralytics are licensed under GNU AGPL version 3 or later. Its source
bundle includes the complete license and scope notice. Commercial users who do
not wish to comply with the AGPL requirements must obtain an Ultralytics
Enterprise License.
