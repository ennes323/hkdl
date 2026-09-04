# HKDL

HKDL authors self-contained ML Variants and records immutable execution
history. The 1.2 release line separates user-authored research JSON from mechanically
managed identities, revisions, Run attempts, Models and results. It adds the v2
content-addressed store, explicit legacy migration, Code/Options snapshots,
safe rename/deletion, and a local Experiment web view with comparisons,
learning curves, Run inspection and explicit Changes commits. Multi-seed
training, named evaluation cases, Variant-owned export, checkpoint retry,
local logs/metrics, shared locked environments and opt-in MLflow remain supported.

Version 1.2.1 completes the scope originally intended for 1.2.0, whose JSON
authoring release was prioritized. It includes the refined Experiment/Variant
web layout and multiple metric charts with shared Run selection, explicit loss
overlay, and quieter refresh feedback. It also adds recoverable whole-Experiment
and whole-Variant deletion and one-sided committed Variant Code promotion.
These are explicit CLI operations; updating does not run them automatically.
This is a release-specific patch-numbering decision, not a claim that the
release contains bug fixes only. Broader web editing is planned for the 1.3 line
and is not included here; existing Changes commit controls remain available.

Version 1.2.2 corrects two lifecycle contracts shipped in 1.2.1. A completed
Experiment or Variant deletion now releases the deleted entity's names for a
fresh identity, and Variant Code promotion now consumes the Source after the
Target commit instead of requiring a second deletion. Immutable CAS, binding,
Run, Model and OptionSet evidence remains preserved. Updating alone does not
delete, rename, promote or migrate research state. This is a narrow,
owner-selected patch exception for the two incorrect 1.2.1 lifecycle meanings;
it is not a general allowance for incompatible patch changes.

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

### Upgrading from 1.2.1

Use `hkdl update` from a clean public `main` checkout. Version 1.2.2 reinstalls
Core without an automatic schema migration. Existing legacy YAML, v2 YAML and
v2 JSON workspaces retain their authored data, recorded history and immutable
Template/Variant source. No new dependency or Template bundle version is
needed. Keep the usual workspace backup; update is not a downgrade or recovery
tool.

The direct-update verification for this candidate targets public 1.2.1. Older
versions should first follow their verified transitions through 1.2.0 and
1.2.1. This does not claim a tested direct jump from 1.2.0, 1.1.5 or 1.0.x to
1.2.2.

### Legacy upgrade: 1.1.5 to 1.2.0

Use `hkdl update` from a public `main` checkout with an `origin` remote and no
tracked local changes. It asks for consent, fast-forwards source and reinstalls
the environment. It does not convert research data. Keep a complete workspace
backup before any separately approved migration, and finish active Runs first.

- Existing YAML workspaces continue using their legacy format and execution
  history after the source update. Existing Variant source and immutable
  Template `1.0.x` bytes are unchanged.
- For v2 management, review `hkdl migrate --all --dry-run -o json` first.
  Only after approving its readiness, space requirements and changes, run
  `hkdl migrate --all` and confirm the cutover. Legacy files remain preserved.
- For JSON authoring, separately review
  `hkdl migrate --authoring --dry-run -o json` in the active v2 workspace.
  Apply only the approved plan with `--authoring --yes` and optionally
  `--expect-plan sha256:<reviewed-digest>`. See [Schema compatibility](#schema-compatibility)
  for tracker conflicts and recovery. Never change either marker by hand.

New empty workspaces using the bundled JSON Templates initialize v2 during the
first `experiment create`; no migration command is needed. While an interrupted
initialization is pending, ordinary operations are blocked: repeat the exact original
`experiment create` command to resume validation/publication. Changed or
unexpected files are preserved and stop recovery instead of being overwritten.
If creation already completed, the normal existing-name error on a repeated
create is expected; inspect or commit that Experiment rather than recreating it.

After a v2 cutover, do not run an older HKDL against that workspace. Preserved
legacy files alone are not a downgrade mechanism; use a separate complete
pre-migration backup if returning to the earlier release is necessary.

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

Before v2 activation, range status queries may maintain
`outputs/.hkdl-index.sqlite3` as a disposable projection of authoritative
authored and generated files. With `.hkdl/store/CURRENT` set to `v2`, the
separate projection is `.hkdl/store/v2/index.sqlite3`, rebuilt from immutable
objects and bindings. Neither database is authority. V2 inspection reads the
graph directly; captured terminal records, metrics, and logs do not require
generated output files. Older Attempts without captured evidence retain a
compatibility reader, and active logs/metrics still use working files. Keep
legacy outputs unless their cleanup is separately approved; activation alone
is not permission to remove them.

```text
hkdl index status
hkdl index rebuild
```

`index status` inspects projection health without writing. `index rebuild`
validates current authority, builds a sibling candidate database, and
atomically replaces only the projection. Neither command migrates or rewrites
authored or generated records.

Inspect and commit one Experiment through its local web view:

```text
hkdl web demo
hkdl web demo --port 8765
```

The foreground server binds only to `127.0.0.1`, prints its local URL, and
stops on interrupt. The selected Experiment is fixed for the server lifetime;
there is no global Experiment picker. Current authored Variants and preserved
generated-history-only Variant identities remain visibly separate, and active
persisted Run states do not claim process liveness. In active v2 workspaces,
the Changes view can commit
validated Experiment or Variant Code revisions; Options remain next-Run inputs.
Open **View Run** to inspect captured Code/Options, one Run's learning curve,
recorded stop reason and recent worker log. These are historical inputs, not
the current draft; refresh remains manual and missing data is shown explicitly.

The web view does not edit JSON, launch, retry, cancel, rename, delete,
authenticate, or poll automatically.

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
checkpoint may be used to continue training. Retry preserves the captured
research inputs; `--tracker none|local|mlflow|local+mlflow` selects tracking for
the new attempt without editing the parent. Without an override, tracking is
resolved from the current workspace setting for JSON authoring or the current
Variant's tracker for legacy YAML authoring.

## Delete a Variant

Preview the complete Variant-owned closure first:

```text
hkdl variant delete demo baseline --dry-run
```

The command removes only the selected Variant's active binding closure and
moves its current authored root plus every output root retained under current
or historical Experiment and Variant names to recoverable transaction trash.
Its parent Experiment, sibling Variants, immutable CAS and derivation history,
shared environments, and external MLflow state remain. Every owned Run must be
terminal and lease-free. After deletion completes, its current and historical
names may be claimed by a new Variant entity while old evidence remains
addressable by immutable hashes.

When active child Variants exist, dry-run returns exit 5 and lists every direct
and transitive descendant. A normal invocation strongly recommends deleting or
reorganizing those children first. HKDL can continue their active lineage around
the deleted Variant without rewriting immutable derivation history, but doing so
requires a dedicated `[y/N]` approval before the separate final deletion
confirmation. Declining the lineage step returns exit 5 without asking the final
question; declining only the final deletion question is a normal cancellation.
There is no `--yes` bypass. Any changed child set, binding HEAD, owned closure or
lease after confirmation rejects the operation without silent expansion.

## Promote Variant Code

Promote committed Code from a validated Source Variant into an unchanged Target
Variant while keeping the Target identity and Options:

```text
hkdl variant promote demo tuned --to baseline --dry-run
hkdl variant promote demo tuned --to baseline
```

Promotion copies the Source's committed `code.json` meaning and complete `src/`
tree into a new Target revision. The Target `options.json` remains unchanged;
the Source Variant and its active owned closure are deleted after the Target
revision commits, while immutable Source provenance remains addressable by
hash. Source must derive from Target and Target Code must not have changed from
that integration point. Dirty drafts, unrelated lineage, Target divergence,
active Target execution, nonterminal or leased Source Runs, and active Source
descendants return exit 5 before prompting.

Dry-run is read-only. A ready apply shows the exact Code delta and asks one
`[y/N]` question that explicitly includes Source deletion; there is no `--yes`
bypass. Promotion and Source deletion are separately journaled around their
binding transactions, so pre-Target-HEAD failure restores the old Target and a
post-HEAD retry completes the already-confirmed Target publication and Source
deletion. The deleted Source names become reusable. Full three-way conflict
resolution, Options merge, cross-Experiment promotion and Experiment merge are
not supported.

## Delete an Experiment

Preview the complete deletion closure first:

```text
hkdl experiment delete demo --dry-run
```

Run the command without `--dry-run` to see the same plan and answer the final
`[y/N]` question. Experiment deletion is intentionally all-or-nothing and has
no `--yes` bypass. Any nonterminal Run or held Run lease blocks the operation;
HKDL reports every blocker with its persisted status, lease state, reason, and
next action before returning exit 5 without prompting.

On confirmation, HKDL revalidates the exact plan, moves the current authored
Experiment root and all output roots retained under its current or historical
names to recoverable transaction trash, and removes the selected Experiment's
active bindings in one transaction. Other Experiments, immutable CAS objects
and blobs, binding history, shared environments, and external MLflow state are
preserved. After deletion completes, the deleted Experiment name may be claimed
by a new entity without rewriting prior binding transactions.
If a confirmed deletion is interrupted, repeat the same non-dry-run command;
HKDL rolls back unpublished work or completes an already-published deletion
before starting any new plan.

## Schema compatibility

New empty workspaces using the bundled schema-2 Templates create
`experiment.json`, `code.json`, and `options.json`. The JSON files contain
research inputs; names, hashes, revision history, and tracker settings are
managed separately. Existing YAML workspaces keep schema-1 `experiment.yaml`
and `variant.yaml` until an explicit authoring migration.

There are three separate migration operations:

| Operation | Scope |
| --- | --- |
| `hkdl migrate <path>` | Validate one legacy authored YAML file. Schema 1 is current for this single-file boundary; no rewrite is registered. |
| `hkdl migrate --all --dry-run` | Preview importing the complete legacy workspace into v2 object authority. Apply uses `--all` with confirmation and preserves existing authored/output bytes. |
| `hkdl migrate --authoring --dry-run` | Preview YAML-to-JSON draft conversion and historical Run/Model graph replay in an already active v2 workspace, including JSON workspaces with older graph evidence. Apply requires explicit `--authoring --yes`. |

Authoring migration preserves uncommitted research edits and existing output
bytes. It reports graph remapping, tracker/lease conflicts, space estimates,
and a plan digest. If legacy tracker defaults disagree, explicitly select the
future default with `--tracker-default local` (also `none`, `mlflow`, or
`local,mlflow`). This preserves historical tracker evidence; repeat the same
selection on apply. `--expect-plan sha256:<64hex>` on apply binds the operation
to the reviewed digest. Apply takes a short exclusive maintenance window;
ordinary commands return exit 5 and the web returns 503 until it completes or
an interrupted cutover is explicitly recovered.

For single-file validation:

```text
hkdl migrate experiments/demo/experiment.yaml
hkdl migrate experiments/demo/baseline/variant.yaml
```

`CURRENT` selects v2 storage authority; the separate
`.hkdl/store/AUTHORING_CURRENT` marker selects JSON authoring. A workspace can
have v2 storage while still authoring YAML. Do not edit either marker manually
or infer that output files are disposable from its presence. Review migration
readiness, blockers, and the reported changes before applying. Unsupported
single-file schema versions fail without changing the target; generated Run
and Model records cannot be migrated in place by `migrate <path>`.

## ResNet18 fixtures

`resnet18@1.1.0` supplies schema-2 JSON authoring and includes the attributed
small TF-Flowers JPEG fixture, two evaluation cases (`default` and `daisy-only`),
prediction result output, ONNX
export support, checkpoint continuation, and optional MLflow dependencies.
It records loss and per-batch wall time and performs no dataset or
pretrained-weight download at runtime. Immutable `1.0.0` and `1.0.1` remain
available for legacy authoring.

## YOLO26n object detection

`yolo26n@1.1.0` supplies schema-2 JSON authoring and includes a deterministic
synthetic shapes dataset with eight training images, four validation images,
and two classes (`circle` and
`rectangle`). It trains the architecture from scratch for ten epochs and
records loss and per-epoch wall time. It performs no dataset or
pretrained-weight download at runtime. Immutable `1.0.0` and `1.0.1` remain
available for legacy authoring.

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

For JSON authoring, the workspace default is stored in `.hkdl/settings.json`;
an absent settings file means `local`. Inspect or change it with:

```text
hkdl settings show
hkdl settings tracker set local
```

`run train` and `run retry` accept `--tracker` to override that default for one
invocation. Eval and Export use the workspace default. Legacy YAML Variants
instead keep `tracker.backend` in `variant.yaml`, with bundled `1.0.1` Variants
defaulting to `local`. Tracker settings do not belong in schema-2 Code/Options.
Training scalars, including `train.batch_seconds` for ResNet18 and
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

The settings command and `--tracker` accept `none`, `local`, `mlflow`, or
`local+mlflow`. In legacy YAML use `tracker.backend: none` to disable tracking,
`mlflow` for MLflow only, or `[local, mlflow]` for both. MLflow requires an
external `MLFLOW_TRACKING_URI`.
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
