# HKDL User Guide for Coding Agents

## Default role

This is a runnable HKDL source checkout.

Unless the user explicitly asks to develop HKDL itself, help them operate
experiments through the HKDL CLI. Do not modify HKDL Core or the bundled
Template catalog as part of normal experiment work.

<!-- hkdl:user-guidance-contract version="1" -->

## Guidance ownership

This projected `AGENTS.md` is managed by HKDL. Do not edit it for local
preferences. If `AGENTS.user.md` exists at the repository root, read it after
this document before starting work. That optional file is user-owned: HKDL
does not distribute, replace, migrate, or delete it, and source updates must
preserve it.

User guidance may refine local execution policy, resource limits, and reporting
preferences. It does not waive HKDL's authoritative-data, immutability, path,
validation, or confirmation rules in this document.

## Start

- Work from the repository root.
- If `.venv/bin/hkdl` is unavailable, run `./setup.sh`.
- Invoke the CLI as `.venv/bin/hkdl`; coding agents do not need shell
  activation. For an interactive Bash or Zsh session, `source ./activate.sh`
  activates the root environment and registers contextual completion for that
  session without modifying shell startup files.
- Inspect existing state before creating anything:

  ```text
  .venv/bin/hkdl --version
  .venv/bin/hkdl template list
  .venv/bin/hkdl experiment list
  .venv/bin/hkdl status
  ```

- Use `<command> --help` when the requested operation is not covered below.
  Do not invent flags, addresses, or filesystem layouts.

## Update HKDL

Run `.venv/bin/hkdl update` only when the user explicitly asks to update this
public source checkout. Report the current and available versions and let the
command request confirmation. Use `--yes` only after the user has already
approved the shown update.

Do not reset, stash, discard, or merge local changes to make an update pass.
If the command reports a dirty checkout, branch mismatch, divergence, or major
version transition, stop and report it. If source update succeeds but setup
fails, preserve the updated checkout and follow the reported `./setup.sh`
recovery instruction.

Version 1.2.0 keeps existing YAML workspaces on their legacy authority after a
source update. Do not implicitly run either workspace migration. V2 management
and JSON authoring are separate opt-in cutovers; review each dry-run and obtain
approval. After cutover, do not use an older HKDL on the same workspace or treat
retained legacy files as a complete downgrade backup.

## Normal workflow

Create an Experiment and Variant:

```text
.venv/bin/hkdl experiment create <experiment> -t <template>
.venv/bin/hkdl variant create <experiment> <variant>
```

The latest numeric Template version is used unless the user requests an exact
version. New empty workspaces use schema-2 `experiment.json`, `code.json`, and
`options.json` with the current bundled Templates. Existing YAML workspaces
retain their format until an explicit authoring migration.

In a truly empty workspace, the first JSON Experiment also initializes its v2
identity and committed revision. An interrupted initialization blocks ordinary
commands. Repeat the exact original `experiment create` invocation to resume;
do not delete the bootstrap journal or edit markers to bypass recovery. If
drafts, markers or bindings changed, stop and preserve the reported state.

After changing authored Variant configuration, validate it:

```text
.venv/bin/hkdl variant check <experiment> <variant>
```

Train one or more seeds:

```text
.venv/bin/hkdl run train <experiment> <variant> <group> -s <seeds>
```

If the user does not specify execution options, retain the CLI defaults:
seed `0` and device `auto`.

Inspect the resulting state and Models:

```text
.venv/bin/hkdl status <experiment> <variant>
.venv/bin/hkdl status <experiment> <variant> --full
.venv/bin/hkdl status <experiment> --table
.venv/bin/hkdl run metrics <experiment> <variant> <run-id>
.venv/bin/hkdl run logs <experiment> <variant> <run-id>
.venv/bin/hkdl model list <experiment> <variant>
.venv/bin/hkdl storage
.venv/bin/hkdl environment prune --dry-run
```

Before v2 activation, `status` may maintain the disposable SQLite projection at
`outputs/.hkdl-index.sqlite3` from authoritative authored and generated files.
With `.hkdl/store/CURRENT` set to `v2`, its separate projection is
`.hkdl/store/v2/index.sqlite3`, derived from immutable objects and bindings.
Captured Run inspection reads the graph and verified blobs without requiring
generated output files or a working SQLite index. Older Attempts without
captured evidence use an isolated compatibility reader; active logs/metrics
still use working files. Preserve legacy outputs until cleanup is explicitly
approved. Never work around a missing, stale, incompatible, corrupt, or
unavailable projection by editing generated records or object-store contents.

Inspect or explicitly rebuild the projection with:

```text
.venv/bin/hkdl index status
.venv/bin/hkdl index rebuild
```

`index status` is read-only. `index rebuild` validates the active authority,
builds a sibling candidate, and atomically replaces only the disposable
projection. It does not migrate or rewrite authored or generated files.

`status` uses the compact brief view by default. Use `--full` for timestamps,
configured Train dimensions, metric summaries, checkpoints, trackers, and Eval
details. Use `--table` to compare aggregate Eval results across Variants and
Training Groups. The table is text-only and cannot be combined with `--full`.
For automation, use `--output json`; JSON always returns the full structure,
with or without `--full`.

JSON authoring uses the workspace tracker setting, which defaults to `local`
when `.hkdl/settings.json` is absent. Inspect it with `hkdl settings show`;
`hkdl settings tracker set <selector>` changes the workspace default.
`run train` and `run retry` accept a per-invocation `--tracker` selector:
`none`, `local`, `mlflow`, or `local+mlflow`. Legacy YAML Variants retain their
authored `tracker.backend`; bundled `1.0.1` Variants default to `local`.
ResNet reports `train.batch_seconds` per optimizer step and
YOLO reports `train.epoch_seconds` per completed epoch. Do not enable MLflow
unless the user requests it and provides an external tracking service.

To watch new local Train metrics in an attached terminal, use:

```text
.venv/bin/hkdl run metrics <experiment> <variant> <run-id> --follow
```

`--follow` is a text-only live view for locally tracked Train Runs. Use the
regular command after completion when the full persisted metric history is
needed.

`run logs` returns the raw merged stdout and stderr captured from the action
worker, including ordinary child processes that inherit those descriptors. It
does not capture setup, preflight, direct terminal writes, or detached daemons.
The log has no redaction or size limit: never print credentials, tokens, or
other secrets from Variant code.

`storage` reports logical bytes for authored Experiment content, legacy and
shared Variant environments, generated outputs, and their total. It is
read-only: do not describe it as cleanup, pruning, or available disk space.

Variant actions reuse repository-local immutable environments when their locked
inputs and runtime identity match. Use `environment prune --dry-run` to inspect
cleanup. The default confirmed prune removes legacy Variant environments,
incomplete cache entries, and unreferenced shared environments. `--all` also
selects referenced but inactive shared environments. Active environments are
always skipped. Do not use `--yes` without prior confirmation from the user.

Evaluate an existing Evaluation Case:

```text
.venv/bin/hkdl run eval <experiment> <variant> <group> <case> -s <seed|all>
```

Export one exact Model ID:

```text
.venv/bin/hkdl run export <experiment> <variant> <model-id>
```

Retry a failed, interrupted, or abandoned Run as a new Run:

```text
.venv/bin/hkdl run retry <experiment> <variant> <run-id>
```

Retry preserves captured Code/Options and the parent remains sealed. Tracking
for the child uses the current workspace or legacy Variant setting unless
`--tracker` is supplied; it need not match the parent's tracker.

## Migration boundaries

`hkdl migrate <path>` only validates a legacy authored `experiment.yaml` or
`variant.yaml`; schema 1 is current at that single-file boundary.
`hkdl migrate --all --dry-run` previews the full legacy workspace import into
v2 authority without rewriting existing authored or output files on apply.
`hkdl migrate --authoring --dry-run` previews separate YAML-to-JSON draft
conversion and historical Run/Model graph replay in an already active v2
store, including normalization of older graph evidence in JSON workspaces.
Uncommitted research edits remain uncommitted and outputs remain unchanged.
For conflicting legacy tracker defaults, an explicit
`--tracker-default none|local|mlflow|local,mlflow` selects only the future
workspace default. Preserve historical tracker evidence and repeat the choice
on approved apply; it is included in the plan digest. Omission retains conflict
blocking, and malformed settings or held leases cannot be overridden.

The storage marker `CURRENT` and authoring marker `AUTHORING_CURRENT` under
`.hkdl/store/` have separate roles: active v2 storage does not imply JSON
authoring or independence from output files. Never change these markers by
hand. Apply a migration only when the user requests it, after reviewing the
dry-run's readiness and changes; `--authoring` requires `--dry-run` or `--yes`.
Use `--expect-plan sha256:<64hex>` with apply when approval names a specific
digest. Apply uses a journaled maintenance window; normal commands and web
reads are blocked until completion or explicit recovery. Never treat a new
post-recovery plan as implicitly approved.

## Long-running operations

When an HKDL command may outlast the current agent turn:

- Use `run metrics ... --follow` only for attached live observation; it does
  not replace durable completion signaling.
- Run it in a host-managed background terminal or scheduler. Do not assume
  `nohup` survives the host's command boundary.
- Store logs and an atomically published exit marker under
  `.hkdl/monitors/<job-id>/`, never under `outputs/`.
- Treat that monitor log as the whole-command lifecycle record. Run-owned
  `worker.log` covers only action-worker output and does not replace the monitor
  exit marker.
- Return the terminal session or job ID, log path, and checkpoint path.
- When the host supports scheduled follow-ups, attach a same-thread heartbeat
  that observes the exit marker and verifies final Run state with
  `.venv/bin/hkdl status ... --output json`.
- Treat `done`, `failed`, `interrupted`, and `abandoned` as terminal. If the
  process ended while its Run remains active, report the mismatch. Never edit
  generated state or retry automatically.
- Stop the heartbeat after its first terminal report.

## Ownership and safety

- `experiments/` contains authored Experiment and Variant inputs.
- Modify authored JSON, legacy YAML, or Variant source only when required by
  the user's experiment request.
- Preserve `schema_version` and Template provenance. Do not insert names,
  hashes, IDs, revisions, timestamps, or tracker settings into research JSON;
  preserve embedded identity fields in legacy YAML.
- Run `variant check` after editing a Variant.
- In active v2 workspaces, commit the Experiment question separately from
  Variant Code. Code/source changes require `variant commit` and are blocked
  while active Run/Model bindings exist; Options are validated and captured
  automatically by the next Run. The web Changes view uses these same rules
  and is unavailable before v2 activation.
- `outputs/` contains generated Run/Model projections; `.hkdl/store/` owns v2
  objects and bindings after activation. Never edit, move, normalize, or delete
  these directly. Explicitly requested rename or deletion must use the CLI
  after reviewing its dry-run. Deletion requires `--yes`; rename applies
  without a further prompt when `--dry-run` is omitted.
- Never work around a contract error by modifying generated files.
- Use `run retry` for a stopped Run; do not reuse or overwrite its Run ID.
- Keep single-file validation, full import, and authoring conversion separate
  as described above; do not apply a migration as an automatic error repair.
- Keep MLflow credentials and tracking URIs outside authored and generated
  files.
- Ask before deleting authored work, starting an unexpectedly expensive run,
  or enabling an external tracking service.
- After an operation, report the created Run or Model IDs and the resulting
  status.
