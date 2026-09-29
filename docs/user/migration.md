# Research migration and recovery

Migration changes research storage or authoring authority. It is separate from a program [update](updating.md) and must not be used as automatic error repair. Use the documentation for the HKDL version executing the migration.

For a public 1.2.2 source checkout moving to 2.0.0, start with the [release-specific transition guide](migrating-to-2.0.md). Use the corrected new executable for migration; do not first convert with the 1.2.2 migrator.

## 1. Identify the requested operation

From the research root, inspect `hkdl --version` and `hkdl migrate --help`.
HKDL 2.0 requires graph storage and JSON
authoring for ordinary operations; `hkdl status --output json` rejects older
workspaces with migration guidance. Use the explicit migration preview to
inspect those workspaces. If pending recovery is reported, follow the recovery
section below instead of attempting another migration. New workspaces initialize
directly to graph storage and JSON during first Experiment creation.

Choose exactly one mode based on the workspace state and the requested transition:

| Mode | Purpose |
| --- | --- |
| `hkdl migrate PATH` | Validate one repository-owned legacy `experiment.yaml` or `variant.yaml`; not a workspace cutover. |
| `hkdl migrate --all` | Import legacy storage into content-addressed v2 storage while preserving authored files and outputs. |
| `hkdl migrate --authoring` | Convert authored YAML to JSON and normalize historical graph identity in an already active v2 workspace. Active v2 storage alone does not imply JSON authoring. |

Do not run both workspace migrations merely because they exist. If both transitions are requested, complete and verify storage import before separately previewing authoring conversion.

## 2. Prepare

1. Confirm the exact research root and requested migration with the owner.
2. Finish active Runs and stop worker/web activity holding workspace leases. Never bypass a lease or relabel a stale Run by editing its files.
3. Back up the complete research workspace while it is quiescent, including hidden `.hkdl` state, authored files, outputs, and custom Templates. Verify that the backup is readable before proceeding.
4. Preserve uncommitted edits and historical tracker evidence. Do not manually change `CURRENT`, `AUTHORING_CURRENT`, binding HEADs, settings, or recovery journals to make a migration pass.

## 3. Preview and approve

For storage import:

```sh
hkdl migrate --all --dry-run --output json
```

For authoring conversion:

```sh
hkdl migrate --authoring --dry-run --output json
```

Review readiness, malformed records, leases, stale Runs, required space, proposed file/graph changes, and the plan digest. A blocked plan is not approval to discard conflicting data. Resolve each blocker and preview again.

If authoring reports conflicting tracker defaults, have the owner choose `none`, `local`, `mlflow`, or `local,mlflow`. Repeat the preview with `--tracker-default SELECTOR`. That selection affects future tracking; historical tracker identities remain preserved. Invalid settings and leases cannot be overridden. Keep credentials and tracking URIs out of research files.

Obtain approval for the concrete plan. Retain its output and, for authoring, its `sha256:` plan digest. Do not edit the workspace between preview and apply.

## 4. Apply the approved operation

For an approved storage import:

```sh
hkdl migrate --all --yes
```

For approved authoring conversion, replace the placeholder below with the exact reviewed digest:

```sh
hkdl migrate --authoring --yes --expect-plan sha256:<64hex>
```

Repeat the same `--tracker-default SELECTOR` if it was used in the preview. `--expect-plan` applies only to authoring conversion; a changed plan is rejected. Storage import rechecks authority and leases during apply, but does not accept an approved-plan digest from the earlier CLI preview. If the workspace changed, preview and obtain approval again before applying.

## 5. Verify

1. Check the command's exit status and completion report; do not infer success from newly created files.
2. After authoring conversion, repeat `hkdl migrate --authoring --dry-run --output json` and confirm `already_current` without pending changes or blockers. Storage import does not offer this idempotent preview: after cutover, `--all` reports `HKDL v2 is already active` as a command error. Do not treat that error as a request to repair or repeat import. Storage import alone leaves YAML authoring in place; ordinary operations remain unavailable until the separately approved authoring conversion completes.
3. Run `hkdl status --output json` and inspect the expected Experiments, Variants, Runs and Models. Confirm preserved uncommitted edits and outputs against the backup.
4. Keep the backup and previous installation. Retained legacy files alone are not a complete downgrade backup; do not run an older HKDL against cutover or reused-name state.

## Recovery after interruption

1. Preserve the error, original command, approved preview and workspace bytes. Do not launch research work while migration recovery is pending.
2. For interrupted storage import, repeat the original `hkdl migrate --all --yes` only when inputs and the approved scope remain unchanged. Matching partial objects/bindings can be resumed; do not edit them manually.
3. For interrupted authoring conversion, repeat the exact original apply command, including `--expect-plan` and any tracker selection. Recovery restores the pre-cutover state or completes the recorded transition according to its commit point.
4. Unexpected file edits, changed plans or unrelated graph movement are conflicts. Stop and report them; never delete journals or force markers. If recovery reports additional changes requiring a new plan, run a new preview and obtain new approval.
5. After successful recovery, repeat the verification steps above. If Core itself cannot start, first use the independent installer [repair procedure](updating.md#recover-a-managed-installation); program repair does not repair research authority.
