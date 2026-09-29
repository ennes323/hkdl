# Moving from HKDL 1.2.2 to 2.0.0

This is the direct transition from the public 1.2.2 source release
(`82fa70dadd8451d85e10e2240cdbc0bda85a9dde`) to 2.0.0. It is an explicit
release-specific exception to the usual advance deprecation notice: ordinary
operations in 2.0 require graph storage and JSON authoring. Older formats remain
available to the explicit migration commands, not ordinary research operations.
Existing Variant code and captured execution history are preserved; this is not
an instruction to rewrite Template or framework APIs.

Use this procedure only with the exact `v2.0.0` tag and matching artifacts from
the [GitHub Release](https://github.com/hukuhaka/hkdl/releases/tag/v2.0.0).
Confirm that the release and its transition verification are available before
starting. Do not substitute a moving `main`, an arbitrary development build,
or an unpublished bundle for the selected release.

The order is **back up → update source → use the new CLI to migrate → verify**.
Continuing with a source checkout is supported. Managed installation is optional
and comes after migration; public 1.2.2 did not ship a managed installation.

## 1. Preserve the original workspace

Follow the [migration preparation steps](migration.md#2-prepare). Finish Runs and
stop worker/web activity, then make and check a complete, quiescent backup outside
the checkout. Include hidden files, `.git`, `.hkdl`, authored files, outputs,
custom Templates and local changes. A Git commit or an `experiments/` copy alone
is not a complete backup. Retain the old version, command outputs and backup
until the new environment and research have been verified.

At the public source checkout root, inspect:

```sh
.venv/bin/hkdl --version
git branch --show-current
git remote -v
git status --short
```

Confirm version 1.2.2, branch `main`, and the expected public HKDL `origin`.
Preserve tracked local changes and resolve possible untracked-file collisions
before updating; do not reset or delete them to make the checkout pass. Local
Core or Template changes may require a separately reviewed integration. If the
checkout is older, divergent or otherwise different from this starting point,
stop and establish its supported route first.

## 2. Update source and its environment

Do not run 1.2.2's `hkdl migrate` as preparation. Its storage migrator can accept
missing completed-Train Models, select the wrong current Experiment after an
A→B→A edit history, or miss newly malformed paths during apply. The 2.0 migrator
contains the fixes.

Do not use 1.2.2's `hkdl update` for this transition: its public-source validation
requires root agent files that are no longer distributed, and its version check
also rejects major transitions. Do not manufacture those files or bypass checks.
Fetch the exact release target instead:

```sh
git fetch origin tag v2.0.0
git rev-parse 'v2.0.0^{commit}'
git log --oneline HEAD..v2.0.0
git diff --stat HEAD..v2.0.0
git show v2.0.0:pyproject.toml
```

Compare the resolved commit and version with the published release information,
read its release notes, and approve that exact target. Then:

```sh
git merge --ff-only v2.0.0
./setup.sh
.venv/bin/hkdl --version
.venv/bin/hkdl migrate --help
```

Stop if the fast-forward is refused. Do not force, reset or merge divergent
history to complete this procedure. If setup fails after source advances,
preserve the error and rerun the selected checkout's `./setup.sh` after resolving
the environment issue; the source has not automatically rolled back.

Confirm that `.venv/bin/hkdl --version` reports **2.0.0**. Use this explicit
executable from the research root for every migration and verification below.
An activated shell or bare `hkdl` could still select a different installation.
Do not run ordinary research commands, including `status`, before completing
required migrations.

## 3. Preview and apply only the required transitions

First preview authoring with the new CLI:

```sh
.venv/bin/hkdl migrate --authoring --dry-run --output json
```

A preview inspects the workspace without converting it. Choose the route from
its report, not just from the presence of YAML or JSON files:

| Existing workspace | Required action |
| --- | --- |
| Legacy storage and YAML; report says v2 must be active | Preview and apply storage import below, then obtain a fresh authoring preview and apply it. |
| Active graph storage and YAML | Review and apply authoring conversion. Do not repeat storage import. |
| Active graph storage and JSON | Inspect the authoring preview. If `already_current` is true, no apply is needed. Otherwise review the proposed historical graph normalization before applying. |
| Pending recovery, malformed records, conflicts or active leases | Stop this sequence and follow [recovery](migration.md#recovery-after-interruption) or resolve the reported blocker. Do not treat every blocked authoring preview as a request for storage import. |

For legacy storage only, preview:

```sh
.venv/bin/hkdl migrate --all --dry-run --output json
```

Review readiness and every reported issue under the [common preview procedure](migration.md#3-preview-and-approve).
After approval of that concrete storage plan, with inputs unchanged:

```sh
.venv/bin/hkdl migrate --all --yes
```

Check the storage command's successful completion before continuing. Storage
import preserves authored YAML, so this intermediate workspace still cannot run
ordinary 2.0 operations. Do not repeat `--all` after success: it reports
`HKDL v2 is already active` as an error, not a request for repair. Now obtain a
fresh authoring preview:

```sh
.venv/bin/hkdl migrate --authoring --dry-run --output json
```

For authoring conversion or historical graph normalization, retain the newly
reviewed preview and its `sha256:` plan digest. After approval, replace the
placeholder with the exact digest:

```sh
.venv/bin/hkdl migrate --authoring --yes --expect-plan 'sha256:<64hex>'
```

If tracker defaults require a choice, use the same `--tracker-default SELECTOR`
on the preview and apply, as described in the common procedure. Do not guess or
remove historical tracker records. A changed plan requires a fresh preview and
approval. Storage import does not support `--expect-plan`; its apply rechecks
inputs and leases, and changes between preview and apply require a new review.

## 4. Verify before resuming work

Check every apply's exit status and completion report, then run:

```sh
.venv/bin/hkdl migrate --authoring --dry-run --output json
.venv/bin/hkdl status --output json
```

Confirm `already_current` with no pending changes or blockers, and inspect the
expected Experiments, Variants, Runs and Models. Compare authored meaning and
uncommitted edits, historical Model/Run relationships, results, captured sources
and output files with the backup. YAML-to-JSON conversion intentionally changes
file names and representation; those authored files need a semantic comparison.
Historical outputs and existing immutable graph objects must retain their bytes.

Keep the backup. Resume research only after these checks succeed. You may keep
using this source checkout; no managed registration is needed for that route.

## 5. Optional: adopt into a fresh managed 2.0.0 installation

Do this only after section 4 succeeds. Obtain the **installer, bundle and checksum
file supplied together by the same 2.0.0 release**. Verify both artifacts against
that checksum file before executing the installer. For example, from their
download directory, with the actual checksum filename substituted:

```sh
shasum -a 256 -c CHECKSUMS.sha256
```

Choose a fresh installation root and a launcher directory outside the research
checkout, without replacing another HKDL installation or launcher. Substitute
absolute paths and the published bundle SHA-256 in the following commands:

```sh
uv run --no-project --python 3.12 /path/to/INSTALLER.pyz \
  --root /path/to/hkdl-2-install install /path/to/BUNDLE.zip \
  --sha256 PUBLISHED_BUNDLE_SHA256 --bin-dir /path/to/hkdl-2-bin
uv run --no-project --python 3.12 /path/to/INSTALLER.pyz \
  --root /path/to/hkdl-2-install status
/path/to/hkdl-2-bin/hkdl --version
/path/to/hkdl-2-bin/hkdl workspace init /absolute/path/to/source-checkout
```

Review the installer's preview before confirming activation. Confirm that its
status and the explicit launcher both select 2.0.0 before `workspace init`.
Adoption checks the now-2.0.0 source version and migrated research, then registers
the same research directory. It does not move the research, delete the source
checkout, or perform a data migration. Review any agent-guidance choice; existing
user or edited guidance is preserved unless you explicitly choose replacement.

Source-only or locally modified Templates block adoption. Keep using the source
checkout until those Templates are explicitly reconciled; do not delete them or
edit receipts to force registration. Existing Variant copies are not rewritten.

From the same research root, verify with the explicit managed launcher:

```sh
/path/to/hkdl-2-bin/hkdl status --output json
```

Compare with section 4, then use that launcher consistently. A pre-existing
managed development installation is outside this public 1.2.2 transition; its
own compatibility checks still apply. This route does not relax the installer's
general cross-major update rejection.

## If the transition stops

Preserve errors, previews and current workspace bytes. Follow [migration recovery](migration.md#recovery-after-interruption)
using the same `.venv/bin/hkdl` 2.0.0 executable and original approved command,
including the authoring digest and tracker selection. Recovery may restore the
pre-cutover state or finish a committed transition. Do not remove journals,
partial immutable objects or authority markers, and do not start research while
recovery remains pending.

For a managed-program failure after adoption, inspect and use the [installer
repair procedure](updating.md#recover-a-managed-installation) with the same
installation root and bundle. Program repair does not restore research data.

Returning to 1.2.2 means restoring the **complete quiescent pre-transition backup
to a separate location** and using the matching old program/environment there.
Preserve the failed or migrated workspace for diagnosis. Never run 1.2.2 against
the migrated workspace, copy only a few legacy files back, or flip authority
markers to simulate rollback. Work created after the backup is not included in
that restoration.
