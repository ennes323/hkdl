# Updating HKDL

Use the update procedure for your installation type. Updating HKDL changes the program and its execution environment; it does not automatically migrate research data or replace existing Variant code.

For the direct public 1.2.2 → 2.0.0 transition, follow [Moving from 1.2.2 to 2.0.0](migrating-to-2.0.md) instead of the ordinary update commands below. That guide fixes the source-update and migration order and identifies the exact release target.

## Before updating

1. From the research root, inspect `hkdl --version` and `hkdl update --help`. Use this document for that version. If Core cannot start, use the independent installer recovery procedure below.
2. Identify whether you use a managed installation or a public source checkout. Select the intended release and read its release notes, including required intermediate versions.
3. Preserve local changes and maintain a readable backup of the complete research workspace, including hidden state. An update is not a backup or research downgrade mechanism.
4. Finish affected Runs and stop worker/web activity holding installation or workspace leases. Do not bypass lease or compatibility blockers.

## Managed installation

1. Select the supplied release bundle explicitly. For a local archive, run:

   ```sh
   hkdl update BUNDLE.zip
   ```

   Replace `BUNDLE.zip` with the selected release bundle. HTTPS URLs require the release's explicit `--sha256` value; consult `hkdl update --help` for syntax.
2. Review the displayed installed/target versions, compatibility of every registered workspace, and guidance actions. The command asks for confirmation before preparing or activating the candidate. Decline if the plan is not approved; use `--yes` only after approval of the shown update.
3. If the direct transition is unsupported, follow the reported intermediate path one release at a time, reviewing each update. HKDL does not execute that path automatically.
4. After confirmation, HKDL prepares and verifies a separate installation before switching the active version. If preparation fails, the existing active installation stays selected. Old installations are retained.
5. Check the exit status and reported active version, then run `hkdl --version` and `hkdl status --output json` in the research workspace. Confirm expected research identities and review any guidance warnings below. Do not migrate or delete old installations as an automatic follow-up.

## Source checkout

1. Run from the public HKDL checkout. It must be on public `main`, have an `origin` remote, and contain no tracked local changes. Review and preserve changes if the update is blocked; do not discard them to satisfy the check.
2. Run `hkdl update` and review the installed/available versions and proposed changes before confirming.
3. After approval, the command fast-forwards the public source and reinstalls the environment.
4. Check the command result, `hkdl --version`, and `hkdl status --output json`. If environment installation fails after the source advances, preserve the error, inspect the checkout state, and rerun that checkout's `./setup.sh`; do not assume the source was rolled back or reset research files.

### Older source updater compatibility

Older public releases require source-root agent files when validating an update
target. Those files are no longer distributed. Such an updater can reject this
release as `origin/main is not a public HKDL release` before changing source.
Do not create dummy agent files to bypass that check.

For this specific transition, use the following manual source-update procedure:

1. Verify you are at the public checkout root on `main` with the expected `origin`.
   Run `git status --short` and preserve local changes and untracked-file conflicts
   before proceeding. Back up research as described above.
2. Run `git fetch origin main`, then inspect `git log --oneline HEAD..FETCH_HEAD`
   and `git diff --stat HEAD..FETCH_HEAD`. Verify the selected release's version,
   release notes and compatibility requirements; do not cross an unapproved major
   version or intermediate transition.
3. Obtain approval for that exact fetched target, then run
   `git merge --ff-only FETCH_HEAD`. If it refuses, stop; do not reset, stash or
   merge divergent histories to force the update.
4. Run `./setup.sh` and verify the installed version. Complete any release-required
   [migration](migration.md) before checking research status; older workspaces are
   rejected by ordinary operations until conversion finishes. Once updated, the
   current updater no longer requires source-root agent files.

This is an explicit source-update transition, not a managed-installation repair
or permission to migrate research. Published predecessor-to-candidate evidence
must still be checked during release preparation.

## Workspace agent guidance

The managed update preview lists the guidance action for each registered workspace. After successful activation, HKDL refreshes only `AGENTS.md` files whose bytes still match their last HKDL ownership receipt. It preserves user-owned, edited, missing, or unsafe paths and prints a link to the selected version's guide.

A guidance change after the preview is preserved and reported. A guidance write failure after activation does not roll back or misreport the active program version; HKDL warns that the guide is incomplete. Rerun `hkdl workspace init PATH` to review and recover guidance. An interrupted receipt write leaves the file unowned, requiring explicit adoption before future automatic replacement.

During initialization, an existing user or modified `AGENTS.md` offers three choices: keep (also the default on EOF), backup-replace, or preserve as `AGENTS.user.md` and install the default. Existing `AGENTS.user.md` files are never overwritten. Replacements retain the old bytes in the reported backup or user file.

## What an update preserves

Both installation methods preserve:

- Authored Experiment and Variant files.
- Existing Runs, Models, outputs, and immutable history.
- Source captured by previous executions.
- Existing Variant copies of Template code.

Source-checkout updates also preserve the user-owned `AGENTS.user.md`.

A new Template version affects new authoring when selected. It does not patch existing Variants or change the source used by a Retry of an older Run.

## Compatibility

Supported updates follow verified release transitions. A shared major version does not guarantee that every older version can update directly to the latest release.

Follow any required intermediate release or migration instructions. Release-specific requirements belong in the [release notes](https://github.com/hukuhaka/hkdl/releases); the commands on this page remain the standard update entrypoints.

Updates do not provide research-data downgrade or backup restoration.

## When migration is required

Program updates and research-data migrations are separate operations. Existing YAML authoring and storage formats are not automatically converted by an update.

Follow [Research migration and recovery](migration.md) for mode selection, backup, preview, approval, application, verification, and interrupted-cutover recovery. Do not edit storage or authoring authority markers manually.

## Recover a managed installation

1. Use the supplied standalone installer, which can show help and installation status without importing Core:

   ```sh
   uv run --no-project --python 3.12 INSTALLER.pyz --help
   uv run --no-project --python 3.12 INSTALLER.pyz status
   ```

   Replace `INSTALLER.pyz` with the supplied artifact. For a custom installation root, pass the original `--root PATH` before `status` or `install`. The installer's help links to the HKDL release with which it was built, not its separate installer protocol version.
2. Record the selected installation version and bundle identity from status. Select the same release bundle for repair; a different bundle is an update requiring its own review. If metadata is damaged and status fails, stop and retain the error; do not reconstruct receipts or the `current` pointer by hand.
3. Read `INSTALLER.pyz install --help` using the same Python invocation. Preview and approve repair by running:

   ```sh
   uv run --no-project --python 3.12 INSTALLER.pyz install BUNDLE.zip --repair
   ```

4. Repair prepares a replacement environment and retains the previous installation. It requires working Python, uv, and intact installation-manager metadata. If preparation fails, inspect status before retrying the same bundle.
5. Verify installer status, `hkdl --version`, and research `hkdl status --output json` after success. Review any guidance warnings. Repair does not restore research from backup or resolve migration journals.
