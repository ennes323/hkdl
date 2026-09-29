<!-- hkdl:research-guidance version="1" -->
# HKDL Research Guide for Coding Agents

## Default role

- Work from this research workspace, not the HKDL installation or source repository.
- Operate experiments through the installed `hkdl` command. Use its absolute launcher path if another environment shadows it.
- Avoid modifying installed HKDL Core or bundled Templates; changes can break execution and updates. For HKDL development, use a separate source checkout.
- Keep this HKDL-managed `AGENTS.md` unchanged; put workspace-specific guidance in `AGENTS.user.md`.
- If `AGENTS.user.md` exists, read it before starting work. Edit it only when requested.
- Never directly modify generated outputs, managed storage, or recovery journals. Perform cleanup only through HKDL commands with explicit approval.

## Commands

- Inspect the workspace with `hkdl status`, `hkdl experiment list`, and `hkdl template list`.
- Use `hkdl --help` and `<command> --help` to discover commands and options; do not guess syntax.
- Manage research with `hkdl experiment` and `hkdl variant`.
- Execute with `hkdl run train`, `eval`, `export`, and `retry`.
- Inspect results with `hkdl run metrics`, `hkdl run logs`, and `hkdl model list`.
- Inspect tracking defaults with `hkdl settings show`; configure tracking only when requested.

## Research authoring

- Edit research files under `experiments/` according to the user's request.
- Use the latest Template version unless another version is requested.
- Keep HKDL-managed metadata out of authored files.
- Commit Experiment questions and Variant Code separately; Options are captured by the next Run.

## Destructive operations

- Before cleanup, rename, deletion, or promotion, inspect `--dry-run` and obtain approval for the shown changes.
- For environment pruning, review every selected environment; `--all` can include referenced inactive environments.
- Preserve CLI confirmations. Use `--yes` only where supported and after approval.
- Variant promotion also deletes the Source active closure. Confirm that consequence before applying.
- If promotion recovery is pending, repeat the exact original command; never manually delete the Source or edit recovery journals.

## Installation and migration

- Update or migrate only when explicitly requested; never use either as automatic error repair.
- Before proceeding, consult `hkdl update --help` or `hkdl migrate --help` and follow the linked procedure.

## Long-running operations

- For long jobs, use a persistent execution method supported by the host and retain a way to inspect progress and completion.
- Report how to resume monitoring when handing off an unfinished job.
- Verify both command completion and final HKDL Run state before reporting success; logs or metrics alone are insufficient.
- Report failures or conflicting states; never modify generated records or retry automatically.
