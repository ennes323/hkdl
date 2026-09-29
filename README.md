# HKDL

HKDL is a local tool for managing machine learning experiments and preserving their execution history. It captures the code and configuration used by each run, records its results, and keeps previous runs intact as experiments evolve.

HKDL also supports working with LLM-based coding agents. `hkdl workspace init` provides an `AGENTS.md` guide at the research root for operating experiments while preserving research data and execution history.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then use the standalone installer and release bundle supplied together:

```sh
uv run --no-project --python 3.12 INSTALLER.pyz install BUNDLE.zip
```

Replace the filenames with the installer and bundle supplied together on the selected [GitHub Release](https://github.com/hukuhaka/hkdl/releases). Check the attached checksum file before installation; a package version alone does not establish artifact availability.

The installer previews the bundle and asks before activation. Add the printed launcher directory to `PATH`; it defaults to `~/.local/bin`. Python 3.12 and locked dependencies may be downloaded by uv.

Create a separate research workspace:

```sh
hkdl workspace init /path/to/research
cd /path/to/research
```

HKDL and its execution environment live outside this directory. The workspace holds your experiments, results, and local agent guidance; it does not need a source checkout or root virtual environment.

Source-checkout installation remains available:

```sh
git clone https://github.com/hukuhaka/hkdl.git
cd hkdl
./setup.sh
source ./activate.sh
```

## Run your first experiment

Create an Experiment from the ResNet18 image-classification Template, then create a Variant:

```sh
hkdl experiment create demo --template resnet18
hkdl variant create demo baseline
```

Train on CPU:

```sh
hkdl run train demo baseline smoke --seed 0 -d cpu
```

The Template includes a small image dataset and trains without downloading additional data or pretrained weights.

Evaluate the trained Model:

```sh
hkdl run eval demo baseline smoke default --seed 0
hkdl model list demo baseline
```

Inspect the training log and metrics:

```sh
hkdl run logs demo baseline run-001
hkdl run metrics demo baseline run-001
```

Use `hkdl --help` or add `--help` to a command to see its available options.

## How experiments are organized

An **Experiment** groups related research. Each **Variant** contains independently editable code and configuration.

Training, evaluation, and export each create a **Run** that records the inputs and outcome of that execution. Successful training also creates a **Model**, which can be evaluated or exported.

Editing a Variant does not rewrite previous Runs. Retrying a failed or interrupted execution creates a new Run linked to its parent:

```sh
hkdl run retry demo baseline run-001
```

Training can resume from a valid checkpoint when the captured Variant code supports it.

## Features

- **Reusable Templates:** Start with bundled ResNet18 classification or YOLO26n object-detection examples, then edit the copied Variant source.
- **Repeatable execution:** Capture research inputs, train with multiple seeds, and evaluate named cases.
- **Result inspection:** Read logs and metrics from the CLI or compare Runs and learning curves in the local web interface.
- **Tracking:** Record metrics locally or connect an external MLflow server.
- **Environment reuse:** Share matching execution environments while protecting environments used by active Runs.
- **Explicit lifecycle operations:** Preview rename, deletion, migration, and committed Code promotion before applying them.

Existing Variants retain their copied Template source when HKDL is updated. New Template versions do not alter previous experiments or captured execution inputs.

## Work with coding agents

Ask your coding agent to read the research root's `AGENTS.md` before working on experiments. It describes the HKDL workflow, CLI usage, data ownership, and operations that require confirmation.

For local preferences, create an optional `AGENTS.user.md` at the research root instead of editing the HKDL-managed `AGENTS.md`. Use it for execution limits, preferred devices, and reporting conventions. For example:

```md
# Local experiment guidance

- Use CPU unless I explicitly request another device.
- Ask before starting training that is expected to take more than 10 minutes.
- Report Run IDs, final status, and key metrics in Korean.
```

`workspace init` installs the default guide when `AGENTS.md` is absent. For an existing or modified file, choose to keep it, back it up and replace it, or preserve it as `AGENTS.user.md` before installing the default. An existing `AGENTS.user.md` is never overwritten.

The default guide instructs agents to read `AGENTS.user.md` when present. Managed updates refresh only unmodified HKDL-owned guidance; user files and local edits remain unchanged. Rerun `hkdl workspace init` to explicitly adopt the installed guide, or consult the version-specific GitHub link printed by HKDL. Git tracking in an independent workspace is user-controlled.

Local preferences supplement the standard guide while retaining its data-preservation and confirmation rules.

## Update

Moving from public 1.2.2 to 2.0.0 requires an explicit source update and any needed research migration. Start with [Moving from 1.2.2 to 2.0.0](docs/user/migrating-to-2.0.md); use the new CLI for migration before resuming research. The guide identifies the exact release and artifacts required for that transition.

For a managed installation, select the supplied release bundle:

```sh
hkdl update BUNDLE.zip
```

For a public source checkout, use `hkdl update` without a bundle.

HKDL previews the update and asks for confirmation. See [Updating HKDL](docs/user/updating.md) for requirements, data preservation, and migration guidance.

Version-specific changes are listed in the [release notes](https://github.com/hukuhaka/hkdl/releases).

## License

HKDL Core is licensed under MIT. Bundled data and Templates may have additional terms:

- The TF-Flowers fixture is provided under CC BY 4.0 with its included attribution.
- The Ultralytics-based YOLO26n Template and applicable derived code and artifacts are subject to AGPL-3.0-or-later. An alternative commercial license is available from Ultralytics.

See the included license and attribution files for details.
