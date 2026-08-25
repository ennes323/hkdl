#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required" >&2
    exit 1
fi

venv_dir=.venv
venv_python="$venv_dir/bin/python"

fail() {
    echo "error: $*" >&2
    exit 1
}

if [ -L "$venv_dir" ]; then
    fail ".venv is a symlink; refusing to replace it"
fi

if [ -e "$venv_dir" ] && [ ! -d "$venv_dir" ]; then
    fail ".venv exists but is not a directory; refusing to replace it"
fi

if [ -d "$venv_dir" ]; then
    if [ ! -f "$venv_dir/pyvenv.cfg" ]; then
        fail ".venv exists but is not a Python virtual environment; refusing to replace it"
    fi

    if [ ! -f "$venv_python" ] || [ ! -x "$venv_python" ]; then
        echo "Root .venv is incomplete; recreating it." >&2
        if ! uv venv --clear --relocatable "$venv_dir"; then
            fail "could not recreate the incomplete root .venv"
        fi
    else
        if ! uv venv --allow-existing --relocatable "$venv_dir"; then
            fail "could not normalize the root .venv"
        fi
    fi
else
    if ! uv venv --relocatable "$venv_dir"; then
        fail "could not create the root .venv"
    fi
fi

if ! uv sync --locked --no-editable --reinstall; then
    fail "locked dependency synchronization failed"
fi

if ! "$venv_python" -c "import hkdl"; then
    fail "hkdl could not be imported from the root .venv"
fi
if ! "$venv_dir/bin/hkdl" --help >/dev/null; then
    fail "the installed hkdl CLI could not start"
fi
if ! "$venv_dir/bin/hkdl" --version >/dev/null; then
    fail "the installed hkdl CLI could not report its version"
fi
if ! "$venv_dir/bin/hkdl" completion zsh >/dev/null; then
    fail "the installed hkdl CLI could not generate Zsh completion"
fi

echo "HKDL is ready. Run: source ./activate.sh"
