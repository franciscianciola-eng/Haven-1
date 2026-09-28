#!/bin/bash
# Starts Haven and opens its window. The first time, it sets itself up: that needs the internet and
# takes a few minutes. It uses uv (https://docs.astral.sh/uv/) to get Python and Haven's parts, which
# go into the .venv folder here, so nothing else on this computer is changed.
cd "$(dirname "$0")" || exit 1

pause() { read -r -p "Press Enter to close this window. " _; }

UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
if [ -z "$UV" ]; then
  echo "Getting uv, a small tool that installs Python and Haven's parts..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
  else
    wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
  fi
  UV="$HOME/.local/bin/uv"
fi
if [ ! -x "$UV" ]; then
  echo "Couldn't get uv. Check the internet connection, or install uv yourself (https://docs.astral.sh/uv/), then try again."
  pause
  exit 1
fi

if [ ! -x .venv/bin/haven ]; then
  echo "Setting Haven up. The first time takes a few minutes..."
  if [ ! -x .venv/bin/python ]; then
    "$UV" venv --python 3.12 .venv || { echo "Couldn't set up Python."; pause; exit 1; }
  fi
  # --torch-backend=auto picks the PyTorch build for this computer's graphics card, if it has one.
  "$UV" pip install --python .venv/bin/python --torch-backend=auto -e ".[cortex]" \
    || "$UV" pip install --python .venv/bin/python -e ".[cortex]" \
    || { echo "Couldn't install Haven's parts. Check the internet connection and try again."; pause; exit 1; }
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1  # on a Mac, anything its graphics chip can't do runs on the processor
.venv/bin/haven app "$@"
status=$?
if [ "$status" -ne 0 ]; then pause; fi
exit "$status"
