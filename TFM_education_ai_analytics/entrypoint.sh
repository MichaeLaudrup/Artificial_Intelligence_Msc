#!/bin/bash
# Entrypoint script for the TFM GPU container
# Includes workaround for Antigravity v1.16.5 devcontainer bug
# See: https://discuss.ai.google.dev/t/can-no-longer-connect-to-devcontainer-after-updating-to-v1-16-5/121479

# Workaround: Antigravity looks for node at .../bin/{commit}/node
# but installs to .../bin/{version}-{commit}/node
# This background loop creates symlinks as soon as the server directory appears
(
  SERVER_DIR="$HOME/.antigravity-server/bin"
  mkdir -p "$SERVER_DIR"
  for i in $(seq 1 60); do
    for dir in "$SERVER_DIR"/*-*; do
      if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        commit="${dirname#*-}"
        symlink="$SERVER_DIR/$commit"
        if [ ! -e "$symlink" ]; then
          ln -s "$dir" "$symlink" 2>/dev/null
        fi
      fi
    done
    sleep 2
  done
) &

# Robust bootstrap for portable runs on any host machine.
# The project source is bind-mounted at /workspace, but the venv is stored in
# /opt so it is not overwritten by host files.
PROJECT_DIR="/workspace/TFM_education_ai_analytics"
VENV_DIR="/opt/tfm-venv"
VENV_PY="$VENV_DIR/bin/python"
BOOTSTRAP_DIR="$VENV_DIR/.bootstrap"
HASH_FILE="$BOOTSTRAP_DIR/pyproject.hash"

if [ ! -x "$VENV_PY" ]; then
  echo "[entrypoint] Creating runtime venv at $VENV_DIR..."
  python -m venv --system-site-packages "$VENV_DIR"
fi

export PATH="$VENV_DIR/bin:$PATH"

if [ -d "$PROJECT_DIR" ] && [ -f "$PROJECT_DIR/pyproject.toml" ]; then
  mkdir -p "$BOOTSTRAP_DIR"
  CURRENT_HASH=$(sha256sum "$PROJECT_DIR/pyproject.toml" | awk '{print $1}')
  STORED_HASH=""
  if [ -f "$HASH_FILE" ]; then
    STORED_HASH=$(cat "$HASH_FILE")
  fi

  if [ "$CURRENT_HASH" != "$STORED_HASH" ] || ! "$VENV_PY" -c "import educational_ai_analytics" >/dev/null 2>&1; then
    echo "[entrypoint] Syncing project dependencies from pyproject.toml..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install --no-cache-dir -e "$PROJECT_DIR"
    echo "$CURRENT_HASH" > "$HASH_FILE"
  fi
fi

# Execute the original command
exec "$@"
