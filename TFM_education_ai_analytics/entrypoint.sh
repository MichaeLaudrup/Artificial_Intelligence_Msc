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

# Execute the original command
exec "$@"
