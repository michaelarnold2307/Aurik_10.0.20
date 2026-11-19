#!/usr/bin/env bash
# §4.4+§10.2 Compliance-Pytest Runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv_aurik/bin/python"

# Robuste Terminal-Capability-Defaults fuer Snap/VS-Code-Subprozesse
export TERM="${TERM:-xterm-256color}"
export TERMINFO="${TERMINFO:-/usr/share/terminfo}"
export TERMINFO_DIRS="${TERMINFO_DIRS:-/usr/share/terminfo:/lib/terminfo:/etc/terminfo}"

cd "$PROJECT_ROOT"
"$PYTHON" -m pytest tests/unit \
  -p no:xdist \
  --override-ini="addopts=--strict-markers --import-mode=importlib" \
  --timeout=30 \
  --tb=short \
  -q \
  --disable-warnings \
  --no-header \
  2>&1 | tail -60
