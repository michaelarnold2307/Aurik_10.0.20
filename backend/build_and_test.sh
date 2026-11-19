#!/usr/bin/env bash
# Aurik Desktop Build-&-Test-Helfer — kein Server, kein Docker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv_aurik/bin/python"

# Robuste Terminal-Capability-Defaults fuer Snap/VS-Code-Subprozesse
export TERM="${TERM:-xterm-256color}"
export TERMINFO="${TERMINFO:-/usr/share/terminfo}"
export TERMINFO_DIRS="${TERMINFO_DIRS:-/usr/share/terminfo:/lib/terminfo:/etc/terminfo}"

if [[ ! -x "$PYTHON" ]]; then
  echo "[Aurik] venv-Python nicht gefunden: $PYTHON" >&2
  echo "[Aurik] Bitte zuerst: bash scripts/install_aurik.sh" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

echo "[Aurik] Syntaxprüfung Kernmodule..."
"$PYTHON" -m py_compile \
  backend/core/unified_restorer_v3.py \
  backend/core/defect_scanner.py \
  backend/core/gp_parameter_optimizer.py

echo "[Aurik] Unit-Smoke-Tests..."
"$PYTHON" -m pytest tests/unit \
  -p no:xdist \
  --override-ini="addopts=--strict-markers --import-mode=importlib" \
  --timeout=30 \
  --tb=short \
  -q \
  --disable-warnings \
  --no-header \
  --maxfail=3

echo "[Aurik] Desktop Build-&-Test-Helfer erfolgreich abgeschlossen."
