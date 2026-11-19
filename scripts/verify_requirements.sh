#!/usr/bin/env bash
# scripts/verify_requirements.sh — Wrapper-Skript für verify_requirements.py
# Ausführen: bash scripts/verify_requirements.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv_aurik/bin/python"

# Fallback auf System-Python wenn kein venv vorhanden
if [[ ! -f "$PYTHON" ]]; then
    PYTHON="python3"
fi

exec "$PYTHON" "${SCRIPT_DIR}/verify_requirements.py" "$@"
