#!/usr/bin/env bash
# scripts/count_pytest_ids.sh — Dynamische Testzählung für Aurik 9
#
# Zählt alle gesammelten Pytest-IDs (inkl. parametrisierter Tests)
# und die Anzahl der def test_-Funktionen im Quelltext.
#
# Verwendung:
#   ./scripts/count_pytest_ids.sh
#   ./scripts/count_pytest_ids.sh --json   # JSON-Ausgabe für CI
#
# Referenz: copilot-instructions.md, Specs 07/08 (D-6 dynamischer Testzähler)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv_aurik/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "FEHLER: Python-venv nicht gefunden: $VENV_PYTHON" >&2
    exit 1
fi

# Pytest-IDs sammeln (inkl. parametrisierter Tests)
PYTEST_IDS=$("$VENV_PYTHON" -m pytest tests --collect-only -q \
    -p no:xdist \
    --override-ini="addopts=--strict-markers --import-mode=importlib" \
    --disable-warnings --no-header 2>/dev/null \
    | grep -c '::' || echo 0)

# def test_-Funktionen im Quelltext zählen
DEF_TEST_COUNT=$(grep -r --include='*.py' -c 'def test_' "$PROJECT_ROOT/tests/" \
    | awk -F: '{s+=$2} END {print s}')

if [[ "${1:-}" == "--json" ]]; then
    echo "{\"pytest_ids\": $PYTEST_IDS, \"def_test_functions\": $DEF_TEST_COUNT}"
else
    echo "Gesammelte Pytest-IDs (parametrisiert): $PYTEST_IDS"
    echo "def test_-Funktionen im Quelltext:      $DEF_TEST_COUNT"
fi
