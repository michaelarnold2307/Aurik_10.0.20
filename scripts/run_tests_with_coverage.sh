#!/usr/bin/env bash
# run_tests_with_coverage.sh
# Führt Tests mit Coverage-Report aus

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv_aurik/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "FEHLER: venv-Python nicht gefunden: $PYTHON" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

echo "=== Aurik Test Suite mit Coverage ==="
echo ""

# Cleanup alte Coverage-Daten
rm -rf .coverage htmlcov/ coverage.xml 2>/dev/null || true

# Führe Tests mit Coverage aus
echo "📊 Starte Tests mit Coverage-Tracking..."
"$PYTHON" -m pytest tests/ \
    -p no:xdist \
    --override-ini="addopts=--strict-markers --import-mode=importlib" \
    --maxfail=5 \
    --disable-warnings \
    --tb=short \
    --cov=. \
    --cov-config=.coveragerc \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    -n auto \
    || true

echo ""
echo "✅ Tests abgeschlossen!"
echo ""
echo "📈 Coverage-Reports:"
echo "  - HTML: htmlcov/index.html"
echo "  - XML: coverage.xml"
echo ""
echo "Öffne HTML-Report mit:"
echo "  xdg-open htmlcov/index.html"
