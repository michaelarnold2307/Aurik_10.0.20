#!/usr/bin/env bash
# run_performance_benchmarks.sh
# Führt Performance-Benchmarks aus und generiert Reports

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv_aurik/bin/python"

# Robuste Terminal-Capability-Defaults fuer Snap/VS-Code-Subprozesse
export TERM="${TERM:-xterm-256color}"
export TERMINFO="${TERMINFO:-/usr/share/terminfo}"
export TERMINFO_DIRS="${TERMINFO_DIRS:-/usr/share/terminfo:/lib/terminfo:/etc/terminfo}"

if [[ ! -x "$PYTHON" ]]; then
    echo "FEHLER: venv-Python nicht gefunden: $PYTHON" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

echo "=== Aurik Performance Benchmarks ==="
echo ""

# Erstelle Benchmark-Output-Verzeichnis
mkdir -p benchmarks/results

echo "🚀 Starte DSP Performance Benchmarks..."
"$PYTHON" -m pytest benchmarks/test_dsp_performance.py \
    --benchmark-only \
    --benchmark-autosave \
    --benchmark-save-data \
    --benchmark-json=benchmarks/results/dsp_benchmarks.json \
    --benchmark-histogram=benchmarks/results/dsp_histogram \
    -v

echo ""
echo "🚀 Starte Pipeline Performance Benchmarks..."
"$PYTHON" -m pytest benchmarks/test_pipeline_performance.py \
    --benchmark-only \
    --benchmark-autosave \
    --benchmark-save-data \
    --benchmark-json=benchmarks/results/pipeline_benchmarks.json \
    --benchmark-histogram=benchmarks/results/pipeline_histogram \
    -v

echo ""
echo "✅ Benchmarks abgeschlossen!"
echo ""
echo "📊 Ergebnisse:"
echo "  - DSP Benchmarks: benchmarks/results/dsp_benchmarks.json"
echo "  - Pipeline Benchmarks: benchmarks/results/pipeline_benchmarks.json"
echo "  - Histogramme: benchmarks/results/*_histogram.svg"
echo ""
echo "Vergleiche Benchmarks mit:"
echo "  pytest-benchmark compare benchmarks/results/*.json"
