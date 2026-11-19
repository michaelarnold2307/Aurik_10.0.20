#!/bin/bash
#
# Aurik 9 — Monitoring-Ökosystem Quick-Start
# ============================================
#
# Startet sofort die vollständige Analyse mit GUI und Echtzeit-Monitoring.
#
# Usage:
#   ./start_quality_monitoring.sh
#   ./start_quality_monitoring.sh --headless
#   ./start_quality_monitoring.sh --audio path/to/song.mp3

set -e

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_ROOT"

# Default-Audio suchen
AUDIO_PATH=""
HEADLESS=""
VERBOSE=""

# Args parsen
while [[ $# -gt 0 ]]; do
    case $1 in
    --audio)
        AUDIO_PATH="$2"
        shift 2
        ;;
    --headless)
        HEADLESS="--headless"
        shift
        ;;
    --verbose)
        VERBOSE="--verbose"
        shift
        ;;
    *)
        echo "Unbekannte Option: $1"
        exit 1
        ;;
    esac
done

# Audio suchen wenn nicht angegeben
if [ -z "$AUDIO_PATH" ]; then
    if [ -f "test_audio/Testkünstlerin (Schlager) - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3" ]; then
        AUDIO_PATH="test_audio/Testkünstlerin (Schlager) - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"
    else
        # Findet erste MP3 in test_audio
        AUDIO_PATH=$(find test_audio -name "*.mp3" -type f | head -1)
        if [ -z "$AUDIO_PATH" ]; then
            echo "✗ Keine Audio-Datei gefunden"
            echo "  Bitte test_audio/ befüllen oder Audio mit --audio angeben"
            exit 1
        fi
    fi
fi

echo ""
echo "================================================================================"
echo "AURIK 9 — QUALITÄTS-MONITORING ÖKOSYSTEM"
echo "================================================================================"
echo ""
echo "Audio:    $AUDIO_PATH"
echo "Mode:     restoration"
echo ""

# Venv aktivieren
if [ -f ".venv_aurik/bin/activate" ]; then
    source .venv_aurik/bin/activate
else
    echo "✗ .venv_aurik nicht gefunden"
    echo "  Bitte führe zuerst setup aus"
    exit 1
fi

# Python-Verfügbarkeit prüfen
python -c "import scipy.signal, numpy" 2>/dev/null || {
    echo "✗ Abhängigkeiten fehlen"
    exit 1
}

# Orchestrator starten
CMD="python scripts/orchestrate_quality_monitoring.py --audio \"$AUDIO_PATH\""
if [ -n "$HEADLESS" ]; then
    CMD="$CMD $HEADLESS"
fi
if [ -n "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

echo "Starte Orchestrator..."
echo ""
eval "$CMD"
EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Monitoring abgeschlossen"
else
    echo "✗ Monitoring mit Fehler beendet (Code: $EXIT_CODE)"
fi
echo "================================================================================"
echo ""
echo "Ergebnisse:"
echo "  - Analyse-Daten: analysis_results/"
echo "  - Audio-Exports:  output_audio/"
echo "  - Logs:           *.log"
echo ""

exit $EXIT_CODE
