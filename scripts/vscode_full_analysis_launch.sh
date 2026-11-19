#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:-/media/michael/Software 4TB/Aurik_Standalone}"

# Voll-Analyse-Modus: startet VS Code ohne zusätzliche Extension-Blockliste.
# Für maximale Stabilität im Daily-Workload weiterhin scripts/vscode_stable_launch.sh nutzen.
code "$WORKSPACE_DIR"
