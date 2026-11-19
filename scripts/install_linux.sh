#!/usr/bin/env bash
# =============================================================================
# Aurik 10.1.0 — Linux-Installation (Kompatibilitäts-Wrapper)
#
# Der kanonische Installer ist scripts/install_aurik.sh (Komplett-Installation
# mit OS-Erkennung, apt-Abhängigkeiten, venv, PyTorch-Auswahl, Modell-Prüfung
# und Launcher-Integration). Dieser Wrapper erhält den historischen Aufruf
# `bash scripts/install_linux.sh` aufrecht und delegiert vollständig.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/install_aurik.sh" "$@"
