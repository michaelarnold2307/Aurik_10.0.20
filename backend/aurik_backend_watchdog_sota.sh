#!/usr/bin/env bash
# Kompatibilitätswrapper: Der kanonische Desktop-Watchdog lebt in
# backend/aurik_backend_watchdog.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/aurik_backend_watchdog.sh" "$@"
