#!/usr/bin/env bash
# Aurik Desktop-Watchdog (v9.12.9-hotfix.1)
#
# Aurik ist eine lokale Desktop-App, kein Serverdienst. Dieser Watchdog prüft
# den GUI-Prozess und startet ihn über den kanonischen Launcher neu. Kein
# systemctl, kein Docker, kein HTTP-Healthcheck.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
START_SCRIPT="$PROJECT_ROOT/run_aurik.sh"
PID_FILE="${AURIK_GUI_PID_FILE:-$PROJECT_ROOT/temp_repro/aurik_gui.pid}"
LOG_FILE="${AURIK_WATCHDOG_LOG:-$PROJECT_ROOT/logs/aurik_watchdog.log}"

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"

log() {
    local msg="$1"
    printf '[Aurik-Watchdog] %s\n' "$msg" | tee -a "$LOG_FILE"
    if command -v logger >/dev/null 2>&1; then
        logger -t aurik-watchdog "$msg" || true
    fi
}

pid_is_running() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

_pid=""
if [[ -f "$PID_FILE" ]]; then
    _pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if pid_is_running "$_pid"; then
    log "Aurik GUI läuft fehlerfrei (PID $_pid)."
    exit 0
fi

_pgrep_pid="$(pgrep -f "[A]urik910/main.py" | head -n 1 || true)"
if pid_is_running "$_pgrep_pid"; then
    printf '%s\n' "$_pgrep_pid" >"$PID_FILE"
    log "Aurik GUI läuft fehlerfrei (PID $_pgrep_pid, PID-Datei aktualisiert)."
    exit 0
fi

if [[ ! -x "$START_SCRIPT" ]]; then
    log "Launcher nicht ausführbar: $START_SCRIPT"
    exit 1
fi

log "Aurik GUI nicht aktiv — starte über run_aurik.sh."
"$START_SCRIPT"
