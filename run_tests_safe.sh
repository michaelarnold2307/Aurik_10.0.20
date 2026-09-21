#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_tests_safe.sh — Crash-sicherer Pytest-Launcher für Aurik 9
# ══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM: Wenn pytest unter VS Code (Snap) läuft, ist der Python-Prozess ein
# Kind-Prozess des VS Code-Prozessbaums. Ein OOM-Kill des Python-Prozesses durch
# den Linux-Kernel killt mitunter den gesamten VS Code-Prozessbaum → Absturz.
#
# LÖSUNG: Dieser Wrapper isoliert den Test-Prozess vollständig aus dem
# VS Code-Prozessbaum via systemd-run cgroup (bevorzugt) oder setsid+ulimit.
# Speicher-Cap: Python wird gekillt, NICHT VS Code.
#
# VERWENDUNG:
#   ./run_tests_safe.sh [pytest-argumente]
#
# BEISPIELE:
#   ./run_tests_safe.sh tests/unit -q --timeout=30
#   ./run_tests_safe.sh tests/unit tests/musical_goals --maxfail=5
#   AURIK_MEM_GB=12 ./run_tests_safe.sh tests/ -m "not ml and not e2e"
#
# UMGEBUNGSVARIABLEN:
#   AURIK_MEM_GB=8          Speicher-Cap in GB (Default: 8)
#   AURIK_TEST_RSS_LIMIT_MB=7000   RSS-Watchdog-Limit in conftest.py
#   AURIK_LOG_FILE=...      Pfad für Log-Datei (Default: logs/pytest_safe.log)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# GUI-Tests (PyQt5) ohne Display: offscreen erzwingen — sonst bricht QApplication
# im Headless-Cgroup mit nativem Abort ab (Voll-Suite-Befund 2026-09-21, Chunk 25:
# test_modern_window_defect_summary_widget_content.py, "Fatal Python error: Aborted").
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM=offscreen
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${SCRIPT_DIR}/.venv_aurik/bin/python"
MEM_GB="${AURIK_MEM_GB:-8}"
MEM_BYTES=$(( MEM_GB * 1024 * 1024 * 1024 ))
MEM_MB=$(( MEM_GB * 1024 ))
MEM_KB=$(( MEM_GB * 1024 * 1024 ))
SWAP_MB="${AURIK_SWAP_MB:-2048}"
LOG_FILE="${AURIK_LOG_FILE:-${SCRIPT_DIR}/logs/pytest_safe.log}"
RSS_LIMIT_MB="${AURIK_TEST_RSS_LIMIT_MB:-$(( MEM_GB * 1024 * 85 / 100 ))}"
GLOBAL_TIMEOUT_S="${AURIK_GLOBAL_TIMEOUT_S:-5400}"
STATUS_FILE="${AURIK_STATUS_FILE:-${SCRIPT_DIR}/logs/pytest_safe.status}"
REPORT_FILE="${AURIK_REPORT_FILE:-${SCRIPT_DIR}/logs/pytest_safe.mini_report.txt}"
STREAM_MODE_RAW="${AURIK_STREAM_TO_TERMINAL:-auto}"
STREAM_TO_TERMINAL="0"
TERMINAL_LINE_BUDGET="${AURIK_TERMINAL_LINE_BUDGET:-400}"
TERMINAL_PROGRESS_EVERY="${AURIK_TERMINAL_PROGRESS_EVERY:-2000}"

# Auto-Modus: In VS-Code-Terminals standardmaessig quiet (stabil), ausserhalb live.
_IN_VSCODE_TERMINAL=0
if [[ "${TERM_PROGRAM:-}" == "vscode" || -n "${VSCODE_PID:-}" || -n "${VSCODE_IPC_HOOK_CLI:-}" ]]; then
    _IN_VSCODE_TERMINAL=1
fi

case "${STREAM_MODE_RAW,,}" in
    1|true|yes|on|live)
        STREAM_TO_TERMINAL="1"
        ;;
    0|false|no|off|quiet)
        STREAM_TO_TERMINAL="0"
        ;;
    auto|"")
        if [[ "$_IN_VSCODE_TERMINAL" -eq 1 ]]; then
            STREAM_TO_TERMINAL="0"
        else
            STREAM_TO_TERMINAL="1"
        fi
        ;;
    *)
        echo "[safe-runner] Warnung: Unbekannter Wert fuer AURIK_STREAM_TO_TERMINAL='${STREAM_MODE_RAW}', nutze auto." >&2
        if [[ "$_IN_VSCODE_TERMINAL" -eq 1 ]]; then
            STREAM_TO_TERMINAL="0"
        else
            STREAM_TO_TERMINAL="1"
        fi
        ;;
esac

# Robuste Terminal-Capability-Umgebung fuer Snap/VS-Code-Subprozesse
export TERM="${TERM:-xterm-256color}"
export TERMINFO="${TERMINFO:-/usr/share/terminfo}"
export TERMINFO_DIRS="${TERMINFO_DIRS:-/usr/share/terminfo:/lib/terminfo:/etc/terminfo}"

# Konftest-Watchdog-Limit aus Speicher-Cap ableiten (85 % des Caps)
export AURIK_TEST_RSS_LIMIT_MB="$RSS_LIMIT_MB"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "${SCRIPT_DIR}/logs/locks"

_is_heavy_invocation() {
    local arg
    local joined=" $* "

    for arg in "$@"; do
        case "$arg" in
            --run-heavy-tests|--run-gui-tests)
                return 0
                ;;
        esac
    done

    if [[ "$joined" == *" -m "* ]]; then
        if [[ "$joined" == *" ml "* || "$joined" == *" slow "* || "$joined" == *" e2e "* || "$joined" == *" competitive "* || "$joined" == *" amrb "* ]]; then
            return 0
        fi
    fi

    for arg in "$@"; do
        local low="${arg,,}"
        case "$low" in
            *tests/test_uat_acceptance_criteria.py*|*tests/normative/test_amrb_ci_gate.py*|*tests/normative/test_competitive_ci_gate.py*|*tests/integration*|*tests/normative*)
                return 0
                ;;
        esac
    done

    return 1
}

_classify_exit() {
    local rc="$1"
    case "$rc" in
        0) echo "PASS" ;;
        124) echo "TIMEOUT" ;;
        137) echo "OOM_OR_SIGKILL" ;;
        143) echo "SIGTERM_ABORT" ;;
        98) echo "DUPLICATE_RUN_BLOCKED" ;;
        *)
            if [[ "$rc" -gt 128 ]]; then
                echo "SIGNAL_$((rc - 128))"
            else
                echo "FAIL"
            fi
            ;;
    esac
}

_write_status_and_report() {
    local rc="$1"
    shift
    local args_text
    local cls
    local now
    local summary
    local first_fail
    cls="$(_classify_exit "$rc")"
    now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    args_text="$*"
    summary="$(grep -E "=+ .* in [0-9.]+s|[0-9]+ passed" "$LOG_FILE" | tail -n 1 || true)"
    first_fail="$(grep -m1 -E "^FAILED |^ERROR |^E   |AssertionError|Traceback" "$LOG_FILE" || true)"

    {
        echo "timestamp=${now}"
        echo "exit_code=${rc}"
        echo "classification=${cls}"
        echo "args=${args_text}"
    } > "$STATUS_FILE"

    {
        echo "Aurik Pytest Mini Report"
        echo "timestamp_utc: ${now}"
        echo "classification: ${cls}"
        echo "exit_code: ${rc}"
        echo "memory_cap_gb: ${MEM_GB}"
        echo "swap_cap_mb: ${SWAP_MB}"
        echo "args: ${args_text}"
        if [[ -n "$summary" ]]; then
            echo "summary: ${summary}"
        fi
        if [[ -n "$first_fail" ]]; then
            echo "first_failure_hint: ${first_fail}"
        fi
        echo "top_warnings:"
        grep -E "WARNING|⚠️|WARN" "$LOG_FILE" | head -n 5 || true
    } > "$REPORT_FILE"
}

# Duplicate-Run-Guard: blockiert nur wirklich identische gleichzeitige Aufrufe.
# Wichtige Runtime-Parameter (Memory/Swap/Stream/Log) gehen in den Fingerprint ein,
# damit unterschiedliche Profile nicht faelschlich als Duplikat gelten.
_run_fingerprint="$(
    printf '%s\0' \
        "$@" \
        "mem_gb=${MEM_GB}" \
        "swap_mb=${SWAP_MB}" \
        "rss_limit_mb=${RSS_LIMIT_MB}" \
        "stream_mode=${STREAM_TO_TERMINAL}" \
        "log_file=${LOG_FILE}" \
        "python=${PYTHON}" \
        "cwd=${SCRIPT_DIR}" \
    | sha256sum | awk '{print $1}'
)"
_lock_file="${SCRIPT_DIR}/logs/locks/pytest_${_run_fingerprint}.lock"
if command -v flock &>/dev/null; then
    exec 9>"$_lock_file"
    if ! flock -n 9; then
        echo "[safe-runner] Abbruch: identischer Testlauf ist bereits aktiv (${_lock_file})." >&2
        _write_status_and_report 98 "$@"
        exit 98
    fi
fi

_HEAVY_SERIAL_LOCK_ACTIVE=0
if command -v flock &>/dev/null && _is_heavy_invocation "$@"; then
    _heavy_lock_file="${SCRIPT_DIR}/logs/locks/pytest_heavy_global.lock"
    exec 8>"$_heavy_lock_file"
    _HEAVY_SERIAL_LOCK_ACTIVE=1
    echo "[safe-runner] Heavy-Serialisierung aktiv: warte auf globalen Heavy-Lock (${_heavy_lock_file})."
    flock 8
fi

echo "══════════════════════════════════════════════════════"
echo " Aurik Safe Test Runner"
echo " Speicher-Cap : ${MEM_GB} GB"
echo " Swap-Cap     : ${SWAP_MB} MB"
echo " RSS-Watchdog : ${RSS_LIMIT_MB} MB"
echo " Global-Timeout: ${GLOBAL_TIMEOUT_S} s"
echo " Log          : ${LOG_FILE}"
echo " Terminal-Budget: ${TERMINAL_LINE_BUDGET} Zeilen (live)"
if [[ "$_HEAVY_SERIAL_LOCK_ACTIVE" -eq 1 ]]; then
    echo " Heavy-Lock   : global serialisiert"
fi
if [[ "$STREAM_TO_TERMINAL" == "1" ]]; then
    if [[ "${STREAM_MODE_RAW,,}" == "auto" || -z "$STREAM_MODE_RAW" ]]; then
        echo " Terminal-I/O : live (auto)"
    else
        echo " Terminal-I/O : live (erzwungen)"
    fi
else
    if [[ "${STREAM_MODE_RAW,,}" == "auto" || -z "$STREAM_MODE_RAW" ]]; then
        echo " Terminal-I/O : quiet (auto fuer VS Code; Details im Log)"
    else
        echo " Terminal-I/O : quiet (erzwungen; Details im Log)"
    fi
fi
echo " Argumente    : $*"
echo "══════════════════════════════════════════════════════"

_print_quiet_summary() {
    local rc="$1"
    local cls
    cls="$(_classify_exit "$rc")"
    if [[ "$rc" -eq 0 ]]; then
        local summary
        summary="$(grep -E "=+ .* in [0-9.]+s|[0-9]+ passed" "$LOG_FILE" | tail -n 1 || true)"
        if [[ -n "$summary" ]]; then
            echo "[safe-runner] Erfolg (${cls}): ${summary}"
        else
            echo "[safe-runner] Erfolg (${cls}) (Details: ${LOG_FILE})"
        fi
    else
        if [[ ! -s "$LOG_FILE" ]]; then
            {
                echo "[safe-runner] WARN: Keine Pytest-Ausgabe im Log erfasst (Datei leer)."
                echo "[safe-runner] Hinweis: Prozess könnte vor erster Ausgabe beendet worden sein."
            } >> "$LOG_FILE"
        fi
        echo "[safe-runner] Fehler (${cls}, Exit ${rc}). Letzte 120 Log-Zeilen:" >&2
        tail -n 120 "$LOG_FILE" >&2 || true
    fi
}

_stream_with_budget() {
    awk -v max_lines="$TERMINAL_LINE_BUDGET" -v progress_every="$TERMINAL_PROGRESS_EVERY" -v log_file="$LOG_FILE" '
        {
            if (NR <= max_lines) {
                print $0;
                next;
            }
            if (NR == max_lines + 1) {
                printf("[safe-runner] Terminal-Budget erreicht (%d Zeilen). Weitere Ausgabe wird unterdrueckt; Log laeuft weiter: %s\n", max_lines, log_file);
                next;
            }
            if (progress_every > 0 && (NR % progress_every) == 0) {
                printf("[safe-runner] ...%d Zeilen verarbeitet (Terminal-Ausgabe gedrosselt)...\n", NR);
            }
        }
        END {
            if (NR > max_lines) {
                printf("[safe-runner] Terminal-Drossel aktiv: %d/%d Zeilen angezeigt. Vollstaendiges Log: %s\n", max_lines, NR, log_file);
            }
        }
    '
}

# ── Methode 1: systemd-run (beste Isolation via cgroup) ──────────────────────
# Erstellt eine eigene cgroup mit hartem Speicher-Limit. Der Python-Prozess
# wird vom Kernel in seiner eigenen cgroup gekillt — VS Code ist vollständig
# getrennt.
# Prüfe ob systemd-run --user --scope --collect unterstützt wird (systemd ≥ 236).
_SYSTEMD_OK=0
if command -v systemd-run &>/dev/null; then
    if systemd-run --user --scope --collect -- true 2>/dev/null; then
        _SYSTEMD_OK=1
    fi
fi

if [[ "$_SYSTEMD_OK" -eq 1 ]]; then
    echo "[safe-runner] Methode: systemd-run cgroup (MemoryMax=${MEM_GB}G, MemorySwapMax=${SWAP_MB}M)"
    set +e
    if [[ "$STREAM_TO_TERMINAL" == "1" ]]; then
        timeout --foreground --signal=TERM --kill-after=60 "${GLOBAL_TIMEOUT_S}" \
        systemd-run \
            --user \
            --scope \
            --collect \
            --quiet \
            --setenv=TERM="$TERM" \
            --setenv=TERMINFO="$TERMINFO" \
            --setenv=TERMINFO_DIRS="$TERMINFO_DIRS" \
            -p "MemoryMax=${MEM_GB}G" \
            -p "MemorySwapMax=${SWAP_MB}M" \
            -p "CPUWeight=50" \
            -p "TasksMax=512" \
            -- \
            "$PYTHON" -m pytest "$@" \
            --override-ini="addopts=--strict-markers --import-mode=importlib" \
            -p no:xdist \
            --disable-warnings \
            --no-header \
            2>&1 | tee "$LOG_FILE" | _stream_with_budget
        _rc=${PIPESTATUS[0]}
    else
        {
            echo "[safe-runner] start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
            echo "[safe-runner] mode=systemd-run quiet"
            echo "[safe-runner] pytest_args=$*"
        } > "$LOG_FILE"
        timeout --foreground --signal=TERM --kill-after=60 "${GLOBAL_TIMEOUT_S}" \
        systemd-run \
            --user \
            --scope \
            --collect \
            --quiet \
            --setenv=TERM="$TERM" \
            --setenv=TERMINFO="$TERMINFO" \
            --setenv=TERMINFO_DIRS="$TERMINFO_DIRS" \
            -p "MemoryMax=${MEM_GB}G" \
            -p "MemorySwapMax=${SWAP_MB}M" \
            -p "CPUWeight=50" \
            -p "TasksMax=512" \
            -- \
            "$PYTHON" -m pytest "$@" \
            --override-ini="addopts=--strict-markers --import-mode=importlib" \
            -p no:xdist \
            --disable-warnings \
            --no-header \
            >> "$LOG_FILE" 2>&1
        _rc=$?
        _print_quiet_summary "$_rc"
    fi
    _write_status_and_report "$_rc" "$@"
    set -e
    exit "$_rc"
fi

# ── Methode 2: setsid + ulimit (Fallback ohne systemd) ───────────────────────
# setsid: Trennt den Prozess von VS Codes Session → eigene Prozessgruppe.
# ulimit -v: Virtuelle Memory Cap. Wenn Python dieses Limit überschreitet,
# erhält es ENOMEM → Python beendet sich, VS Code überlebt.
echo "[safe-runner] Methode: setsid + ulimit -v ${MEM_MB}M"

(
    # Eigene Session → eigene Prozessgruppe → kein Signal-Forwarding zu VS Code
    if [[ "$STREAM_TO_TERMINAL" == "1" ]]; then
        exec timeout --foreground --signal=TERM --kill-after=60 "${GLOBAL_TIMEOUT_S}" setsid bash -c "
            ulimit -v $MEM_KB 2>/dev/null || true
            ulimit -m $MEM_KB 2>/dev/null || true
            exec '$PYTHON' -m pytest \"\$@\" \
                --override-ini='addopts=--strict-markers --import-mode=importlib' \
                -p no:xdist \
                --disable-warnings \
                --no-header
            " -- "$@" 2>&1 | tee "$LOG_FILE" | _stream_with_budget
    else
        set +e
        {
            echo "[safe-runner] start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
            echo "[safe-runner] mode=setsid quiet"
            echo "[safe-runner] pytest_args=$*"
        } > "$LOG_FILE"
        timeout --foreground --signal=TERM --kill-after=60 "${GLOBAL_TIMEOUT_S}" setsid bash -c "
            ulimit -v $MEM_KB 2>/dev/null || true
            ulimit -m $MEM_KB 2>/dev/null || true
            exec '$PYTHON' -m pytest \"\$@\" \
                --override-ini='addopts=--strict-markers --import-mode=importlib' \
                -p no:xdist \
                --disable-warnings \
                --no-header
        " -- "$@" >> "$LOG_FILE" 2>&1
        _rc=$?
        set -e
        _print_quiet_summary "$_rc"
        _write_status_and_report "$_rc" "$@"
        exit "$_rc"
    fi
)
