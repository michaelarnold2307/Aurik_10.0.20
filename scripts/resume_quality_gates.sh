#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTEST_CLEAN="${ROOT_DIR}/scripts/pytest_clean.sh"
SAFE_LOG="${ROOT_DIR}/logs/pytest_safe.log"
SAFE_STATUS="${ROOT_DIR}/logs/pytest_safe.status"
SAFE_REPORT="${ROOT_DIR}/logs/pytest_safe.mini_report.txt"
GATE_LOG_DIR="${ROOT_DIR}/logs/gate_resume"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
SUMMARY_FILE="${GATE_LOG_DIR}/resume_summary_${RUN_TS}.txt"

mkdir -p "${GATE_LOG_DIR}"

classify_exit() {
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

cleanup_stale_gate_processes() {
    local pattern="$1"
    pkill -f "$pattern" >/dev/null 2>&1 || true
}

run_gate() {
    local gate_name="$1"
    shift

    echo "[resume-gates] Starte ${gate_name} ..."
    set +e
    "$@"
    local rc=$?
    set -e

    local cls
    cls="$(classify_exit "$rc")"
    local gate_ts
    gate_ts="$(date +"%Y%m%d_%H%M%S")"

    if [[ -f "${SAFE_LOG}" ]]; then
        cp "${SAFE_LOG}" "${GATE_LOG_DIR}/${gate_name}_${gate_ts}.log" || true
    fi
    if [[ -f "${SAFE_REPORT}" ]]; then
        cp "${SAFE_REPORT}" "${GATE_LOG_DIR}/${gate_name}_${gate_ts}.mini_report.txt" || true
    fi
    if [[ -f "${SAFE_STATUS}" ]]; then
        cp "${SAFE_STATUS}" "${GATE_LOG_DIR}/${gate_name}_${gate_ts}.status" || true
    fi

    {
        echo "gate=${gate_name}"
        echo "exit_code=${rc}"
        echo "classification=${cls}"
        echo "timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "---"
    } >> "${SUMMARY_FILE}"

    echo "[resume-gates] ${gate_name}: ${cls} (Exit ${rc})"
    return "$rc"
}

if [[ ! -x "${PYTEST_CLEAN}" ]]; then
    echo "[resume-gates] Fehler: pytest_clean.sh nicht gefunden: ${PYTEST_CLEAN}" >&2
    exit 2
fi

echo "[resume-gates] Resume gestartet: ${RUN_TS}" | tee -a "${SUMMARY_FILE}"

# Bereinigung verwaister Prozesse nach Crash/Reboot
cleanup_stale_gate_processes "test_amrb_ci_gate.py"
cleanup_stale_gate_processes "test_competitive_ci_gate.py"

# 1) AMRB Gate
if ! run_gate "amrb" \
    env QT_QPA_PLATFORM=offscreen AURIK_MEM_GB=16 AURIK_SWAP_MB=4096 AURIK_STREAM_TO_TERMINAL=0 AURIK_TERMINAL_LINE_BUDGET=250 \
    "${PYTEST_CLEAN}" tests/normative/test_amrb_ci_gate.py -p no:xdist --run-heavy-tests --run-gui-tests --timeout=600 --tb=short -q --disable-warnings --no-header --show-capture=no --maxfail=1
then
    echo "[resume-gates] Abbruch: AMRB Gate fehlgeschlagen. Details: ${SUMMARY_FILE}" >&2
    exit 1
fi

# 2) Competitive Gate
if ! run_gate "competitive" \
    env AURIK_COMPETITIVE_BENCHMARK_TIMEOUT_S=660 AURIK_COMPETITIVE_BENCHMARK_GRACE_S=75 QT_QPA_PLATFORM=offscreen AURIK_MEM_GB=16 AURIK_SWAP_MB=4096 AURIK_STREAM_TO_TERMINAL=0 AURIK_TERMINAL_LINE_BUDGET=250 \
    "${PYTEST_CLEAN}" tests/normative/test_competitive_ci_gate.py -p no:xdist --run-heavy-tests --run-gui-tests --timeout=1200 --tb=short -q --disable-warnings --no-header --show-capture=no --maxfail=1
then
    echo "[resume-gates] Abbruch: Competitive Gate fehlgeschlagen. Details: ${SUMMARY_FILE}" >&2
    exit 1
fi

echo "[resume-gates] Fertig. Zusammenfassung: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
