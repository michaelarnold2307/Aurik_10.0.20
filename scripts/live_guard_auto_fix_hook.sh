#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCIDENT_PATH="${1:-}"
ATTEMPT="${2:-1}"
ATTEMPT_DIR="${INCIDENT_PATH}/attempt_${ATTEMPT}"
ENV_OUT="${ATTEMPT_DIR}/repair_env.sh"
DIAG_OUT="${ATTEMPT_DIR}/repair_diag.log"

mkdir -p "${ATTEMPT_DIR}"

echo "[hook] incident=${INCIDENT_PATH} attempt=${ATTEMPT}"

auto_detect_mert_timeout=0
if [[ -f "${INCIDENT_PATH}/uat_r5_r12.log" ]]; then
    if grep -q "Timeout (>180.0s)" "${INCIDENT_PATH}/uat_r5_r12.log" && grep -q "mert_plugin.py" "${INCIDENT_PATH}/uat_r5_r12.log"; then
        auto_detect_mert_timeout=1
    fi
fi

{
    echo "timestamp=$(date -Iseconds)"
    echo "attempt=${ATTEMPT}"
    echo "detected_mert_timeout=${auto_detect_mert_timeout}"
    echo "cwd=${ROOT_DIR}"
    echo "python=$(command -v "${ROOT_DIR}/.venv_aurik/bin/python" || true)"
    echo "meminfo:";
    grep -E "MemTotal|MemAvailable|SwapTotal|SwapFree" /proc/meminfo || true
    echo "top_python:";
    ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pmem | grep -E "python|pytest" | head -n 20 || true
} >"${DIAG_OUT}"

# Record potentially overlapping UAT runs for diagnostics only.
# Never terminate external processes from this hook.
if pgrep -fa "pytest.*test_uat_acceptance_criteria" >/dev/null 2>&1; then
    pgrep -fa "pytest.*test_uat_acceptance_criteria" >"${ATTEMPT_DIR}/observed_uat_pids.log" || true
fi

# Provide per-attempt environment caps for deterministic CPU behavior.
{
    echo "export OPENBLAS_NUM_THREADS=1"
    echo "export OMP_NUM_THREADS=1"
    echo "export MKL_NUM_THREADS=1"
    echo "export VECLIB_MAXIMUM_THREADS=1"
    echo "export NUMEXPR_NUM_THREADS=1"
    # Safe validation profile: keep validation deterministic and avoid
    # heavyweight model init in automated incident-repair attempts.
    echo "export AURIK_SAFE_VALIDATION_PROFILE=1"
    # Keep heavy model init deterministic in automated repair attempts.
    if [[ "${auto_detect_mert_timeout}" == "1" ]]; then
        echo "export AURIK_LIVE_GUARD_DETECTED_MERT_TIMEOUT=1"
    fi
} >"${ENV_OUT}"

echo "[hook] wrote ${ENV_OUT} and ${DIAG_OUT}"
