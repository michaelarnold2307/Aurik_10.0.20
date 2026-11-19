#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv_aurik/bin/python"
BACKEND_LOG="${ROOT_DIR}/logs/aurik_backend.log"
REPORT_DIR="${ROOT_DIR}/reports/live_guard"
STATE_FILE="${REPORT_DIR}/state.env"
INCIDENT_DIR="${REPORT_DIR}/incidents"

INTERVAL_SEC="${AURIK_LIVE_GUARD_INTERVAL:-8}"
AUTO_REPAIR_ATTEMPTS="${AURIK_LIVE_GUARD_AUTO_REPAIR_ATTEMPTS:-2}"
AUTO_REPAIR_HOOK="${AURIK_LIVE_GUARD_REPAIR_HOOK:-${ROOT_DIR}/scripts/live_guard_auto_fix_hook.sh}"

mkdir -p "${REPORT_DIR}" "${INCIDENT_DIR}" "${ROOT_DIR}/logs"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Missing Python interpreter: ${PYTHON_BIN}" >&2
    exit 1
fi

touch "${BACKEND_LOG}"

if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
fi

LAST_DONE_COUNT="${LAST_DONE_COUNT:-0}"

count_done_runs() {
    grep -c "AurikDenker.denke() abgeschlossen" "${BACKEND_LOG}" || true
}

write_state() {
    cat >"${STATE_FILE}" <<EOF
LAST_DONE_COUNT=${LAST_DONE_COUNT}
EOF
}

run_incident_deep_analysis() {
    local incident_tag="$1"
    local incident_path="${INCIDENT_DIR}/${incident_tag}"
    mkdir -p "${incident_path}"

    {
        echo "timestamp=${incident_tag}"
        echo "reason=runtime_or_compliance_violation"
        echo "backend_log=${BACKEND_LOG}"
    } >"${incident_path}/meta.txt"

    tail -n 500 "${BACKEND_LOG}" >"${incident_path}/backend_tail.log" || true

    set +e
    "${PYTHON_BIN}" audit/runtime_spec_check.py >"${incident_path}/runtime_spec_check.log" 2>&1
    local runtime_rc=$?
    "${PYTHON_BIN}" scripts/compliance_check.py >"${incident_path}/compliance_check.log" 2>&1
    local compliance_rc=$?

    "${PYTHON_BIN}" -m pytest tests/test_uat_acceptance_criteria.py -p no:xdist \
        -k "test_restoration_criteria and (R5 or R6 or R7 or R8 or R9 or R10 or R11 or R12)" \
        --run-heavy-tests \
        --override-ini="addopts=--strict-markers --import-mode=importlib" \
        --timeout=180 --tb=short -q --disable-warnings --no-header \
        >"${incident_path}/uat_r5_r12.log" 2>&1
    local uat_rc=$?
    set -e

    {
        echo "runtime_spec_rc=${runtime_rc}"
        echo "compliance_rc=${compliance_rc}"
        echo "uat_r5_r12_rc=${uat_rc}"
    } >"${incident_path}/summary.env"

    run_incident_auto_repair "${incident_path}"
}

run_incident_auto_repair() {
    local incident_path="$1"
    local repaired=0

    for ((attempt = 1; attempt <= AUTO_REPAIR_ATTEMPTS; attempt++)); do
        local attempt_dir="${incident_path}/attempt_${attempt}"
        local attempt_env_file="${attempt_dir}/repair_env.sh"
        mkdir -p "${attempt_dir}"

        echo "[live-guard] incident auto-repair attempt=${attempt}/${AUTO_REPAIR_ATTEMPTS}" | tee -a "${REPORT_DIR}/live_guard.log"

        set +e
        local hook_rc=0
        if [[ -x "${AUTO_REPAIR_HOOK}" ]]; then
            "${AUTO_REPAIR_HOOK}" "${incident_path}" "${attempt}" >"${attempt_dir}/repair_hook.log" 2>&1
            hook_rc=$?
        fi

        if [[ -f "${attempt_env_file}" ]]; then
            # shellcheck disable=SC1090
            source "${attempt_env_file}"
        fi

        "${PYTHON_BIN}" audit/runtime_spec_check.py >"${attempt_dir}/runtime_spec_check.log" 2>&1
        local runtime_rc=$?

        "${PYTHON_BIN}" scripts/compliance_check.py >"${attempt_dir}/compliance_check.log" 2>&1
        local compliance_rc=$?

        "${PYTHON_BIN}" -m pytest tests/test_uat_acceptance_criteria.py -p no:xdist \
            -k "test_restoration_criteria and R10" \
            --run-heavy-tests \
            --override-ini="addopts=--strict-markers --import-mode=importlib" \
            --timeout=180 --tb=short -q --disable-warnings --no-header -vv \
            >"${attempt_dir}/uat_r10.log" 2>&1
        local r10_rc=$?

        local full_rc=99
        if (( runtime_rc == 0 && compliance_rc == 0 && r10_rc == 0 )); then
            "${PYTHON_BIN}" -m pytest tests/test_uat_acceptance_criteria.py -p no:xdist \
                -k "test_restoration_criteria and (R5 or R6 or R7 or R8 or R9 or R10 or R11 or R12)" \
                --run-heavy-tests \
                --override-ini="addopts=--strict-markers --import-mode=importlib" \
                --timeout=180 --tb=short -q --disable-warnings --no-header \
                >"${attempt_dir}/uat_r5_r12.log" 2>&1
            full_rc=$?
        fi
        set -e

        {
            echo "repair_hook_rc=${hook_rc}"
            echo "runtime_spec_rc=${runtime_rc}"
            echo "compliance_rc=${compliance_rc}"
            echo "uat_r10_rc=${r10_rc}"
            echo "uat_r5_r12_rc=${full_rc}"
        } >"${attempt_dir}/summary.env"

        if (( runtime_rc == 0 && compliance_rc == 0 && r10_rc == 0 && full_rc == 0 )); then
            repaired=1
            echo "[live-guard] incident resolved on attempt=${attempt}" | tee -a "${REPORT_DIR}/live_guard.log"
            break
        fi
    done

    if (( repaired == 1 )); then
        echo "status=recovered" >"${incident_path}/status.env"
    else
        echo "status=degraded" >"${incident_path}/status.env"
    fi

    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/live_guard_report.py" \
        --incident-dir "${INCIDENT_DIR}" \
        --output "${REPORT_DIR}/daily_report.txt" \
        >>"${REPORT_DIR}/live_guard.log" 2>&1 || true
}

echo "[live-guard] started interval=${INTERVAL_SEC}s report_dir=${REPORT_DIR}" | tee -a "${REPORT_DIR}/live_guard.log"

while true; do
    done_count="$(count_done_runs)"

    if [[ "${done_count}" =~ ^[0-9]+$ ]] && (( done_count > LAST_DONE_COUNT )); then
        ts="$(date +"%Y%m%d_%H%M%S")"
        echo "[live-guard] new completed run detected: ${LAST_DONE_COUNT} -> ${done_count} @ ${ts}" | tee -a "${REPORT_DIR}/live_guard.log"

        set +e
        "${PYTHON_BIN}" audit/runtime_spec_check.py >"${REPORT_DIR}/runtime_spec_last.log" 2>&1
        runtime_rc=$?
        "${PYTHON_BIN}" scripts/compliance_check.py >"${REPORT_DIR}/compliance_last.log" 2>&1
        compliance_rc=$?
        set -e

        if (( runtime_rc != 0 || compliance_rc != 0 )); then
            echo "[live-guard] violation detected runtime_rc=${runtime_rc} compliance_rc=${compliance_rc} -> deep analysis" | tee -a "${REPORT_DIR}/live_guard.log"
            run_incident_deep_analysis "${ts}"
        else
            echo "[live-guard] checks OK for latest run" | tee -a "${REPORT_DIR}/live_guard.log"
        fi

        LAST_DONE_COUNT="${done_count}"
        write_state
    fi

    sleep "${INTERVAL_SEC}"
done
