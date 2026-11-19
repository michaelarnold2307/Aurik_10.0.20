#!/usr/bin/env bash
set -euo pipefail

# Startet pytest mit stabilem Warning-Filter fuer die bekannte Trio-Excepthook-Meldung.
# Alle Argumente werden unveraendert an pytest weitergereicht.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv_aurik/bin/python"
SAFE_RUNNER_BIN="${ROOT_DIR}/run_tests_safe.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Fehler: Python-Interpreter nicht gefunden: ${PYTHON_BIN}" >&2
  exit 1
fi

# VS-Code-Crash-Schutz:
# In VS-Code-Terminals laufen Langtests standardmäßig isoliert in eigener cgroup,
# damit OOM/Signal den Editor nicht mitreißen.
_IN_VSCODE_TERMINAL=0
if [[ "${TERM_PROGRAM:-}" == "vscode" || -n "${VSCODE_PID:-}" || -n "${VSCODE_IPC_HOOK_CLI:-}" ]]; then
  _IN_VSCODE_TERMINAL=1
fi

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

_USE_SAFE_RUNNER="${AURIK_PYTEST_CLEAN_USE_SAFE:-1}"
if [[ "${_USE_SAFE_RUNNER}" != "0" && -x "${SAFE_RUNNER_BIN}" ]]; then
  if [[ "${_IN_VSCODE_TERMINAL}" -eq 1 ]] || _is_heavy_invocation "$@"; then
  export PYTHONWARNINGS="ignore::RuntimeWarning:trio._core._multierror"
  exec "${SAFE_RUNNER_BIN}" "$@"
  fi
fi

exec "${PYTHON_BIN}" -W "ignore::RuntimeWarning:trio._core._multierror" -m pytest "$@"
