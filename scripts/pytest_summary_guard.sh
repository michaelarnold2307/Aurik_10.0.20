#!/usr/bin/env bash

set -o pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/pytest_summary_guard.sh <pytest args...>"
  echo "Example: scripts/pytest_summary_guard.sh tests/normative/ -q -ra"
  exit 2
fi

tmp_log="$(mktemp -t aurik-pytest-XXXXXX.log)"

cleanup() {
  rm -f "$tmp_log"
}
trap cleanup EXIT

echo "[pytest-guard] Running: ./.venv_aurik/bin/python -m pytest $*"

./.venv_aurik/bin/python -m pytest "$@" 2>&1 | tee "$tmp_log"
pytest_ec=${PIPESTATUS[0]}

summary_line="$(grep -E '^=+ .* in .* =+$' "$tmp_log" | tail -1)"
if [[ -z "$summary_line" ]]; then
  # Pytest -q often emits short summary lines without === wrappers.
  summary_line="$(grep -E '([0-9]+ passed|[0-9]+ failed|[0-9]+ error|[0-9]+ skipped|[0-9]+ deselected).*in ' "$tmp_log" | tail -1)"
fi
if [[ -z "$summary_line" ]]; then
  summary_line="(no pytest summary line found)"
fi

echo
echo "[pytest-guard] Pytest exit code: $pytest_ec"
echo "[pytest-guard] Summary: $summary_line"

skip_lines="$(grep '^SKIPPED \[' "$tmp_log" || true)"
if [[ -n "$skip_lines" ]]; then
  echo "[pytest-guard] Skipped tests:"
  echo "$skip_lines"
fi

if [[ $pytest_ec -eq 0 ]]; then
  echo "[pytest-guard] RESULT: PASS"
else
  echo "[pytest-guard] RESULT: FAIL"
fi

exit $pytest_ec
