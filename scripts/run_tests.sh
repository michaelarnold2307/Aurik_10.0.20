#!/usr/bin/env bash
# Aurik Test-Wrapper — repo-relativ, venv-kanonisch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv_aurik/bin/python"

if [[ ! -x "$PYTHON" ]]; then
	echo "FEHLER: venv-Python nicht gefunden: $PYTHON" >&2
	echo "Bitte zuerst: bash scripts/install_aurik.sh" >&2
	exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" -m pytest tests \
	-p no:xdist \
	--override-ini="addopts=--strict-markers --import-mode=importlib" \
	--maxfail=1 \
	--disable-warnings \
	--no-header \
	--tb=short "$@"
