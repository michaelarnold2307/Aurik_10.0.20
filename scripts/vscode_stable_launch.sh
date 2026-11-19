#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${1:-/media/michael/Software 4TB/Aurik_Standalone}"

# Stabiler Start: schwere Hintergrunddienste und Analyse-Extensions deaktivieren,
# damit VS Code Insiders nicht in den OOM-Kill läuft.
code \
  --disable-extension continue.continue \
  --disable-extension jeroenv.github-copilot-with-context \
  --disable-extension github.copilot \
  --disable-extension github.copilot-chat \
  --disable-extension hbenl.vscode-test-explorer \
  --disable-extension github.vscode-github-actions \
  --disable-extension ms-python.autopep8 \
  --disable-extension ms-python.vscode-python-envs \
  --disable-extension ms-python.pylint \
  --disable-extension ms-python.flake8 \
  --disable-extension ms-python.mypy-type-checker \
  --disable-extension ms-vscode.cpptools \
  --disable-extension snyk-security.snyk-vulnerability-scanner \
  "$WORKSPACE_DIR"
