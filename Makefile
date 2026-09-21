# =============================================================================
# Makefile — Aurik 9 Code-Qualitäts-Befehle (SOTA, März 2026)
# =============================================================================
# Verwendung:
#   make fmt          — Alle Dateien auto-formatieren (Black + isort + autoflake)
#   make lint         — Alle Linter ausführen (Ruff + Flake8)
#   make typecheck    — Mypy Typprüfung
#   make quality      — Vollständiger Qualitäts-Check
#   make test         — Unit-Tests ausführen
#   make clean        — Temporäre Dateien aufräumen
# =============================================================================

PYTHON := .venv_aurik/bin/python
PYMODULE := $(PYTHON) -m
PYTEST_CLEAN := ./scripts/pytest_clean.sh

# Robuste Terminal-Capability-Defaults (wichtig fuer Snap/VS-Code-Subprozesse)
TERM ?= xterm-256color
TERMINFO ?= /usr/share/terminfo
TERMINFO_DIRS ?= /usr/share/terminfo:/lib/terminfo:/etc/terminfo
export TERM TERMINFO TERMINFO_DIRS

# Verzeichnisse für Code-Qualität
SRC_DIRS := dsp plugins backend denker Aurik10 cli/aurik_cli.py
EXCLUDE_DIRS := models .venv_aurik build dist output_audio sessions logs __pycache__

# Farben für Terminal-Ausgabe
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
RESET := \033[0m

.PHONY: help fmt lint typecheck quality compliance compliance-full test test-clean test-spec-gate test-spec-gate-core test-amrb-gate test-competitive-gate test-release-gui clean pre-commit-install \
        black isort autoflake ruff flake8 pylint mypy bandit

# ---------------------------------------------------------------------------
# HILFE
# ---------------------------------------------------------------------------
help:
	@echo "$(GREEN)Aurik 9 — Code-Qualitäts-Befehle$(RESET)"
	@echo ""
	@echo "  $(YELLOW)make fmt$(RESET)              — Auto-Formatierung (Black + isort + autoflake)"
	@echo "  $(YELLOW)make lint$(RESET)             — Linting (Ruff + Flake8)"
	@echo "  $(YELLOW)make typecheck$(RESET)        — Statische Typprüfung (mypy)"
	@echo "  $(YELLOW)make quality$(RESET)          — Vollständiger Qualitäts-Check"
	@echo "  $(YELLOW)make test$(RESET)             — Unit-Tests"
	@echo "  $(YELLOW)make test-clean$(RESET)       — Unit-Tests via clean pytest wrapper"
	@echo "  $(YELLOW)make test-spec-gate$(RESET)   — Integration+Normative ungefiltert (inkl. heavy/gui)"
	@echo "  $(YELLOW)make test-spec-gate-core$(RESET) — Spec-Gate ohne AMRB-Langläufer"
	@echo "  $(YELLOW)make test-amrb-gate$(RESET)   — AMRB-Langläufer separat"
	@echo "  $(YELLOW)make test-competitive-gate$(RESET) — Competitive-Langläufer separat"
	@echo "  $(YELLOW)make test-all$(RESET)         — Alle Tests inkl. Integration"
	@echo "  $(YELLOW)make clean$(RESET)            — Temporäre Dateien löschen"
	@echo "  $(YELLOW)make pre-commit-install$(RESET) — Pre-commit-Hooks installieren"
	@echo "  $(YELLOW)make compliance$(RESET)       — VERBOTEN-Regeln prüfen (errors only)"
	@echo "  $(YELLOW)make compliance-full$(RESET)  — VERBOTEN + Warnings (vollständiger Scan)"
	@echo ""

# ---------------------------------------------------------------------------
# FORMATIERUNG
# ---------------------------------------------------------------------------
fmt: black isort autoflake
	@echo "$(GREEN)✅ Formatierung abgeschlossen$(RESET)"

black:
	@echo "$(YELLOW)⚙ Black: Code-Formatierung...$(RESET)"
	$(PYMODULE) black \
		--line-length=120 \
		--exclude="models/|\.venv_aurik/|build/|dist/" \
		$(SRC_DIRS) tests benchmarks scripts

isort:
	@echo "$(YELLOW)⚙ isort: Import-Sortierung...$(RESET)"
	$(PYMODULE) isort \
		--profile=black \
		--line-length=120 \
		$(SRC_DIRS) tests

autoflake:
	@echo "$(YELLOW)⚙ autoflake: Ungenutzte Imports entfernen...$(RESET)"
	$(PYMODULE) autoflake \
		--in-place \
		--recursive \
		--remove-all-unused-imports \
		--remove-unused-variables \
		--ignore-init-module-imports \
		$(SRC_DIRS)

pyupgrade:
	@echo "$(YELLOW)⚙ pyupgrade: Python 3.10+ Modernisierung...$(RESET)"
	find $(SRC_DIRS) -name "*.py" -not -path "*/__pycache__/*" \
		-exec $(PYMODULE) pyupgrade --py310-plus {} \;

# ---------------------------------------------------------------------------
# LINTING
# ---------------------------------------------------------------------------
lint: ruff flake8
	@echo "$(GREEN)✅ Linting abgeschlossen$(RESET)"

ruff:
	@echo "$(YELLOW)⚙ Ruff: Schnell-Linting...$(RESET)"
	$(PYMODULE) ruff check \
		--fix \
		--respect-gitignore \
		$(SRC_DIRS) tests || true

flake8:
	@echo "$(YELLOW)⚙ Flake8: PEP-8-Prüfung...$(RESET)"
	$(PYMODULE) flake8 \
		--config=.flake8 \
		$(SRC_DIRS) || true

pylint:
	@echo "$(YELLOW)⚙ Pylint: Tiefenanalyse...$(RESET)"
	$(PYMODULE) pylint \
		--jobs=4 \
		--score=yes \
		$(SRC_DIRS) 2>&1 | tee reports/pylint_report.txt || true

# ---------------------------------------------------------------------------
# TYPPRÜFUNG
# ---------------------------------------------------------------------------
typecheck: mypy
	@echo "$(GREEN)✅ Typprüfung abgeschlossen$(RESET)"

mypy:
	@echo "$(YELLOW)⚙ mypy: Statische Typprüfung...$(RESET)"
	$(PYMODULE) mypy \
		backend dsp plugins denker \
		--ignore-missing-imports \
		--show-error-codes \
		2>&1 | tee reports/mypy_report.txt || true

# ---------------------------------------------------------------------------
# SICHERHEIT
# ---------------------------------------------------------------------------
security:
	@echo "$(YELLOW)⚙ Bandit: Sicherheits-Scan...$(RESET)"
	$(PYMODULE) bandit \
		-r core dsp plugins backend denker \
		-l -l \
		--skip B101,B311,B603,B607 \
		-f json \
		-o reports/bandit_report.json || true
	@echo "$(GREEN)✅ Sicherheits-Scan abgeschlossen$(RESET)"

# ---------------------------------------------------------------------------
# VOLLSTÄNDIGER QUALITÄTS-CHECK
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# VERBOTEN-REGELN COMPLIANCE CHECK
# ---------------------------------------------------------------------------
compliance:
	@echo "$(YELLOW)⚙ Aurik Compliance-Check (VERBOTEN-Regeln)...$(RESET)"
	$(PYTHON) scripts/compliance_check.py --errors-only
	@echo "$(GREEN)✅ Compliance-Check bestanden$(RESET)"

compliance-full:
	@echo "$(YELLOW)⚙ Aurik Compliance-Check (inkl. Warnings + f-strings)...$(RESET)"
	$(PYTHON) scripts/compliance_check.py --fix-fstrings

quality:
	@mkdir -p reports
	@echo "$(GREEN)═══════════════════════════════════════$(RESET)"
	@echo "$(GREEN) Aurik 9 — Code-Qualitäts-Prüfung$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════$(RESET)"
	@$(MAKE) compliance
	@$(MAKE) fmt
	@$(MAKE) lint
	@$(MAKE) typecheck
	@$(MAKE) security
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════$(RESET)"
	@echo "$(GREEN) ✅ Qualitäts-Prüfung abgeschlossen$(RESET)"
	@echo "$(GREEN) Reports in: reports/$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════$(RESET)"

# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------
test:
	@echo "$(YELLOW)⚙ Unit-Tests...$(RESET)"
	$(PYTEST_CLEAN) tests/unit \
		-p no:xdist \
		--override-ini="addopts=--strict-markers --import-mode=importlib" \
		--timeout=30 --tb=short -q --disable-warnings --no-header

test-clean: test

test-spec-gate:
	@echo "$(YELLOW)⚙ Full Spec Gate (integration + normative, ungefiltert)...$(RESET)"
	QT_QPA_PLATFORM=offscreen $(PYTEST_CLEAN) tests/integration tests/normative \
		-p no:xdist \
		--run-heavy-tests --run-gui-tests \
		--override-ini="addopts=--strict-markers --import-mode=importlib" \
		--timeout=90 --tb=line -q --disable-warnings --no-header --maxfail=5

test-spec-gate-core:
	@echo "$(YELLOW)⚙ Spec Gate Core (ohne AMRB/Competitive-Langläufer)...$(RESET)"
	QT_QPA_PLATFORM=offscreen $(PYTEST_CLEAN) tests/integration tests/normative \
		-p no:xdist \
		--run-heavy-tests --run-gui-tests \
		-m "not amrb and not competitive" \
		--override-ini="addopts=--strict-markers --import-mode=importlib" \
		--timeout=90 --tb=line -q --disable-warnings --no-header --maxfail=5

test-amrb-gate:
	@echo "$(YELLOW)⚙ AMRB Gate (langlaufend, separat)...$(RESET)"
	QT_QPA_PLATFORM=offscreen $(PYTEST_CLEAN) tests/normative/test_amrb_ci_gate.py \
		-p no:xdist \
		--run-heavy-tests --run-gui-tests \
		--timeout=600 --tb=short -q --disable-warnings --no-header --maxfail=1

test-competitive-gate:
	@echo "$(YELLOW)⚙ Competitive Gate (langlaufend, separat)...$(RESET)"
	AURIK_COMPETITIVE_BENCHMARK_TIMEOUT_S=180 QT_QPA_PLATFORM=offscreen $(PYTEST_CLEAN) tests/normative/test_competitive_ci_gate.py \
		-p no:xdist \
		--run-heavy-tests --run-gui-tests \
		--timeout=1200 --tb=short -q --disable-warnings --no-header --maxfail=1

test-release-gui:
	@echo "$(YELLOW)⚙ Release-GUI-Gate (SIGTERM/Notfall-Checkpoint & GUI-Invarianten, headless)...$(RESET)"
	QT_QPA_PLATFORM=offscreen $(PYTEST_CLEAN) \
		tests/normative/test_stability_invariants.py \
		tests/normative/test_e2e_gui_smoke.py \
		tests/normative/test_kmv_stufe2.py \
		-p no:xdist \
		-m gui \
		--run-heavy-tests --run-gui-tests \
		--timeout=120 --tb=short -q --disable-warnings --no-header --maxfail=5

test-all:
	@echo "$(YELLOW)⚙ Alle Tests (2 Worker)...$(RESET)"
	$(PYTEST_CLEAN) tests \
		-n 2 --dist=loadfile \
		--override-ini="addopts=--strict-markers --import-mode=importlib" \
		--timeout=60 --tb=short -q --disable-warnings --no-header --maxfail=5

test-coverage:
	@echo "$(YELLOW)⚙ Tests mit Coverage...$(RESET)"
	$(PYTEST_CLEAN) tests/unit \
		--cov=core --cov=dsp --cov=plugins --cov=backend --cov=denker \
		--cov-report=html:reports/coverage \
		--cov-report=term-missing \
		--cov-fail-under=70 \
		-q

# ---------------------------------------------------------------------------
# PRE-COMMIT
# ---------------------------------------------------------------------------
pre-commit-install:
	@echo "$(YELLOW)⚙ Pre-commit-Hooks installieren...$(RESET)"
	$(PYMODULE) pre_commit install
	$(PYMODULE) pre_commit install --hook-type commit-msg
	@echo "$(GREEN)✅ Pre-commit-Hooks installiert$(RESET)"

pre-commit-run:
	@echo "$(YELLOW)⚙ Pre-commit auf alle Dateien anwenden...$(RESET)"
	$(PYMODULE) pre_commit run --all-files

pre-commit-update:
	@echo "$(YELLOW)⚙ Pre-commit-Hooks aktualisieren...$(RESET)"
	$(PYMODULE) pre_commit autoupdate

# ---------------------------------------------------------------------------
# AUFRÄUMEN
# ---------------------------------------------------------------------------
clean:
	@echo "$(YELLOW)⚙ Temporäre Dateien löschen...$(RESET)"
	find . -type d -name "__pycache__" -not -path "./.venv_aurik/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv_aurik/*" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -not -path "./.venv_aurik/*" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
	@echo "$(GREEN)✅ Aufräumen abgeschlossen$(RESET)"

# ---------------------------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------------------------
info:
	@echo "$(GREEN)Aurik 9 — Tool-Versionen$(RESET)"
	@$(PYTHON) -m black --version
	@$(PYTHON) -m isort --version-number
	@$(PYTHON) -m flake8 --version
	@$(PYTHON) -m ruff --version
	@$(PYTHON) -m mypy --version
	@$(PYTHON) -m pylint --version | head -1
	@$(PYTHON) -m pytest --version
	@echo ""
	@echo "Python:" $(shell $(PYTHON) --version)
