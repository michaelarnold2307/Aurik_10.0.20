# CI/CD Pipeline Documentation

> **Stand:** 2026-09-22 · Aurik 10.1.0
> Die historischen Workflows `ci_enhanced.yml` / `ci.yml` / `release.yml` /
> `validate_musical_goals.yml` wurden ersetzt. Maßgeblich sind ausschließlich
> die vier unten genannten Workflows.

## 📋 Overview

Aurik nutzt GitHub Actions für Continuous Integration, Qualitäts-Gates und die
Release-Freigabe. Alle Workflows liegen in `.github/workflows/`.

### Available Workflows

| Workflow | Trigger | Purpose | Status |
| --- | --- | --- | --- |
| `ci-lite.yml` | Push/PR | CI Lite: 11 Gates (Lint, Type, Determinismus, GUI-Smoke, Coverage u. a.) | ✅ Active |
| `ci-cross-platform.yml` | Push (main) / manuell | Ubuntu/Windows/macOS-Matrix (§15.4) | ✅ Active |
| `nightly-quality.yml` | Cron 02:00 / manuell | AMRB + Drift + normative Nacht-Gates | ✅ Active |
| `solo-release-gate.yml` | Push (main) / manuell | Spec-Evidenz + Heavy-Tests (CLI/GUI-Parität) | ✅ Active |

---

## 🔄 CI Lite (`ci-lite.yml`)

Der Hauptworkflow für Push/PR. Jobs:

- **pr-evidence-gate** (PR-only): Evidenzblock-Pflicht bei Spec-Änderungen
  (`## Evidenzblock`, `Seed`, `95 %-CI`, `Maintainer Sign-off`) +
  TASK_CHANGES.md-Abdeckung (`scripts/change_ledger.py check`).
- **snyk-guard**: Security-Scan (`scripts/snyk_guard.py`).
- **determinism-gate**: `test_full_pipeline_determinism` + P2-Audit.
- **export-guard**: Audio-/Metadaten-Export-Vertrag.
- **normative-guard**: `test_no_production_stubs` + VERBOTEN-Compliance.
- **gui-smoke-gate**: Offscreen-PyQt-Suite (`QT_QPA_PLATFORM=offscreen`).
- **evaluation-gate**: Evaluations-Selbsttest + objektives Gate.
- **scope-lint-gate**: pylint Scope-Bugs in `backend/core/phases`.
- **type-gate**: mypy 2.1.0 (`backend/core backend/api plugins Aurik10 cli`).
- **lint-gate**: ruff 0.16.7 (`F821,F601,B009,I001`) — Version über
  `required-version` in der pyproject erzwungen.
- **coverage-gate**: `scripts/release_must_coverage_check.py` (RELEASE_MUST).

## 🔄 Cross-Platform (`ci-cross-platform.yml`)

Matrix Ubuntu 22.04 / Windows 2022 / macOS 14 (ARM64), Python 3.10.

- `platform-compat`-Job: statischer Kompat-Check (hartkodierte Windows-Pfade,
  CRLF, Case-Konflikte).
- Test-Job je OS: exakte Produktions-Pins (numpy 1.26.4, scipy 1.15.3,
  soundfile 0.13.1, librosa 0.11.0, **numba 0.64.0**, pytest 9.0.2);
  Tests ohne `NUMBA_DISABLE_JIT` (Produktionskonfiguration); voller Log mit
  Faulthandler, bei Rot Upload als Artefakt. Scope:
  `tests/unit` + `test_no_production_stubs`, Marker `not slow/gpu/onnx`.

## 🔄 Nightly Quality (`nightly-quality.yml`)

Nächtliche Gates (Cron 02:00): Spec-Evidenz der letzten 24 h, AMRB,
Drift-Checks (`spec_drift_check.py`) und normative Gates.

## 🔄 Solo Release Gate (`solo-release-gate.yml`)

Der maßgebliche Release-Check: Spec-Evidenz-Artefakt auf main +
normative Heavy-Tests inkl. CLI/GUI-Parität (echte Vollpipeline, 60-min-Limit).

---

## 🛠️ Lokale Gates (Pre-Commit)

Vor jedem Commit laufen die Hooks aus `.pre-commit-config.yaml` (fail-closed):
`aurik-compliance`, `aurik-verboten-linter`, `gebote-verifier`,
`aurik-bug-prevention`, ruff (kritischer Scope + Format), `aurik-id-registry`,
`aurik-file-lifecycle`, `aurik-symbol-duplicates`,
`aurik-horordnung-calibration`, `aurik-code-weakness`,
`aurik-spec-integration`, `aurik-unit-smoke` u. a.

Manuell, CI-exakt:

```bash
ruff check --select F821,F601,B009,I001 backend/ plugins/ denker/ Aurik10/ cli/ scripts/
ruff check backend/ plugins/ denker/ Aurik10/ cli/ scripts/
ruff format --check backend/ plugins/ denker/ Aurik10/ cli/ scripts/
mypy --config-file pyproject.toml backend/core backend/api plugins Aurik10 cli
python -m pytest tests/unit -m "not slow and not gpu and not onnx" --maxfail=3
```

Die ruff-Version wird über `required-version` in der pyproject erzwungen —
lokal, im Hook und in der CI gilt dieselbe Version.

---

## 🚀 Release-Ablauf

1. **Konsistenz prüfen:** `python scripts/check_version_consistency.py`
   (pyproject/README/CHANGELOG müssen übereinstimmen).
2. **Gates:** CI Lite + Solo Release Gate grün; Cross-Platform mindestens
   Ubuntu/macOS grün, Windows-Befunde dokumentiert.
3. **Tag & Push:** Tag auf `main` setzen und pushen; Solo Release Gate läuft
   auf main.
4. **Release:** GitHub Release mit CHANGELOG-Abschnitt; Installer-Artefakte
   (`install_aurik.sh` / `install_aurik.bat`) beilegen.
5. **Modell-Bereitstellung:** Der Installer muss die Gewichts-Bereitstellung
   für `models/` dokumentieren (ohne Modelle läuft Aurik auf DSP-Pfaden).

## 📊 Status-Badges

```markdown
[![CI Lite](https://github.com/michaelarnold2307/Aurik_10.0.20/actions/workflows/ci-lite.yml/badge.svg)](https://github.com/michaelarnold2307/Aurik_10.0.20/actions/workflows/ci-lite.yml)
[![Solo Release Gate](https://github.com/michaelarnold2307/Aurik_10.0.20/actions/workflows/solo-release-gate.yml/badge.svg)](https://github.com/michaelarnold2307/Aurik_10.0.20/actions/workflows/solo-release-gate.yml)
```

---

## ⚠️ Bekannte Einschränkungen

- **Windows:** numba/LLVM-Absturz (SIGSEGV, Exit 139) unter Beobachtung —
  Diagnose über das hochgeladene Test-Log-Artefakt des Cross-Platform-Jobs.
- **GPU/ONNX-Klassen** laufen mangels GPU-Runner nur lokal (Spec §15.4 plant
  ROCm auf Ubuntu; auf GitHub-hosted Runnern nicht verfügbar).
