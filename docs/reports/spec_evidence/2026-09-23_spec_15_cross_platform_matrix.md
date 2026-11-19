# Spec-Evidenz (2026-09-23): Cross-Plattform-Matrix — macOS entfernt, Windows 10/11 x64

**Betroffene Spec**: `.github/specs/15_world_class_gap_closure.md` (Ist-Stand,
Wurzelursache, Implementierungsschritte 4.1–4.4, Erfolgskriterien).

## Evidenzblock

**Belege:**

1. **CI-Befund 2026-09-23 (Run 35840059803)**: `actions/setup-python` konnte
   `3.10.12` weder für `macos-14` (arm64) noch für `windows-2022` provisionieren:
   „The version '3.10.12' with architecture 'arm64' was not found for macOS 14.8.9“
   bzw. „… 'x64' … not found for Windows 2022“.
2. **Manifest-Verifikation** (`actions/python-versions/versions-manifest.json`,
   abgerufen 2026-09-23): Der Eintrag `3.10.12` enthält ausschließlich
   darwin-x64- und linux-Builds — **kein arm64- und kein Windows-x64-Build**.
   Die letzte Windows-3.10.x-Version mit x64-Build im Manifest ist `3.10.11`.
3. **Umsetzung**:
   - `ci-cross-platform.yml`: Matrix auf `ubuntu-22.04` (exakt 3.10.12) und
     `windows-2022` (`3.10.11`, dokumentierte Runner-Ausnahme) reduziert;
     macOS-Job entfernt.
   - Windows-10/11-**Zielmaschinen** bleiben über `install_aurik.bat` und
     `scripts/install_windows.ps1` exakt auf **Python 3.10.12 x64** gepinnt
     (neu: 64-Bit-Pflicht, 32-Bit wird abgelehnt).
   - `scripts/platform_compat_check.py`: prüft jetzt zusätzlich exakte
     Python-Version + x64 sowie developer-lokale Absolutpfade (`/media/...`).
   - CI-Checkout mit `lfs: true` (ML-Modelle via Git-LFS).

## Seed

Kein stochastischer Prozess geändert. Alle betroffenen Tests sind
deterministisch (`np.random.default_rng(42)` in Testgeneratoren unverändert);
CI-Gates sind statische Checks ohne Seed.

## 95 %-CI

Lokale Punktmessungen (Python 3.12.3; Abweichung nur im neuen
Versions-Check — CI läuft auf 3.10.12/3.10.11):

- 112 passed (calibration_matrix, phase_53, psychoacoustic_masking — inkl.
  neuer Era-/Beat-Sync-/Residual-Floor-Tests)
- 2292 passed / 1 skipped (Phasenselektion, Rescheduler, §0a-Guard, Tier-1,
  Gap-Fixes)
- 34 passed (Phasen-DSP-Rewritten + Beat-Sync)
- `platform_compat_check.py`: Developer-local paths 0, Path Separators 0,
  Line Endings 0, Case Conflicts 0

Konfidenzintervall der GitHub-Runner-Laufzeit folgt mit dem nächsten grünen
Lauf; die betroffenen Gates sind deterministische statische Prüfungen.

## Maintainer Sign-off

- [x] Michael Arnold (michaelarnold2307), 2026-09-23
