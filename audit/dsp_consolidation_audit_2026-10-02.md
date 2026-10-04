# DSP-Konsolidierungs-Audit — `backend/core/dsp` (2026-10-02)

> SOTA-/Produktions-Audit des konsolidierten DSP-Pakets nach Zusammenlegung des
> ehemaligen Root-Paketes `dsp/` in `backend/core/dsp/`. Normative Grundlage:
> AGENTS.md §3 (Schnell-Referenz), `.github/copilot-instructions.md`,
> `.github/VERBOTEN.md`. Vorab-Audit mit historischem Bezugspunkt:
> `audit/non_sota_code_audit_2026-04-15.md`.

## 1. Inventar

| Größe | Wert | --- | --- |
| Python-Dateien in `backend/core/dsp/` (rekursiv) | **376** |
| Top-Level-Module ohne `_`-Präfix (Discovery-Relevanz) | **367** |
| Subpakete | `aurik_deesser_pro/`, `optimized/` |
| Module mit deklariertem `DSPContract` | **121** |

Subpakete:

- `aurik_deesser_pro/` — eigenständige ML-Pipeline (MaskNet/Music-Vocal) mit eigener
  `Dockerfile.aurik_deesser_pro`; keine externen Python-Referenzen → als
  Docker-bereites Standalone-Unit zu behandeln, nicht als Teil der statischen
  Registry-Oberfläche.
- `optimized/` — Cython/numexpr-Beschleunigungshelfer (`cython_loops.pyx`,
  `fft_cache.py`, `numexpr_ops.py`); einziger externer Konsument:
  `scripts/benchmark_dsp.py`.

## 2. Import-Gesundheit (Produktions-Discovery)

Methode: Simulation der Eager-Discovery von `dsp_module_registry.DSPModuleRegistry`
(Import aller Top-Level-Module als `backend.core.dsp.<mod>`; dasselbe Muster nutzt
`unified_restorer_v3._compute_dsp_registry_result()` pro Restauration).

| Lauf | TOTAL | AVAILABLE | WITH_CONTRACT | FAILED | --- |--- | --- | --- | --- |
| Vor Fix (2026-10-02) | 367 | 364 | 121 | **3** |
| Nach Fix (2026-10-02) | 367 | **367** | 121 | **0** |

Die drei Fehlschläge waren **vorbestehend** (CWD-/Docker-Kopplung), keine
Migrations-Regressions — die betroffenen Dateien waren bei der Konsolidierung als
reine Renames übergegangen (`git diff --stat` → `| 0`, keine Zeilenänderung):

1. `optimize_dsp_chain.py` — `FileNotFoundError: config/config_dsp_chain_example.yaml`
   (CWD-relativer Pfad; existiert nur aus Repo-Wurzel).
2. `update_yaml_with_dsp_modules.py` — `KeyError: 'dsp_modules'`; die Zielfile mit dem
   erwarteten Schema ist `tests/sota_dsp_module_list.yaml` (Top-Level-Key
   `dsp_modules`, Felder `modul_file`/`modul_class`). `tests/sota/sota_dsp_module_list.yaml`
   ist ein Platzhalter mit anderem Schema (`modules:`) und wird von keinem Code referenziert.
3. `waveunet_infer.py` — `ImportError: Config`; importiert das Docker-Volume
   `/workspace/Wave-U-Net/Config` ungeschützt (nur im Deesser-Docker-Container nutzbar).

Angewendete Fixes (§V7 (copilot-instructions.md): Ursache statt Symptom, keine
CWD-Annahme mehr):

- `optimize_dsp_chain.py`: `_REPO_ROOT = Path(__file__).resolve().parents[3]`;
  `AUDIT_DIR` und `POLICY_PATH` auf Repo-Wurzel verankert.
- `update_yaml_with_dsp_modules.py`: `YAML_PATH` → `tests/sota_dsp_module_list.yaml`
  (korrektes Schema), `DSP_PATH` → eigenes Verzeichnis (`__file__`).
- `waveunet_infer.py`: Docker-only-Import in `try/except ImportError` mit
  `logger.warning(...)` (§V6 (copilot-instructions.md): Silent-Failure-Verbot);
  Aufruf außerhalb des Containers wirft explizites `RuntimeError` in `main()`.

Nebenwirkung (beabsichtigt, dokumentiert): Der Import von
`update_yaml_with_dsp_modules.py` schreibt die YAML zurück und hat dabei zwei
bisherige `null`-Einträge in `tests/sota_dsp_module_list.yaml` aufgelöst:
`Adaptive Comb Filter → adaptive_comb_filter.py/AdaptiveCombFilter`,
`Harmonic Enhancer → vocal_presence_enhancer.py/HarmonicEnhancer`.

Bekanntes Restverhalten (nicht Teil dieses Fixes): Beide Dev-Skripte haben
modulseitige Effekte (Laden/Schreiben bei Import) — veraltetes Design, aber
vorbestehend und für die Discovery jetzt schmerzfrei. `optimize_dsp_chain.py` globt
`*_audit.json` nicht-rekursiv aus der Repo-Wurzel; die tatsächlichen Dateien liegen in
`reports/` bzw. `audit/`, d. h. es werden i. d. R. keine Audit-JSONs gesammelt
(vorbestehend).

## 3. Produktionsfläche

Distincte kanonische Modulnamen (`backend.core.dsp.<mod>`) mit statischer Referenz:

| Scope | Distinct Module |
| --- | --- |
| Striktes Produktions-Runtime (`backend/ plugins/ denker/ Aurik10/ cli/ forensics/`, exkl. `_vendor_*`) | **179** |
| + `policy/` (Policy-Katalog) | **180** |
| Alle gescannten Verzeichnisse (+ `scripts/ tests/`) | 238 |

Top-Consumer (Referenzanzahl, alle Scope):

```text
66 psychoacoustics        33 hallucination_guard
59 lpc_formant_tracker    31 noise_texture_guard
57 formant_system         30 hnr_guard
55 spectral_color_guard   29 transient_guard
52 onset_guard            29 dtw_groove
41 temporal_masking       28 warmth_guard
40 audibility_gate        28 vibrato_guard
35 mikrodynamik_guard
```

Katalog-Interpretation: Der Policy-Katalog (`policy/dsp_policy_engine.py`,
`backend/core/song_calibration_profile.py`) referenziert Guard-Namen **nur als
Entscheidungs-Metadaten** — die Ausführung-Disposition erfolgt über
`backend/core/regulator/_dsp_applier.py`. Ein Katalogeintrag allein ist also kein
Produktions-Import.

## 4. Orphan-/Dead-Code-Analyse

Methode (zweistufig, Daten in `/tmp/opencode/classify.txt`, `pc_check.txt`,
`true_orphans.txt`):

1. **Stamm-Scan**: Alle 367 Top-Level-Module gegen absolute Referenzen
   `backend.core.dsp.<stamm>` außerhalb `dsp/`. Ergebnis:
   `INTERNAL_REL=3`, `REF_ONLY=139`, `ORPHAN=52` (kein absoluter Stamm-Import).
2. **PascalCase-Klassen-Check** für die 52 Orphans: Referenzen an den exportierten
   Klassennamen außerhalb `dsp/`.

Ergebnis:

| Klasse | Anzahl | Bedeutung | |--- | --- | --- |
| Echte Orphans (kein Stamm-, kein Klassenreferenz) | **24** | Dead-Code-Kandidaten |
| Katalog-only (1–2 einzelne Klassenreferenzen, nur in `policy/dsp_policy_engine.py` / `backend/core/song_calibration_profile.py`) | **28** | Entscheidungs-Metadaten ohne Live-Ausführung |

Besonderfälle unter den 24:

- `optimize_dsp_chain`, `update_yaml_with_dsp_modules` — bewusste Dev-/Wartungs-Utilities
  (kein Produktions-Import beabsichtigt) → **aus der Cleanup-Kandidatenliste ausklammern**.
- `sota_mdx23c_separator` — **superseded**: MDX23C wird über ONNX-Config
  (`config/config_model_selection.yaml:17`, `config/docker-compose.yml`) und KIM-Plugins
  (`plugins/kim_vocal_enhancer_plugin.py`, `plugins/kim_music_enhancer_plugin.py`) geladen;
  der Python-Wrapper ist Legacy.

Echte Orphans (24, vollständige Liste; `*` = bewusste Dev-/Wartungs-Utility):

```text
adaptive_ar_prediction_burg        ki_inpainting
adaptive_ar_prediction_levinson    modulation_deesser
adaptive_crosscorrelation          noise_burst_remover
adaptive_per_band_snr              optimize_dsp_chain *
adaptive_rms_energy                reverb_amount_estimator
adaptive_zero_crossing             rlp_decision
binaural_enhancer                  safety_net
confidence_engine                  shellac_mono_strategy
dccrn_client                       shellac_quality_report_template
dialog_intelligibility             sota_mdx23c_separator
envelope_matching                  truepeak_detector
update_yaml_with_dsp_modules *     vibrato_continuity_guard
```

→ **22 Cleanup-Kandidaten** (ohne die zwei Dev-Utilities `optimize_dsp_chain` und
`update_yaml_with_dsp_modules`).

## 5. Subpaket-/Extern-Referenzen

- `aurik_deesser_pro/`: keine externen Python-Referenzen → Docker-Standalone.
- `optimized/`: einziger externer Konsument `scripts/benchmark_dsp.py`.
- MDX23C: siehe §4 (ONNX + KIM-Plugins, kein Wrapper-Import).

## 6. Verifikations-Evidenz (alle Läufe am 2026-10-02 unter `.venv_aurik`)

| Gate | Ergebnis ||--- | --- |
| `python -m py_compile` der drei fixierten Dev-Skripte | OK |
| Discovery-Simulation (`/tmp/opencode/dsp_discovery_audit.py`) nach Fix | **367/367, FAILED=0** (mit §V6-Warnung von waveunet_infer) |
| ruff-critical `--select F821,F601,B009,I001` auf den drei fixierten Skripten (ruff 0.16.7) | All checks passed, EXIT=0 |
| `scripts/aurik_verboten_linter.py --ci --errors-only backend/core/dsp/` | no errors, EXIT=0 |
| `scripts/compliance_check.py --errors-only` (vollständiger CI-Lauf) | bestanden, **1316 Dateien**, EXIT=0 |
| Psychoakustik-Suite | 58 passed |
| Unified-Restorer / v99 / ML-Fallback-Suite | 685 passed (nur vorbestehende librosa-Warnung) |
| `tests/normative/test_no_production_stubs.py` | 7 passed |
| Spectral-Denoiser / v95 Dynamic-Import-Suite | 179 passed, 14 deselected |
| `compileall backend/core/dsp/` (vor Fix-Runde) | EXIT=0 |

## 7. In dieser Aufgabe geänderte Dateien

| Datei | Änderung || --- | --- |
| `Setupfile/build_deb.sh` | Root-`dsp` aus Kopier-Schleife + Kommentar entfernt (`bash -n` OK; altes generiertes Baum-Verzeichnis durch `.gitignore:203` abgedeckt) |
| `backend/core/dsp/optimize_dsp_chain.py` | `_REPO_ROOT = Path(__file__).resolve().parents[3]`; `AUDIT_DIR`, `POLICY_PATH` verankert |
| `backend/core/dsp/update_yaml_with_dsp_modules.py` | `YAML_PATH` → `tests/sota_dsp_module_list.yaml` (korrektes Schema); `DSP_PATH` → eigenes Verzeichnis |
| `backend/core/dsp/waveunet_infer.py` | Docker-only-Import geschützt, §V6-Warnung, expliziter `RuntimeError` in `main()` |
| `tests/sota_dsp_module_list.yaml` | Write-back-Nebenwirkung: 2 Einträge von `null` auf echte Module aufgelöst (siehe §2) |

## 8. Follow-up-Aufgaben (noch offen)

1. **Dead-Code-Cleanup** der 22 Kandidaten aus §4 (eigene PR, mit Registry-Evidenz
   via `scripts/repo_graph.py --duplicates` und Pre-Commit-Gates). Bewusst hier
   nur dokumentiert, nicht ausgeführt — Scope-Trennung zur Konsolidierung.
2. **Registry-Import-Kosten**: Die Eager-Discovery importiert bei jeder Restauration
   alle 367 Module; Orphans zahlen Importkosten ohne Nutzen (kein
   Korrektheitsproblem). Optional: Discovery-Whitelist oder Lazy-Import — nur als
   Performance-Follow-up, nicht normativ.
3. **Dev-Skript-Hygiene** (optional): modulseitige Effekte in `optimize_dsp_chain.py` /
   `update_yaml_with_dsp_modules.py` nach `main()` verschieben; nicht-rekursiver
   Audit-JSON-Glob von `optimize_dsp_chain.py` auf `reports/`+`audit/` ausweiten.

## 9. Bekannte Grenzen der Analyse

- Statisches Referenz-Matching auf `backend.core.dsp.<mod>`: dynamisch
  konstruierte Modulnamen (z. B. f-Strings, die den Namen zur Laufzeit bilden)
  können unterzählt werden; die Discovery-Simulation (§2) deckt diesen Fall ab, da sie
   die tatsächliche Import-Auflösung repliziert.
- Katalog-only-Klassifikation beruht auf der Evidenz, dass die einzigen
  Klassenreferenzen in den beiden Policy-/Kalibrierungsdateien liegen; eine spätere
  Ausführung-Disposition über neue `_dsp_applier`-Pfade müsste die Liste erneut prüfen.
