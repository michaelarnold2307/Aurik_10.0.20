# Spec-Evidenz: Gate-Haertung und Paritaetsstatus

## Evidenzblock

- Spec-Datei: `.github/specs/00_SPEC_INDEX.md`, `.github/specs/01_musical_goals.md`, `.github/specs/07_quality_and_tests.md`, `.github/specs/13_human_ear_quality.md`, `.github/specs/26_defekt_reparatur_trainingsvertrag.md`
- Abschnitt: Spec 07 §8.1 und Ressourcenprioritaet; Spec 26 DRT-12/DRT-16; Paritaetsabgrenzung in Specs 01/13
- Aenderungstyp: Normative Einordnung, Evidenz- und Freigabevertrag
- Alte Regel: Marker im PR-Text oder in einem Bericht konnten das Evidenz-Gate bestehen, ohne dass Felder ausgefuellt oder Messbelege geprueft wurden.
- Neue Regel: PR-Body und Main-Push-Bericht muessen ausgefuellte Evidenzfelder enthalten; begruendetes N/A ist erlaubt. DRT-16-Paritaetsbehauptungen muessen einen maschinengeprueften JSON-Beleg referenzieren.
- Hypothese: Ein einheitliches, fail-closed Inhaltsgate verhindert, dass Dokumentationsmarker als Qualitaets- oder Paritaetsnachweis missverstanden werden.
- Referenzen/Standards: `AGENTS.md` §4; `docs/guides/SPEC_EVIDENZBLOCK_TEMPLATE.md`; Spec 26 DRT-16; `.github/WETTBEWERBS_PARITAET.md`
- Datensaetze/Szenarien: Historische Declip-Ganz-Titel-Kontrolle und Clipping-Witness-Q-Bericht; keine neue RX-12- oder Hoerpanel-Messung in diesem Umsetzungsschritt.
- Umfang (n): Historischer Clipping-Kontrollbericht: 3 Paare (Seed 42); Clipping-Witness-Bericht vom 2026-11-21: 10 Regionen, davon 0 Q-bewertbar.
- Primarmetrik: Ganz-Titel `musik_mos` und `defektlast`; fuer den aktuellen n=10-Eintrag meldet das Wettbewerbsregister 1/10 bessere Ergebnisse und ΔMOS −0,001.
- Effektstaerke: Der dokumentierte Declip-Kandidat verfehlt das Verbesserungskriterium; der Ganz-Titel-Never-worsen-Nachweis ist ebenfalls nicht bestanden (mos_min −0,474, defektlast_max +0,397).
- 95 %-CI: N/A — die referenzierten Declip-Berichte enthalten fuer diese Ergebnisse kein 95-%-Konfidenzintervall; es wird keines abgeleitet oder behauptet.
- Signifikanztest + p-Wert: N/A — fuer die dokumentierten n=3/n=10 Declip-Befunde ist hier kein p-Wert belegt.
- Seed(s): 42 fuer den historischen Ganz-Titel-Kontrollbericht; kein neuer stochastischer Audiolauf in diesem Prozess-/Dokumentationsschritt.
- Commit: Grundlage der bereits eingecheckten Spec-Aenderungen: `4ac58597`; Proxy-Gate-Aenderungen und dieser Bericht sind Teil des noch offenen Worktrees.
- Skript/Befehl: `python3 -B -m pytest tests/unit/test_spec_evidence_gate.py tests/unit/test_drt_contract_gate.py tests/unit/test_flag_freigabe.py tests/unit/test_aspade_declipper.py::test_flag_default_aus -q --disable-warnings --no-header`; Proxy-Gates mit Frischetest: `python3 -B -m pytest tests/unit/test_studio_oqs_gate.py tests/unit/test_real_audio_restoration_quality_gate.py tests/normative/test_real_audio_restoration_quality_gate.py tests/unit/test_0c_degraded_export_contract.py -q --disable-warnings --no-header`; Wiederholung ohne Frischetest: `python3 -B -m pytest tests/unit/test_studio_oqs_gate.py tests/unit/test_real_audio_restoration_quality_gate.py tests/normative/test_real_audio_restoration_quality_gate.py tests/unit/test_0c_degraded_export_contract.py -k 'not test_daily_gate_stored_recently' -q --disable-warnings --no-header`
- Artefaktpfade: `docs/reports/current/2026-11-15_declip_ganztitel_kontrolle.json`; `docs/reports/current/2026-11-21_witness_q_erstbelege_clipping.json`; `.github/WETTBEWERBS_PARITAET.md`; Testausgabe aus dem oben genannten Befehl
- Risiko fuer P1/P2: Ein strengeres Evidence-Gate kann bisher unvollstaendige PR-Berichte blockieren; das ist beabsichtigt und zeigt fehlende Evidenz statt Klangfehler an.
- Risiko fuer Artefakte: In der Legacy-A-SPADE-Routerspur wird ein Kandidat jetzt nur noch akzeptiert, wenn sein Harmonik-Proxy sowohl gegen den Eingang als auch gegen den aktuellen PCHIP-Amtsinhaber sinkt. Das kann die Übernahme eines Kandidaten verhindern, ist aber kein Ganz-Titel- oder Hörnachweis; `use_aspade_declipper` bleibt AUS.
- Bekannte Unsicherheiten: Keine neue Hörprobe und kein RX-12-Vergleich durchgeführt. Der Declip-Status bleibt `LUECKE`; `use_aspade_declipper` bleibt Default `False`.
- Rollback-Kriterium: Ruecknahme nur, falls der gemeinsame Validator gueltige Evidenzberichte oder DRT-16-Belege faelschlich ablehnt; Marker-only-Pass und unbelegte Paritaetsbehauptungen duerfen nicht wieder zugelassen werden.
- Entscheidung: CONDITIONAL — Prozess-Gate implementiert; keine Freigabe oder Paritaetsbehauptung fuer Declip oder Klangqualitaet.
- Maintainer Sign-off: AUSSTEHEND — die Umsetzung ist angefordert, aber ein separater Qualitaets-/Evidenz-Sign-off wird nicht fingiert.
- Datum: 2026-11-22

## Validierung

- Evidenz-, DRT- und Declip-Freigabetests: 34 bestanden.
- Follow-up 2026-11-22: Der neue Regressionstest reproduzierte vor dem Fix, dass A-SPADE einen besseren PCHIP-Amtsinhaber verdrängen konnte, obwohl der Proxy nur gegenüber dem Eingang sank. Nach dem zusätzlichen Amtsinhaber-Vergleich bestanden die fokussierten Phase-07-/A-SPADE-Suiten mit 80 Tests; Ruff und `git diff --check` bestanden. Keine neue Ganz-Titel-, RX- oder Hörmessung.
- Evidenz-Provenienz 2026-11-22: Der UAT-Report-Generator verwendete fest codierte Versionen `9.10.77`/`10.0.0`; Scorecard, Abschlussbericht und JSON beziehen die Versionsnummer jetzt aus `backend/core/version.py` und brechen bei fehlender/ungültiger Quelle explizit ab. Versions-Regressionstest und Generator-Suite: 8 bestanden; direkte CLI-Ausführung meldet 10.14.0. Das ist Provenienz-Härtung, kein UAT- oder Qualitätsnachweis.
- Zielumgebung 2026-11-22: Die 121 fokussierten Evidenz-, DRT-, Declip- und Report-Generator-Tests bestanden in 12,94 s mit `.venv_aurik` (Python 3.10.12, PyTorch 2.11.0+rocm7.2, ROCm-Gerät Radeon RX 7900 XTX/gfx1100). Der Host hat einen Ryzen 7 3800X, 32 GiB RAM und rund 24 GiB GPU-VRAM. Diese Testausführung belegt Laufzeit-/Vertragskompatibilität, nicht Audioqualität oder Performance.
- Vollständiger UAT-Neulauf: nicht ausgeführt. Zum Prüfzeitpunkt lief bereits ein CPU-intensiver Produktionsdatenlauf; ein paralleler UAT wäre ressourcenbeeinflusst und könnte dessen Fortschritt verfälschen. Die UAT-Frische bleibt offen und ist in einem trainingsfreien Fenster mit `.venv_aurik/bin/python -B` zu erneuern; bis dahin keine Aktualisierung des Daily-Gates und keine Qualitätsbehauptung.
- Declip-Flag AUS und bit-identischer Bestands-Pfad: 2 bestanden (`python3 -B -m pytest tests/unit/test_aspade_declipper.py::test_flag_default_aus tests/unit/test_aspade_declipper.py::test_flag_aus_challenger_nie_gerufen_und_bit_identisch -q --disable-warnings --no-header`).
- Proxy-Gate-Lauf mit Frischepruefung: 23 bestanden, 1 uebersprungen, 1 fehlgeschlagen. Die Frischepruefung meldet die letzte UAT-Analyse vom 2026-11-15 als 7 Tage alt (Grenze: 48 Stunden).
- Wiederholung ohne ausschliesslich diesen veralteten Frischetest: 23 bestanden, 1 uebersprungen, 1 abgewählt.
- Breiter Lauf der geaenderten Testmodule ohne den ausschliesslich frische UAT-Evidenz verlangenden Test:
  ```sh
  AURIK_FORCE_CPU=1 ./.venv_aurik/bin/python -B -m pytest --maxfail=1 -q --disable-warnings --no-header \
    -k 'not test_daily_gate_stored_recently' \
    tests/integration/test_aurik_denker_e2e.py \
    tests/normative/test_parameter_register_gate.py \
    tests/normative/test_real_audio_restoration_quality_gate.py \
    tests/test_e2e_v9_10_41.py \
    tests/test_full_chain_ml_hybrid.py \
    tests/test_optimization_integration.py \
    tests/test_uat_acceptance_criteria.py \
    tests/unit/test_0c_degraded_export_contract.py \
    tests/unit/test_aspade_declipper_plugin.py \
    tests/unit/test_chunked_streaming_layouts.py \
    tests/unit/test_drt_contract_gate.py \
    tests/unit/test_real_audio_restoration_quality_gate.py \
    tests/unit/test_reparatur_phasen_attribution.py \
    tests/unit/test_sota_hearing_modules.py \
    tests/unit/test_studio_oqs_gate.py \
    tests/unit/test_uat_report_generator.py \
    tests/unit/test_witness_detektor.py \
    tests/unit/test_witness_kopf.py \
    tests/unit/test_wohlklang_zeuge.py \
    tests/unit/test_spec_evidence_gate.py
  ```
  Ergebnis: 200 bestanden, 2 uebersprungen, 28 Heavy-Tests durch die Standardkonfiguration abgewählt, 1 Warnung. Damit sind die nicht-schweren Tests gruen; dies ersetzt weder die Heavy-Tests noch den UAT-Neulauf.
- Projektweite Abschlusspruefungen: Ruff ueber geaenderte Python-Dateien, `scripts/compliance_check.py --errors-only`, `scripts/compliance/drt_contract_gate.py`, `scripts/release_must_coverage_check.py`, `scripts/spec_drift_check.py` und `git diff --check` bestanden. Der Evidenzbericht-Validator bleibt ausschliesslich wegen des ausstehenden Maintainer-Sign-offs rot.
- Der Evidenzbericht-Validator meldet ausschliesslich den ausstehenden Maintainer-Sign-off; dieser Fail-closed-Blocker ist beabsichtigt.
- Diese Befunde sind keine Hoerabnahme. Fuer eine gruene Hauptbranch-Evidenzfreigabe muss der Maintainer-Sign-off nach inhaltlicher Pruefung nachgetragen werden.
