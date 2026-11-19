# Hörordnungs-Ausbau — von der Norm zur Wirkung (D-K3-73, D-K3-74)

**Datum:** 2026-11-08
**Klasse:** K3 (Norm/Guard-Wirkung), keine Modell-/Audio-Änderung
**Register:** D-K3-73 (Hörordnung nannte unwirksame Größen — geschlossen),
D-K3-74 (umgebungsabhängiger `test_panns_rocm_provider_filter` — geschlossen)
**Flag-Flips:** **keine.** Keine Klangwirkung ohne §4-Verfahren.

## 1. Befund

Die normative Hör-Spitze (`.github/instructions/hoerordnung.instructions.md`)
nannte in §6/§8 Gate-Größen und psychoakustische Träger, die im Code **nicht
wirkten** — die wiederkehrende Fehlerklasse „Wortlatt statt Wirkung", die in
diesem Repo bereits Kette 4 (Phantom-Verdrahtung) und W-3 (Dauer-Achsen-Defekt)
getroffen hat. Sechs konkrete Lücken:

| # | Lücke | Wirkung vorher |
| --- | --- | --- |
| a | Fluktuationsstärke (vacil) und Loudness-Sprung waren normativ §6-Gate-Größen, wurden aber nicht gemessen | Kriterium existierte nur als Text |
| b | Das Pumpen-Kriterium war eine **Phantom-Norm**: `_run_inviting_gate_measure` bekam keine `reference_audio`-Referenz durchgereicht | delta-basiertes Kriterium strukturell unmöglich |
| c | Ebene 2 verlangt Maskierung „zeitlich UND spektral lokal"; `residual_audibility` maß nur spektral | Forward-Masking unberücksichtigt |
| d | Binaurale Stereo-Abbildung (ITD/ILD/IACC) war Mechanik, fehlte aber in der Ebene-1-Invarianten-Tabelle (§3) | sechste Hör-Invariante unverankert |
| e | Die neuen Ebene-4-Kriterien landeten in `_meta["inviting_gate_song_level"]`, die GUI-Ampel las sie nicht | letzte Meile fehlte |
| f | §8-Träger-Tabelle: unklar, ob die genannten Träger wirklich verdrahtet sind | ungeprüft |

## 2. Umsetzung

**(a) Ebene-4-Kriterien gemessen** — `backend/core/inviting_sound_gate.py`:
`max_vacil` (60-s-Zentrum; 5-s-Fenster lösen 4-Hz-Modulation nicht auf) und
`loudness_jump_max` (`compute_loudness_phon` je Fenster). Beide **delta-basiert**
gegen den Input (Guard-Kalibrierungs-Grundsatz, AGENTS.md §3):
`max_vacil > max_vacil_input × 1.20` **und** `> 0.25 vacil`;
`loudness_jump_max > loudness_jump_max_input + 1.0 phon` (JND-Marge).

**Ehrlich dokumentierter Messbefund:** Der musikalische Grundpuls 2–4 Hz liegt im
Fluktuationsband. Echte Musik (Elke-1977-Referenz) treibt Input **und** Output auf
10.0 vacil — ein absolutes Kriterium wäre ein False-Positive-Desaster gewesen und
wurde deshalb verworfen. Ohne Referenz ist `max_vacil` rein diagnostisch.

**(b) Verdrahtung** — `backend/core/unified_restorer_v3.py`:
`_run_inviting_gate_measure(..., reference_audio=...)`; beide Aufrufstellen
reichen durch (`original_audio_for_goals` im Restore-Pfad, `audio` auf
Song-Ebene). Context-Dict führt `max_vacil`, `max_vacil_input`,
`fluctuation_regression`, `loudness_jump_max*`. Wiring-Guard-Test pinnt das
Durchreichen (Muster C: die Prüfung berührt den entscheidenden Teil).

**(c) Ebene-2-Zeit-Dimension** — `backend/core/dsp/audibility_targets.py`:
Liegt der Residual-Schwerpunkt in einer Forward-Masking-Zone
(`temporal_masking.compute_zones`), wird die effektive Schwelle um bis zu **6 dB**
angehoben (konservativ unter Moore 2003 ≈ 10–20 dB; die bestehende 6-dB-Marge
bleibt). Neue Ergebnisschlüssel `threshold_db_eff`, `temporal_boost_db`,
`in_forward_mask_zone`; Vertrag: Residual und Kontext sind zeitaligniert;
**fail-closed** — Fehler ⇒ `boost = 0`, keine Aufweichung der Schwelle.

**(d) Ebene-1-Audit** — keine Phantom-Verdrahtung gefunden:
`_evaluate_stereo_safety_guard` (Rollback auf HPI-Best/Original),
`level_1_invariants_guard` (Blend-Rücknahme), `wohlklang_ordnung_gate`,
`witness_correction_loop`, `_should_skip_masked_phase`, `pre_echo_detector` —
alle echt aufgerufen. Die binaurale Abbildung ist jetzt als **sechste
Hör-Invariante** in §3 verankert (Warn ab JND 1, Hard-Fail ab JND 2 → Rollback,
delta-basiert gegen den Input).

**(e) GUI-Ampel** — `Aurik10/ui/hearing_gates_summary.py` liest
`inviting_gate_song_level`; delta-basierte Regressionen machen die Ampel
**GELB** (advisory, nie ROT — Guard-Kalibrierungs-Grundsatz). Absolutwerte
(asper/sharpness) erscheinen nur als advisory-Zeile.

**(f) Kalibrierungs-Harness** — `scripts/horordnung_calibration.py` um die
Invarianten 15 (Pumpen-Monotonie + Schwellen-Trennung), 16 (Tonality
Sinus/Rauschen), 17 (Forward-Masking Klick/kein-Klick), 18 (+12-dB-Stufe) und
19 (Temporal-Masking hebt die effektive Schwelle; ohne Klick nicht) erweitert.

**Nebenbefund (§G5 (GEBOTE.md))** — `tests/unit/test_panns_rocm_provider_filter.py`
kippte umgebungsabhängig: der Test prüfte den Provider-Filter, hing aber an
`_gpu_device_present()`, also am torch-Backend des **Test**-Pythons
(System-Python ohne ROCm-torch ⇒ False trotz laufender GPU; Produktions-
`venv_rocm72` liefert True). Fix: `_gpu_device_present` wird deterministisch
gemockt — der Test prüft den Filter, nicht die Präsenz.

## 3. Verifikation

In dieser Sitzung selbst gemessen (System-Python 3.12, pytest 7.4.4):

    tests/unit/test_inviting_sound_gate.py
    tests/unit/test_hearing_gates_summary.py
    tests/normative/test_audibility_targets.py
    tests/unit/test_p0_1_end_gate_chunking.py
    tests/unit/test_panns_rocm_provider_filter.py
    → 67 passed in 6,04 s

    python3 -B scripts/horordnung_calibration.py
    → 19/19 Invarianten erfüllt ("KALIBRIERUNG: alle Invarianten erfüllt")

Von den Vorgänger-Läufen derselben Serie belegt (Handover §2): 120/120
Audibility-Unit-Tests, 89/89 Stereo-Tests, 24/24 Level-1-Guard-Tests,
245/245 PMGG-/Goal-Tests, 30/30 Ampel-Tests, 60 Gate-Tests; voller
`tests/unit`-Lauf 9227 passed / 19 skipped / 1 failed (der eine Fehler war der
oben behobene `test_panns_rocm_provider_filter`).

## 4. Offene Reste (menschlich)

- **Hörpanel-Kalibrierung** der Vorbelegungen: `max_vacil`-Boden 0.25,
  1.20-Faktor und 1.0-phon-Marge tragen den Status
  `vorbelegung_hoerpanel_offen`.
- MUSHRA n ≥ 30 (Hebel D), W-Stufen-Freigaben (Zeile 6, §4-Verfahren),
  Batch-Pfad-Freigabe (§2.46h).
- **Commit-Blocker außerhalb dieser Serie:** `tests/normative/
  test_real_audio_restoration_quality_gate.py::TestDailyGateStatus::
  test_daily_gate_stored_recently` ist rot, weil die UAT-Analyse 10 Tage alt ist
  (Limit 48 h, D-K3-34(d)). Ein `audit/daily_real_audio_gate.py`-Lauf stempelt
  nur `generated_at` neu und wäre ein Formfix — verlangt ist ein **echter**
  `audit/uat_report_generator.py`-Lauf (GPU-gebunden, im trainingsfreien
  Fenster; D-K3-13-UAT).

## 5. Regelkonformität

- Kein Flag-Flip, keine Klangänderung, keine Modell-/Checkpoint-Änderung.
- Keine Hör-/Lizenz-Entscheidung eigenmächtig.
- Determinismus §G5 (copilot-instructions.md): keine Zeitabhängigkeit in der
  Entscheidungslogik; Kalibrierungs-Harness ohne Zufalls-Seeds.
- Laufende GPU-Ketten nicht gestört (Kette 3 + `m9_tf_gridnet` liefen weiter).
