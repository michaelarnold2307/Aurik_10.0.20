# Spec-SOTA-Audit — Spiegeln die Vorgaben das 2026-Maximum für das menschliche Gehör?

**Datum:** 2026-11-09 · **Auftrag:** Audit, ob die Vorgaben/Specs das Maximum an
Restaurierungsqualität widerspiegeln, das 2026 auf Desktop-PCs möglich ist.
**Methode:** Abgleich der normativen Kette (copilot-instructions, Hörordnung,
Specs 01–22, dsp/musical_goals-Instruktionen) gegen (a) den psychoakustischen
Literaturstand (Zwicker/Fastl, Moore/Glasberg, ISO 532-1/2, IEC STI, binaurale
JNDs), (b) die SOTA-Modell-Landschaft 2026 (Demucs/DeepFilterNet/VoiceFixer-
Klasse, DNSMOS/NISQA/PESQ-Klasse, MUSHRA-Protokolle) und (c) die offenen
Zeilen des SOTA-Defizit-Registers. **Alle Aussagen sind am Repo gemessen.**

## Executive Summary

Die Hörordnung ist in den **Kern-Dimensionen auf 2026-SOTA-Niveau**: vier
Entscheidungs-Ebenen mit Veto-Invarianten, zweidimensionale Maskierung
(spektral ISO 11172-3/Moore-Glasberg + zeitlich Forward-Masking), vier
psychoakustische Gate-Größen (Roughness/Sharpness/Loudness/Fluktuationsstärke,
delta-basiert), binaurale JND-Tabelle (ITD/ILD/IACC/BMLD) mit
Hard-Fail-Rollback, 19 Kalibrierungs-Invarianten. **Fünf konkrete Lücken**
trennen den Stand vom erreichbaren Maximum — zwei davon sind autonom
schließbar (GPU-frei), drei sind menschlich/daten-gebunden.

## Bewertungsmatrix

| Dimension | Status | Beleg |
|---|---|---|
| Maskierung spektral (Bark/ERB, level-abhängig, asymmetrisch) | ✅ SOTA | Hörordnung §4, Kalibrierung 13/14 |
| Maskierung zeitlich (Forward-Masking in `residual_audibility`) | ✅ SOTA | Hörordnung §4 (2026-11-08), Invariante 19 |
| Lautheit stationär (ISO 532-1, Zwicker) | ✅ SOTA | `zwicker_loudness.py` |
| Lautheit zeitvariabel (ISO 532-2) | 🟡 näherungsweise | `dynamic_loudness_model.py` (ERB, STL/LTL) — **532-2 nirgends normativ benannt** |
| Rauigkeit/Fluktuationsstärke (Zwicker) | ✅ SOTA | `zwicker_metrics.py`, Gate + V42 |
| Schärfe (Bismarck) | ✅ | `inviting_sound_gate.py` |
| Binaurale Cues (ITD/ILD/IACC/BMLD) + Hard-Fail | ✅ SOTA | §8b, `interaural_cues.py` |
| Personalisierte HRTF (CIPIC/HRIR-Klasse) | 🟡 dokumentiert, nicht erzwungen | §8b Ehrlichkeits-Klausel |
| Sprach-/Gesangs-Verständlichkeit als Invariante (SII/STI) | 🔴 **fehlt** | nur `consonant_clarity ≥ 0.85` (Spec 01 §2.35d-ii) als Proxy |
| Cross-Ear/Binaurale Lautheit in der Audibility-Entscheidung | 🔴 **fehlt** | BMLD-Proxy existiert (§8b), `residual_audibility` ist mono |
| Sprach-MOS-Metriken (DNSMOS/NISQA/PESQ) | ✅ | 152/83/62 Treffer |
| Musik-MOS auf Hör-Labels (Preference-Orakel F15) | 🔴 offen | D-K4-11 |
| MUSHRA-Human-Panel mit Stimuli | 🔴 offen | D-K4-9 (Protokoll fertig, keine Stimuli) |
| Modell-Challenger (M12 WF-V4/TP-V2) | ✅ trainiert, Messung offen | D-K3-85 |
| Wohlklang-Zeuge je Lauf (musik_mos/defektlast) | ✅ neu | D-K3-83 |

## Die fünf Lücken im Detail

### L1 — SII/Verständlichkeits-Invariante (Ebene 1) — autonom schließbar, mittel
`consonant_clarity ≥ 0.85` ist ein Energie-Proxy. Der Speech Intelligibility
Index (ANSI S3.5) bzw. STI (IEC 60268-16) misst die **gewichtete**
Verständlichkeit (Bänder × Modulationstransfer × Hörschwellen) — für
Gesangs-Restaurierung die präzisere Schutzgröße (De-Esser/Denoiser opfern
genau die Bänder, die SII gewichtet). Vorschlag: SII-Delta (Input vs. Output)
als **sechste-plus-eins Invariante** mit Panel-Vorbelegung — Muster C-Test +
Harness-Invariante 20. KEIN neuer Absolut-Schwellwert ohne Panel.

### L2 — ISO 532-2 als normative Referenz benennen — autonom, klein
Das dynamische ERB-Modell (STL/LTL) existiert und ist kalibriert; die Norm
verweist nirgends auf ISO 532-2 (0 Treffer). Eine Doku-Schärfung („zeitvariable
Lautheit entspricht 532-2-Struktur; dokumentierte Abweichungen: …“) macht die
Spezifikation audit-fest ohne Code-Risiko.

### L3 — Cross-Ear Audibility (Ebene 2) — ✅ GESCHLOSSEN (stereo_pair, bis 6 dB BMLD-Bonus)
`residual_audibility` misst mono. Binaurale Freisetzung (BMLD ≥ 15 dB
NπS0 vs. N0S0, §8b) bedeutet: Ein Restdefekt, der in beiden Ohren identisch
ankommt, ist bis zu ~15 dB besser maskiert als einer mit interauraler
Phasen-Differenz. SOTA wäre eine optionale BMLD-Korrektur der effektiven
Schwelle bei Stereo-Eingaben (konservativ, wie die 6-dB-Temporal-Anhebung).

### L4 — MUSHRA-Stimuli (D-K4-9) — menschlich, Daten-gebunden
Protokoll + Builder fertig; es fehlen die Stimuli-Erzeugung und die
Panel-Durchführung. Das ist der Flaschenhals aller Vorbelegungs-Kalibrierungen.

### L5 — Musik-Preference-Orakel F15 (D-K4-11) — menschlich, Labels fehlen
Ohne Hör-Labels kein musik-spezifisches MOS-Orakel; musik_mos/defektlast
(Spearman 0,95) sind der beste verfügbare Proxy (D-K3-81-Befund).

## Priorisierte Empfehlung

1. **L1 + L2 jetzt** (GPU-frei, schließt die letzten autonomen Norm-Lücken).
2. **L3 als nächster Eingriff** nach L1 (konservativ, Harness-gepinnt).
3. **L4/L5 menschlich** — Voraussetzung für die Panel-Kalibrierung aller
   Vorbelegungen und die endgültige Freigabe der M12-Modelle.
4. **B3-Neumessung M12 + UAT-Evidenz** im freien GPU-Fenster (vorbereitet).

## Umsetzungs-Stand (2026-11-09)

- **L1 ✅:** `backend/core/dsp/speech_intelligibility.py` (SII-Proxy, ANSI-S3.5-
  Band-Wichtung, `sii_delta`), §3-Tabellenzeile (Report-Only, Panel-Vorbelegung),
  Harness-Invariante 20, 9 Vertragstests. **Nachtrag (Träger-Audit, D-K3-86):**
  der erste Stand hatte **null** Aufrufstellen — der Befund wäre in keinem Lauf
  entstanden. Jetzt Träger `UnifiedRestorerV3._sii_zeuge_meta` →
  `metadata["sii_zeuge"]` in Einzel-, Chunked- und Options-Pfad (REPORT-ONLY,
  3 Schreibstellen, keine Lesestelle); Messfenster zeitlich gemittelt (vorher
  EIN 42-ms-Fenster ⇒ bei Stille-/Fade-Anfang blind).
  Eingangs-Trägerprüfung `eingang_brauchbar`: `None`/Strings/zu kurze Puffer ⇒
  `verfuegbar=False` + Warnung statt stillem „Delta 0“.
- **L2 ✅:** Hörordnung §6 — zeitvariable Lautheit als ISO-532-2/DIN-45631-
  Struktur normativ verankert.
- **L3 🟡 (Fähigkeit geschlossen, Träger offen):** `residual_audibility(stereo_pair=…)`
  — diotischer Rest hebt die effektive Schwelle um bis zu 6 dB (konservativ
  unter der 15-dB-BMLD-Literatur), Harness-Invariante 21, normativer
  Wirkungs-Test. **Ehrlich (D-K3-86):** kein Produktionspfad übergibt
  `stereo_pair` (`repair_objective_met`/`verify_declick_repair` raveln mono) —
  die Aktivierung ist klangwirksam und läuft über §4/Flag, nicht autonom.
  Verifiziert: Harness 21/21, normative Audibility 11/11, unit-Kern 42/42.

## Fazit

Die Vorgaben spiegeln das 2026-Maximum **bis auf L4/L5** wider — L1/L2 sind
vollständig geschlossen (Norm + Träger), L3 ist als Fähigkeit geschlossen und
als Entscheidungspfad-Aktivierung ehrlich offen (§4/Flag). Verbleibend: L4/L5
(menschlich) und die GPU-gebundenen Läufe (B3-Neumessung M12 nach D-K3-85,
UAT-Evidenz) — beide vorbereitet. Kein Befund zeigt eine Regression des
erreichten Niveaus.
