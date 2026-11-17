# Hörordnung — Psychoakustische Wahrheits-Ordnung (normativ)

> **Status:** Normative Spitze für Hör-Entscheidungen. Diese Datei steht in der
> normativen Kette (AGENTS.md §1) **über** `musical_goals.instructions.md`,
> `dsp.instructions.md` und den Goal-Specs — aber **nur** für die Frage, welche
> Größe bei Konflikt den Ausschlag gibt. Die Berechnung der Messgrößen bleibt
> unverändert dort normativ, wo sie definiert ist.
>
> **Grundsatz (vgl. 01_musical_goals.md, Einleitung):** „Der CausalDefectReasoner
> kann irren — das Ohr nicht." Diese Datei operationalisiert diesen Satz.

Arbeitssprache: Deutsch. Alle §-Zitate mit Quelle.

---

## 1. Die Rolle der Messwerte: Zeugen, nicht Richter

Technische Metriken (MUSHRA/NSIM/MCD, HPI, die 15 Musical Goals, VQI,
Goosebumps, HPE) sind **Evidenz**. Kein einzelner Messwert darf allein eine
Entscheidung mit destruktiver Wirkung tragen (Re-Run, Veto, Rollback, Phase-
Strip). Solche Entscheidungen laufen immer über die Ebenen in §2.

Ausnahme: Die Härte-Invarianten der Ebene 1 (§3) sind selbst die oberste
Entscheidungsinstanz — sie sind die operationalisierte Form dessen, was das
Ohr niemals akzeptiert.

**GPU-Backends sind Zeugen-Träger (2026-09-18, §III.9 (copilot-instructions.md)):**
Ein Beschleuniger (ROCm/MIGraphX/CUDA) darf Messwerte nur liefern, wenn er
numerisch gegen den kanonischen ONNX-CPU-Zeugen paritätsverifiziert ist
(rel ≤ 1e-3 auf strukturierten Feeds). Numerisch abweichende Kernels
(6 dokumentierte ORT-ROCm-Defekte, rel 0,19–0,98) hätten sonst als
„Evidenz“ entschieden — das verletzt die Zeugen-Rolle. Deshalb: Torch-ROCm-
Kerne mit Paritäts- und Determinismus-Nachweis (§G5 (copilot-instructions.md)),
ONNX als reiner CPU-Fallback (§V6 (copilot-instructions.md)).

**Gesangs-Gate (2026-09-18):** Gesangsgeführte Bearbeitung (Phonem-DSP,
LyricsGuidedEnhancement) ist eine Never-worsen-Ausnahme: Sie darf nur auf
bestätigtem Gesang wirken. Sprachmodelle halluzinieren auf Instrumental-
Musik (gemessen: 24 Wörter/s — physiologisch unmöglich) — deshalb gilt:
`vocal_presence` (p(no_speech), Token-Konfidenz, Wortdichte ≤ 12 W/s) unter
0,15 → Signal bleibt bit-identisch; schwache Wörter (confidence < 0,30)
werden übersprungen; unsichere Präsenz skaliert die Stärke linear. Das Ohr
entscheidet nie auf der Grundlage halluzinierter „Evidenz“.

## 2. Die vier Ebenen

Jede Ebene dominiert die darunterliegende. Eine Entscheidung auf Ebene n darf
niemals eine Invariante der Ebene < n verletzen.

1. **Hör-Invarianten** — unverhandelbar (siehe §3)
2. **Audibility-Schicht** — nur Hörbares wird repariert (§4)
3. **Wohlklang-Ordnung** — lexikografische Ziel-Hierarchie (§5)
4. **Einladungs-Gate** — positiver Wohlklang-Nachweis (§6)

## 3. Ebene 1 — Hör-Invarianten (Veto-Bereich)

Verletzt eine Phase eine dieser Invarianten, wird die **verursachende Phase
zurückgenommen oder geblendet** — nicht erst am Pipeline-Ende „wiederhergestellt".

| Invariante | Messung (bestehend, normativ dort) | Schwelle |
|---|---|---|
| Stimm-Identität | `singer_identity_cosine` (Spec 01 §2.35c/d) | ≥ 0.92 (kein Unterschreiten durch irgendeine Phase) |
| Konsonanten-/Atem-Energie | `consonant_clarity` (Spec 01 §2.35d-ii) | ≥ 0.85 |
| Vibrato-Erhalt | `vibrato_precision` (Spec 01 §2.35d-i) | Rate-Fehler ≤ 0.3 Hz, Tiefen-Erhalt ≥ 0.85 |
| Dynamikbogen | EmotionalArc (Spec 01 §2.35e) | Arc-Korrelation ≥ Schwelle aus §2.35e-ii |
| Atem-Zeitstruktur | BreathEmotionClassifier-Segmente | Anzahl/Position der Atemer dürfen sich durch NR nicht um > 10 % ändern |
| Stereo-Abbildung (binaural) | ITD/ILD/IACC gegen den Input (delta-basiert) — Messgrößen + JNDs in §8b | Warn ab JND 1 (Log), **Hard-Fail ab JND 2 → Rollback** der verursachenden Phase auf Input/HPI-Best (`STEREO_SAFETY_HARD_FAIL`) |
| Sprach-/Gesangsverständlichkeit (SII-Proxy, L1 aus dem Spec-SOTA-Audit 2026-11-09) | `sii_delta` gegen den Input (delta-basiert, ANSI-S3.5-Band-Wichtung, 18 Bänder 160 Hz–8 kHz), Träger `UnifiedRestorerV3._sii_zeuge_meta` → `metadata["sii_zeuge"]` in Einzel-, Chunked- und Options-Pfad | Verschlechterung > 0.05 SII (Panel-Vorbelegung `vorbelegung_hoerpanel_offen`) ⇒ **Befund (Report-Only, `hoer_instanz=vorzulegen`)**; die Rücknahme-Schwelle (Blend) bleibt Panel-Entscheidung — Harness-Invariante 20 |

**Verboten:** Eine Ebene-1-Verletzung durch „Recovery-Kaskade" zu kaschieren,
wenn die Ursache eine identifizierbare Einzelphase ist (Ursache statt Symptom —
§V7 (copilot-instructions.md)).

**Umsetzung (kanonisch):** `backend/core/dsp/level_1_invariants_guard.py`
(`check_level_1_invariants`/`Level1Result`) misst alle fünf Invarianten nach
jeder Phase; der zentrale Phasen-Hook (`unified_restorer_v3.py`) blendet die
verursachende Phase bei Verletzung Richtung Phasen-Eingang zurück
(`blend_factor < 1.0`). Für die definierenden Härte-Fälle greift dieselbe
Stelle mit harter Phasen-Rücknahme: §SCK-R (Spektralfarben-Korrelation < 0.60)
und §WBG-R (Einzelphasen-Wärmeband-Verlust > 3 dB, Restoration) — Metadatum
`hearing_invariant_retreat`, kumulativer Wärme-Tracker wird zurückgedreht
(Ursache statt Symptom — §V7 (copilot-instructions.md); keine
End-Gate-Kompensation). Die sechste Invariante (Stereo-Abbildung) ist über
`_evaluate_stereo_safety_guard` verdrahtet (Verankerung 2026-11-08):
Warnstufe loggt (`itd_drift_gt_30us` u. a.), Hard-Fail
(`STEREO_SAFETY_HARD_FAIL`) rollt auf `_hpi_best_rollback_audio` bzw. das
Original zurück — delta-basiert gegen den Input (§8b-Regel 1: nur eine
Verschlechterung ist ein Befund; analoge Träger tragen inhärenten
L/R-Versatz).

## 4. Ebene 2 — Audibility: Reparaturziel ist die Maskierungsschwelle

Reparatur gilt als **abgeschlossen**, wenn ein Defekt **unter der psychoakustischen
Maskierungsschwelle** liegt — nicht wenn sein Messwert Null ist.

**Reparatur-Auftrag (§G13 (copilot-instructions.md), Verbesserungs-Pflicht):**
Never-worsen ist der Boden, nicht das Ziel. Für JEDE der 65 DefectTypes gilt
messbar: (a) Rest ≤ Maskierungsschwelle − 6 dB Marge (`objective_met`),
(b) Reparaturquote Q ≥ 0,5 (defekt_delta, `REPARATURQUOTE_ZIEL`),
(c) Kontrollregionen Δ ≈ 0, (d) Frage B „kein hörbares Musikmaterial
entfernt" NIE OK — die Hör-Invarianten (Ebene 1) dominieren jede
Verbesserung. Konsistente Erreichbarkeit über das kanonische Messprotokoll
(Seed 42, ≥ 6 Paare, p25-Pegelmatching, JND 0,05) und die Coverage-Gates
(`scripts/defect_coverage_check.py`): ohne Messung ist eine Klasse offen,
nie „bestanden". Physikalisch irreduzible Reste werden klassifiziert (§G10
(copilot-instructions.md)). Stärke-Änderungen aus Q-Befunden nur über §4.

- **Erzwingbare Instanz (kanonisch):** `backend/core/dsp/audibility_targets.py`
  — `residual_audibility` (Nutzung als `is_audible`) misst mit EINER Messung in
  der `masking_model`-Domäne Hörbarkeit (`audible`), Zielabnahme
  (`objective_met` = `delta_db <= threshold_db − margin_db`, Default-Marge 6 dB)
  und liefert die Dosierung über `repair_strength_for` (Stärke ∝
  Hörbarkeits-Überschuss, geclippt [0.25, 1.0], zentral
  §V7 (copilot-instructions.md)-skaliert). Nachgangsprüfung jeder Reparatur:
  `repair_objective_met`/`verify_declick_repair` — Frage A „kein hörbarer
  Restdefekt?" (Rest unter Ziel), Frage B „kein hörbares Musikmaterial
  entfernt?" (**NIE OK**); Golden-Ear-Referenzkette (`witness_chain`,
  `golden_ear_corpus.reference_pair`); hartes Gate vor Export
  (§0c (copilot-instructions.md): „degraded" statt stiller Verschlechterung).
- **Ohren-Kalibrierung statt Modell-Autorität:** Hörbarkeitsgrenzen werden pro
  Defektklasse am Hörpanel gemessen (Golden-Ear-Korpus
  `backend/core/golden_ear_corpus.py` — `inject_at_margin`-Rückführung,
  `calibration_report`; 2AFC-Hörpanel-Kette `scripts/mushra_harness.py` und
  `scripts/hoerpanel_player.py`) und kalibrieren die Instanz —
  Literatur-Vorbelegungen tragen Status `vorbelegung_hoerpanel_offen`.

- Maßgeblich: Masking-Modell nach ISO 11172-3 (Bark) für den NR-Masking-Guard
  (§2.62 (dsp.instructions.md)); für Residuum-Salience und P1-3-Masking-JND gilt
  seit v10.0.8 die Moore-&-Glasberg-Spreizung auf ERB-Distanz (§2.69d
  (dsp.instructions.md)) — asymmetrisch (Aufwärts-Masking > Abwärts) und
  level-abhängig (SOTA-Upgrade, Kalibrierung §8a: Checks 13/14). Beide zeitlich
  **und** spektral lokal.
- **Temporal-Masking-Integration (2026-11-08):** Die erzwingbare Instanz
  `audibility_targets.residual_audibility` misst jetzt auch die ZEIT-Dimension:
  Liegt der Residual-Schwerpunkt in einer Forward-Masking-Zone des Kontexts
  (`temporal_masking.compute_zones`, Zwicker & Fastl §4.2), wird die effektive
  Schwelle um bis zu **6 dB** angehoben (konservativ unter den Literaturwerten
  ≈10–20 dB, Moore 2003; die bestehende 6-dB-Marge bleibt zusätzlich).
  Vertrag: `residual` und `context` müssen zeitaligniert sein (voll-längige
  Arrays; der Schwerpunkt des Rests trägt die Song-Position). Fail-closed:
  Messfehler ⇒ boost=0 (keine Aufweichung). Ergebnis-Transparenz:
  `threshold_db_eff`/`temporal_boost_db`/`in_forward_mask_zone` im Report.
  Harness-Invariante 19; Wirkungs-Test `test_temporal_masking_kippt_audible_konservativ`.
- **Cross-Ear/BMLD-Bonus (L3 aus dem Spec-SOTA-Audit, 2026-11-09):**
  `residual_audibility` akzeptiert optional ein `stereo_pair` (Rest, Kontext).
  Ein **diotischer** Rest (beide Ohren kohärent, `lowband_corr` 250 Hz–1 kHz)
  wird durch die binaurale Verarbeitung besser maskiert (BMLD bis ~15 dB
  NπS0 vs. N0S0, §8b) — die effektive Schwelle wird konservativ um bis zu
  **6 dB** angehoben (proportional zur Rest-Kohärenz; inkohärenter Rest ⇒
  kein Bonus). Fail-closed: Messfehler ⇒ Bonus 0. Ohne `stereo_pair` bleibt
  das Verhalten unverändert (Rückwärtskompatibilität). Ergebnis-Transparenz:
  `binaural_boost_db`/`bmld_rest_corr`. Harness-Invariante 21; Wirkungs-Test
  `test_cross_ear_bmld_bonus`.
  **Träger-Status (ehrlich, 2026-11-09):** die Fähigkeit ist kalibriert und
  testgedeckt, aber **noch von keinem Produktionspfad übergeben** —
  `repair_objective_met`/`verify_declick_repair` (die einzigen Audibility-
  Verbraucher) arbeiten mono. Eine Aktivierung im Entscheidungspfad ist
  klangwirksam und läuft deshalb ausschließlich über §4/Flag; bis dahin bleibt
  die effektive Schwelle bit-identisch. Kein Phantom-Anspruch einer Wirkung.
- Jede Reparatur-Entscheidung (Phase an/aus, Stärke, Re-Scan) fragt zuerst:
  **„Ist dieser Defekt über der Maskierungsschwelle hörbar?"** Nur dann wird
  repariert.
- Berichte (z. B. §v10.703 Defekt-Countdown, §B2 Per-Defekt-Reduktion) müssen
  „hörbar" als **„über Maskierungsschwelle"** ausweisen. Ein Re-Scan, der nach
  erfolgreicher Reparatur neue „Defekte" unterhalb der Schwelle meldet, darf
  nicht als Misserfolg gezählt werden.
- **Mindestanforderung an PerceptualSalience:** Der Salience-Filter muss real
  maskieren. Läuft er über einen Abschnitt mit einem Anteil `salient ≥ 0.99`
  bei breitbandigem Musikmaterial, ist er als Pass-Through zu behandeln und
  darf keine Audibility-Entscheidung tragen (Befund 2026-08-23: 12969/12969
  salient, mean=1.000).
- **Vollständigkeits-Pflicht (universell, §G10 (copilot-instructions.md)):** Das
  Ziel „unter der Schwelle" gilt für ALLE DefectTypes über den GESAMTEN Song
  jedes beliebigen Imports (§G12 (copilot-instructions.md)) — kein Typ gilt ohne
  Messung als unhörbar (unbewiesen ≠ bestanden). Der Export trägt den
  per-Typ-Audibility-Report (`defect_residual_audibility` ≤ 0 dB je Typ);
  physikalisch irreduzible Reste werden als solche klassifiziert und begründet.
  Erfolgs-/„vollständig"-Meldungen ohne diesen Nachweis sind unzulässig.

## 5. Ebene 3 — Lexikografische Wohlklang-Ordnung

Bei Zielkonflikt gilt die Reihenfolge strikt — ein höherrangiges Ziel darf für
kein niederrangiges gesenkt werden. Der Pareto-Tie-Break nach Hörpriorität
(Spec 01 §2.36) wird von der End-Gate-Ausnahme zur Regel für **alle** Phasen
und für die FeedbackChain-Boosts.

1. **Natürlichkeit** (P1: DNSMOS/SingMOS, HNR, Mikrodynamik — Spec 01 §2.34, §MKK (dsp.instructions.md))
2. **Wärme** (§WBG 200–800 Hz-Balance — §V25 (dsp.instructions.md), waerme P4)
3. **Klarheit / Durchhörbarkeit** (Maskierungs-SMR — transparenz, sep_fidelity)
4. **Brillanz** (bounded durch Material-BW-Ceiling — §6.2c, §2.46a Chain-End-Ceiling)

**Verboten:** Ein Brillanz-/Transparenz-Boost, der Wärme oder Natürlichkeit
senkt — unabhängig davon, ob der Einzel-Score des Boost-Ziels steigt
(Teamwork- statt Dominanz-Prinzip, Spec 01 §1.2c).

Seit v10.1.0 wird Ebene 3 zusätzlich maschinell auditiert: `WohlklangOrdnungGate`
(`backend/core/wohlklang_ordnung_gate.py`) verifiziert die Dominanz-Konformität
von Kandidaten-Entscheidungen auf Basis von `goal_priority_protocol.py` und meldet
Verstöße an die GUI-Ampel (`hearing_gates_summary.py`). Kein neuer Schwellwert,
keine Regeländerung — vgl. §8, Zeile „3“.

## 6. Ebene 4 — Einladungs-Gate (positiver Nachweis)

„Wohlklang, in den sich das Ohr hineinlegt" ist ein **positives** Kriterium und
wird als Fenster-Gate gemessen, nicht als Einzelwert:

- **Zeitverlauf-Gate:** Roughness (Zwicker), Sharpness (Bismarck), Loudness
  (ERB, ISO 532-1) über Fenster (z. B. 5 s, überlappend). Das Gate ist erfüllt,
  wenn keine Roughness-Spitze `asper > 0.5` in Stimmen-/Klimax-Zonen liegt und
  der Sharpness-Verlauf keine Sprünge > 0.2 acum zwischen benachbarten Fenstern
  aufweist. (Die Größen sind bereits als Gewichtungs-Input in Spec 01 §2.56
  Stufe 3 definiert — hier werden sie zum eigenständigen Gate.)
- **Loudness-Sprünge (Zwicker/ISO 532-1, phon) — delta-basiert (2026-11-08):**
  Die Loudness wird je Fenster gemessen (`compute_loudness_phon` je 5-s-Fenster);
  **Zeitvariable Lautheit (ISO 532-2 / DIN-45631-Struktur):**
  `compute_time_varying_loudness` (`zwicker_loudness.py`) — gefensterte
  Kurzzeit-Lautheit + Angriffs-/Abkling-Zeitbewertung + N5/N10-Perzentile —
  ist die normative Referenz für Pegel-Zeitverläufe, wo Fenster-Mittelwerte
  nicht genügen; dokumentierte Abweichungen (deterministisch, keine ad-hoc-
  Steigungs-Interpolation) gelten analog zum stationären Modell.
  Befund ist nur eine **Regression der Sprunggröße gegenüber dem Input**
  (`loudness_jump_max > loudness_jump_max_input + 1.0 phon`, JND-Marge) —
  natürliche Musik-Dynamik erzeugt große Sprünge, die kein Artefakt sind
  (Guard-Kalibrierungs-Grundsatz). Ohne Input-Referenz rein diagnostisch.
  Träger: `inviting_sound_gate.py` (`loudness_jump_max`/`loudness_jump_regression`);
  Harness-Invariante 18 (`scripts/horordnung_calibration.py`).
- **Fluktuationsstärke (Zwicker, vacil) — Pumpen-Gate (2026-11-08):**
  NR-Pumpen/Kompressor-Atmen ist ein langsamer (≈4 Hz), global wahrnehmbarer
  Effekt und wird als Ganzsignal-Größe auf einem 60-s-Zentrum gemessen
  (die Welch-Auflösung der Messung verlangt ≥ 2-s-Segmente — 5-s-Fenster
  lösen 4-Hz-Modulation nicht auf; Messbefund 2026-11-08). **Das Kriterium
  ist delta-basiert** (Guard-Kalibrierungs-Grundsatz: absolute Normalwerte
  sind kein Hard-Fail — der musikalische Grundpuls 2–4 Hz liegt im
  Fluktuationsband und treibt echte Musik bis zum Mess-Clip; Messbefund
  2026-11-08 auf der Elke-1977-Referenz: Input und Output beide 10.0 vacil):
  Pumpen-Befund nur wenn `max_vacil > max_vacil_input × 1.20` UND
  `max_vacil > 0.25` (absoluter Boden; V42-Logik auf Song-Ebene). Ohne
  Input-Referenz ist der vacil-Wert rein diagnostisch. Schwelle = Panel-
  Vorbelegung `vorbelegung_hoerpanel_offen` (Golden-Ear-Konvention §4;
  1 vacil = 100 % AM bei 4 Hz; gemessen: 15 % Pump-Tiefe auf
  Rauschhintergrund = 0.27 vacil, ungepumpt < 0.01). Träger:
  `inviting_sound_gate.py` (`max_vacil`/`fluctuation_regression`,
  `reference_audio`-Parameter); Harness-Invariante 15
  (`scripts/horordnung_calibration.py`).
- **Ermüdungs-Abbruch:** Fatigue (experience_runtime) > 0.40 beendet die
  Optimierung (gilt schon in OneTakeExport; hier generell für End-Gate/FC).
  Die audio-basierte Listening-Fatigue (Spektralbalance/Crest/Mikrodynamik,
  `listening_fatigue_metric.py`) wird seit v10.1.0 komponenten-getrieben
  präventiert: `anti_fatigue_pass.py` korrigiert High-Shelf und
  Mikrodynamik-Expansion mit Do-No-Harm; OneTakeExport nutzt denselben Plan
  statt blindem High-Shelf. Kein neuer Schwellwert, keine Regeländerung.
- **Bezugspunkt:** Die Goosebumps-/HPE-Bewertung bleibt erhalten, aber ein
  „EXZELLENT" ersetzt das Einladungs-Gate nicht — beide müssen unabhängig halten.

## 7. Konfliktregel — die Hör-Instanz entscheidet

Wenn zwei Messgrößen einander widersprechen, gilt:

1. Hält Ebene 1 (alle Hör-Invarianten erfüllt)?
2. Hält eine **Hör-Instanz** (GoosebumpsQualityChecker oder HPE) auf hohem
   Niveau (Goosebumps ≥ 0.75 bzw. HPE stabil hoch)?

Wenn beides ja: Die widersprechende automatische Metrik wird als
**Messartefakt-Verdacht** behandelt. Sie darf dann **keinen** Re-Run, kein
Veto, keinen Rollback und keinen Phase-Strip auslösen; stattdessen wird eine
Warnung geloggt und die Messung ggf. auf einem korrekten Fenster wiederholt
(Bezug: Alignment-Artefakt-Guard der Wohlklang-Garantie, 2026-08-23).

Wenn Ebene 1 verletzt ist, hat die Hör-Instanz **keinen** Vorrang — die
Invariante dominiert.

**Zwei-Ebenen-Zeugen-Pflicht für Aktivierungs-Entscheidungen (2026-11-15,
DRT-5/8/12-Verankerung):** Eine Hör-Entscheidung über die Aktivierung eines
Reparatur-Kanals ist nur entscheidungsreif, wenn BEIDE Zeugen-Ebenen belegt
sind — (a) die Regions-Ebene (Witness-Q je Defektklasse, leak-frei) UND
(b) die Ganz-Titel-Ebene mit beiden Metriken inkl. Kontroll-Lauf ohne den
Kandidaten (musical_goals.instructions.md „Ganz-Titel-Zwei-Metrik-Never-worsen").
Fehlt eine Ebene, ist die Messung nicht entscheidungsreif — die Hör-Instanz
kann die Lücke benennen, aber nicht durch Hören ersetzen (Beleg 2026-11-15:
Jets-vinyl bestand die Regions-Ebene, kollabierte aber auf Ganz-Titel-Ebene
musik_mos −0,551 — erst die zweite Ebene machte die Überrestauration sichtbar).

**Sub-JND-Sperre für Kompositions-Kandidaten (2026-11-17):** Ein
Kompositions-Kandidat der B3-Suche (`scripts/b3_komposition_suche.py`) wird
nur dann für eine Hör-Session empfohlen (`kandidat_empfohlen=true`), wenn die
beste Komposition gegen die Baseline eine **Mindest-JND-Marge auf BEIDEN
Metriken** zeigt — musik_mos ≥ +JND UND defektlast ≤ −JND (JND =
`JND_TOLERANZ` aus `backend/core/dsp/wohlklang_zeuge.py`, EINE Quelle).
Sub-JND-Kandidaten verbrauchen keine Hör-Sitzung der menschlichen Instanz
(Befund 2026-11-11: beste Komposition W-3,W-8 lag sub-JND); stattdessen
Ursachen-Analyse (§V7 (copilot-instructions.md)).

**Zeugen (Report-Only):** `musik_mos` und `defektlast` (zertifiziert,
Spearman 0,95) sowie der **Defekt-Delta-Witness** (`aurik.defektdelta/3`,
§G191 (GEBOTE.md); /3 trägt zusätzlich den gain-invarianten
Sättigungskanal 1/Crest, D-K3-95/97) sind Zeugen, keine Richter: Sie
dokumentieren, wo und wie Reparatur wirkt (Regionen-Deltas,
Reparaturquote Q), lösen aber **keinen**
Re-Run, kein Veto, keinen Rollback und keinen Phase-Strip aus. **Ausnahme
(DRT-10/M1, 2026-11-17):** Der Ganz-Titel-Produktions-Verdikt hat keine reine
Zeugen-Rolle — Überrestauration (defektlast-Gewinn ≥ JND bei musik_mos-Verlust
≥ JND) degradiert **jeden** Lauf (§0c) und löst, sofern die Hör-Instanz nicht
hoch hält (HPI < 0,75 — die §7-Schutzbedingung „Ebene 1 + hohe Hör-Instanz"
greift nur bei HPI ≥ 0,75), den **Identitäts-Rollback** auf den Eingang aus;
bei hoher Hör-Instanz gilt der Messartefakt-Verdacht (kein Rollback). Der
Defekt-Delta-Witness trägt **kein** Richtungszertifikat, solange seine
Regionen-Urteile nicht gegen Hör-Labels kalibriert sind (D-K4-9); bis dahin
darf er B3-Beschlüsse nur **begleiten**, nie eine Bedingung ersetzen.

## 8. Maschinen-Wahrheit — wo die Ebenen leben

Die Ebenen binden Komponenten an ihre Rolle (Berechnung bleibt bei den
jeweiligen Mess-Definitionen; hier zählt nur der Entscheidungsfluss):

| Ebene | Träger |
|---|---|
| 1 | VQI-Gate + singer_identity-Rollback + EmotionalArc (UV3, Spec 01 §2.35c–e), ConsonantClarity; **§SCK-R/§WBG-R** Phasen-Rücknahme im zentralen Phasen-Call (`unified_restorer_v3.py`, Konkretisierung zu V24/V25 in dsp.instructions.md), **Level-1-Guard-Blend** (`level_1_invariants_guard.check_level_1_invariants` → `blend_factor < 1.0`), **Witness-Veto** (`witness_correction_loop.py`: `compute_witness_veto`/`apply_witness_veto` → `global_scalar`/`family_scalars`) |
| 2 | `compute_masking_threshold_iso11172` (dsp §2.62), PerceptualSalience (Pass-Through-Guard), `residuum_masking.py` (3. Blend-Term), `_should_skip_masked_phase` (Stufe B), §v10.703-Countdown (Stufe A: ERB-maskierte Events), **`audibility_targets.residual_audibility`/`repair_objective_met`/`repair_strength_for`** (erzwingbare Hörbarkeits-Instanz, Ziel „Maske − Marge", Default 6 dB; Alias `is_audible`; **Temporal-Masking-Integration 2026-11-08: Forward-Masking-Zonen heben die effektive Schwelle um bis zu 6 dB — Harness-Invariante 19**; **Cross-Ear/BMLD-Bonus 2026-11-09: diotischer Rest hebt die effektive Schwelle um bis zu 6 dB — Harness-Invariante 21**), **`golden_ear_corpus.py`** (Ohren-Kalibrierung: `inject_at_margin`/`calibration_report`; Hörpanel-Status `vorbelegung_hoerpanel_offen`), **`temporal_masking.py`** (Forward-Masking-Zonen, Zwicker & Fastl 1999 §4.2; NR-Stärke-Boost nur in maskierten Nachklang-Fenstern — Harness-Invariante 17), **`pre_echo_detector.py`/`pre_echo_model.py`** (Pre-/Post-Masking-Artefakt-Schutz), **`tonality_gate.py`** (§23-TONALITY: tonales/sauberes Signal ⇒ spectral_denoiser-Passthrough; Harness-Invariante 16) |
| 3 | GoalPriorityProtocol (Spec 01 §2.34) + `HEARING_TIER_MAP`/`hearing_tier()` (Hörordnungs-Dominanzstufe), `goal_weights` (§2.56), PhaseConductor/PMGG; Guards in FeedbackChain (intern + UV3-Callback) und End-Gate-Ranking; `wohlklang_ordnung_gate.py` (`WohlklangOrdnungGate`, maschineller Audit der Dominanz-Konformität, meldet Verstöße an die GUI-Ampel) |
| 4 | `inviting_sound_gate.py` (Fenster-Gate: Roughness/Sharpness + **Fluktuationsstärke `max_vacil` — Pumpen-Gate 2026-11-08** (Harness-Invariante 15) + **Lautheits-Sprünge `loudness_jump_max` — delta-basiert, 1-phon-JND-Marge, Harness-Invariante 18**), experience_runtime (fatigue_index), GoosebumpsQualityChecker, OneTakeExport |
| §7 Konfliktregel | Wohlklang-Garantie-Alignment-Guard, af-false-positive-Handling, MQA-Verdict-Kennzeichnung („Messartefakt-Verdacht“) |

Änderungen an diesen Trägern bleiben der jeweiligen Spec unterworfen; diese
Datei regelt nur den Entscheidungsfluss zwischen ihnen.

## 8a. Kalibrierungs-Wächter

`scripts/horordnung_calibration.py` prüft die psychoakustischen Invarianten der
Hörordnungs-Module gegen synthetische Referenz-Signale mit bekannten
Eigenschaften (Roughness-/Sharpness-/Residuum-Monotonie, Maskierungs-Richtung,
Hörstufen-Konsistenz). Verletzung ⇒ Exit 1.

- Verdrahtet als Pre-Commit-Hook `aurik-horordnung-calibration`
  (läuft bei Änderungen an den psychoakustischen Trägern).
- Neue Schwellwerte oder Modell-Änderungen an Ebene 2/4 MÜSSEN im Harness
  eine Invariante ergänzen — sonst verschiebt sich die Kalibrierung still.
- Das Harness ist die maschinelle Vorstufe der Panel-Kalibrierung (§7, Abschluss
  der echten Hörertests bleibt menschliche Prozessarbeit).

## 8b. Hörfähigkeits-SOTA — interaurale Fähigkeiten des menschlichen Gehörs

„Lückenlos aufs Gehör zugeschnitten“ ist nur ehrlich, wenn die
binauralen Fähigkeiten des Hörsystems maschinell bewertet und geschützt
werden. Kanonische Implementierung: `backend/core/dsp/interaural_cues.py`
(§HRTF/interaural, verdrahtet in `_evaluate_stereo_safety_guard` der UV3).

| Fähigkeit | Messgröße | Hör-JND (Warn-/Hard-Fail) | Modell |
|---|---|---|---|
| Zeitliche Lokalisation | ITD (Kreuzkorrelation, parabolische Sub-Sample-Interpolation, Fenster-Median) | ~30 µs / 60 µs (Ton 500 Hz–1 kHz) | Woodworth (1938)/Kuhn (1977) |
| Pegel-Lokalisation | ILD (mittelbandig, 1/3-Oktaven 315–3150 Hz) | ~1 dB / 2 dB (Mills 1960) | Kopfschatten, Duda & Martens (1998) |
| Räumliche Bündelung | IACC (Maximum der normierten Kreuzkorrelation, ±1 ms) | Δ 0.08 / 0.15 | Standard-IACC-Definition |
| Maskierungs-Freisetzung | BMLD-Proxy (Lowband-Korrelation 250 Hz–1 kHz) | ≥15 dB Freisetzung NπS0 vs. N0S0 | Levitt (1971), Durlach (1972) |
| Richtungs-Ortsgedächtnis | `apply_interaural_cues` (Kopfmodell: Woodworth-ITD + Sphären-ILD + 1/r-Distanz) | deterministische Mono→Stereo-Cue-Platzierung | First-Order-HRTF |

Regeln:

1. **Delta-Guard:** Nur eine Verschlechterung der Cues GEGENÜBER dem Input ist
ein Befund (analoge Träger tragen inhärenten L/R-Versatz).
2. **Layout-Invariante (§V7):** Alle Cue-Messungen normalisieren über
`backend.core.audio_layout` — kein hartes (N,2)/(2,N)-Annehmen; Mono-Kollaps
⇒ maximaler Befund (itd_drift=1e6 µs).
3. **Ehrlichkeits-Klausel:** Das sphärische Kopfmodell ist die deterministische
First-Order-HRTF-Näherung. Personalisierte HRIR-Datensätze (z. B. CIPIC)
sind dokumentierte, aber nicht erzwungene Erweiterung — keine Messdaten-
Behauptung ohne Messdaten.
4. Neue Träger auf Ebene 1/2 MÜSSEN ihre interauralen Invarianten im
`aurik-horordnung-calibration`-Harness ergänzen (§8a).

## 9. Verhältnis zur normativen Kette

- Kein Widerspruch zu den Mess-Definitionen in `musical_goals.instructions.md`,
  `dsp.instructions.md` oder Spec 01 — diese bleiben für die **Berechnung**
  normativ.
- Bei Konflikt zwischen einer Rollen-Regel dieser Datei und einer
  Mess-Definition gilt: Die Messung bleibt wie definiert, die
  **Entscheidungsnutzung** folgt dieser Datei. Der Konflikt ist im
  PR-Evidenzblock zu dokumentieren (AGENTS.md §4).
- Regeländerungen hier ⇒ betroffene Gates/Verifier in Skripten nachziehen
  (AGENTS.md §2, Absatz „Regeländerung ⇒ Skript nachziehen").
