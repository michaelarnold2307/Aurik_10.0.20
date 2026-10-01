# Lückenschluss-Gesamtplan — Aurik zum ultimativen Restaurierungsprogramm für das menschliche Gehör

> **Status:** Lebender Gesamtplan (2026-09-28) · konsolidiert ALLE bekannten
> offenen Lücken aus der normativen Kette und den Gap-Analysen an einer Stelle.
> **Zielbild:** Maximaler Wohlklang und unhörbare Defekte für das menschliche
> Gehör bei ALLEN Importsongs — ohne erforderliche Nutzereingriffe (0-Config).
> **Normative Verankerung:** `.github/instructions/hoerordnung.instructions.md`
> (Audibility = Maskierungsschwelle statt Mess-Null; lexikografische
> Wohlklang-Ordnung; Metriken sind Zeugen), `.github/copilot-instructions.md`
> (§0c degraded-Export, §G5 Determinismus, §V6 Silent-Failure-Verbot, §V7
> Workaround-Verbot, §V8 Song-Isolation), Spec 06 §7.2d (Messungs-Programm),
> `policy/scope_guardrails.yaml` (Neue-Phasen-Evidenzvertrag).
>
> **Quellen (Detail-Tiefe):** `docs/TODOS_SOTA_ROADMAP.md` (SOTA-Maßnahmen,
> Modell-Portfolio, Performance-Katalog PERF-R1–R17, Resthebel-Matrix),
> `docs/PHASE_SOTA_GAP_ANALYSE.md` (Ausbaustufen je Domäne, §SR-Audit,
> L3-Schließungen, Guardrail-Paket, Wohlklang-Optimum),
> `docs/WITNESS_SOTA_GAP_ANALYSE.md` (Witness-SOTA-Inventar),
> `docs/AURIK_V10_ERKENNTNISSE.md`, `docs/ANALYSE_VOKAL_MUSIK_NACH_RESTAURATION.md`,
> `docs/GESAMTKONZEPT_PERFORMANCE_WOHLKLANG.md`,
> `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md` (beratend).

---

## 1. Erfolgsdefinition und Messkriterien

1. **Unhörbarkeit:** Nach der Restaurierung liegt KEIN Defekt mehr über der
   Maskierungsschwelle des menschlichen Gehörs (ERB-bandweise nach ISO 389-7,
   Johnston-Spreading). „Mess-Null" ist kein Ziel — sub-audible Reste sind
   normativ zulässig (Hörordnung §4; Praxisbeispiele dieser Session:
   `flutter_spectral_sidebands` 1,7 cents, `scrape_flutter` 0,86 cents —
   beide < Frequenz-JND ≈ 3,4 cents, Klumpp & Eady 1956).
2. **Maximaler Wohlklang:** lexikografische Wohlklang-Ordnung der Hörordnung
   (Ebene 1 Hör-Invarianten → Ebene 2 Audibility → Ebene 3 Wohlklang); HPE
   (`human_pleasantness_estimator`) ist die gemeinsame Zielfunktion
   (Zwicker-Schärfe, Rauigkeit, Lautheit, Tonalität, Fluktuationstärke).
3. **0-Config:** Jeder Importsong durchläuft Voranalyse → Restaurierung →
   Export ohne Nutzereingriff; bei fehlgeschlagenem Quality-Gate wird das
   bestmögliche sichere Ergebnis mit Status „degraded" exportiert (§0c) —
   Hardstop ohne Datei ist unzulässig.
4. **Akzeptanz je Lücke:** Never-worsen (delta-basiert, §G5-Determinismus),
   §V6-Fallback mit Warnung, HPI-Verbesserung ≥ 0,02 (Neue-Phasen-Vertrag),
   JND-Kalibrierung statt absoluter Schwellen.

---

## 2. Stand dieser Session-Reihe (2026-09-27/28) — bereits geschlossen

| Block | Inhalt | Beleg |
|---|---|---|
| A1 | 3 Phasen-Exceptions (phase_06/61/19) — Reproducer: RUN statt SKIP | Commit 520bbf5a |
| A2 | Hör-Instanz konsolidiert (kanonische Maskierungs-Instanz + UV3-Wiring, 9 Tests) | §483 PHASE_SOTA_GAP_ANALYSE |
| A3 | phase_12-Familie auf Musik: wow 37,2→0,0 cents, multiband 16,5→0,0 cents, sidebands sub-audibel | §329 ff. |
| A4 | Scanner-Lücken Tape-/Dropout-Familie — alle 9 Fälle sev ≥ 0,15 | §374 |
| A5 | Restliche Scanner-Lücken: hiss 0,559, clipping 0,225, proximity 0,287, pre_echo 0,637, dyn_comp 0,620, compression 0,426, quantization 0,545, transport_bump 0,803, **flutter 0,587** (Magnituden-Gate + Rauschboden-normalisierte Kohärenz), **stylus 0,372** (Gleichtakt + Peak-Norm), **IGD 1,0** (Summton-Messung + Harmonik-Pruning); 2 sub-audibel SKIP; Harness-Rerun: **0 Scanner-Lücken** | §404 + heute |
| A6a | speed_calibration_error: Mapper phase_31-first + DSP-Tuning-Fallback (polyphon, 12-TET, IQR-Gate) + direktes Polyphasen-Resampling (1024/round(1024·ratio)) + Harness-Cents-Metrik: **7,4→0,14 cents** | heute |

---

## 3. Katalog aller bekannten offenen Lücken

### 3.1 L3-Phasen-Wirksamkeit — gemessene Reparatur-Lücken (A6b–A6f)

Quelle: §SR-Audit-Läufe (`repair_effectiveness_harness.py`), Musik-Träger.

**Vinyl/wow_flutter-Familie — die fünf gemessenen Fälle sind GESCHLOSSEN (2026-09-28):**

| Defekt | Phase | Ergebnis (phys vorher→nachher) |
|---|---|---|
| inner_groove_distortion | phase_60 | **0,01199→0,00357 ✓** (Band 400 Hz–8 kHz + Positionsgewicht 0,65 + Steigung 0,85 + Clamp-Entfernung begründet) |
| groove_echo | phase_61 | **0,0778→0,0334 ✓** (globaler Vorläufer-Pfad: RPM-Lag-Überschuss + g-Leiter + Energie-Gate) |
| riaa_curve_error | phase_04 | **0,4894→0,1655 ✓** (adaptive De-Emphasis + Enum-feste Material-Normalisierung — 2 Produktions-Bugs dabei gefunden) |
| stylus_damage | phase_09 | **0,8591→0,2812 ✓** (Skewness-Dekompression der positiven Flanke) |
| speed_calibration_error | phase_31 | **7,40→0,14 cents ✓** (DSP-Tuning-Fallback + direktes Polyphasen-Resampling) |

**Tape-/Dropout-Familie (A4-validiert, Reparatur → Restkatalog unten):**

| Fall | Phase | Befund |
|---|---|---|
| dropout_splice | phase_64 | sev 0,201 erreicht, phys-Wirkung fehlt noch |
| amplitude_drift | phase_40 | sev 1,000 erreicht, phys-Wirkung fehlt noch |
| azimuth_error | phase_25 | sev 1,000 erreicht, phys-Wirkung fehlt noch |
| tape_head_clog | phase_56 | sev 0,192 erreicht, phys-Wirkung fehlt noch |
| dolby_nr_mismatch | phase_54 | sev 0,567 erreicht, phys-Wirkung fehlt noch |
| nr_breathing | phase_54 | sev 0,600 erreicht, phys-Wirkung fehlt noch |
| print_through | (LMS-Pfad) | phys 0,182→0,181 marginal |
| dropouts | phase_24 | sev 0,168→0,028 (wirkt; Metrik-Kalibrierung folgt) |

**Gesamt-Liste der ursprünglich gemessenen 29 Phasen-Lücken** (Priorisierung je
Fall nach dem §7.4c-Muster: Root-Cause → Fix → phys-Metrik-Gate):
phase_12-Familie (✅ A3), phase_24 (dropout/dropout_oxide/sticky_shed),
phase_64 (tape_splice), phase_03/05/49/14/56/59/63/65, motor_interference,
bias_error, riaa_curve_error (✅ A6d), head_wear, mpeg_frame_loss,
jitter_artifacts, overload_distortion, pitch_drift, reverb_excess,
room_mode_resonance, tape_head_level_dip, vocal_harshness,
digital_artifacts, clicks (52→37 Kanten reichen nicht fürs Gate),
speed_calibration (✅ A6a), inner_groove (✅ A6b), groove_echo (✅ A6c),
stylus (✅ A6e).

**A6f-Stand (Gesamtlauf 2026-09-28): 19 OK · 43 Phasen-Lücken · 0
Scanner-Lücken · 3 SKIP.** Restkatalog der 43 Fälle (nächste Sessions,
je Fall nach dem §7.4c-Muster): dropout/dropout_oxide/dropout_head_contact/
sticky_shed (phase_24), dropout_splice (phase_64), distortion (phase_07),
clicks (phase_01), hiss (phase_03 — sev steigt 0,559→0,939!),
high_freq_noise (phase_03), low_freq_rumble (phase_05), modulation_noise
(phase_59), azimuth_error (phase_25), bias_error + dolby_nr_mismatch
(phase_04), head_wear + tape_head_clog (phase_56), hf_remanence_loss +
bandwidth_loss (phase_06), print_through (phase_57), nr_breathing +
tape_head_level_dip (phase_54), pre_echo (phase_23), transport_bump
(phase_12), amplitude_drift (phase_40), pitch_drift (phase_31),
jitter_artifacts + mpeg_frame_loss + digital_artifacts (phase_23),
overload_distortion (phase_09), vocal_harshness (phase_65),
intermodulation_distortion (phase_63), phase_issues + phase_rotation
(phase_14), quantization_noise (phase_03), proximity_effect_excess
(phase_04), reverb_excess (phase_49), room_mode_resonance (phase_04),
clipping (phase_07), sibilance (phase_19).

### 3.2 Witness-SOTA-Lücken (A7)

Quelle: `docs/WITNESS_SOTA_GAP_ANALYSE.md`. Der Witness hat 9
Delta-Metriken + P1–P4 (Maskierung, Rauigkeit, Räumlich, Pre-Echo) ✅.
Noch offen (im Inventar als „fehlt" markiert):

| Lücke | Schritt |
|---|---|
| **Sibilanz-Härte** (5–8 kHz Rauigkeit/Resonanz) | Band-weise Vassilakis-Verfeinerung von P2; Finding `sibilance_harshness` gegen Hör-JNDs |
| **Muddiness** (250-Hz-LF-Verdeckung) | Maskierungs-Analyse im 250-Hz-Band als eigener Witness-Kanal (Aufbau wie P1) |
| **Zeitvariante Maskierung** | P4 ist nur ein Pre-Echo-Proxy — ein echtes Forward-Masking-Modell (postmaskierte Delta-Energie hinter Onsets) als Verfeinerung |
| Rauigkeits-Verfeinerung | P2 band-weise statt globaler Hüllkurven-Fluktuation |

**Bewusst NICHT (bleibt ausgeschlossen):** PEAQ/POLQA/MOS-Prädiktoren,
gelernte Qualitäts-Prädiktoren im Gate-Loop (Rollenbruch/Determinismus —
siehe Fazit der Witness-Analyse).

### 3.3 Phasen-SOTA-Ausbaustufen je Domäne (A8)

Quelle: `docs/PHASE_SOTA_GAP_ANALYSE.md` §1/§6 + TODO-Liste.

| Stufe | Status | Schritt |
|---|---|---|
| CR-V1 (BANQUET-Klick-Detektion) | ✅ 36b452b4 | — |
| HU-V1 (Kalman-Hum-Drift) | ✅ a4428e50 | — |
| WF-V1/V2/V3 (Resampling, Log-f-Zentroid, Kalman) | ✅ ba151793/86a10e10 | — |
| WF-CASS (Scrape-Flutter-Rest) | ✅ | — |
| **WF-V4 (neuraler Warp-Schätzer)** | 🔴 BLOCKIERT | Checkpoint-Quelle klären (2025/26-Modell) + Download + Verdrahtung mit Never-worsen-Gate gegen phase_12-Warp — eigene Session |
| IN-V1/V2 (Inpainting-Naht-Gates) | ✅ 09124e12 | — |
| **HR-V1 (BigVGAN-Repair)** | 🟡 verdrahtet, Aktivierung aus | F3-GPU-Befund abwarten (Aktivierungsvertrag fail-closed); `scripts/validate_hr_v1.py` steht |
| DR-V1 (neurales RT60) | 🟡 V1 verdrahtet (konservativ) | Präziser neuraler RT60-Regressor (GPU-Training) als Folge-Schritt |
| **TP-V1 (BEATs-Onset-Konsens)** | 🟡 Zeuge verdrahtet | Adaptive-Schwelle-Stufe; **Tagger-Head** fehlt (Head auf Tokens trainieren oder Tagger-ONNX beschaffen — GPU) |
| **TP-V2 (neurale Phasen-Schätzung Transienten)** | 🔴 | Modell + Quelle fehlen |
| **C4 DDSP (EQ/Dynamik-Parameter-Prädiktion)** | 🟡 | F5: BEATs-Embedding-Probe war positiv (0,7e-3…1,8e-3, richtungsrichtig); V1 = CLAP→BEATs-Tausch im C4-Harness (480 Paare, GPU); Aktivierung erst bei val-MAE < Mittelwert-Baseline 0,2455 |
| **D3 (Multiresolution-Split 2–5 kHz)** | niedrig | optional für Denoise-Kette (H1/H2 fusionieren bereits bandweise) |
| **D4 (A1-Loss für APPLADE/DGT)** | niedrig | A1-Maskierungs-Loss künftig auch für APPLADE (DGT-Domain) — Waveform-Variante existiert |
| SP-V1 (geteiltes bandbegrenztes Resampling phase_12/31) | 🟡 teils erledigt | heute durch direktes Polyphasen-Resampling im phase_31-DSP-Fallback gedeckt; vollständige Komponenten-Teilung bleibt |
| B8 (Spektral-Reparatur-Naht-Gates) | ✅ via IN-V1/V2 | — |

### 3.4 ML-Finetunes & GPU-Pfade (F-Reihe)

Quelle: TODO-Liste Punkte 2–20 + Performance-Katalog P4.

| Maßnahme | Status | Schritt |
|---|---|---|
| **ML-V1 FlashSR-Musik-Finetune (F4)** | 🔄 läuft (Epoche 0/12, Val 0,4562) | AERO-vs-FlashSR-Kandidatenwahl ✅ (beide unter Baseline; FlashSR bleibt Basis, 14× Echtzeit); Akzeptanz: Never-worsen ΔSDR/ΔSegSNR ≥ 0 je Segment |
| **ML-V2 BigVGAN-v2 Musik-/Vokal-Finetune (F3)** | 🔜 GPU | Repair-Pfad B5-gegated; danach HR-V1-Aktivierungsurteil |
| **VOCAL-INPAINT S2 (DiffWave-Vokal-Finetune, F1)** | 🔄 läuft | Gate: mean Val-SDR ≥ 0 dB (S1-Ziellücke −1,7 dB); danach S3-Entdrosselung + S4-Verifikation |
| VOCAL-INPAINT S4-Reste | 🟡 | Per-Segment-Gate (ΔSDR ≥ 0) und Witness cos ≥ 0,92 in allen Fenstern marginal offen — CQTdiff+-Modellqualität am Gate, GPU-Finetune ist der Hebel |
| **C4-BEATs (F5)** | 🔜 nach F3 | s. 3.3 |
| **Tagger-Head (BEATs)** | 🔴 | Head trainieren oder Tagger-ONNX beschaffen |
| **Präziser RT60-Regressor** | 🔴 | GPU-Training |
| BSR-GPU (ONNX-EP-Sinn) | funktional überholt | ORT-ROCm-EP bleibt numerisch defekt (Upstream-Fix nötig); Torch-ROCm-Kern live (41,8×, Parität 1e-5) — nur S3 im ONNX-EP-Sinn offen |
| Quantisierung (int8/bf16) | gesperrt | nur mit gemessenem rel ≤ 1e-3 je Modell (BSR-Präzedenz) |

### 3.5 Performance-Resthebel (Wohlklang-neutral)

Quelle: Performance-Maßnahmenkatalog + Resthebel-Matrix. 32,0× RT ist mit
r10 erreicht; die Hebel, die noch NICHT ausgeschöpft sind:

| Hebel | Schritt |
|---|---|
| Engine-Ebene ≈ 50 % des Laufs (Qualitätsprüfung 136 s, Musical Goals 72 s, Nachbearbeitung 46 s …) | Kaskaden-Messkette: VERSA/PANNs/MERT laufen kandidaten-seitig über Alphas — GPU-Ports im R3-Muster (Parität rel ≤ 1e-3) bzw. ONNX-Run-Kosten (~2,0 s steady-state je Alpha) senken |
| Per-Phasen-Loop-Overhead (PLM-Eviction/OOM-Probes/Steering ≈ 1,6 s je Phase) | Eigene Session; PMGG/CALIB/Coalition-Anteil im 2,3-s-Gap phasenabhängig zerlegen (R13-Tooling existiert) |
| UTMOSv2-Erst-Load (timm-Hub, 4 Folds) + FeedbackChain 23,5 s je Song-Tail | einmalig je Song — dokumentiert, ggf. amortisieren |
| **Chunk-Vergrößerung** (`AURIK_CHUNK_S`, R16) | OFFEN: Qualitäts-Entscheid — ändert Chunk-Grenzen (nicht bit-identisch); Evidenz-Lauf AURIK_CHUNK_S=120 mit Export-Quality ≥ Referenz + Hör-Check |
| Chunk-Grenzen-Gaps (phase_41) | R13-trennbar — im Chunk-Kontext bewerten |

**Bewusst NICHT verfolgt** (Qualitätskompromiss): Chunk-Parallelisierung
(§V7-Regelkreis), Song-Level-Hoist BANQUET/DFN/Pitch (nicht äquivalent),
Quantisierung ohne Parity, Phasen-Skip außerhalb der Audibility-Gates.

### 3.6 Governance- und Prozess-Lücken (kein DSP, aber Garantie-relevant)

Quelle: AGENTS.md §2/§5/§6.

| Lücke | Schritt |
|---|---|
| Linter/Skripte hartkodiert (`compliance_check.py`, `aurik_verboten_linter.py`, `gebote_verifier.py`) | Bei JEDER Regeländerung im Markdown parallel das Skript nachziehen — sonst prüft CI eine andere Regel als dokumentiert |
| `.github/GEBOTE.md` beansprucht Vorrang, wird aber nicht enforced | Entweder Enforcement in den Verifier aufnehmen oder Referenzcharakter ehrlich deklarieren; Konflikte nie stillschweigend entscheiden (PR-Evidenzblock) |
| ID-Kollisionen (§G/§V-IDs nicht eindeutig über Dokumente) | Zitierdisziplin „mit Quelle"; Registry/`id_registry_check.py` pflegen; Bereinigungsplan `docs/ID_COLLISION_MAP.md` fortführen |
| Veraltete Dokumente (`.github/VERBOTE.md`, `GEBOTEN.md`, `.agents/skills/*`-Stubs) | Als nicht-normativ markiert — bei Berührung auf die normative Kette verweisen statt Duplikate pflegen |
| Spec-Drift-Checks (nur Specs 01–08 + ausgewählte Dateien gehasht) | Abdeckung auf weitere normative Dateien ausdehnen, wenn sie sich als drift-anfällig erweisen |
| `[RELEASE_MUST]`-Deckung | `release_must_coverage_check.py` hält die 1:1-Deckung; bei neuen Anforderungen Test mitliefern |
| FILE_REGISTRY/Repo-Karte | Write-Gate einhalten (auch für dieses Dokument — Eintrag unten) |

### 3.7 Bekannte Negativbefunde (dokumentiert, NICHT erneut versuchen)

| Befund | Konsequenz |
|---|---|
| UTMOS auf Vollmix richtungsblind bis invertiert (ML-V4) | kein Musik-MOS-Gate im Audio-Modus; PQS-DSP-Gate bevorzugen; Delta-Gate-Richtung bei Vollmix geprüft |
| MP-SENet auf Musik −5,9…−8,5 dB out-of-domain (ML-V3) | de-wired belassen; kein Musik-Finetune empfohlen (Rolle durch Pipeline abgedeckt) |
| CLAP-Head lernte nichts über Mittelwert-Baseline (C4-Probe) | BEATs-Encoder als C4-Embedding-Basis (F5) |
| ORT-ROCm-Kernels numerisch defekt (6 Modelle, rel 0,19–0,98) | ONNX bleibt CPU-Fallback; paritätsverifizierte Torch-ROCm-Kerne sind der GPU-Weg; §III.9 normativ |
| FlashSR/AERO nativ unter Never-worsen auf Musik | ohne Finetune nicht aktivierbar — Finetune-Gates bindend |

### 3.8 Nutzereingriffsfreiheit — verbleibende Garantie-Lücken

| Lücke | Schritt |
|---|---|
| phase_67 + phase_ambience_polish ohne Echt-Audio-Evidenz (Guardrail-Test rot, intendierter Ratchet) | **A9:** ≥5 reale Audiofälle je Phase aus `corpus/*` (Crackle-Set + reverb/ambience), HPI-Delta ≥ 0,02 via `run_real_audio_corpus_test.py` + Golden-Set-Infrastruktur; danach Review — `max_phases.limit` bleibt bis dahin unverändert |
| Golden-Set-Manifest erst 8 annotierte Fälle | auf ≥5 Crackle- + ≥5 Ambience-Fälle erweitern (`audit/real_audio_defect_golden_manifest.json`) |
| **MUSHRA-Pilot** | ITU-R BS.1534-3-konformer Hörtest als externe Hör-Evidenz der 0-Config-Ergebnisse — Pilot aufsetzen |
| Import-Fallstudie „Trio Schweizer – 13 Tage" | offene Extraktions-Grenze bei irregulärem Drift (GT-bewiesen eingeengt, §682–766) — als Regressionstest in die Harness-Familie aufnehmen |
| Overprocessing-Schutz De-Essing (ERKENNTNISSE 3.3) | material-adaptiv, aber Overprocessing-Schutz-Verifikation ergänzen |
| Blinde Flecken der Defekterkennung (ERKENNTNISSE 4.1) | Liste gegen die 62 DefectTypes abgleichen — jeder Typ braucht einen Musik-Träger-Fall im L3-Harness |
| Integrations-Lücke (ERKENNTNISSE 1.2, teilweise behoben) + Pipeline-Ordnung (1.3, in Analyse) | Ordnungs-Analyse abschließen; Rest-Integrationspunkte dokumentieren |
| Masking-Modell: Stärken & Lücken (ERKENNTNISSE 2.1) + Perceptual Salience nicht vollständig psychoakustisch (2.2) | in die A7-Witness-Arbeit einbeziehen (Audibility-Einstufung jedes Findings) |
| Sänger-Identität (ERKENNTNISSE 3.1) | Resemblyzer-Witness bereits im VOCAL-INPAINT-S4-Harness — auf weitere Vokal-Phasen ausdehnen |

---

## 4. Empfohlene Abarbeitungs-Reihenfolge (Hör-Gewinn je Aufwand)

1. **A6b–A6e** (4 gemessene Vinyl/wow_flutter-Phasen-Lücken) — jede mit
   phys-Metrik-Gate und Never-worsen, §7.4c-Muster.
2. **A6f** Gesamt-Harness alle Familien → restliche Phasen-Lücken der
   29er-Liste einzeln schließen.
3. **A7** Witness: Sibilanz-Härte + Muddiness + zeitvariante Maskierung
   (die Hörordnung verlangt Audibility-Wissen des Zeugen).
4. **A9** Guardrail-Evidenz (phase_67/ambience_polish, HPI ≥ 0,02) + MUSHRA-Pilot
   — sonst bleibt der Scope-Ratchet rot und neue Phasen-Evidenz blockiert.
5. **A8** Phasen-SOTA-Ausbaustufen: WF-V4-Checkpoint klären, HR-V1-Aktivierung
   nach F3, TP-V1-adaptive-Schwelle + Tagger-Head, TP-V2-Quelle, C4-F5,
   DR-V1-Regressor, D3/D4 (niedrig).
6. **ML-Finetunes:** F4/F1-Läufe abschließen → F3 → F5; Gates (Never-worsen,
   Val-SDR ≥ 0, val-MAE < Baseline) sind bindend — kein Blind-Aktivieren.
7. **Performance-Resthebel** (3.5) — qualitätsneutral, nach den
   Qualitäts-Blöcken; AURIK_CHUNK_S-Entscheid mit Supervised-Evidenz.
8. **Governance** (3.6) kontinuierlich bei jeder Regeländerung, nicht als
   eigener Block.

---

## 5. Verbindliche Akzeptanzkriterien (für jeden Lückenschluss)

- **Never-worsen:** Delta-basiert gegen den Input; absolute Produktions-
  Normalwerte sind kein Hard-Fail (Guard-Kalibrierung).
- **Determinismus §G5:** gleicher Input + Version ⇒ bit-identischer Output;
  Seeds input-abgeleitet (blake2b), kein `time.time()` in Entscheidungslogik.
- **§V6:** jeder ML→DSP-Fallback mit `logger.warning()` + Begründung;
  fail-closed.
- **§V7:** Ursache statt Symptom; keine phasen-individuellen Schwellwerte;
  Stärken zentral über `global_scalar`.
- **Hörordnung:** Befunde gegen die Maskierungsschwelle prüfen; sub-audible
  Reste dokumentiert als SKIP akzeptieren (JND-Begründung).
- **Evidenz:** phys-Metrik vorher→nachher je Fall; bei neuen Phasen ≥5
  Echt-Audio-Fälle + HPI ≥ 0,02; MUSHRA für Hör-Endbestätigung.

---

## 6. Zuordnung Lücke → Quelle (für die Detail-Arbeit)

| Lücke | Detail-Quelle |
|---|---|
| 3.1 Phasen-Lücken | `PHASE_SOTA_GAP_ANALYSE.md` §294–440; Harness-Ergebnisse `/tmp/harness_*.log` |
| 3.2 Witness | `WITNESS_SOTA_GAP_ANALYSE.md` §3–5 |
| 3.3 Ausbaustufen | `PHASE_SOTA_GAP_ANALYSE.md` §1–§6 |
| 3.4 ML/GPU | `TODOS_SOTA_ROADMAP.md` Punkte 2–20 + Modell-Portfolio |
| 3.5 Performance | `TODOS_SOTA_ROADMAP.md` PERFORMANCE-KATALOG + PERF-R1–R17 + Resthebel-Matrix |
| 3.6 Governance | `AGENTS.md` §2/§4/§5/§6 |
| 3.7 Negativbefunde | `TODOS_SOTA_ROADMAP.md` (ML-V3/V4, C4-Probe, §III.9) |
| 3.8/3.9 Echt-Audio & Garantie | `PHASE_SOTA_GAP_ANALYSE.md` §802–911; `AURIK_V10_ERKENNTNISSE.md`; `ANALYSE_VOKAL_MUSIK_NACH_RESTAURATION.md` |
