# Wohlklang-Vertrag — jede klangverändernde Stufe braucht sechs Belege

> **Normative Grundlage:** §v10.802/§G8/§G9 (copilot-instructions.md) ·
> **Enforced durch:** `scripts/wohlklang_gate.py` (Pre-Commit-Hook
> `aurik-wohlklang-gate`, fail-closed)

## Warum dieser Vertrag existiert

Produktionsbefund 2026-10-06: Die BigVGAN-Reparaturstufe HR-V1 war **produktiv
aktiv** (`applied: True`, 18 Bänder, Signal verändert) — auf der Grundlage eines
A/B-Belegs, der auf **20 s Material auf der GPU** lief, während der
Produktionspfad **CPU/ONNX** ist und **13,2× RT je Passage** kostet
(330 % des Budgets für die gesamte Phase-Pipeline). Zusätzlich sagte die
Aktenlage weiterhin „Flag OFF".

Der Fehler war **nicht** das Modell, sondern die **Beweislage**. Dieser Vertrag
macht daraus eine prüfbare Bedingung: Eine Stufe darf nur dann als _aktiviert_
gelten, wenn alle sechs Belege vorliegen — und der Gate-Abgleich E5 verhindert,
dass Vertrag und Code auseinanderlaufen.

## Die sechs Belege

| # | Beleg | Anforderung |
| --- | --- | --- |
| 1 | **≥3 echte Songs** | vollständige Titel, keine 20-s-Ausschnitte, keine Testtöne |
| 2 | **Produktionspfad** | derselbe Pfad/EP/Modus wie im Export, nicht ein Nebenpfad |
| 3 | **Eingefrorene Baseline** | versionsgetaggter Vorher-Stand (§G5 copilot-instructions.md: gleiche Version ⇒ gleicher Output) |
| 4 | **Blindes A/B für einen Menschen** | die Hör-Entscheidung — nicht der Proxy, nicht das Modell |
| 5 | **Budget-Zahl** | RT-Beitrag gegen die Budget-Tabelle (copilot-instructions.md) |
| 6 | **Metrik-Identität** | jede Gate-Zahl als Paar (Wert, Formel, Referenz-Set + Seed, Artefakt) — Tabelle „Gate-Zahlen“ unten |

**Status-Enum:** `aktiviert` (alle sechs erfüllt) · `ausnahme` (befristete,
begründete Ausnahme — erscheint als Warnung) · `gesperrt` (Soll = AUS).

**Regeln:** E1 Struktur/IDs · E2 zitierte Pfade müssen existieren ·
E3 `aktiviert` ohne `OFFEN` · E4 `ausnahme` braucht `Grund:` (≥ 20 Zeichen) und
`Review:` · E5 **Schalter-Abgleich Code ↔ Vertrag** · E6 `gesperrt` ⇒ Soll AUS ·
E7 **Gate-Zahlen & Metrik-Identität (Paar-Pflicht)** · jede veröffentlichte Bewertung: Name · Messgröße · Referenz-Set + Seed · Belegpfad (alle Felder gefüllt; Status „aktiv“ nur bei vollständiger Identität) ·
**Betriebsregeln (im Release-Report belegt, nicht Gate-geprüft):** B1 **Fallback-Quote 0 (ungeplant)** — kein ungeplanter ML→DSP-Fallback im Referenz-Korpus; Zählung `backend/core/fallback_ledger.py` mit Klassifikation (`geplant` = bewusst gesperrte Flags §v10.802 (copilot-instructions.md) bzw. offene Zeugen-Modelle — separat ausgewiesen; `defekt` > 0 blockiert den Flag-Flip) · B2 **Flag-Disziplin** — maximal 1–2 neue Flags pro Release; kein neuer K4-Challenger, solange mehr als 3 offene Kandidaten warten · B3 **Formeller Objektivmetrik-Beschluss** — der Beleg „blindes A/B“ kann alternativ über das Verfahren im Abschnitt „Formeller Objektivmetrik-Beschluss“ erbracht werden (Hör-Vorbehalt!).

## Gate-Zahlen & Metrik-Identität (Beleg 6)

Jede Zahl, die ein Tor öffnet oder schließt, wird hier geführt als Paar
(**Wert, Formel, Referenz-Set + Seed, Artefakt**). Eine Zeile ohne
vollständige Metrik-Identität trägt Status `offen` und darf keinen
Statuswechsel begründen (E7). Genau dieses Muster ließ die ungeklärte
„0,73"-Zahl (KP-7) jahrelang weiterleben — bis die Kalibrierung gegen das
eingefrorene Alt-Artefakt die Metrik identifizierte
(`docs/reports/current/2026-10-27_kp7_hf_gate_v2.json`).

| ID | Gate-Zahl | Wert | Formel | Referenz-Set + Seed | Artefakt | Status |
| --- | --- | --- | --- | --- | --- | --- |
| G-1 | HF-Gain-Gate (D-K1-5) | ≥ 1,02 | HF-Energie(out)/HF-Energie(in) > 3 kHz | milde HF-Dämpfung 6 dB/oct ab 2 kHz, Seed 42 (`scripts/validate_bw_v5_hf_gate.py`) | `models/bw_reconstructor/retrain_a1_20261025/bw_reconstructor_v5.onnx` (Report `docs/reports/current/2026-10-27_kp7_hf_gate_v2.json`) | aktiv |
| G-2 | AM-Peak-Signifikanz (D-K3-54) | k = 1,40 ∧ Schärfe ±2 Bins | Peak > k·Perzentil(abs(ΣS_w), 99,9 %) des Störgrunds | 48 Rausch-Realisierungen + dk328-Matrix, Seed 42 (`tests/unit/test_dk354_am_peak_calibration.py`) | `backend/core/defect_scanner.py` (`_AM_PEAK_SIG_K`) | aktiv |
| G-3 | ΔSDR-Gate (D-K4-1, A-SPADE) | ΔSDR ≥ 0 dB (Never-worsen) ∧ Blindes A/B trägt | ΔSDR = SDR(A-SPADE) − SDR(PCHIP-Baseline) auf geclipptem Real-Material | Geclipptes Real-Material des Referenz-Korpus, Seed 42 (`scripts/release_candidate_pipeline.py`) | `scripts/train_aspade_declipper.py` (Delta-SDR-Report) | aktiv (Messung offen) |
| G-4 | Never-worsen-Grenzen (D-K4-2, TF-GridNet) | Score_nachher ≥ Score_vorher + Hör-Sign-off | ΔScore = Score(nachher) − Score(vorher) je Stufe ≥ 0 | 3 Referenz-Songs (AM_Contra, Al_James, Motor_Tapes laut Vertrag W-2), Seed 42 (`scripts/release_candidate_pipeline.py`) | `scripts/release_candidate_pipeline.py` (Metrik-Report je Kandidat) | aktiv (Messung offen) |
| G-5 | SI-SDR-Vergleich SGMSE-Kerne (D-K0-3) | Kopf-an-Kopf-Vergleiche gewonnen (Hör) ∧ ΔSI-SDR ≥ 0 | ΔSI-SDR = SI-SDR(Musik-Core) − SI-SDR(WSJ0/WPE-Kern) auf Referenz-Fenstern | 3-Wege-Fenster w04/w20, Seeds 42/43 (`scripts/build_sgmse_hoerprobe_3weg.py`) | `models/sgmse_plus/sgmse_musik_core.onnx` + `docs/reports/current/2026-10-23_hoerprobe_sgmse_3weg.json` | aktiv (Messung offen) |

## Vertrag

| ID | Stufe | Schalter (Datei::Name) | Soll | ≥3 echte Songs | Produktionspfad | Baseline | Blindes A/B | Budget | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W-1 | HR-V1 additiv (BigVGAN, phase_07) | `plugins/bigvgan_v2_plugin.py::BIGVGAN_V2_HR_ACTIVATED` | AN | OFFEN (A/B nur 20 s GPU-Material: `docs/reports/current/2026-09-16_hr_v1_bigvgan_ab_validation.md`) | am kanonischen Helfer CPU/ONNX gemessen (`docs/reports/current/2026-10-06_p1_3_bigvgan_f3_rollout_entscheid.md`); **wirkt seit dem Kanal-Fix 2026-10-07** (`applied=True`, 47/32/33/22 Bänder auf 4 Ausschnitten, außerhalb der Deckel-Fenster bit-identisch — D-K3-6) | Hörprobe liegt bereit (PRODUKTIONSPFAD): `output_audio/mushra/thresholds_wohlklang_w1_hrv1/` (Anleitung `docs/reports/current/2026-10-07_hoerprobe_anleitung.md`) | OFFEN (C4-Hörstichprobe) | 1,04 × RT je Passage = 62 s je Audio-Minute | ausnahme (Grund: Maintainer-Entscheid 2026-10-06 Weg 2 — Längen-Deckel + eine Aufrufstelle; **Ursache 2026-10-07 behoben (D-K3-6): der Pfad war auf Stereo bit-identisch wirkungslos, weil das Gate über die Baseline-Kanäle iterierte und den Mono-Kandidaten indexierte — betroffen waren auch §B4/§B5; Fix an EINER kanonischen Stelle, Hörprobe jetzt Produktionspfad**; Review: C4-Hörstichprobe entscheidet über Verbleib) |
| W-2 | SCNet-4-Stems (Separation) | `backend/core/music_model_flags.py::use_scnet_music` | AUS | 3 (AM Contra 165 s; Al James 169 s; Motor Tapes 107 s) | am kanonischen Plugin gemessen, nicht in der Produktionskette | fair gemessene Demucs v4 (`docs/reports/current/2026-10-06_p1_2_scnet_vs_demucs_fair_ab.md`) | OFFEN (C4; Hörpaare in `output/scnet_ab_2026-10-06_fair`) | OFFEN (CPU ~7× langsamer als Demucs v4) | gesperrt |
| W-3 | SGMSE+ Musik-Core (Denoise/Dereverb) | `backend/core/music_model_flags.py::use_sgmse_musik` | AUS | OFFEN | Artefakt vorhanden (`models/sgmse_plus/sgmse_musik_core.onnx`), Pfad gesperrt | OFFEN | OFFEN | OFFEN | gesperrt |
| W-4 | MP-SENet Musik-Kern (Vokal-NR) | `backend/core/music_model_flags.py::use_mp_senet_musik` | AN | 3 Teilstücke (30 s) — Voll-Song OFFEN | über resolve_model_path, Artefakt `models/mp_senet/finetuned/mp_senet_musik.onnx` | OFFEN | OFFEN | OFFEN | ausnahme (Grund: Aktivierung 2026-09-20 auf 3 × 30-s-Teilstücken (seg-SNR +5,3…+11,3 dB) statt auf Voll-Songs; Review: Voll-Song- und Blind-Nachweis) |
| W-5 | DFN Musik-NR | `backend/core/music_model_flags.py::use_df_musik` | AN | OFFEN | Datenpfad `models/deepfilternet_v3_ii/finetuned` | OFFEN | OFFEN | OFFEN | ausnahme (Grund: Bestandsaktivierung vor Einführung dieses Vertrags; Review: Evidenz-Matrix WP-2) |
| W-6 | MIIPHER-DiT (Gesangsverbesserung) | `backend/core/music_model_flags.py::use_miipher_dit` | AN | OFFEN | Artefaktverzeichnis `models/miipher_dit` | OFFEN | OFFEN | OFFEN | ausnahme (Grund: Bestandsaktivierung vor Einführung dieses Vertrags; Review: Evidenz-Matrix WP-2) |
| W-7 | Harmonic-Inpainting-DiT | `backend/core/music_model_flags.py::use_harmonic_inpainting` | AN | OFFEN | Artefaktverzeichnis `models/harmonic_inpainting` | OFFEN | OFFEN | OFFEN | ausnahme (Grund: Bestandsaktivierung vor Einführung dieses Vertrags; Review: Evidenz-Matrix WP-2) |
| W-8 | BW-Reconstructor v5 | `backend/core/music_model_flags.py::use_bw_v5` | AUS | OFFEN | OFFEN | OFFEN | OFFEN | OFFEN | gesperrt (A1-HF-Gain-Gate 2026-10-27 bestanden — Beleg `docs/reports/current/2026-10-27_kp7_hf_gate_v2.json`; Hör-Sign-off offen, +17 dB HF-Boost-Artefakt-Risiko, §v10.802 (copilot-instructions.md)) |
| W-9 | Whisper-Denoiser | `backend/core/music_model_flags.py::use_whisper_denoiser` | AUS | — | — | — | — | — | gesperrt (deprecated Rev. 2026-08-16) |
| W-10 | UTMOSv2 Musik-MOS (Zeuge) | `backend/core/music_model_flags.py::use_utmos_music` | AUS | — | — | — | — | — | gesperrt (Sprach-Orakel, kein Veto-Recht bis Musik-Kalibrierung) |
| W-11 | Silero-VAD Musik-Freigabe | `backend/core/music_model_flags.py::use_silero_vad_music` | AUS | — | — | — | — | — | gesperrt (sprachtrainiertes VAD darf Musik-Gate nicht steuern); KP-4 (2026-10-25): musikadaptiver PANNs-Singing-Offset liegt report-only vor (`backend/core/dsp/music_vad_oracle.py`, `panns_vad_offset_db` in phase_18-meta) — Aktivierung erst nach diskriminativer Gesangs-Evidenz + Never-worsen + Hörordnungs-Sign-off |
| W-12 | Resemblyzer Musik-Freigabe | `backend/core/music_model_flags.py::use_resemblyzer_music` | AUS | — | — | — | — | — | gesperrt (sprachtrainierter Embedder, §III.11 copilot-instructions.md) |
| W-13 | BigVGAN-F3 Studio-Finisher-Stufe-1 | `backend/core/music_model_flags.py::use_bigvgan_f3_studio` | AUS | — | — | — | — | — | gesperrt (KP-5: Kandidat-Pfad im §1.4-Block vorbereitet; Aktivierung erst nach W-1-Hörprobe D-K0-1/D-K2-3 + Hörordnungs-Sign-off §v10.802 copilot-instructions.md; Vocos bleibt §V6-Fallback) |
| A-1 | Audibility-Budget (Dämpfungs-Obergrenze, phase_03/phase_43) | `backend/core/music_model_flags.py::use_audibility_budget` | AUS | OFFEN (B3-Messung im trainingsfreien Fenster) | quell-gepinnt: `backend/core/phases/phase_03_denoise.py::_denoise_mono_mrsa`, `phase_43_ml_deesser.py::_deess_channel` | Flag-AUS bit-identisch (30 Vertragstests, `tests/unit/test_audibility_budget*.py`) | OFFEN | OFFEN (RT-Kosten je Zone; nachmessen) | gesperrt (Implementierung 2026-11-07; Aktivierung nur §4-Verfahren nach B3-Messung + Hör-Sign-off) |

## Was dieses Gate nicht behauptet

- **Keine Stufe steht auf `aktiviert`.** Genau das ist der Befund: Für **keine**
  eingeschaltete Qualitätsstufe liegen derzeit alle sechs Belege vor.
  Fünf Ausnahmen sind sichtbar und befristet, sieben Stufen sind gesperrt.
- Der Vertrag ersetzt **keine** Hör-Entscheidung. Er erzwingt nur, dass eine
  Aktivierung ohne Beleg **auffällt** statt still zu geschehen.
- `ausnahme` ist **kein** Freibrief: Der Gate-Lauf weist jede Ausnahme als
  Warnung aus, mit Grund und Review-Trigger.

## Abarbeitungs-Reihenfolge (aus dem Gate ablesbar)

1. **W-1 HR-V1:** C4-Hörstichprobe (Ausschnitt-Behandlung) → Beleg 1 und 4.
2. **W-2 SCNet:** C4 (Hörpaare liegen bereit) + RT-Beitrag gegen das Budget.
3. **W-4 MP-SENet / W-5 DFN / W-6 MIIPHER / W-7 Harmonic-Inpainting:**
   Voll-Song- und Blind-Nachweis je Stufe — die größte offene Beweislücke.
4. Erst danach neue Stufen aktivieren.

## Formeller Objektivmetrik-Beschluss — Alternative zur Hör-Abnahme

**Norm:** `docs/TODOS_SOTA_ROADMAP.md` („MUSHRA n≥30 terminieren **oder**
formal Objektivmetrik-Beschluss“). Die Hörordnung bleibt die
Wahrheits-Ordnung; dieses Verfahren ist der dokumentierte
**Vertretungs-Beschluss** für den Beleg „blindes A/B“ — mit
**Hör-Vorbehalt** (eine spätere Hör-Abnahme gilt vorrangig).

Erbringung (fail-closed — **alle** Bedingungen Pflicht):

1. **Vergleichende Messung** Kandidat vs. Referenz auf ≥ 3 echten Songs
   (Referenz-Korpus), eingefrorene Baseline, dokumentierter Seed.
2. **Never-worsen:** keine Metrik signifikant schlechter (Δ ≥ 0 bzw. innerhalb
   95 %-CI/JND) — Gate-Zahl G-4.
3. **Signifikanter Vorteil** in der Zielmetrik (G-1/G-2/G-3), 95 %-CI über
   der Schwelle. Die Zielmetrik MUSS ein gültiges **Richtungszertifikat**
   tragen (`docs/reports/current/qualitaetsmetrik_richtungszertifikat.json`,
   `scripts/quality_metric_direction_gate.py`) — blind/invertierte Metriken
   sind gesperrt (Stand: `defektlast` zertifiziert; UTMOS/PQS blind).
4. **B1:** ungeplante ML→DSP-Rückfälle = 0 (`backend/core/fallback_ledger.py`).
5. **Blindheit:** Auswertung auf anonymisierten Paaren
   (`scripts/ab_harness.py`-Schlüssel), Auswertung ohne Label-Kenntnis.
6. **Provenienz:** Mess-Report mit Quelle je Zeile; Buchung als Beleg mit
   Vermerk „Objektivbeschluss“.

**Hör-Vorbehalt:** Die Stufe läuft danach als „Ausnahme (Objektivbeschluss)“.
Die Hör-Abnahme hebt sie auf „aktiv“; ein Hör-Widerspruch setzt sie auf
„gesperrt“ zurück (Rollback, §V7 (copilot-instructions.md)).
Automatisierte Abwicklung (eine Zeile): `python3 -B scripts/flag_freigabe.py
--flag <use_x> --objektivbeschluss <report.json>`.

## Ziel-Koordinate der Hör-Instanz (Block C, 2026-11-05)

Aus der EAR-VAE-v1↔v2-Blind-Runde (17 Paare, Entblindung 12:3 für v1,
p = 0,035) hat die Hör-Instanz die verbindliche Wunsch-Richtung für
künftige klangverändernde Iterationen notiert: **heller UND weniger Rauschen**
(v2-Helligkeit bei v1-Sauberkeit) — belegt in
`docs/reports/current/2026-11-05_hoer_signoff.md`.

Leitplanke für Stufen, die dieser Richtung dienen:

- Brillanz bleibt **gebunden** durch das Material-BW-Ceiling (Hörordnung §5,
  Tier-Kette: Natürlichkeit > Wärme > Klarheit > Brillanz).
- Kein HF-Boost über ein Gain-Gate — W-1 Höhen-Behandlung = **NO-GO**
  (0/8 Trials, Artefakte auf der behandelten Seite) ist das Referenzbeispiel.
- Erwünscht: Rauschreduktion zuerst, dann begrenzte Brillanz
  (Stufe-2-Nachbehandlung, P1-7-Muster).
