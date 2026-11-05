# K4 — Wohlklang-Stufen-Objektivierung: entscheidungsreif für das menschliche Gehör (2026-11-05)

**Bearbeitung:** Manager (Worker-Durchläufe scheiterten dreimal an der Worktree-Basis-Falle
`65c6ef4d`; Wahrheits-Checks der Worker waren korrekt). **Gearbeitet auf Stand `4361777f`+**
(v10.12.51, nach K0/K1 `40086bbd` und K2 `4361777f`).
**Regel eingehalten:** KEINE Aktivierungen, KEINE Flag-Flips (§v10.802 (copilot-instructions.md),
§4-Verfahren `scripts/flag_freigabe.py`); keine erfundenen Zahlen (§V7 (copilot-instructions.md)).

---

## 1. Objektiv-Befunde W-1…W-13 (B3-Verfahren, Seed 42)

Datenquellen: `docs/reports/current/b3_stufenmessung_W-*.json`,
`docs/reports/current/b3_objektivbeschluss_W-*.json`,
`.github/WOHLKLANG_CLAIMS.md` (Vertrags-Status), Hörordnung §2/§7
(`.github/instructions/hoerordnung.instructions.md`).

| Stufe | Flag | B3-Befund (Trials / never_worsen / Vorteile / p) | Hörordnung-Ebene-Check | Entscheidungsreif? | Rest-Blocker |
| --- | --- | --- | --- | --- | --- |
| **W-1** | `BIGVGAN_V2_HR_ACTIVATED` (ausnahme) | 5 / **False** / 2/5 / 1,0 — **einzige Stufe mit verfehltem Never-worsen** | Widerspruch: HF-Gain-Beleg BESTÄTIGT vs. musik_mos ABGELEHNT ⇒ §7-Konfliktregel | **JA — Hör-Instanz entscheidet** | keine (nur Ohr) |
| **W-2** | `use_scnet_music` (gesperrt) | 6 / True / 2/6 / 0,6875 (Nachmessung 2026-11-04, `b3_objektivbeschluss_W-2.json`) | never-worsen ⇒ Ebene 1–3 unverletzt | ja — objektiv ABGELEHNT (kein Nutzen signifikant), Sicherheit belegt | Aktivierung nur §4 + Hör-Bestätigung |
| **W-3** | `use_sgmse_musik` (gesperrt) | **kein Mess-Record** — der W-2/W-3-Lauf (Report `2026-11-04_sitzung_w2_w3_b3_messung.md` §4) hat nur W-2 geschrieben | — | **NEIN** — Messung fehlt | Stufe im Lauf gesperrt/übersprungen; Nachlauf `b3_stufenmessung.py --stages W-3` nötig |
| **W-4** | `use_mp_senet_musik` (ausnahme) | 5 / True / 2/5 / 1,0 | never-worsen ⇒ unverletzt | ja — objektiv ABGELEHNT | Review-Bedingung Voll-Song-Blindnachweis (Vertrag) |
| **W-5** | `use_df_musik` (ausnahme) | 5 / True / 0/5 / 0,0625 | never-worsen ⇒ unverletzt | ja — objektiv ABGELEHNT (kein Vorteil) | Evidenz-Matrix WP-2 |
| **W-6** | `use_miipher_dit` (ausnahme) | 6 / True / 1/6 / 0,219 | never-worsen ⇒ unverletzt | ja — objektiv ABGELEHNT | Evidenz-Matrix WP-2 |
| **W-7** | `use_harmonic_inpainting` (ausnahme) | 6 / True / 1/6 / 0,219 | never-worsen ⇒ unverletzt | ja — objektiv ABGELEHNT | Evidenz-Matrix WP-2 |
| **W-8** | `use_bw_v5` (gesperrt) | kein B3-Lauf (Flag AUS) | A1-HF-Gain-Gate 2026-10-27 **bestanden** (kalibriert, `2026-10-27_kp7_hf_gate_v2.json` `gate_passed=True`) | ja — gesperrt mit Grund | Hör-Sign-off (+17 dB HF-Boost-Artefakt-Risiko, §v10.802) |
| **W-9** | `use_whisper_denoiser` (gesperrt) | — | deprecated Rev. 2026-08-16 | ja — dauerhaft gesperrt | keine |
| **W-10** | `use_utmos_music` (gesperrt) | — | Sprach-Orakel; Negativbefund richtungsblind/invertiert (KP-2, `2026-10-25_kp2_musik_mos_orakel.md`) | ja — gesperrt mit Grund | Musik-Kalibrierung; alternativ zertifiziertes `musik_mos` (Spearman 0,95, 2026-10-28) — Umstellung = Flag-Entscheidung |
| **W-11** | `use_silero_vad_music` (gesperrt) | — | sprachtrainiertes VAD darf Musik-Gate nicht steuern; PANNs-Singing-Offset liegt report-only vor (KP-4) | ja — gesperrt mit Grund | Freigabe des PANNs-Pfads = Flag-Entscheidung |
| **W-12** | `use_resemblyzer_music` (gesperrt) | — | sprachtrainierter Embedder (§III.11 (copilot-instructions.md)) | ja — gesperrt mit Grund | Musik-Finetune (Daten/Training) |
| **W-13** | `use_bigvgan_f3_studio` (gesperrt) | — | Kandidat-Pfad im §1.4-Block vorbereitet (KP-5) | ja — gesperrt mit Grund | Aktivierung erst nach W-1-Hörprobe + Hörordnungs-Sign-off (§v10.802) |

**Router-Fix-Evidenz (W-6/W-7):** Die B1-Rückfälle (ungeplante ML→DSP-Ersatzpfade)
sanken durch den Router-Fix 2026-11-02 messbar: W-6 **11 164 → 364**, W-7
**18 220 → 2 236** (Vorher-Records `b3_objektivbeschluss_W-6/7_vor_routerfix_20261102.json`
mit `fehler`-Status dokumentieren den vorherigen Bruchzustand). Die B3-Beschlüsse
selbst blieben ABGELEHNT (nie signifikant) — der Fix erhöht die Robustheit, nicht
die gemessene Qualität.

**Gesamturteil:** 12 von 13 Stufen sind **objektiv entscheidungsreif** (7 mit
B3-Messung, 5 als „gesperrt mit dokumentiertem Grund"). **W-3 ist die einzige
Stufe ohne Mess-Record** (Nachlauf nötig). **W-1 ist die einzige Stufe, die eine
menschliche Entscheidung BRAUCHT** (Widerspruch nach §7) — alle übrigen
Aktivierungen sind Ermessens-/Hör-Fragen über bereits vorhandene Objektiv-Befunde.

---

## 2. K4-Einträge D-K4-1…D-K4-11 — Kriterium → Handlung → Status

| Eintrag | Kriterium (Register) | Handlung/Befund 2026-11-05 | Status |
| --- | --- | --- | --- |
| D-K4-1 (A-SPADE-Declipping) | Musik-Finetune + A/B vor Rollout (§III.11) | Code liegt vor (`backend/core/dsp/aspade_declipper.py`, Challenger in `phase_07_declipper.py`); Finetune auf MUSDB18HQ + A/B über `scripts/b3_stufenmessung.py` | **offen — präziser Blocker:** GPU-Trainingsfenster (D3-Kette läuft) + A/B-Messung; kein Flag-Flip ohne Verfahren |
| D-K4-2 (TF-GridNet-Denoiser) | Musik-Finetune (§III.11) | Code liegt vor (`backend/core/dsp/tf_gridnet_denoiser.py`); Training ist als Job **M9** eingeplant (`docs/reports/current/train_job_m9_tf_gridnet.json`) | **offen — präziser Blocker:** M9-Training (GPU-Kette) + danach A/B |
| D-K4-3 (Ganzsong-Pipeline + Per-Stem) | Song-globale Größen + Per-Stem als Standard | **Song-Global-Teil ERLEDIGT** durch P0-1 Slice B (Commit `9a8ba267`): HPI/EmotionalArc/VQI/Singer-ID + measure_all laufen einmal je Song nach Assembly; §G10-Audibility-Nachzug erhalten | **teilweise erfüllt** — offen: Per-Stem-Restaurierung als Standard (Architektur-Entscheidung, klangrelevant) |
| D-K4-4 (UTMOSv2 Sprach-Orakel, W-10) | Musik-Kalibrierung vor Veto-Recht | Negativbefund (richtungsblind/invertiert) bleibt bestehen (KP-2); **musik_mos** ist als zertifizierte Musik-Alternative vorhanden (Spearman 0,95, 2026-10-28) | **offen — Flag-Entscheidung:** W-10 gesperrt; Umstellung auf musik_mos = §4-Verfahren |
| D-K4-5 (VoiceEncoder W-12) | Musik-Finetune (§III.11) | Vertrag W-12 gesperrt; Training/Finetune braucht Sprach-freien Gesangs-Datensatz | **offen — präziser Blocker:** Daten + Training |
| D-K4-6 (VersaSingMOS F10) | Kalibrierung auf verarbeitete Musik-Vocals | Messpfad definiert (Modelldomäne musik, `scripts/model_inventory.py --domains`); Kalibrier-Lauf auf Effektketten/Misch-Material steht aus | **offen — präziser Blocker:** Kalibrier-Messung (autonom ausführbar, GPU-freies Fenster nötig) |
| D-K4-7 (WF-V4 Warp-Head) | Neurales Wow/Flutter-Modell | Code + 9 Tests vorhanden (`backend/core/dsp/wf4_warp_head.py`), Challenger-Block in `phase_12_wow_flutter_fix.py` | **offen — präziser Blocker:** Gewichte/Quelle unklar (Datenbeschaffung, §III.13 (copilot-instructions.md)) |
| D-K4-8 (TP-V2 Phasen-Head) | Neurale Phasenschätzung | Code + 8 Tests vorhanden (`backend/core/dsp/tp_v2_phase.py`), Challenger-Block §v10.303 | **offen — präziser Blocker:** Modell + Quelle fehlen (Datenbeschaffung) |
| D-K4-9 (MUSHRA-Panel) | Human-Panel §3.6 / Proxy-Kalibrierung | Protokoll + Pilot-Szenarien (`docs/guides/MUSHRA_STUDIENPROTOKOLL.md` §10) + Stimuli-Builder (`scripts/build_mushra_stimuli.py`, Seed 42, BS.1770) liegen vor | **entscheidungsreif — menschlich-gated:** Durchführung = Hör-Session |
| D-K4-10 (RT60-Regressor) | Messbare RT60-Schätzung statt Heuristik | Code + 17 Tests vorhanden (`backend/core/dsp/rt60_regressor.py`); Datenlücke ehrlich: 1 lokale RIR-Simulation | **offen — präziser Blocker:** echte RIR-Trainingsdaten (D-K1-2/RIR-Bank) |
| D-K4-11 (Preference-Modell F15) | Musik-Preference-Ranking auf Hör-Labels | Kein Modell vorhanden; Labels entstehen erst über Hör-Sessions (K0-Runde, MUSHRA) | **entscheidungsreif — menschlich-gated:** Label-Generierung = Hör-Sessions |

---

## 3. Hör-Ordnungsgemäßheit (Ebene 1–3, „nie gegen Ebene 1")

- Alle B3-gemessenen Stufen mit `never_worsen=True` (W-2/4/5/6/7): die
  richtungszertifizierten Metriken (musik_mos + defektlast) zeigen keine
  Verschlechterung — Ebene 1–3 unverletzt (B1-Zähler `b1_sauber` dokumentiert,
  keine ungeplanten Ersatzpfade in den Nachmess-Läufen).
- **W-1:** nie gegen Ebene 1 ist nicht maschinell entscheidbar (widersprüchliche
  Belege zweier richtungszertifizierter Metriken) ⇒ §7-Verweis an die Hör-Instanz.
- W-8…W-13: gesperrte Flags ⇒ kein Klangpfad berührt ⇒ Ebene 1–3 trivial unverletzt.
- Der Wohlklang-Ordnungs-Enforcer (`backend/core/wohlklang_ordnung_gate.py`,
  Commit `a6a41cf4`) und die Song-Level-Gates (P0-1 Slice B, `9a8ba267`) sind
  unberührt.

## 4. Was die Hör-Instanz entscheiden muss (gebündelt, ~1 Session)

1. **W-1** (Pflicht-Entscheidung): HF-Gain vs. musik_mos-Widerspruch nach §7.
2. **W-2** (Ermessen): SCNet-Musik aktivieren? (objektiv: sicher, aber ohne
   signifikanten Nutzen)
3. **W-8** (Ermessen): BW-v5 freigeben? (Gate bestanden; +17 dB HF-Risiko)
4. **W-10** (Ermessen): UTMOS-Veto durch zertifiziertes musik_mos ersetzen?
5. **W-11** (Ermessen): PANNs-Singing-Offset freigeben?
6. **W-13** (Folge): BigVGAN-F3-Studio nach W-1-Urteil.
7. Aus K0/K1/K2: EAR-VAE v1→v2 (objektiv 16/17 für v2), Material-CRNN-Flag,
   D-K2-3-Primär-Rolle, D-K4-9/D-K4-11 (MUSHRA/Labels).

## 5. Offene Nachläufe (technisch)

- **W-3-B3-Messung** nachholen (`nice -n 15 python3 -B scripts/b3_stufenmessung.py
  --stages W-3`, host-sicher) — einzige fehlende Objektiv-Messung.
- D-K4-6-Kalibrier-Messung (GPU-fensterfrei).
- D3-Kette (M9 = TF-GridNet, D-K4-2) liefert die Trainings-Artefakte.

## 6. Verifikations-Befehle (technisch, ohne Code-Änderung in diesem Report)

```bash
python3 -B scripts/sota_deficit_gate.py        # Register bleibt gate-fähig
python3 -B scripts/wohlklang_gate.py           # W-Vertrag konsistent
```

**Arbeitsstand:** 4361777f+. **Keine Aktivierung, kein Flag-Flip, keine
erfundenen Zahlen** — alle Zahlen mit Quelldatei oben genannt.
