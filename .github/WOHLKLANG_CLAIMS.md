# Wohlklang-Vertrag — jede klangverändernde Stufe braucht fünf Belege

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
gelten, wenn alle fünf Belege vorliegen — und der Gate-Abgleich E5 verhindert,
dass Vertrag und Code auseinanderlaufen.

## Die fünf Belege

| # | Beleg | Anforderung |
| --- | --- | --- |
| 1 | **≥3 echte Songs** | vollständige Titel, keine 20-s-Ausschnitte, keine Testtöne |
| 2 | **Produktionspfad** | derselbe Pfad/EP/Modus wie im Export, nicht ein Nebenpfad |
| 3 | **Eingefrorene Baseline** | versionsgetaggter Vorher-Stand (§G5 copilot-instructions.md: gleiche Version ⇒ gleicher Output) |
| 4 | **Blindes A/B für einen Menschen** | die Hör-Entscheidung — nicht der Proxy, nicht das Modell |
| 5 | **Budget-Zahl** | RT-Beitrag gegen die Budget-Tabelle (copilot-instructions.md) |

**Status-Enum:** `aktiviert` (alle fünf erfüllt) · `ausnahme` (befristete,
begründete Ausnahme — erscheint als Warnung) · `gesperrt` (Soll = AUS).

**Regeln:** E1 Struktur/IDs · E2 zitierte Pfade müssen existieren ·
E3 `aktiviert` ohne `OFFEN` · E4 `ausnahme` braucht `Grund:` (≥ 20 Zeichen) und
`Review:` · E5 **Schalter-Abgleich Code ↔ Vertrag** · E6 `gesperrt` ⇒ Soll AUS ·
P1 Evidenz-Harneske dürfen keine Phantom-Pfade referenzieren ·
P2 Bericht über Phantom-Referenzen im Produktionscode.

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
| W-8 | BW-Reconstructor v5 | `backend/core/music_model_flags.py::use_bw_v5` | AUS | OFFEN | OFFEN | OFFEN | OFFEN | OFFEN | gesperrt (A1-Gate nicht bestanden: 0,73 < 1,02) |
| W-9 | Whisper-Denoiser | `backend/core/music_model_flags.py::use_whisper_denoiser` | AUS | — | — | — | — | — | gesperrt (deprecated Rev. 2026-08-16) |
| W-10 | UTMOSv2 Musik-MOS (Zeuge) | `backend/core/music_model_flags.py::use_utmos_music` | AUS | — | — | — | — | — | gesperrt (Sprach-Orakel, kein Veto-Recht bis Musik-Kalibrierung) |
| W-11 | Silero-VAD Musik-Freigabe | `backend/core/music_model_flags.py::use_silero_vad_music` | AUS | — | — | — | — | — | gesperrt (sprachtrainiertes VAD darf Musik-Gate nicht steuern) |
| W-12 | Resemblyzer Musik-Freigabe | `backend/core/music_model_flags.py::use_resemblyzer_music` | AUS | — | — | — | — | — | gesperrt (sprachtrainierter Embedder, §III.11 copilot-instructions.md) |

## Was dieses Gate nicht behauptet

- **Keine Stufe steht auf `aktiviert`.** Genau das ist der Befund: Für **keine**
  eingeschaltete Qualitätsstufe liegen derzeit alle fünf Belege vor.
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
