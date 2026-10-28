# Tiefenanalyse — was zur Vollständigkeit noch fehlt (2026-10-28)

> **Methodik:** Fünf Ebenen querverglichen — SOTA-Defizit-Register (98 Zeilen),
> LÜCKEN-MATRIX A–D (`docs/TODOS_SOTA_ROADMAP.md:289–372`), Maßnahmen-Katalog
> (Wellen A–D, KP-1…KP-19), Wohlklang-Vertrag (13 Stufen), Roadmap-Rest +
> TODO-Statuszeilen. Stand: Commit e4a4d924.

## E1 · Der Kern: 0 von 13 Wohlklang-Stufen haben die volle Belegkette

`wohlklang_gate` meldet: **aktiviert=0, Ausnahmen=5, gesperrt=8.**

| Lücke | Detail |
| --- | --- |
| 5 Stufen laufen als **Ausnahme** | W-1 HR-V1, W-4 MP-SENet, W-5 DFN, W-6 MIIPHER-DiT, W-7 Harmonic-Inpainting — **AN ohne vollständige Fünf-Beleg-Kette** (fehlend: Hör-Abnahme, teils ≥3 echte Songs/blindes A/B) |
| 8 Stufen **gesperrt** | W-2 SCNet, W-3 SGMSE+, W-8 BW-v5, W-9…W-13 (W-9 Whisper ist per SOTA-Urteil archiviert = erledigt-als-Entscheidung) |

**Das ist die Missionslücke schlechthin:** „Weltbester Wohlklang" = diese
Stufen mit echten Hör-Belegen aktiviert. Die Mechanik ist fertig
(flag_freigabe-Kette, ab_harness, AERO-Studie); es fehlen **nur Hör-Entscheidungen**
(§v10.802) — aber jede einzelne.

## E2 · Fähigkeiten (LÜCKEN-MATRIX B, 12 Einträge)

| Lücke | Rolle | Blocker-Typ |
| --- | --- | --- |
| **Referenzfreie Qualitätsmetrik** | Wohlklang-Entscheid ohne Referenz — UTMOS ist richtungs-**blind/invertiert** | **Mess-Infrastruktur defekt** (höchste technische Lücke) |
| Laufzeit 53× → 32× RT | F-Reihe-Optimierung (Song-Ebene-Struktur ist erledigt, Realität-Rekalibrierung offen) | autonom (Performance-Engineering) |
| Neuraler Warp-Schätzer WF-V4 | Rest-Wow/Flutter unter Hörschwelle | Quelle unklar (Alternativweg: Warp-Head auf FCPE/CRePE lokal trainieren — KP-17) |
| Neurale Phasenschätzung TP-V2 | Transienten-Phasenkohärenz | Modell + Quelle fehlen (KP-18) |
| BEATs-Tagger-Head (527 Klassen) | Instrument-/Genre-Evidenz ohne CLAP | Head-Training oder Tagger-ONNX |
| MERT-Genre-Head | genre-adaptive Goals | GTZAN/MagnaTagATune fehlen (KP-14) |
| Gender-Head + Label-Korpus | Register/Formant-Entscheidungen | ≥50 Labels/Klasse (KP-15) — menschlich/Daten |
| GPU-Finetune-Queue (7!) | DiffWave-Vokal (−1,7 dB!), GaCELA-Vokal, BigVGAN-v2, FlashSR-Musik, DDSP-BEATs, BEATs-Head, Gender-Head | **läuft NICHT in der aktuellen GPU-Kette** (Matrix D.3 vs. Welle C) |
| Präziser RT60 | Dereverb-Steuerung | V1-Witness läuft; echte RIRs sind jetzt da (50 Stück) |
| Human-Panel / MUSHRA-Kalibrierung | Proxy-Kalibrierung | menschlich |

## E3 · Defizit-Register: 41 offen (von 98)

K3: 15 (meist Hygiene/Phantome) · K4: 10 (Modell-Lücken) · K0: 7 (Kernpfad) ·
K1: 5 · K2: 4. Davon **menschlich-gated**: D-K0-3 (SGMSE-Hör), D-K0-9
(EAR-VAE v1↔v2-Urteil), D-K2-5 (AERO), D-K3-21 (SNR-Adaption A/B), D-K2-3/4
(Rollenentscheidungen), §9/7-Lizenzpolitik.

## E4 · Pfad-Hygiene (LÜCKEN-MATRIX C) — autonom abarbeitbar

1. **17 TorchScript-Stubs (K3-3):** deklarierte ML-Pfade, die nie laufen —
   versteckte DSP-Fallbacks, Wartungs-Attrappen.
2. **53 Phantom-`models/`-Referenzen in 1660 Produktionsdateien (D-K3-5)** —
   einzeln prüfen (flow_matching 8×, muq_eval_a1_head 3×, mp_senet ohne
   `finetuned`-Präfix …); bewusst Bericht statt Fail, aber offen.
3. **3 Domänen unbelegt** (bigvgan_v2.onnx 44,1-kHz-Konfiguration, hifi_gan.onnx …).
4. **Namens-Drift:** `models/nvsr/nvsr.onnx` byte-identisch zum
   FlashSR-Produktionsexport (D-K2-4).

## E5 · Offene Engineering-Restposten

- **P1-7** „Keine hörbaren Restdefekte: m1b intern" — **KEIN STATUS** (offen).
- **P1-10** „Defekt-Rückwärtszählung (Chips)" — **KEIN STATUS** (offen).
- **P1-3** nur teilweise: Audibility-Deckung für SCK/WBG/ATI/Formant umgesetzt
  — ob ALLE Schwellwert-Guards abgedeckt sind, ist nicht ausgewiesen.
- **KP-8-Nachlauf:** Re-Export + Parität nach `fine_tune` (Kette erledigt das
  automatisch), README-Tiny-Notiz ist bereits widerlegt/aktualisiert.

## E6 · Entscheidungen (menschlich — überall ist die Mechanik fertig)

W-1…W-13-Hörabnahmen · §9/5–7 · BW-v5-Sign-off (Gate 51,0 ≥ 1,02) ·
AERO-MUSHRA-Runde · D-K3-21-A/B · P1-4 Blind-Panel · MUSHRA n≥30 · Hör-Ritual.

## Priorisierte Restliste (Wirkung × Machbarkeit)

1. **E4 Hygiene abschließen** (WP-2): 53 Phantome + 17 Stubs — Fehlerfreiheit.
2. **GPU-Finetune-Queue D.3 als zweite Kette** starten (DiffWave-Vokal →
   BigVGAN-v2 → FlashSR-Musik → DDSP-BEATs → BEATs-Head; Gender-Head/GaCELA
   daten-blockiert).
3. **Referenzfreie Qualitätsmetrik** ersetzen/sichern (UTMOS-Negativbefund;
   Kandidaten: SingMOS Pro (KP-12), DNSMOS-P835-Kompatibilität prüfen).
4. **P1-7 + P1-10** implementieren; P1-3-Restdeckung ausweisen.
5. **Matrix D.1b/c:** F3-BigVGAN-Rollout-Entscheid + EAR-VAE v1↔v2 als
   vergleichende Messungen vorbereiten (Urteil = Hör).
6. **Laufzeit 53×→32×** (F-Reihe) — Profiling der Hotspots.
