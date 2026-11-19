# K0-Sign-off-Paket — Hör-Runde für die drei Musik-Rollouts

> **Zweck:** Entscheidungsreife Hör-Runde für die drei K0-Musik-Rollouts
> (§v10.802, §III.11 — die Hör-Instanz entscheidet die Aktivierung). Eine Runde,
> ~30–45 min. Danach sind alle drei entschieden (GO/NO-GO) und die Aktivierung
> ist freigegeben bzw. verworfen. **Kein Dauer-Hören.**
>
> **Voraussetzung:** Kopfhörer, ruhige Umgebung, konstante Lautstärke
> (Details in `docs/reports/current/2026-10-07_hoerprobe_anleitung.md`).

## Die drei Entscheidungen

| # | Modell | Frage (Hörordnung Ebene 4/§4) | Hör-Paket | Metrik-Beleg |
|---|---|---|---|---|
| **K0-A** | **SCNet** (`D-K0-5`, `use_scnet_music`) | Klingt der SCNet-Gesangskern klarer/natürlicher als Demucs-v4? | `output_audio/mushra/thresholds_scnet_c4_fair_r2` (12 Trials + 2 Fangfragen) | fair-A/B **+1,96…+3,59 dB SI-SDR**; RT 3,43× gemessen |
| **K0-B** | **SGMSE-Musik** (`D-K0-3`, `use_sgmse_musik`) | Trägt der Musik-Finetune-Core gegen Sprach-Core + WPE-DSP? | `output_audio/mushra/thresholds_sgmse_3weg` (3-Wege-MUSHRA) | ONNX↔ckpt-Parität 6,7e-05; A/B-Metrik gefahren |
| **K0-C** | **BigVGAN-F3** (`D-K0-1`, Rollout) | Ist die Timbre-Konsistenz der gleichmäßig verteilten Ausschnitte hörbar konsistent (C4)? | `output/…/bigvgan_rollout_test` + A/B `docs/reports/current/2026-10-06_p1_3_bigvgan_f3_rollout_entscheid.md` | F3-ONNX paritätsbewiesen (rel ≤ 6,5e-05); HNR +4,42 dB |

## Ablauf (eine Runde, ~30–45 min)

1. **K0-A (SCNet)** — A/B-Hören (`thresholds_scnet_c4_fair_r2`). Frage: klarer/natürlicher?
   → **GO** = `use_scnet_music=True` aktivieren (fair-A/B belegt +1,96…+3,59 dB).
2. **K0-B (SGMSE)** — 3-Wege-MUSHRA (`thresholds_sgmse_3weg`). Frage: Musik-Core > Sprach-Core/WPE?
   → **GO** = `use_sgmse_musik=True` (A/B-Metrik: Musik ~1 dB besser).
3. **K0-C (BigVGAN-F3)** — C4-Timbre (`bigvgan_rollout_test` + F3-Report). Frage: Timbre konsistent?
   → **GO** = F3-Rollout (Kosten +0,9 % — nur bei hörbarem Timbre-Nutzen).

## Entscheidungsblatt (ausfüllen)

| Modell | Urteil (GO / NO-GO / unsicher) | Begründung (Hör-Eindruck) | Aktivierung |
|---|---|---|---|
| K0-A SCNet | ☐ GO ☐ NO-GO ☐ unsicher | | `use_scnet_music` = ☐ |
| K0-B SGMSE | ☐ GO ☐ NO-GO ☐ unsicher | | `use_sgmse_musik` = ☐ |
| K0-C BigVGAN-F3 | ☐ GO ☐ NO-GO ☐ unsicher | | F3-Rollout = ☐ |

## Harte Regeln (Hörordnung)

- **Metriken sind Zeugen, die Hör-Instanz entscheidet** (§4 Hörordnung). Die Metrik-Belege
  (oben) stützen, ersetzen das Urteil aber nicht.
- **Kein Erfolgsdruck:** „nichts Auffälliges" ist eine wichtige Information → NO-GO.
- **§v10.802:** Kein klangveränderndes Modell ohne dieses Sign-off.
- Jede Entscheidung ist **endlich** (GO/NO-GO) → danach operativer Zustand, kein Nachhören.

## Danach (Chain, siehe `docs/SOTA_WELTKLASSE_PLAN.md`)

Sign-off → Aktivierung der GO-Modelle → **Substanz-Hebung** (das hörbare Niveau) →
parallel läuft Pfeiler A (Wohlklang-Balken) → Pfeiler B (Welt-SOTA-Vergleich).

---

## Ergebnis der Hör-Runde (2026-10-30, Hör-Instanz L01)

| Runde | Hör-Ergebnis | Fangfragen | Urteil | Aktivierung |
|---|---|---|---|---|
| **K0-A** SCNet vs Demucs-v4 | SCNet **0** – Demucs **12** | 2/2 | **NO-GO** | `use_scnet_music` = False (bleibt) |
| **K0-B** SGMSE-Musik vs Baselines | musik **2** – Baseline **16** (WPE 12×, WSJ0 4×) | 2/2 | **NO-GO** | `use_sgmse_musik` = False (bleibt) |
| **K0-C** BigVGAN-F3 (Timbre) | *offen* | — | — | — |

**Befund (Hörordnung §4 validiert):** Beide Modelle wurden vom Ohr abgelehnt, obwohl
die SI-SDR-Metrik sie „besser" bescheinigte (SCNet +1,96…+3,59 dB; SGMSE ~1 dB). Die
SI-SDR-Metrik ist hier **nicht wahrnehmungsnah** — sie pries Modelle an, die das Ohr
schlechter findet. **Metriken sind Zeugen, die Hör-Instanz entscheidet.** Die hörbar
besseren Baselines (Demucs-v4, WPE-DSP) bleiben aktiv → kein Qualitätsverlust.

**Entscheidung:** Beide Rollouts verworfen (§v10.802 — kein klangveränderndes Modell
ohne Hör-Sign-off). Kein Offen-Posten außer K0-C.

### K0-C Ergebnis (2026-10-30, Hör-Instanz L01)

| Paar | Hör-Urteil |
|---|---|
| cassette_1980s_wow | Referenz (Basis) **deutlich besser** → F3 schlechter |
| jazz_1950s_scratched | Kandidat (F3) besser |
| reel_1940s_dropout | Kandidat angenehmer, aber **wesentlich dumpfer** (Höhenverlust) |

**Urteil: NO-GO.** F3 ist nicht timbre-konsistent (in Extremfällen dumpf =
Höhenverlust, genau der im Report Z.192-195 befürchtete Timbre-Sprung), dazu
1× klar schlechter vs. 1× besser. Kein konsistenter Gewinn + Timbre-Nachteil +
Kosten +0,9 % ⇒ kein Rollout-Grund. **Basis-BigVGAN bleibt aktiv.**

**Strang H (Hör-Instanz) damit KOMPLETT:** K0-A SCNet NO-GO, K0-B SGMSE NO-GO,
K0-C BigVGAN-F3 NO-GO. Alle drei Musik-Rollouts verworfen — die hörbar besseren
Baselines (Demucs-v4, WPE-DSP, Basis-BigVGAN) bleiben aktiv.
