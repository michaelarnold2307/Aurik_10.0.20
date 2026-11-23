# VinylNet-Retrain (CC0-Leerrillen-Bank) — Testsplit-Re-Eval gegen die Basis

**Datum:** 2026-11-23 · **Werkzeug:** `models/de_crackle/eval.py` (MUSDB18-HQ-Testsplit,
n=16 × 8 s, Seed-Auswahl je Clip, Noise-Bank `data/vinyl_noise_bank`, MERT-Zeuge
`models/mert/mert_330m.onnx`, GPU) · **D-K1-4 / Parity-Zeile Decrackle.**

## Ergebnisse (SI-SDR [dB] / MERT-Sim)

| Schweregrad | unbearbeitet | Basis `vinylnet_base_best.pt` | Retrain `final_infer.pt` |
| --- | --- | --- | --- |
| 0,0 (Clean) | 19,66 / 0,978 | 14,91 / 0,978 | **19,54 / 0,996** |
| 0,3 | 20,43 / 0,988 | 15,99 / 0,988 | **19,76 / 0,997** |
| 0,6 | 16,66 / 0,988 | 13,98 / 0,988 | **17,40 / 0,996** |
| 1,0 | 14,42 / 0,984 | 14,46 / 0,986 | **17,12 / 0,997** |

## Einordnung (ehrlich)

1. **Never-worsen gegen die Basis BESTANDEN und übertroffen:** Auf allen vier
   Schweregraden gewinnt der Retrain deutlich (+2,66…+3,77 dB SI-SDR) und
   verbessert den MERT-Zeugen überall (0,988→0,997 u. ä.).
2. **Clean-Schutz entscheidend verbessert:** Die deployte Basis kostete auf
   sauberem Material −4,75 dB SI-SDR (14,91); der Retrain lässt Clean
   praktisch unangetastet (19,54 vs. 19,66 Referenz, −0,12 dB) — der
   never-worsen-Fall ist damit von „deutlich schädlich“ auf „neutral“
   repariert.
3. **Gegen „nichts tun“:** Auf starkem Knistern (sev 0,6/1,0) liegt der
   Retrain über „unbearbeitet“ (17,40 > 16,66 bzw. 17,12 > 14,42); auf
   leichtem Knistern (sev 0,3) bleibt er knapp darunter (19,76 vs. 20,43) —
   keine Überlegenheits-Behauptung für leichte Fälle.
4. **Klickdetektion (AR-Detektor) ist unverändert die Schwachstelle**
   (F1 0,009–0,094) — der Gewinn stammt aus dem Restaurations-Kern, nicht
   aus der Detektion.

## Konsequenz

Der Retrain (`models/de_crackle/runs/vinylnet/final_infer.pt`, aus dem
100/100-Epochen-Lauf mit CC0-Leerrillen-Bank, SI-SDR +4,89 dB, Log
`/tmp/vinylnet_resume.log`) ist der **objektiv überlegene Decrackle-Kandidat**.
Das Deployment (Austausch von `vinylnet_base_best.pt`) ist **klangwirksam**
⇒ §4-Verfahren mit Hör-Instanz (Repo-Konvention D-K1-5/D-K3-94) — nicht
autonom. Nächste Schritte (menschlich): (a) Deployment-Entscheidung,
(b) RX-12-Vergleich (DRT-16), (c) §4-Hör-Sign-off. Kein Flag-Flip, kein Deploy.
