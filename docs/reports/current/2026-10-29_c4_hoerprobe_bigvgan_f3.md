# C4-Hörprobe — BigVGAN-F3 Timbre-Konsistenz (Hörordnung Ebene 4)

> **Zweck:** Aktivierungs-Gate für das stille Kapital **BigVGAN-F3**
> (D-K0-1, `models/bigvgan/bigvgan_v2_f3.onnx`). F3 ist via „Weg 2" in
> `phase_07_harmonic_restoration` implementiert (eine Aufrufstelle,
> Längen-Deckel `BIGVGAN_V2_HR_MAX_DUTY`), aber **rollout-gesperrt** bis zum
> C4-Sign-off. Dieses Paket ist der Weg zum Sign-off.
>
> **Warum der größte Hebel:** F3 ist das am weitesten gediehene stille
> Kapital (**+4,4 dB HNR** gemessen, ONNX-paritätisch, Rollout-Entscheid
> dokumentiert). Die C4-Hörprobe trifft den Kern des Ziels auf **zwei** Achsen:
> (1) sie aktiviert einen realen Qualitäts-Gewinn, (2) sie liefert den
> Hörordnungs-Beweis „für das menschliche Ohr" — und schaltet damit zugleich
> die anderen stillen Käufner (SGMSE-Musik, SCNet) frei.

## Abnahmekriterium — Hörordnung Ebene 4 (Einladungs-Gate, positiver Nachweis)

Der F3-Restaurierungspfad wird als **timbre-konsistent** abgenommen, wenn auf
den gleichmäßig verteilten Ausschnitten (Längen-Deckel `BIGVGAN_V2_HR_MAX_DUTY`):

* **keine Roughness-Spitze** `asper > 0.5` in Stimmen-/Klimax-Zonen,
* **keine Sharpness-Sprünge** `acum > 0.2` zwischen benachbarten 5-s-Fenstern
  (überlappend, Roughness/Sharpness/Loudness-Gate),
* **Fatigue < 0.40** (sonst Abbruch),
* **§SCK-R** (Ebene 1): Spektralfarben-Korrelation `≥ 0.60` (sonst
  Phasen-Rücknahme).

Quelle: `.github/instructions/hoerordnung.instructions.md` §6 (Ebene 4) +
§3 (§SCK-R) + §7 (Konfliktregel). Ein „EXZELLENT" (Goosebumps/HPE) ersetzt
das Gate **nicht**.

## Hör-Anleitung (MUSHRA, ITU-R BS.1534, doppelblind)

Das Paket wird über `scripts/mushra_harness.py` aufgebaut — drei Schritte:

```bash
# 1) Stimuli: deterministische Restauration der Corpus-Paare (F3 via Weg 2)
python3 -B scripts/mushra_harness.py prepare --modes balanced maximum --seconds 12

# 2) Doppelblinde Hör-Sessions je Trial [clean (hidden ref), 3.5-kHz-LP-Anker, Aurik(+F3)]
python3 -B scripts/mushra_harness.py sessions --listeners 5 --out c4_bigvgan_f3_run1

# 3) Nach dem Hören: MUSHRA-Scores 0–100, 95 %-CI, Hör-Urteil
python3 -B scripts/mushra_harness.py analyze --answers c4_bigvgan_f3_run1/answers.csv
```

Beim Hören gilt: **Timbre-Konsistenz** der F3-Ausschnitte prüfen — keine
Rauheit/„Metallisch"-Färbung, keine plötzlichen Brillanz-Sprünge zwischen
benachbarten Fenstern, stimmliche Natürlichkeit (Hörordnung Ebene 1–3).

## Sign-off (Hörordnung-Instanz)

| Feld | Eintrag |
|---|---|
| Hör-Datum | |
| Hörer/Instanz | |
| Trials gesamt / gültig | |
| Roughness-Verstöße (`asper > 0.5`) | 0 erwartet |
| Sharpness-Sprünge (`acum > 0.2`) | 0 erwartet |
| §SCK-R Spektralfarben-Korr. | ≥ 0.60 erwartet |
| Fatigue-Spitzen (> 0.40) | 0 erwartet |
| **Urteil** | ☐ ABNAHME (Timbre konsistent) · ☐ RÜCKNAHME |
| Begründung / Klangprobe-Notiz | |

**Bei ABNAHME:** F3-Rollout freischalten (`D-K0-1` → geschlossen;
Budget-Ausnahme „Weg 2" als aktiv dokumentieren). Das Sign-off ist die
**Hörordnungs-Instanz** (menschliches Ohr) — nicht die Metrik.

**Bei RÜCKNAHME:** §3 (§SCK-R) Phasen-Rücknahme / §7 Konfliktregel anwenden,
F3 bleibt gesperrt; Begründung im Register `D-K0-1` festhalten.

## Verwandte stille Käufner (dasselbe C4-Gate)

* **D-K0-3 SGMSE-Musik** (`use_sgmse_musik=False`) — Metrik-A/B inconclusive
  (SI-SDR straft Diffusion), Hörabnahme ist der Treiber.
* **D-K0-5 SCNet** (`use_scnet_music=False`) — C4-Runde 2 gebaut
  (`thresholds_scnet_c4_fair_r2`, 12 Trials + 2 Fangfragen).

Ein positives C4-Sign-off für F3 etabliert das Verfahren; die anderen beiden
können mit demselben Gate abgeschlossen werden.

## Belege

* Rollout-Entscheid (Weg 2, Kosten, +0,9 %): `docs/reports/current/2026-10-06_p1_3_bigvgan_f3_rollout_entscheid.md`
* F3-A/B (HNR +4,42 dB): `docs/reports/current/2026-09-16_hr_v1_bigvgan_ab_validation.md`
* Hörordnung Ebene 4 + §SCK-R: `.github/instructions/hoerordnung.instructions.md`
* F3-Artefakt: `models/bigvgan/bigvgan_v2_f3.onnx` (Parität rel ≤ 6,5e-05)
* Defizit-Register: `.github/SOTA_DEFICIT_REGISTER.md` `D-K0-1`
