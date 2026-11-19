# Vollmessung Release-Kandidaten — Auswertung (Versuch 2)

> **Beleg:** `docs/reports/current/2026-10-27_release_candidates_voll.json`
> (9 Kandidaten × 3 Referenz-Clips × 10 s, Seed 20261027, 27/27 Cases OK).
> Versuch 1 (Timeout 300 s) scheiterte durchgängig am Worker-Timeout —
> Beleg: `docs/reports/current/2026-10-27_release_candidates_versuch1_300s_timeout.json`.

## Metrik-Bild (Ø über 3 Clips je Kandidat)

| Flag | rmsΔ (dB) | RauschbodenΔ (dB) | clip_ratio | stereo_corr | Fallbacks |
| --- | --- | --- | --- | --- | --- |
| use_scnet_music | −2,60 | −2,73 | 0,000 | 0,866 | 9 |
| use_sgmse_musik | −2,60 | −2,74 | 0,000 | 0,866 | 9 |
| use_bigvgan_f3_studio | −2,60 | −2,74 | 0,000 | 0,866 | 10 |
| use_bw_v5 | −2,60 | −2,74 | 0,000 | 0,866 | 9 |
| use_wf4_warp_head | −2,60 | −2,74 | 0,000 | 0,866 | 11 |
| use_tp_v2_phase | −2,60 | −2,74 | 0,000 | 0,866 | 9 |
| use_rt60_regressor | **−2,07** | **−2,04** | 0,000 | 0,871 | 9 |
| use_aspade_declipper | −2,60 | −2,74 | 0,000 | 0,866 | 9 |
| use_tf_gridnet_quality | −2,60 | −2,74 | 0,000 | 0,866 | 12 |

(Quelle: `summary` + `cases` des Beleg-Reports.)

## Befunde

1. **Fallback-Quote > 0 (B1 verletzt):** 9–12 ML→DSP-Rückfälle je Lauf
   (≈ 3 pro Fenster systematisch, +1–3 flag-spezifisch; `use_tf_gridnet_quality`
   12). Das Release-Gate blockiert daher zurecht **alle** Flag-Flips
   (`flag_freigabe = "blockiert (Fallback-Quote > 0, Pfeiler 5)"`).
   Nächste Maßnahme: die systematischen ~3 Rückfälle je Fenster identifizieren
   (Komponenten-Aufschlüsselung durchreichen) und auf Budget 0 bringen.
2. **8 von 9 Kandidaten sind metrisch identisch** (−2,60 dB): ihre
   Flag-Modelle verändern das Ergebnis auf diesem Korpus noch nicht — die
   eigentliche Differenzierung kommt mit den GPU-Runden (SOTA-M8/M9,
   Blocker `hardware_gpu`). Ausnahme `use_rt60_regressor` (−2,07 dB,
   weniger aggressive Rauschsenkung, stereo_corr 0,871).
3. **Invarianten gehalten:** clip_ratio 0,000 und finite über alle 27 Cases;
   stereo_corr ≥ 0,866 (Layout-EINE-Quelle hält).
4. **Laufzeit-Kennzahl:** CPU-Vollkette > 5 min je 20-s-Fenster (Versuch 1),
   ~7–10 min je 10-s-Fenster inkl. Modell-Ladezeit (Versuch 2) — der
   GPU-Blocker ist damit auch betriebswirtschaftlich belegt.

## Konsequenz für den Wohlklang-Vertrag

B1 (Fallback-Quote 0) ist über die Messung automatisch enforced: kein
Flag-Flip über `scripts/flag_freigabe.py`, solange der Zähler > 0 zeigt.
Die Gate-Zahlen G-3/G-4/G-5 bleiben „aktiv (Messung offen)" bis die
GPU-Runden die Kandidaten trennscharf messen.
