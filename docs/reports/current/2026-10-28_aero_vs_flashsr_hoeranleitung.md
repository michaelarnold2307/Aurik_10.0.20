# Hör-Runde AERO vs. FlashSR — Anleitung (§9/6, D-K2-5)

> **Vorbereitet:** 2026-10-28. Die Entscheidung ist Ihr SOTA-Urteil
> (§9/6 der A2-Liste: „MUSHRA-Challenger-Runde (bisheriger A/B-Verlust gegen
> FlashSR)"). Diese Runde ersetzt den bisherigen reinen A/B durch eine
> doppelblinde, pegelgematchte Präferenz-Studie.

## Studie

- **Ort:** `output_audio/mushra/thresholds_aero_vs_flashsr/`
  (meta.json, trial_key.json, answers_template.csv, trials/)
- **Aufgabe (task=preference):** „Welches klingt besser?" — kein „richtig",
  nur Präferenz.
- **Trials:** 3 Items (cassette/reel/vinyl, je 10 s aus dem Referenz-Korpus
  `test_audio/tape|vinyl/…`) + 1 Fangfrage (Referenz vs. 3,5-kHz-Anker).
- **Bedingungen je Item:** AERO (12-kHz-Degradation → 48 kHz) gegen FlashSR
  (16-kHz-Bandlimit → Bandbreiten-Extension), **level-matched**, Seiten
  seed-deterministisch vertauscht (Seed 42) — doppelblind.
- **Stimulus-Herkunft:** `output_audio/mushra/aero_vs_flashsr_stimuli/`
  (Referenz-, AERO-, FlashSR-WAVs je Item; Manifest
  `output_audio/mushra/aero_vs_flashsr_manifest.json`).

## Hören

1. Player starten:
   `python3 -B scripts/hoerpanel_player.py --study output_audio/mushra/thresholds_aero_vs_flashsr --port 8765`
2. Pro Trial das bessere Fragment wählen; bei der Fangfrage das
   **unverfälschte** Fragment (der hörbar dumpfere 3,5-kHz-Anker ist falsch).
3. Antworten in `answers.csv` (Vorlage: `answers_template.csv`):
   `listener, trial_id, answer_interval` (`A`/`B`).

## Auswertung (danach)

```
python3 -B scripts/mushra_harness.py thresholds-fit \
    --study output_audio/mushra/thresholds_aero_vs_flashsr
```

- `fit.json` → `verdict`:
  - `PRAEFERENZ_BELEGT`: signifikante Präferenz (95 %-CI über Zufall).
  - `KEINE_PRAEFERENZ`: kein signifikanter Unterschied.
  - `ZU_WENIGE_TRIALS` / `UNENTSCHEIDEN`: ehrlich — mehr hören.
- **Rollen-Zuordnung (wichtig!):** Die „reference"-Rolle = **AERO**,
  die „candidate"-Rolle = **FlashSR** (siehe Manifest `rollen`). Die
  Trefferquote im fit-Report ist damit der **AERO-Präferenz-Anteil**.
- Fangfrage verfehlt ⇒ Hörer fliegt aus der Wertung (gilt als nicht
  hinhörend).

## Danach (Entscheidung §9/6)

- Bei belegtem AERO-Vorteil: Freigabe-Kette gemäß §v10.802 (copilot-instructions.md)
  (Hör-Sign-off → `scripts/flag_freigabe.py`).
- Bei `KEINE_PRAEFERENZ`/AERO-Verlust: SOTA-Urteil „FlashSR beibehalten",
  AERO archivieren (`scripts/kandidaten_archiv.py --id D-K2-5`).
