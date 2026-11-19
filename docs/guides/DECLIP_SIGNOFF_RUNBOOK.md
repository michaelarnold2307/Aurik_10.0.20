# Declip-Sign-off-Runbook — Musik-Declip-Kopf (§4, menschlich)

> **Zweck:** Der Mensch bekommt den EIN-Befehl-Weg von den erzeugten Hörpaaren
> bis zum Flag-Flip (oder zur Ablehnung). Nichts hier flippt ohne Sign-off —
> `flag_freigabe.py` ist fail-closed (verifiziert 2026-11-14, siehe unten).
> Stand: 2026-11-14. Quellen: `docs/reports/current/2026-11-14_hoerpaket_declip_evidenz.md`,
> Handover-Punkte 17–19/23.

## 0. Was schon belegt ist (objektiv, ohne Hören)

| Beleg | Ergebnis |
| --- | --- |
| Leak-freier Produktions-Gate-Lauf, 10 MUSDB-Titel 61–90 (nie im Training) | **4/10 besser, 0/10 schlechter** über JND — Never-worsen ✓ |
| k398-Gegenmaterial leave-one-out | **3/3 besser** (Boogie +15,6 dB SDR) |
| **k398-ΔSDR-Gate NEUMESSUNG 2026-11-17** | **BESTANDEN** (Träumer −13,722 → −12,916 dB; Boogie/Jets never-worse; pchip-nw ✓, harmonik ✓) — objektiver Teil der D-K4-1-Aktivierung erfüllt, §4-only |
| Produktionspfad-Erreichbarkeit | 6/6 Skip-Schichten gefixt; Probe v11: Phase läuft, Flag-Wirkung delta 1,86 |
| Hörpaket | 3 blinde A/B-Paare, 10 s, 48 kHz, p25-Pegelmatching (rel. Delta 0,00 %), Schlüssel getrennt — NEU RENDERN nach dem 48-kHz-Erkenntnis-Fix (`python3 -B scripts/erstelle_aspade_hoerpaare.py --produktionspfad`), sobald die GPU frei ist (M9-Retrain läuft) |

## 1. Hörpaket prüfen

```bash
cd "/media/michael/Software 4TB/Aurik_Standalone"
ls study_stimuli/hoerproben/declip_produktionspfad/
# erwartet: 3 Paare × {A,B}.wav (Boogie/Träumer/Jets) + manifest.json + schluessel.json
```

## 2. Blind hören + Urteil

A/B je Paar anhören (Reihenfolge der Schlüssel erst NACH dem Urteil öffnen).
Urteil als `aurik.ab_stats/1`-Beleg festhalten (mindestens `signifikant`,
`siege_a`, `siege_b`, `p_wert_vorzeichen_test`).

**Ablehnung (nicht signifikant):** kein Flip — `flag_freigabe.py` bricht mit
`ABBRUCH: Verdikt nicht buchbar` ab (Exit 2), das Flag bleibt AUS. Genau
dieser Pfad ist unten als Dry-Run belegt.

## 3. Sign-off buchen (nur bei signifikantem Vorteil)

```bash
.venv_aurik/bin/python -B scripts/flag_freigabe.py \
  --flag use_aspade_declipper \
  --verdikt <beleg-aurik.ab_stats.json> \
  --signoff docs/reports/current/<datum>_declip_hoer_signoff.md
# prüfen: Ausgabe "Register gebucht, Flag geflippt …" + Nach-Gates grün
git add -A && git commit -m "feat(ml-sota): Flag-Flip use_aspade_declipper — Hör-Sign-off"
```

## 4. Rollback (falls das Ohr später kippt)

```bash
# Flag in backend/core/music_model_flags.py zurück auf False,
# Register-Zeile D-K4-1 wieder auf offen, Tests zurück, committen.
```

## 5. Fail-closed-Beweis (2026-11-14, durchgeführt)

- Nicht-signifikanter Test-Verdikt (`signifikant=False`) ⇒ `ABBRUCH: Verdikt
  nicht buchbar`, Exit 2, `backend/core/music_model_flags.py` byte-identisch.
- Fehlende Verdikt-Datei ⇒ Exit 2.
- `tests/unit/test_flag_freigabe.py`: 7/7 grün.
- Hörpaket-Vollständigkeit: 6 WAVs (48 kHz, 10 s, mono), `manifest.json`
  (`aurik.hoerpaare/1`, 3 Paare, Seed 42), `schluessel.json` vorhanden.
