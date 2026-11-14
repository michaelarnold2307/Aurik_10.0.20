# Declip-Sign-off-Runbook — Musik-Declip-Kopf (§4, menschlich)

> **Wozu:** Ein-Befehl-Ablauf für den blinden A/B-Hör-Sign-off des
> Musik-Declip-Kopfs (`use_aspade_declipper`). Der Flag-Flip darf NUR hier,
> vom Menschen, ausgelöst werden — `flag_freigabe.py` ist fail-closed und
> lehnt alles ab, was kein belegter Vorteil ist.
> Stand: 2026-11-14. Evidenz-Bundle:
> `docs/reports/current/2026-11-14_hoerpaket_declip_evidenz.md`.

## 0. Voraussetzungen (verifiziert 2026-11-14)

- Hörproben: `study_stimuli/hoerproben/declip_produktionspfad/` — 3 blinde
  A/B-Paare (Boogie/Träumer/Jets), 48 kHz, 10 s, p25-Pegelmatching
  (rel. Delta 0,00 %), Schlüssel in `schluessel.json` (getrennt aufbewahrt).
- Objektive Gate-Belege (bereits erbracht): MUSDB leak-frei **4/10 besser,
  0/10 schlechter**; k398-LOO **3/3** (Boogie +15,6 dB SDR);
  Produktionspfad-Erreichbarkeit 6/6 Schichten belegt.
- `flag_freigabe.py` fail-closed **getestet**: 7/7 Vertragstests grün;
  Ablehnungs-Pfad verifiziert (nicht-signifikanter Verdikt ⇒
  „ABBRUCH: Verdikt nicht buchbar“, Exit 2, Flag-Datei byte-identisch).

## 1. Blind hören

```bash
cd "/media/michael/Software 4TB/Aurik_Standalone"
# Paare in study_stimuli/hoerproben/declip_produktionspfad/ abspielen
# (ffplay/paplay/VLC — NICHT schluessel.json öffnen, solange gehört wird)
```

Jedes Paar: A oder B klingt besser? (Beleg-Schema `aurik.ab_stats/1`.)

## 2. Urteil aufzeichnen

`scripts/hoer_entscheidung.py` nutzt dieselben Schemata; für den
Produktionspfad-Flag-Flip ist ein `aurik.ab_stats/1`-Beleg nötig:

```json
{
  "schema": "aurik.ab_stats",
  "signifikant": true,
  "siege_a": <n>, "siege_b": <m>,
  "p_wert_vorzeichen_test": <p>
}
```

Wichtig: `flag_freigabe.py` akzeptiert NUR `signifikant=true` mit
ungleichem Sieg-Bild — alles andere ist eine ehrliche Ablehnung
(fail-closed, oben verifiziert).

## 3. Sign-off + Flag-Flip (nur dieser Schritt aktiviert)

```bash
.venv_aurik/bin/python -B scripts/flag_freigabe.py \
  --flag use_aspade_declipper \
  --verdikt <pfad-zum-ab_stats-beleg.json> \
  --signoff docs/reports/current/<datum>_declip_signoff.md \
  --commit
```

Danach Kontrolle:

```bash
grep -n "use_aspade_declipper" backend/core/music_model_flags.py
# muss "True ... — AKTIV nach Hör-Sign-off (Flag-Flip <datum>, Beleg: ...)" zeigen
```

## 4. Rollback (Flag zurück auf AUS)

Falls das Hör-Urteil später kippt:

```bash
# Flag-Zeile in backend/core/music_model_flags.py manuell auf
# use_aspade_declipper: bool = False  # SOTA-M8 (D-K4-1): ...
# zurücksetzen und committen (Conventional: fix(flags): ...) —
# die zugehörige Register-Zeile (D-K4-1) wieder auf "offen" setzen.
```

## 5. Ablehnung (häufigster Fall)

Nicht signifikant? Nichts flippen. `flag_freigabe.py` bricht mit
„Verdikt nicht buchbar“ ab (Exit 2) — das Flag bleibt AUS, und genau so
ist der Vertrag: never-worsen ist der Boden, §4 der Richter.
