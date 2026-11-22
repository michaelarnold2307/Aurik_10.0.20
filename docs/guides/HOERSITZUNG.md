# Hörsitzung — ein Befehl, keine Programmierarbeit

Ersetzt das manuelle Anhören einzelner Pakete (Declip, Witness-Köpfe, M5, Stereo, Dereverb).
Normativ: Spec 07 §8.8. Das Werkzeug erzeugt Evidenz, **bucht aber nie ein Flag** und
fingiert nie ein Urteil.

## Ablauf

```bash
cd "/media/michael/Software 4TB/Aurik_Standalone"
python3 -B scripts/hoersitzung.py --hoerer DEINNAME
```

Der Befehl baut die Sitzung beim ersten Aufruf (wenige Sekunden), spielt alle Proben
verblindet ab und wertet am Ende automatisch aus. Ohne Argumente: `los`.

- Ort: `output_audio/hoersitzung/sitzung_s42/` (git-ignoriert).
- Dauer: im Plan angegeben (aktuell ≈ 5 Minuten für 12 Proben, davon 6 Kontrollen).
- Abbrechen jederzeit mit `q`; derselbe Befehl setzt genau dort fort (jede Antwort ist sofort gesichert).

## Hörumgebung (einmalig)

Gute Kopfhörer, ruhiger Raum, Systemlautstärke einmal einstellen (die erste Probe ist
dafür gedacht) und nicht mehr verändern. Keine Klangregler/Enhancer.

## Tasten

| Taste | Wirkung |
|---|---|
| `1` / `2` | die erste / zweite Hälfte klingt besser (natürlicher, angenehmer) |
| `=` | kein hörbarer Unterschied |
| `w` | Probe wiederholen |
| `z` | letzte Antwort zurücknehmen |
| `p` | Pause |
| `q` | beenden (fortsetzbar) |

Jede Probe besteht aus zwei Hälften mit 0,8 s Stille dazwischen. Du erfährst nie, was
Bestand und was Kandidat ist. Zwischen den echten Paaren liegen Kontrollen (Tiefpass-Anker
und identische Paare); sie entscheiden, ob dein Urteil gültig ist.

## Ergebnis lesen

`ergebnis/bericht.md` (Übersicht), `ergebnis/hoer_verdikt_<block>.json`
(`aurik.ab_stats/2`, direkt für `flag_freigabe.py`), `ergebnis/signoff_entwurf_<block>.md`
(nur bei buchbaren Blöcken; **Maintainer Sign-off: AUSSTEHEND** — der Maintainer füllt ihn aus).

Buchbar ist ein Block nur, wenn alles zutrifft: signifikant **für den Kandidaten**
(exakter Vorzeichentest, p < 0,05), gültiger Hörer (Anker-Quote ≥ 0,75), alle Paare
beurteilt, kein simulierter Lauf, Flag vorhanden. Ein signifikantes Votum für den Bestand
bucht nichts.

## Buchen (nur wenn der Bericht „buchbar“ ausweist)

```bash
.venv_aurik/bin/python -B scripts/flag_freigabe.py \
  --flag <flag-aus-dem-bericht> \
  --verdikt output_audio/hoersitzung/sitzung_s42/ergebnis/hoer_verdikt_<block>.json \
  --signoff <ausgefüllter-signoff.md> --commit
```

`aurik.ab_stats/2` trägt die Richtung selbst. Ältere `ab_stats/1`-Belege sind nur mit
`--kandidat-seite A|B` buchbar (richtungsblinde Belege sind abgewiesen).

## Zurückgestellte Blöcke

**Zu kurz zum Hören:** Die 12 Witness-Pakete bestehen aus isolierten Defektregionen von
0,05–0,44 s ohne Musik. Darüber ist kein Wohlklang-Urteil möglich (Befund 2026-11-22: erster
Hörversuch). Blöcke mit weniger als 2 s je Hälfte werden nicht abgespielt (auch nicht mit
`--auch-nicht-entscheidbare`). Nachbau: Paare mit musikalischem Kontext (Defekt eingebettet
in ≥ 4 s Musik) neu erzeugen — `scripts/erstelle_witness_kopf_hoerproben.py` schneidet
bisher nur die Region aus.

**Statistisch unentscheidbar:**

Der exakte Vorzeichentest erreicht bei n Paaren höchstens p = 2/2ⁿ; Signifikanz braucht
n ≥ 6 (ohne Gleichstände). Blöcke mit weniger Paaren (Declip 3, Stereo 2, Jitter 1,
Transport 1, Dereverb 5) werden nicht abgespielt — das kostete Menschenzeit ohne
mögliches Ergebnis. Sie brauchen weitere Paare (Nachbau mit
`scripts/erstelle_aspade_hoerpaare.py` bzw. `scripts/erstelle_hoerproben.py`, schwerer
Lauf — nicht während eines laufenden Trainings). Mit `--auch-nicht-entscheidbare`
werden sie trotzdem gehört (nur Hinweis, nie buchbar).

Pakete mit byte-identischen A/B (aktuell: C4/Vocos) werden mit Grund ausgeschlossen; sie
müssen neu erzeugt werden.

## Grenzen

Ein Hörer ist ein Zeuge, kein Panel. Das MUSHRA-Panel n ≥ 30 bleibt über
`scripts/mushra_harness.py` / `scripts/hoerpanel_player.py` offen. Pegel-Toleranz (0,5 dB)
und Anker-Mindestquote (0,75) sind Vorbelegungen ohne Hörpanel-Kalibrierung.
Mehrere Hörer (`--hoerer` je Person, gleiche Sitzung) zählen je Paar per Mehrheit.
