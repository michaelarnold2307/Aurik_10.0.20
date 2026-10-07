# Hörproben-Anleitung — die menschliche Hör-Instanz (Stand 2026-10-07)

> **Zweck:** Dies ist die einzig verbleibende „Mensch"-Stufe der Wohlklang-Roadmap.
> Zwei Aktivierungen sind ohne Ihr Gehör **nicht** entscheidbar. Diese Anleitung
> führt Schritt für Schritt durch beide Hörproben; sie dauert zusammen ca. 20–30 min.
>
> **Warum Sie und nicht eine Metrik:** Nach `.github/instructions/hoerordnung.instructions.md`
> sind Metriken **Zeugen**, nicht Richter. Ein Modell darf nur dann klangverändernd
> wirken, wenn ein Mensch den Unterschied beurteilt hat (Beleg 4 des
> `.github/WOHLKLANG_CLAIMS.md`-Vertrags: „blindes A/B für einen Menschen").
> Genau dieser Beleg fehlt bei **W-1** und bei **SCNet (C4)**.

---

## 1. Die zwei offenen Entscheidungen

| # | Frage | Was entschieden wird |
| --- | --- | --- |
| **A** | **SCNet (P1-2, C4):** Klingt der SCNet-Gesangskern besser als die Demucs-v4-Baseline? | Ob SCNet die Separation übernimmt (A/B: +1,96…+3,59 dB SI-SDR) oder verworfen wird |
| **B** | **HR-V1 (W-1):** Ist die zusätzliche Höhen-Behandlung der Deckel-Fenster **hörbar**? | Ob HR-V1 wirken darf (siehe Abschnitt 6 — der Produktionspfad ist derzeit wirkungslos, deshalb Hörprobe über den Harness-Weg) |

---

## 2. Vorbereitung (einmalig, 2 Minuten)

1. **Kopfhörer** aufsetzen (keine Lautsprecher, kein Handy-Lautsprecher).
2. **Ruhige Umgebung** — keine Musik nebenbei, keine Gespräche.
3. **Lautstärke einmal einstellen** und danach **nicht mehr ändern**. Wichtig: Wenn Sie
   während der Runde lauter drehen, verzerrt das die Bewertung. Stellen Sie die Lautstärke
   an einem der ersten Ausschnitte auf „angenehm, nicht laut" und lassen Sie sie dann stehen.
4. **Kein Erfolgsdruck:** Es gibt hier kein „Richtig" oder „Falsch" im Tonfall. Wenn Sie
   nichts Auffälliges hören, ist genau das die wichtige Information — wählen Sie dann die
   Seite, die Ihnen _subjektiv_ schlechter erscheint, auch wenn Sie unsicher sind.

---

## 3. Hörprobe A starten — SCNet vs. Demucs (Präferenz)

**Schritt 1 — Player starten.** Im Projektordner ausführen:

```bash
.venv_aurik/bin/python scripts/hoerpanel_player.py \
  --study output_audio/mushra/thresholds_scnet_c4_fair \
  --port 8765
```

Es erscheint eine Zeile wie `Hör-Panel läuft auf http://localhost:8765`. Dieses Terminal
**offen lassen** (Beenden: `Strg+C`).

**Schritt 2 — Im Browser öffnen:** <http://localhost:8765>

**Schritt 3 — Hörer-Kürzel eintragen** (z. B. `M01` für Michael) und das Kästchen
„Ich trage Kopfhörer und bin in ruhiger Umgebung" ankreuzen. Dann **Start**.

**Schritt 4 — Je Runde (Trial):**

1. `▶ A hören` klicken, dann `▶ B hören`. Sie dürfen **beliebig oft** hin- und herschalten.
2. Beide Ausschnitte stammen **vom selben Musikabschnitt** (derselbe Song, dieselbe
   Sekunde) — nur das Gesangs-Separations-Ergebnis unterscheidet sich.
3. Frage: **Welches klingt klarer und natürlicher** (Stimme besser erhalten, weniger
   Artefakte)? Klicken Sie `A klingt besser` oder `B klingt besser`.
4. Tastatur geht schneller: **`A` / `B`** = abspielen, **`1` / `2`** = wählen.

**Schritt 5 — Fertig.** Nach der letzten Runde zeigt der Player „Vielen Dank". Ihre
Antworten sind bereits **automatisch** gespeichert (`answers.csv` im Studienordner).
Sie können das Fenster schließen.

> **Wichtig zu Aufwand und Aussagekraft von A:** Diese Studie hat **3 echte Trials**.
> Das ergibt einen **Hör-Eindruck** (Zeugnis), **kein** statistisch belegtes Ergebnis
> (dafür sind ≥ 8–12 Trials nötig). Behandeln Sie das Ergebnis entsprechend vorsichtig.
> Auf Wunsch erweitere ich die Studie auf mehr Songs/Fenster.

---

## 4. Hörprobe B starten — HR-V1 (Defekt-/Auffälligkeitsfrage)

Gleiches Vorgehen, anderer Studienordner:

```bash
.venv_aurik/bin/python scripts/hoerpanel_player.py \
  --study output_audio/mushra/thresholds_wohlklang_w1_hrv1 \
  --port 8766
```

Dann <http://localhost:8766> öffnen (anderer Port, damit beide parallel laufen können).

**Die Frage hier lautet:** In genau **einem** Intervall wurde eine zusätzliche
Höhen-Behandlung eingeblendet. **Hören Sie sie heraus?** Wenn Sie nichts Auffälliges
vernehmen, wählen Sie die Seite, die Ihnen künstlicher bzw. aufgesetzter erscheint.

**Diese Studie hat 8 echte Trials + 2 Fangfragen.** Sie kann ein statistisch belegtes
Ergebnis liefern (bei 8/8 Treffern liegt die untere 95 %-Grenze bei ≈ 0,68 — deutlich
über der Raterate 0,5).

**Die zwei Fangfragen** sind absichtlich **deutlich** hörbar (der Klang ist auf 3,5 kHz
Bandbreite beschnitten). Wenn Sie eine Fangfrage verfehlen, wird Ihre ganze Sitzung
automatisch als ungültig markiert. Das ist **kein Vorwurf**, sondern Qualitätssicherung:
es bedeutet, dass die Ausschnitte zu leise/zu flüchtig gehört wurden.

---

## 5. Auswerten (nach dem Hören)

```bash
# Hörprobe A (SCNet, Präferenz)
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_scnet_c4_fair --min-trials 3

# Hörprobe B (HR-V1)
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_wohlklang_w1_hrv1 --min-trials 8
```

Das Skript schreibt `fit.json` neben die Antworten und zeigt eine Tabelle. **`--min-trials`
ist die Mindestzahl auswertbarer Trials**; darunter lautet das Verdikt `ZU_WENIGE_TRIALS`
und es entsteht bewusst **kein** Beleg.

### Verdikt-Tabelle

| Verdikt | Bedeutung | Konsequenz |
| --- | --- | --- |
| `PRAEFERENZ_BELEGT` | Sie bevorzugten einen Kandidaten **signifikant** (95 %-Intervall über Raterate) | Kandidat übernimmt (SCNet) |
| `KEINE_PRAEFERENZ` | Kein Unterschied hörbar | Kandidat **nicht** übernehmen (Demucs v4 bleibt — es ist ~7× schneller) |
| `HOERBAR_BELEGT` | Die Änderung ist **hörbar** | Bei HR-V1: erst nach Klangbeurteilung aktivieren, Flag/prüfen |
| `NICHT_BELEGT` | Die Änderung ist **nicht** hörbar | Bei HR-V1: Deckel wirkt transparent → Aktivierung unkritisch |
| `ZU_WENIGE_TRIALS` | Zu wenige Antworten für einen Beleg | **Kein** Beleg — Trials erweitern |
| `UNENTSCHEIDEN` | Alle Hörer per Fangfrage ausgeschlossen | Sitzung wiederholen (Lautstärke/Umgebung prüfen) |

Zusätzlich steht in `fit.json` je Hörer `catch.valid` (Fangfragen bestanden?) und je Klasse
`above_chance` (95 %-Untergrenze > 0,5?).

---

## 6. Warum HR-V1 über einen Harness-Weg gehört wird (wichtig, ehrlich)

**Befund 2026-10-07 (gemessen, nicht vermutet):** Der HR-V1-Produktionspfad
(`plugins/bigvgan_v2_plugin.apply_hr_v1_additive`) **wirkt auf Stereo nicht**:

- Stereo (48 kHz, 2ch) → `applied=False`, beide Ausgabedateien **bit-identisch**
  (max&#124;Δ&#124; = 0,000000)
- Mono → `applied=True`, 66 Bänder freigegeben

Ursache, exakt lokalisiert: `backend/core/dsp/additive_synthesis_gate.py:181`
(`cand_cn[ch]` mit `ch = 1` auf einer Achse der Größe 1) — das Plugin synthetisiert
**mono** und reicht den Kandidaten gegen die **Stereo**-Baseline.

**Konsequenz:** HR-V1 ist seit dem Deckel-Commit (2026-10-06) für reale Stereomusik
**wirkungslos** — der Budget-Nachweis (1,04× RT) gilt, der **Wirkungsnachweis fehlte**.
Registriert als **D-K3-6** in `.github/SOTA_DEFICIT_REGISTER.md`.

Die Hörprobe B wurde deshalb **kanalkorrekt im Harness** gerendert (Mono-Synthese auf
beide Kanäle gespiegelt). Sie hört damit **das, was HR-V1 nach einer Ursachenbehebung
tun würde** — nicht den heutigen Produktionszustand. Das ist in
`output_audio/hr_v1_hoerprobe/generation.json` (`"weg": "HARNESS …"`) festgehalten.

**Konsequenz der Hörprobe B:**

- Ergebnis **`NICHT_BELEGT`** (nicht hörbar) → Ursache beheben und aktivieren ist
  qualitätsneutral; der Weg ist frei.
- Ergebnis **`HOERBAR_BELEGT`** (hörbar) → vor der Aktivierung entscheiden, ob die
  Änderung **wohklangfördernd** oder **aufgesetzt** klingt. Bei „aufgesetzt": W-1 auf
  `gesperrt` setzen (Deckel greift zu kurz/zu randnah) statt zu aktivieren.

---

## 7. Was danach passiert (ich übernehme)

Nach Ihrem Verdikt:

1. **`.github/WOHLKLANG_CLAIMS.md`** wird in der Zeile **W-1** (HR-V1) bzw. der
   SCNet-Zeile auf den belegten Stand gesetzt — Status wechselt von `ausnahme` auf
   `aktiviert` (Belege vollständig) oder `gesperrt`.
2. **`.github/SOTA_DEFICIT_REGISTER.md`**: D-K3-6 wird auf `geschlossen` (behoben)
   oder `bewusst-akzeptiert` gesetzt — jeweils mit Ihrem `fit.json` als Belegpfad.
3. Bei „beheben": die Kanalangleichung im Plugin bzw. der Broadcast im Gate wird
   implementiert, mit §III.9-Paritätsnachweis (`rel ≤ 1e-3`) und Regressionstest.
4. Version-Bump nach §v10.802 (Patch/Minor je nach Änderung) und Changelog-Eintrag.

---

## 8. Grenzen dieser Anleitung (bewusst benannt)

- **A hat 3 Trials** → Hör-Eindruck, kein statistischer Beleg. Für einen echten
  Beleg: mehr Songs (die drei MUSDB-Songs sind vorbereitet, es fehlen nur weitere
  Wiederholungen mit anderen Zeitfenstern).
- **Die HR-V1-Fenster liegen randnah.** Die gleichmäßige Verteilung (bewusst ohne
  Inhaltsauswahl, §V7) legt bei kurzen Ausschnitten Fenster an den **Anfang und das
  Ende**. Dort ist Musik oft ausgeblendet — dann ist die Behandlung schwächer hörbar als
  mitten im Stück. Das ist eine Eigenschaft der Fensterpolitik, keine Ihrer Hörleistung.
- **Ein Hörer ist kein Panel.** Wohlklang-Entscheidungen mit Breitenwirkung sollten
  mehrere Hörer haben (MUSHRA-Studie, Roadmap B2). Für die zwei hier offenen
  Aktivierungen genügt die dokumentierte Einzelabnahme als **Zeugenaussage**.

---

## 9. Kommando-Sammlung (Kopieren → Einfügen)

```bash
cd "/media/michael/Software 4TB/Aurik_Standalone"

# --- A: SCNet vs. Demucs (Präferenz) ---
.venv_aurik/bin/python scripts/hoerpanel_player.py \
  --study output_audio/mushra/thresholds_scnet_c4_fair --port 8765
#   Browser: http://localhost:8765

# --- B: HR-V1 (Auffälligkeit) ---
.venv_aurik/bin/python scripts/hoerpanel_player.py \
  --study output_audio/mushra/thresholds_wohlklang_w1_hrv1 --port 8766
#   Browser: http://localhost:8766

# --- Auswerten ---
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_scnet_c4_fair --min-trials 3
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_wohlklang_w1_hrv1 --min-trials 8

# --- Studien neu bauen (nur falls nötig) ---
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-build \
  --build wohlklang_w1_hrv1 \
  --manifest docs/reports/current/2026-10-07_hoerprobe_w1_hrv1.json --seed 20261007
```

**Dateiorte der Artefakte:**

| Was | Pfad |
| --- | --- |
| SCNet-Hördateien (Referenz) | `output/scnet_ab_2026-10-06_fair/` |
| HR-V1-Hördateien (Harness) | `output_audio/hr_v1_hoerprobe/` |
| Studienordner | `output_audio/mushra/thresholds_*/` |
| Antworten | `output_audio/mushra/thresholds_*/answers.csv` |
| Ergebnis | `output_audio/mushra/thresholds_*/fit.json` |
