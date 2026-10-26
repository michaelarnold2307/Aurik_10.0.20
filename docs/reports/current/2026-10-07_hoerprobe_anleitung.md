# Hörproben-Anleitung — die menschliche Hör-Instanz (Stand 2026-10-07, erweitert 2026-10-23 um Runde C)

> **Zweck:** Dies ist die einzig verbleibende „Mensch"-Stufe der Wohlklang-Roadmap.
> Drei Entscheidungen sind ohne Ihr Gehör **nicht** entscheidbar. Diese Anleitung
> führt Schritt für Schritt durch alle drei Hörproben; sie dauert zusammen ca. 30–45 min.
>
> **Warum Sie und nicht eine Metrik:** Nach `.github/instructions/hoerordnung.instructions.md`
> sind Metriken **Zeugen**, nicht Richter. Ein Modell darf nur dann klangverändernd
> wirken, wenn ein Mensch den Unterschied beurteilt hat (Beleg 4 des
> `.github/WOHLKLANG_CLAIMS.md`-Vertrags: „blindes A/B für einen Menschen").
> Genau dieser Beleg fehlt bei **W-1** und bei **SCNet (C4)**.

---

## 1. Die drei offenen Entscheidungen

| # | Frage | Was entschieden wird |
| --- | --- | --- |
| **A** | **SCNet (P1-2, C4):** Klingt der SCNet-Gesangskern besser als die Demucs-v4-Baseline? | Ob SCNet die Separation übernimmt (A/B: +1,96…+3,59 dB SI-SDR) oder verworfen wird |
| **B** | **HR-V1 (W-1):** Ist die zusätzliche Höhen-Behandlung der Deckel-Fenster **hörbar**? | Ob HR-V1 wirken darf (siehe Abschnitt 7 — der Produktionspfad ist seit dem Kanal-Fix wirksam, deshalb Hörprobe über den Produktionspfad) |
| **C** | **SGMSE+ (D-K0-3):** Trägt der Musik-Finetune-Core gegen Sprach-Core und WPE-DSP? | Ob `use_sgmse_musik` geflippt wird, ob nur WPE bleibt oder beide Cores gesperrt bleiben |

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

## 5. Hörprobe C starten — SGMSE+ 3-Wege (Präferenz)

Gleiches Vorgehen, anderer Studienordner:

```bash
.venv_aurik/bin/python scripts/hoerpanel_player.py \
  --study output_audio/mushra/thresholds_sgmse_3weg \
  --port 8767
```

Dann <http://localhost:8767> öffnen (anderer Port, damit alle parallel laufen können).

**Was Sie hören:** In jedem Trial stammen beide Ausschnitte **vom selben degradierten
Gesangsausschnitt** (synthetischer Nachhall + Rauschen) — nur die Restaurierungs-Verfahren
unterscheiden sich. Drei Vergleichspaare rotieren: Musik-Finetune-Core gegen Sprach-Core,
Musik-Finetune-Core gegen WPE-DSP und Sprach-Core gegen WPE-DSP. Alle Arme sind auf
gleiche Lautheit gebracht (RMS-Anker + Level-Match), damit Lautstärke kein Hinweisreiz ist.

**Die Frage lautet:** Welches klingt klarer und natürlicher — Hall/Rauschen besser entfernt,
Stimme ohne Verfärbung?

**Diese Studie hat 18 echte Trials (6 je Vergleichspaar) + 2 Fangfragen.** Die Auswertung
liefert je Klasse einen Belegwert; die Entscheidung folgt aus den beiden Klassen mit dem
Musik-Core (`sgmse_musik_vs_wsj0`, `sgmse_musik_vs_wpe`): trägt er in **beiden**, wird
`use_sgmse_musik=True`; verliert er gegen WPE, ist der WPE-only-Weg die Entscheidung.

**Die zwei Fangfragen** sind wie bei B absichtlich deutlich hörbar (3,5-kHz-Tiefpass).

---

## 6. Auswerten (nach dem Hören)

```bash
# Hörprobe A (SCNet, Präferenz)
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_scnet_c4_fair --min-trials 3

# Hörprobe B (HR-V1)
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_wohlklang_w1_hrv1 --min-trials 8

# Hörprobe C (SGMSE+ 3-Wege, Präferenz je Klasse)
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_sgmse_3weg --min-trials 4
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

## 7. Stand von HR-V1 (Ursache behoben — Hörprobe entscheidet über den Verbleib)

**Befund 2026-10-07 (gemessen):** Der HR-V1-Produktionspfad
(`plugins/bigvgan_v2_plugin.apply_hr_v1_additive`) **wirkte auf Stereo nicht**:
Stereo → `applied=False`, beide Ausgabedateien bit-identisch; Mono →
`applied=True`, 66 Bänder. Ursache exakt lokalisiert: die Kanal-Schleife in
`backend/core/dsp/additive_synthesis_gate.py` lief über die **Baseline**-Kanäle
und indexierte darin den **Kandidaten** (`cand_cn[ch]` mit `ch = 1` auf einer
Achse der Größe 1) → `IndexError`, den **alle drei** Aufrufer (HR-V1, §B5
Vocoder-Gate, §B4-FlashSR-Gate) still in ihren Ersatzpfad schluckten. Die
maskierungsbewusste Schutzschicht war damit auf Stereo **wirkungslos** —
ungeprüfte Synthese-Energie wurde durchgelassen.

**Behoben am 2026-10-07 (Ursache, nicht Symptom):** Das Gate gleicht die
Kanalzahl jetzt **vor** den Schleifen an — ein 1-Kanal-Kandidat wird auf die
Baseline-Kanalzahl gespiegelt (additive Synthese entsteht häufig im Mono-Bereich),
jede andere Abweichung ist fail-closed (Baseline unverändert). Damit ist der
Fix an **einer** kanonischen Stelle für alle drei Aufrufer wirksam
(§G9 copilot-instructions.md). Gemessen am Produktionspfad:

| Ausschnitt | `applied` | Bänder | außerhalb Deckel-Fenster |
| --- | --- | --- | --- |
| 20 s / 60 s / 100 s / 140 s | **True** (vorher False) | 47 / 32 / 33 / 22 | **bit-identisch** |

**Was das für Ihre Hörprobe B bedeutet:** Sie hört jetzt den **echten
Produktionszustand** — die Paare wurden über `apply_hr_v1_additive` erzeugt
(`"weg": "PRODUKTIONSPFAD"` im Manifest), nicht mehr über einen Harness-Weg.
Die Wirkung ist real (max&#124;Δ&#124; 0,14–0,43 in den Deckel-Fenstern); außerhalb bleibt das
Signal unangetastet.

**Ihre Entscheidung:**

- **`NICHT_BELEGT`** (nicht hörbar) → der Deckel wirkt transparent; HR-V1 kann
  bleiben (W-1 → `aktiviert`).
- **`HOERBAR_BELEGT`** (hörbar) → Sie beurteilen, ob die Höhen-Behandlung
  **wohltut** oder **aufgesetzt** klingt. Bei „aufgesetzt": W-1 auf `gesperrt`
  (Deckel greift zu randnah) statt aktiviert lassen.

---

## 8. Was danach passiert (ich übernehme)

Nach Ihrem Verdikt:

1. **`.github/WOHLKLANG_CLAIMS.md`** wird in der Zeile **W-1** (HR-V1), der SCNet-Zeile
   bzw. der SGMSE-Stufe auf den belegten Stand gesetzt — Status wechselt von `ausnahme`
   auf `aktiviert` (Belege vollständig) oder bleibt/wechselt zu `gesperrt`.
2. **`.github/SOTA_DEFICIT_REGISTER.md`**: D-K3-6, D-K0-5 (SCNet), D-K0-1/D-K0-2 (HR-V1/F3)
   und D-K0-3 (SGMSE) werden je Verdikt auf `geschlossen`, `bewusst-akzeptiert` oder
   `offen` gesetzt — jeweils mit Ihrem `fit.json` als Belegpfad.
3. **Feature-Flags** (`backend/core/music_model_flags.py`): nur bei belegtem Gewinn flippen
   (`use_scnet_music`, `use_sgmse_musik`) — je einen pro Entscheidung, Test-Pins nachziehen.
4. Version-Bump nach §v10.802 (Patch/Minor je nach Änderung) und Changelog-Eintrag.

---

## 9. Grenzen dieser Anleitung (bewusst benannt)

- **A hat 3 Trials** → Hör-Eindruck, kein statistischer Beleg. Für einen echten
  Beleg: mehr Songs (die drei MUSDB-Songs sind vorbereitet, es fehlen nur weitere
  Wiederholungen mit anderen Zeitfenstern).
- **Die HR-V1-Fenster liegen randnah.** Die gleichmäßige Verteilung (bewusst ohne
  Inhaltsauswahl, §V7) legt bei kurzen Ausschnitten Fenster an den **Anfang und das
  Ende**. Dort ist Musik oft ausgeblendet — dann ist die Behandlung schwächer hörbar als
  mitten im Stück. Das ist eine Eigenschaft der Fensterpolitik, keine Ihrer Hörleistung.
- **Ein Hörer ist kein Panel.** Wohlklang-Entscheidungen mit Breitenwirkung sollten
  mehrere Hörer haben (MUSHRA-Studie WP-8). Für die drei hier offenen Entscheidungen
  genügt die dokumentierte Einzelabnahme als **Zeugenaussage** — das Flag-Flip-Votum
  bleibt davon unberührt.

---

## 10. Kommando-Sammlung (Kopieren → Einfügen)

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

# --- C: SGMSE+ 3-Wege (Präferenz) ---
.venv_aurik/bin/python scripts/hoerpanel_player.py \
  --study output_audio/mushra/thresholds_sgmse_3weg --port 8767
#   Browser: http://localhost:8767

# --- Auswerten ---
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_scnet_c4_fair --min-trials 3
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_wohlklang_w1_hrv1 --min-trials 8
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-fit \
  --study output_audio/mushra/thresholds_sgmse_3weg --min-trials 4

# --- Studien neu bauen (nur falls nötig) ---
.venv_aurik/bin/python scripts/mushra_harness.py thresholds-build \
  --build wohlklang_w1_hrv1 \
  --manifest docs/reports/current/2026-10-07_hoerprobe_w1_hrv1.json --seed 20261007
# C: vollständiger Neuaufbau (inkl. Arm-Generierung, ~25 min CPU):
AURIK_FORCE_CPU=1 .venv_aurik/bin/python scripts/build_sgmse_hoerprobe_3weg.py
```

**Dateiorte der Artefakte:**

| Was | Pfad |
| --- | --- |
| SCNet-Hördateien (Referenz) | `output/scnet_ab_2026-10-06_fair/` |
| HR-V1-Hördateien (Harness) | `output_audio/hr_v1_hoerprobe/` |
| SGMSE-Arm-Quellen (degraded/wsj0/musik/wpe, FLOAT-WAV) | `output/sgmse_hoerprobe_3weg/` |
| Studienordner | `output_audio/mushra/thresholds_*/` |
| Antworten | `output_audio/mushra/thresholds_*/answers.csv` |
| Ergebnis | `output_audio/mushra/thresholds_*/fit.json` |
