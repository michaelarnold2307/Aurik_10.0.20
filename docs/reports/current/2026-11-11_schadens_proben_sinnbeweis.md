# Sinn-Beweis der Restauration — was die Pipeline an zugefügtem Schaden tatsächlich repariert

**Datum:** 2026-11-11 · **Register:** D-K3-95 · **Status:** Befund erhoben, Lücke **offen**
**Belege:** `docs/reports/current/2026-11-11_schadensbewertung.json` (Messung),
`~/Musik/schadens_proben/schadens_manifest.json` (Proben)
**Kein Flag-Flip, kein Deploy, kein Klangpfad-Eingriff.**

## 1. Warum diese Messung

Die Sitzung 2026-11-10 hatte festgestellt: auf **sauberen** HD-Quellen ist die
Restauration praktisch transparent (das Original „gewinnt", gesundes
Never-worsen). Daraus folgt die eigentliche Frage, die offen war:

> Wenn das saubere Material unangetastet bleibt — **repariert Aurik dann
> wenigstens Material, das tatsächlich Schaden hat?**

Ohne diese Antwort ist der Nutzen des Systems unbewiesen. Bisher gab es dafür
kein Messwerkzeug, nur Hörproben.

## 2. Was dafür gebaut wurde

| Baustein | Zweck |
| --- | --- |
| `scripts/erstelle_schadens_proben.py` (2026-11-10) | Beschädigt **eigene** saubere Titel deterministisch (4 Rezepte) und lässt den Produktionspfad dagegen laufen — CPU-only, stört die GPU-Kette nicht |
| `scripts/bewerte_schadens_proben.py` (**neu, 2026-11-11**) | Bewertet die Paare objektiv mit `defektlast` (niedriger = besser) und `musik_mos` (höher = besser), **pegelgematcht**, Schreibt `aurik.schadensbewertung/1` |

Die Pegel-Kontrolle ist keine Kosmetik: die Pipeline regelt auf eine
Ziel-Lautheit (`rms_out` liegt bei 10 von 12 Paaren eng in 0,11–0,17, obwohl
`rms_in` von 0,105 bis 0,57 streut). Ein Rohpegel-Vergleich misst diesen
Regelkreis statt der Reparaturgüte — genau die Fehlerklasse, die D-K3-94 beim
FlashSR-Gate ausgeschlossen hat (Gain war dort 0,990–0,995).

## 3. Ergebnis — 12 Paare, 3 Titel × 4 Rezepte, Seed 42

Pegelgematcht, Modus `Restoration` (`balanced`), 10-s-Mittelausschnitte, mono/48 kHz:

| Rezept | n | Δ defektlast (Schaden→restauriert) | Δ musik_mos | besser |
| --- | --- | --- | --- | --- |
| vinyl | 3 | −0,004 | +0,000 | 1/3 |
| kassette | 3 | −0,027 | −0,025 | 1/3 |
| **clipping** | 3 | **+0,167** | **−0,094** | **0/3** |
| mp3_128 | 3 | −0,032 | −0,015 | 1/3 |
| **gesamt** | **12** | — | — | **3/12** |

Schwere Einzelfälle (vollständig im JSON):

- **Jets · clipping:** `defektlast` 0,478 → **0,716**, `musik_mos` 4,06 → **3,68**.
  Das restaurierte Signal ist schlechter als der Schaden — und schlechter als
  das saubere Original (4,71).
- **Träumer · clipping:** `defektlast` 0,273 → **0,503**.
- **Träumer · mp3_128:** `defektlast` 0,223 → **0,382**, `musik_mos` 4,86 → 4,79.

## 4. Drei Befunde, sauber getrennt

**(a) Substanz — Clipping wird schlechter, nicht besser.** 3 von 3
Clipping-Paaren kippen; zwei davon deutlich. Bei diesem Rezept senkt die
Pipeline den Pegel stark (Gain 0,30 / 0,32 / 0,71) und der Klang verliert
hörbar (`musik_mos` −0,09 im Mittel, im Extremfall −0,38).

**(b) Metrik-Blindstelle — `defektlast` erkennt Clipping nicht.** Der
Clipping-Schaden wird von `defektlast` **besser** bewertet als das saubere
Original (0,273 gegen 0,280). Die zertifizierte Metrik (Spearman 0,95) hat
also eine blinde Stelle genau bei der Sättigung; `musik_mos` sieht sie
(4,68 gegen 4,84). Wer nur `defektlast` liest, hält Clipping für harmlos —
das ist ein Befund über die Metrik, nicht über das Audio.

**(c) No-Op — zwei Paare sind bit-identisch.** Träumer·vinyl und
Boogie·vinyl liefern exakt dasselbe Signal zurück (`np.array_equal`, Gain 1,0,
Δ = 0,000). Die Pipeline hat hier **entschieden**, nichts zu tun. Für vinyl
(Bandlimit + Rauschen + Nadel-Klicks) ist das der Kernnutzen, der ausbleibt.

Gesund ist dagegen das Verhalten auf **sauberem** Material: der laufende
Kontrolllauf loggt `§SLR-1 stem-NR rejected: VQI=0.669 < 0.72; keeping
Originalsignal` und `DeepFilterNet Hallucination-Guard: novelty=0.605 > 0.50 →
Rollback` — die Never-worsen-Guards greifen wie spezifiziert.

## 5. Grenzen dieser Messung (bewusst nicht überdehnt)

- **10-s-Mittelausschnitte, mono, 48 kHz.** Das ist ein Befund über diese
  Clips, **kein** Qualitätsurteil über ganze Titel; Pre-Analysis und
  Song-Kontext fehlen teilweise.
- n = 3 je Rezept; die Richtung ist deutlich (0/3 beim Clipping), die
  Effektgröße nicht belastbar.
- Die Rezepte sind **synthetisch** (deterministisch, Seed 42) — kein Ersatz für
  echte historische Aufnahmen.
- Der Lauf ist CPU-only und **ohne RT-Limit** (`AURIK_NO_RT_LIMIT=1`) gefahren;
  ein 10-s-Clip brauchte > 15 min. Das ist **nicht** die Produktions-RT-Kennzahl
  (32,2× RT gemessen auf dem Referenz-Song), sondern der Preis des vollen
  Analysepfads auf der CPU.

## 6. Was daraus folgt

**Nicht** gemacht — und bewusst nicht: kein Eingriff in den Klangpfad, kein
Flag-Flip, keine W-Stufen-Buchung. Die zunächst vermutete Declip-Ursache ist
**gemessen widerlegt** (D-K3-97, `docs/reports/current/2026-11-11_declip_never_worsen_und_ursachenpruefung.md`):
`DeclipperPhase` läuft im UV3-Produktionspfad nicht (einzige Aufrufstelle
`backend/core/coordinated_repair.py:1361`), PCHIP ist formtreu und auf den
Sättigungs-Proben wirkungslos. Der Never-worsen-Guard in der Phase bleibt —
mit gültiger Begründung: PCHIP kostet auf echten Sparse-Hard-Clips bis zu
−18 dB SDR. Der offene Ort ist die **Export-/Entscheidungskette** (Gain 0,30
vs. bit-identisch) und die **Reproduzierbarkeit** des UV3-Pfads.

Nächste Schritte, in dieser Reihenfolge:

1. **Reproduzierbarkeit zuerst:** dasselbe Clipping-Paar zweimal durch den
   Produktionspfad, beide Logs + WAV-Hashes vergleichen (RMS-Rauschen ~6,5e-4
   macht den Pfad potenziell nicht bit-reproduzierbar, D-K3-85). Erst danach
   ist „repariert/nicht repariert" eine Aussage.
2. **Export-/Entscheidungskette:** warum exportiert die Pipeline Jets
   gelegentlich mit Gain 0,30 (verzerrt) und gelegentlich bit-identisch?
   Kandidaten: Loudness-Ziel gegen Qualitäts-Gate, Goal-Compensation-
   Sekundärlauf (−91,1 dBFS, vom Silence-Guard abgefangen).
3. **Ganze Titel statt Snippets** (1 Titel × 4 Rezepte) im trainingsfreien
   GPU-Fenster — klärt, ob der Clipping-Effekt ein Snippet-Artefakt ist.
4. **`defektlast`-Grenzen** bei Sättigung UND dem 0,34-s-Fenster: entweder
   Metrik erweitern oder als Zertifikats-Grenzen dokumentieren (`musik_mos`
   deckt die Sättigung ab).
5. Erst danach eine Rezeptur-Entscheidung — und die braucht die Hör-Instanz.
