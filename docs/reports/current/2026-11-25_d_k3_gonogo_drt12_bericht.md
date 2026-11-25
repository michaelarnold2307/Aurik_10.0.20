# DRT-12 — Gepaarte Kanal↔Bestand-Nachmessung: D-K3-GoNoGo-Kontrolle
**Datum:** 2026-11-25 · **Status:** Nachmessung LÄUFT (Ergebnis-Abschnitt wird nach Abschluss befüllt)

## 1. Zweck

DRT-12 verlangt den gepaarten Vergleich des **17-Kanal-Witness-Satzes** gegen
den Bestands-Kanalsatz auf identischen Schadens-Schnipseln. Die
Vorläufer-Kampagne (2026-11-22/23) war durch einen **Schutz-Arm-Flip**
(D-K3-GoNoGo) kontaminiert: Der Bestands-Pass selbst war zwischen den
Messungen nicht reproduzierbar — ein Kanal↔Bestand-Vergleich auf dieser
Basis wäre irreführend.

Dieser Bericht dokumentiert Root-Cause, Fix, deterministische Probe und das
erneuerte Mess-Protokoll. Die eigentliche Paar-Matrix folgt nach Abschluss
der laufenden Kampagne.

## 2. Root-Cause: D-K3-GoNoGo-Flip im Schutz-Arm

**Ort:** `backend/core/audio_preprocessing_protector.py`

**Mechanismus:**
1. Der GoNoGo-Schutz-Arm schreibt Zwischenstände in ein **Temp-Verzeichnis**
   (`tempfile.gettempdir()`, standardmäßig `/tmp`).
2. Der Kollisions-Schlüssel wurde als `sha1` über **Restaurierungs-Daten**
   gebildet — nicht über den Schaden-Input.
3. Unter Last (RAM-Engpass, parallele Jobs) konnte der Temp-Bestand eines
   fremden/älteren Laufs kollidieren; der Schutz-Arm sah „bereits vorhandene"
   Artefakte und **kippte den Pass auf den Passthrough-Zweig** (0.0-Delta).

**Beleg der Konsequenz:** Paare der 17er-Kampagne zeigten exakt
`delta_musik_mos = 0.0 / delta_defektlast = 0.0` mit `status: ok` — der
Kanal-Pass lief vollständig durch, ohne dass dies als Fehler sichtbar war.
Metrik-Zeugen wiesen den Flip nur indirekt aus (B1≠B2-Diskrepanzen).

## 3. Fix (9 Zeilen, kein Verhaltenswechsel im Normalfall)

Der Kollisions-Schlüssel des Schutz-Arms wird jetzt aus dem **normierten
Schaden-Input** gebildet:

- `sha1(schaden_normiert)` → 4 Bytes → Ganzzahl-Seed im Bereich
  `[0, 2^32 − 1]` → `np.random.default_rng(seed)`.
- Gleicher Schaden-Input ⇒ gleicher Temp-Pfad (kollisionssicher innerhalb
  einer Messung), verschiedene Inputs ⇒ praktisch nie kollidierend.
- Kein `time.time()`, kein Zufall in der Entscheidungslogik — **§G5
  (Determinismus)** bleibt erfüllt; der Seed ist deterministisch aus dem
  Input abgeleitet, die Speicher-Konstante bleibt gleich.

## 4. Deterministische Probe (2026-11-25, `AURIK_FORCE_CPU=1`)

Referenz-Fenster (10 s, Kassette), zwei Läufe ohne Rezept-Flags:

| Bedingung | Δ-Musik-MOS (max_delta) | Befund |
|---|---|---|
| In-Prozess, Pass 1 vs. Pass 2 (Wiederverwendung) | **1.172 vs. 0.951** | Drift **~0.17** — dominante Varianzquelle |
| Zwei frische Einzel-Prozesse | 1.102 vs. 1.103 | Drift **~0.001** — ~100× kleiner |

**Konsequenz:** Die In-Prozess-Wiederverwendung trägt die Varianz, nicht der
Schutz-Arm-Fix. B1/K/B2-Vergleiche, die im selben Prozess laufen, messen
überwiegend diesen Zustands-Leak statt des Kanal↔Bestand-Unterschieds.
Rest-Drift frischer Prozesse liegt bei ~0.001 (unter der
Stabilitätsschwelle).

## 5. Erneuertes Mess-Protokoll

`scripts/gepaarte_messung.py` (Commits `d1cb0654`, `580631e8`):

- **Ein Pass pro Child-Prozess:** B1 (Bestand), K (17 Witness-Flags),
  B2 (Bestand-Stabilitätskontrolle) — je in einem frischen Python-Prozess;
  Zeugen (Δ-Musik-MOS, Δ-Defektlast, Überrestauration) als JSON zurück.
- **Stabilitäts-Gate:** `|B1 − B2| > 0.05` (MOS oder Defektlast) ⇒ Paar wird
  als `instabil_d_k3_gonogo` geflaggt statt gewertet.
- **Cache-Busting:** eindeutiger `input_path` je Pass.
- **Stromausfall-Resume:** Ergebnis-JSONL `data/gepaarte_messung.jsonl`;
  fertige Paare werden beim Neustart übersprungen.

**Laufende Kampagne** (Start 2026-11-25 ~12:58):
- 18 Paare (Rezepte: dropout, kassette, hum, wowflutter, kompression,
  clicks), 3 Pässe je Paar, `nice 19`, CPU-only.
- ETA ~26–31 h bei ~25–35 min/Pass (OOM-Probe aktiv, kein RAM-Risiko).
- Log: `/tmp/gepaarte_repo_kind.log`.

## 6. B3-Verlauf (Kontext aus der Fix-Kette)

Der Schutz-Arm-Härtung ging eine Root-Cause-Kette mit 18 Witness-Kanälen
voraus (B3-Verfahren): p_sauber-Quote **0.94 → 0.67 → 0.99** — letzter Stand
nach dem Schutz-Arm-Fix mit 18 Zeugen (Belege: Sitzungs-Kette
2026-11-22/23). Die hier dokumentierte Probe bestätigt den Fix unter
Einzel-Prozess-Bedingungen; die laufende 18-Paar-Kampagne ist der
DRT-12-Abschlussbeleg.

## 7. Ergebnis-Matrix

> **NOCH OFFEN — wird nach Kampagnen-Abschluss hier befüllt:**
> pro Paar: Bestand-Δ (MOS/DL), Kanal-Δ (MOS/DL), Δ-zu-Bestand,
> Stabilitäts-Flag; Aggregat je Rezept und über alle Paare
> (Nichtunterlegenheit Kanal ≥ Bestand nach Spec 26 DRT-12/DRT-16).
