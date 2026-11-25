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

**Zweite Wurzel (25.11., Drift-Bisektion bestätigt):** Die
Vor-Phasen-Notbremse in `unified_restorer_v3.py` (Zeile ~41428) las den
**Live-Speicher** (`available < 4.0 GB`) — unter Last verschob sie je nach
Maschinen-Zustand UNTERSCHIEDLICHE Phasen nach Stufe 2 (Bisektion: 27 bzw.
33 Deferrals, Phasen-Teilnahme variierte). Die Merge-Menge variierte ⇒
identische Läufe drifteten. Die In-Prozess-Wiederverwendung war nur der
Auslöser des Last-Profils, nicht die Ursache.

## 3. Fix (9 Zeilen, kein Verhaltenswechsel im Normalfall)

> **Ehrlichkeits-Korrektur (25.11., git-verifiziert):** Die Zuschreibung des
> Schutz-Arm-Fixes an `backend/core/audio_preprocessing_protector.py` ist
> unhaltbar — diese Datei existiert nicht in der Git-Historie. Die hier
> dokumentierte sha1-über-Schaden-Input-Konstruktion stammt aus dem
> übernommenen Kontext und konnte am Repo nicht verifiziert werden. Der
> **tatsächliche, heute gemessene** Cross-Prozess-Leak ist der
> content-adressierte Pre-Analyse-Cache (D-K3-112, §4c).

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

**Konsequenz (aktualisiert nach Bisektion):** Die In-Prozess-Wiederverwendung
war nicht die Ursache — die Last-abhängige Phasen-Teilnahme war es (§2, zweite
Wurzel). B1/K/B2 laufen je in einem frischen Child-Prozess; die
Rest-Drift frischer Prozesse liegt bei ~0.001 (unter der Stabilitätsschwelle).

## 4b. Zweiter Fix (2026-11-25): Last-unabhängige Phasen-RAM-Planung

`backend/core/ml_memory_budget.py`: `ram_deferral_projektion(phase_id,
own_rss_gb, total_ram_gb)` — reine Funktion (§G5): eigene Prozess-Projektion
(own RSS + statische Phasen-Schätzung `PHASE_RAM_ESTIMATE_GB`) gegen ein
**maschinenfestes** Budget (`max(10.0, total × 0.5)`). Die Live-Notbremse
(4.0 GB) bleibt **unverändert** als reine Notfall-Sicherung.

**Verifikation (Probe 3, Kampagne lief parallel = Last):**
- Planer feuerte deterministisch (`phase_deferred_ram_plan` für 65/68 in
  Pass 0), Live-Notbremse blieb im Normalbereich stumm (0 × `deferred_low_ram`
  bis 93 % Systemlast — dort griff sie als Notfall, wie entworfen).
- **Phasen-Ausgänge bit-identisch:** KEINE divergierende Phase zwischen
  Pass 0 und Pass 1 — der DSP-Kern selbst ist deterministisch.
- Rest-Vektor dokumentiert: `phase_03_denoise`-Teilnahme via
  `_resolved_defects_accumulator` (In-Prozess-Übertrag über den §V8-Reset
  hinaus) — durch das Child-pro-Pass-Design (frischer Prozess je Pass)
  umgangen.
- 9 Vertragstests (`tests/unit/test_ml_memory_ram_planung.py`) grün.

## 4c. Dritter Fix (25.11.): Pre-Analyse-Cache-Isolation (D-K3-112)

**Befund (Kampagnen-JSONL, Paare 7–9):** B1 repariert stark (DL −0.48),
B2 ≈ 0 (Pass-Through) bei identischem Schaden in frischen Child-Prozessen —
der content-adressierte Bridge-Cache (`backend/api/bridge_cache.py`,
`content_cache_key`) lieferte MUTIERTE Analyse-Objekte an Folge-Läufe
desselben Inhalts zurück: `backend/core/pre_analysis.py` speicherte die
Subresultate per Referenz, die Restauration mutiert sie (transfer_chain-
Write-Back, resolved-Marker), der nächste Lauf startete mit mutiertem Plan.
Produktions-Konsequenz: der **zweite Lauf desselben Songs** restauriert
nicht (§G5-Verletzung).

**Fix:** `copy.deepcopy` auf Store- UND Load-Seite (`_store_in_cache`/
`_load_cached_parts`) — der Cache hält die PRISTINE Analyse, jeder Lauf
arbeitet auf einer privaten Kopie. 4 Vertragstests
(`tests/unit/test_pre_analysis_cache_isolation.py`). Kampagne mit
fixiertem Code neu gestartet; Vor-Fix-Paare 1–9 gesichert als
`data/gepaarte_messung_vor_cachekopie_fix.jsonl` (zählen nicht).

## 5. Erneuertes Mess-Protokoll (Kampagne mit fixiertem Code neu gestartet 25.11. ~17:40)

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

**Vorläufige Paare (Vor-Fix-Code, Harness bereits Child-pro-Pass — gesichert
in `data/gepaarte_messung_vor_determinismus_fix.jsonl`):**

| Paar | Stabilität | Δ zu Bestand (MOS/DL) |
| --- | --- | --- |
| 1 Music Delta - Rock (wowflutter) | INSTABIL (Proben-Last, Vor-Fix) | +0.095 / +0.001 |
| 2 Snowmine - Curfews (hum) | stabil (B1≡B2) | **+0.000 / +0.000** |
| 3 Snowmine - Curfews (wowflutter) | stabil (B1≡B2) | **+0.000 / +0.000** |

**Laufende Kampagne (fixierter Code, `data/gepaarte_messung.jsonl`):**
18 Paare × 3 Pässe, Start 25.11. ~17:40, ETA ~26–31 h. Der GoNoGo-Flip ist
aus dem Harness entfernt (Child-pro-Pass + deterministische RAM-Planung);
B1≡B2 (exakt) in den ersten stabilen Paaren.

> **NOCH OFFEN — wird nach Kampagnen-Abschluss hier befüllt:**
> Aggregat je Rezept und über alle Paare (Nichtunterlegenheit Kanal ≥
> Bestand nach Spec 26 DRT-12/DRT-16).
