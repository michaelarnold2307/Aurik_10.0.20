# Flag-Wirkungs-Gate — „jeder `use_*`-Schalter wirkt im Produktionspfad" (D-K3-92)

**Datum:** 2026-11-10 · **Lücke:** `docs/TODOS_SOTA_ROADMAP.md` Pfeiler 4
(„Den Audit als Gate statt als Suchlauf") · **Status:** Gate gelandet,
ein Sachbefund offen (`use_bw_v5`, nur über §4).

## 1. Warum

Drei Defekte derselben Klasse sind in dieser Woche aufgefallen:

- **D-K3-85:** WF-V4/TP-V2 luden ein nie existierendes Artefakt — die
  B3-AN/AUS-Messung verglich den Pfad **mit sich selbst** und „lehnte" ab.
- **D-K3-90:** Der M5-vocos-Export hatte keine Ladestelle.
- **D-K3-91:** `use_miipher_dit` stand nur in der Flag-Datei; `False` schaltete
  nichts ab (der dokumentierte Rollback existierte nicht).

Der manuelle Quer-Audit (22 Flags, 2026-11-10) fand diese Fälle — aber ein
Suchlauf vergisst. Die Roadmap verlangte deshalb ein **Gate**.

## 2. Befund der systematischen Nachmessung (24 Schalter)

| Kategorie | Anzahl |
| --- | --- |
| `use_*`-Schalter in `backend/core/music_model_flags.py` | 24 |
| mit **Produktions-Lesestelle** | 23 |
| nur in Mess-/Hilfsskripten | **1** (`use_bw_v5`) |
| ohne jede Lesestelle | 0 |

**`use_bw_v5` (W-8, BW-Reconstructor v5):** Der Pfad in
`backend/core/phases/phase_06_frequency_restoration.py` hängt an `_use_nvsr`
(NVSR-Verfügbarkeit), **nicht** am Schalter. Gelesen wird der Name nur als
String in `scripts/b3_komposition_suche.py` (Kandidatenliste W-8) und
`scripts/release_candidate_pipeline.py`. ⇒ Eine B3-Messung von W-8 wäre ein
**No-Op-Artefakt**. Die Verdrahtung ist **klangwirksam** (Default AUS würde
einen heute laufenden Pfad abschalten) und deshalb **kein autonomer Schritt**:
nur §4-Verfahren mit Hör-Sign-off.

## 3. Das Gate

`scripts/flag_wirkung_gate.py` (stdlib-only, deterministisch §G5), verdrahtet
als Pre-Commit `aurik-flag-wirkung-gate` (nur bei Code-Änderungen):

- **Stufe A — Lesestelle:** Jede Definition hat eine Lesestelle außerhalb der
  Flag-Datei. Gezählt werden Name-Loads, Attribut-Zugriffe
  (`mmf.use_x`), `getattr`/`hasattr` mit Flag-Namen als String-Literal und
  Import-Aliase, deren lokaler Name danach **auch geladen** wird.
  Ein toter `import ... as` zählt nicht (D-K3-91-Muster), `setattr` nicht
  (Schreibstelle), `tests/` nie (ein Flag, den nur ein Test liest, ist im
  Betrieb tot).
- **Stufe B — Produktionslesestelle:** Dieselbe Definition, beschränkt auf
  `backend/`, `plugins/`, `denker/`, `Aurik10/`, `cli/`, `forensics/`.
  Nur-Mess-Flags sind ausschließlich mit **begründeter Ausnahme inkl.
  Register-ID** zulässig (`AUSNAHMEN` im Skript, fail-closed).
- **Stufe C — Mess-Kandidaten:** Jeder Flag, der in `scripts/b3_*.py` oder
  `scripts/release_candidate_pipeline.py` als Kandidat genannt wird, muss
  Stufe B erfüllen — sonst wird die Messung als No-Op-Artefakt abgelehnt
  (die M12-WF4/TP-V2-Lehre).

Aufruf: `python3 -B scripts/flag_wirkung_gate.py [--verbose|--ci|--json]`.
Exit 0 = grün, 1 = Verstoß, 2 = nicht ausführbar.

## 4. Methoden-Lehre (drei Korrekturen vor dem belastbaren Lauf)

Die erste Ad-hoc-Analyse war **falsch** und hätte vier Flags zu Unrecht als
„tot" gebrandmarkt:

1. **Verzeichnislücke:** `forensics/` fehlte in der Suchliste —
   `use_material_crnn` (MediumDetector) galt dadurch als tot.
2. **Import-Alias:** `from … import use_x as _on` wurde nicht erfasst —
   `use_era_crnn`, `use_loudness_dpn`, `use_defect_crnn` gälten als tot.
3. **`getattr`-String:** Die häufigste Lesestelle im Repo ist
   `getattr(modul, "use_x", default)`; ein pauschaler String-Ausschluss hätte
   neun weitere Flags falsch negativ gemeldet.

Erst nach allen drei Korrekturen stimmte das Bild (23/24). Das Gate kodiert
genau diese drei Fälle als Vertragstests — sie können nicht wieder verloren
gehen.

## 5. Verifikation

- `pytest tests/unit/test_flag_wirkung_gate.py` ⇒ **12 passed** (11 Mini-Repo-
  Verträge + echtes Repo inkl. Ausnahme-Assertion).
- Gate auf dem Bestand: **24 Schalter, 0 Verstöße, 1 begründete Ausnahme**
  (`use_bw_v5`), Laufzeit ≈ 9,5 s (Substring-Vorfilter vor dem Parsen).
- Pre-Commit-Hook `aurik-flag-wirkung-gate` in `.pre-commit-config.yaml`
  (`files:`-Filter auf Code-Pfade, damit reine Doku-Commits nicht zahlen).

## 6. Grenzen (ehrlich)

- Die Prüfung ist **statisch**. Ein dynamischer Zugriff
  `getattr(mm, name)` mit einer Variablen (z. B. in `b3_stufenmessung.py`)
  wird nicht als Lesestelle erkannt — für Stufe A/B ist das unkritisch, weil
  die Produktionslesestellen konkret sind; Stufe C erfasst die Kandidaten
  zusätzlich über ihre String-Nennung.
- Ein Flag, der gelesen, aber dessen Wert **ignoriert** wird
  (`if getattr(...): pass`), besteht das Gate. Das ist bewusst: Das Gate
  schließt die Lesestellen-Lücke, nicht die Semantik jedes Zweigs.
- Die Ausnahme `use_bw_v5` ist ein **offener Sachbefund**, keine Freigabe:
  Sie steht im Register (D-K3-92, Status `offen`) mit der §4-Bedingung.

## 7. Kein Flag-Flip, kein Training

Beide Flags (`use_bw_v5`, alle übrigen) bleiben unverändert. Die GPU-Kette 2
(D3-3 FlashSR, Epoche 17/30, Versuch 2) lief während dieser Arbeit
ununterbrochen weiter; es wurde kein Training gestartet und keine Voll-Suite
gefahren.
