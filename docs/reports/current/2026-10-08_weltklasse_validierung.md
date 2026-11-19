# Weltklasse-Validierung — „nicht nur theoretisch" (2026-10-08)

> **Auftrag:** Validiere, dass die Vorgaben und Specs Aurik tatsächlich zum
> weltweit besten Musikrestaurierungsprogramm des Wohlklangs für das menschliche
> Gehör machen — nicht nur theoretisch.
>
> **Methode:** Enforcement-Matrix je Gebot: Norm → Code-Instanz → Nachweis im
> REALEN Lauf (10.12.17) → Gate/Test → Lücken. Keine Aussage ohne Beleg.

## 1. Die drei neuen Spitzen-Gebote (10.12.20/21)

| Gebot | Norm (copilot-instructions.md) | Code-Instanz | Nachweis-Ebene |
| --- | --- | --- | --- |
| §G10 | Defekt-Freiheit unter der Hörbarkeitsschwelle (alle 65 Typen, ganzer Song) | `backend/core/dsp/audibility_targets.py` + `backend/core/defect_audibility_gate.py` (§B2) | `audibility_gate`/`defect_residual_audibility` im Export-Meta; `evidence_state` + `gate_verified` |
| §G11 | Weltklasse-Abnahme (Hörerlebnis ≥ 90, timbral ≥ 0,98, Einladungs-Gate, keine Violation) | `holistic_perceptual_gate.py` (HPI), `phase_fazit.py` (Hörerlebnis), `inviting_gate` | Export-Narrativ + HPI-Metadaten im Lauf |
| §G12 | Universeller Import-Vertrag (jedes Material/Genre/Sprache/Länge; Chunked=Voll) | Chunked-Pfad (`_restore_chunked`), P0-1-Deferral, §B3-Determinismus | P0-1-Tests + Lauf-Chunks |

## 2. Wirkungs-Belege aus dem realen Lauf (10.12.17)

| Kriterium | Ist im Lauf | Bewertung |
| --- | --- | --- |
| Hörerlebnis (§G11) | **92/100** (Chunk-0-Narrativ) | ✅ über Ziel |
| timbral (§G11) | **0.999** (HPI-Komponente) | ✅ über Ziel (0,98) |
| HPI-Formel | `HPI = MERT × timbral × artifact_freedom × emotional_arc` — Produktformel | ⚠️ eine 0-Komponente killt HPI (D-K3-66: af-Messung war die Ursache; gefixt) |
| §G10-Verdikt | Chunk-0: `scan_missing` → **„Residuum NICHT belegt, nicht QUALITY GUARANTEED"** (fail-closed, §V6) | ⚠️→✅ **Lücke D-K3-68 heute geschlossen** (s. u.) |
| §B2-Post-Scan | läuft per §P0-1 (d) auf dem **letzten** Chunk | ✅ by design; Daten speisen Song-m1b |

**Beurteilung der Selbst-Ehrlichkeit:** Das System meldet seine Beleg-Lücken
explizit („gate_passed=True ist KEIN Nachweis", fail-closed) — genau die
§G8/§V6-Qualität, die eine Weltklasse-Zusage glaubwürdig macht. Es lügt nicht.

## 3. Heute geschlossene Wirkungslücken

### D-K3-68 — §G10-Garantiekette im Chunked-Pfad (Fix 10.12.21)

- **Lücke (bewiesen):** Chunk 0 erbt sein „scan_missing"-Verdikt nach oben
  (`_meta = dict(_first_result.metadata)`); es gab **keinen song-globalen
  Audibility-Nachzug** → die §G10-Zusage (`gate_verified`) hätte im Export
  **nie** vergeben werden können.
- **Fix:** `_restore_chunked` wertet nach der Assembly das Audibility-Gate auf
  Song-Datenbasis neu aus (Quelle: die Per-Defekt-Reduktion, die auch die
  Song-m1b speist), schreibt `audibility_gate` + `defect_residual_audibility`
  (§G10-Feldname) in den Erb-Träger, `post_scan_ran` **fail-closed** aus der
  Datenlage abgeleitet.
- **Wirkungs-Test:** `tests/unit/test_dk368_song_audibility_chain.py` — inkl.
  Semantik-Test `post_scan_ran=True ⇒ gate_verified`, `False ⇒ scan_missing`.

### Weitere heute geschlossene Kettenbrüche (Bughunt 2026-10-08)

| Fund | Wirkung | Fix |
| --- | --- | --- |
| D-K3-63 | FinalPolish/Dither/§2.73b+c+Banner liefen **pro Chunk** (Dither vor Assembly/OneTakeExport, §V5-Bruch) | 10.12.19 |
| D-K3-66 | Final-Gate-Rollback verwarf **in jedem Lauf** die Wirkung der Verarbeitungsschritte nach dem letzten Carrier-Checkpoint | 10.12.20 |
| D-K3-61 | filtfilt-Kurzsignal-Crash (echter Fund) | 10.12.18 |

## 4. Offene Ketten-Punkte (ehrlich benannt — Weltklasse-Garantie noch nicht vollständig)

| ID | Was offen ist | Bedeutung für die §G10/§G11-Zusage |
| --- | --- | --- |
| D-K3-60 | WCS/Song-Level-Metadaten erreichen den Export (Chunked) nicht sicher | Der §G11-Nachweis (WCS-Gate) kommt beim Nutzer/Export nicht an |
| D-K3-64 | Doppel-Dither-Kette restore × Export (beide Pfade) — im §G4-Rauschbodenbereich | Ketten-Redundanz; Hörstichprobe + Single-Dither-Entscheidung fällig |
| D-K3-67 | Phasen-Heartbeat unter GIL-Blockade wirkungslos (0 CPU in 20 s gemessen!) | Monitoring-Blindheit in Langphasen — Transparenz §G8 |
| D-K3-65 | Song-Analytik-Blöcke (ARC/Goosebumps/MDEM) laufen pro Chunk | Bewertungs-Qualität/Narrativ („Klimax 30 Segmente" = Chunk-Artefakt) |

**Der finale Beweis-Lauf:** Ein Lauf mit 10.12.20/21 muss zeigen:

1. `§G10 Song-Audibility nach Assembly: evidence=evaluated/no_residual_defects`
   und `gate_verified=True` im Export-Meta (`defect_residual_audibility`),
2. **keine** `Final-Ausgabe-Audio-Gate … af=0.000`-Rollback-Zeile mehr (D-K3-66),
3. **keine** Per-Chunk-`FinalPolish`/Dither/Banner-Zeilen (D-K3-63),
4. Hörerlebnis ≥ 90 + timbral ≥ 0,98 im Song-Narrativ.

## 5. Fazit

- **Theoretisch:** ✅ Vollständig — §G10–§G12 normativ verankert (copilot-
  instructions, Hörordnung Ebene 2, Specs 01/13, ID-Registry, AGENTS.md) mit
  Zitierdisziplin und Ambiguitäts-Mapping.
- **Infrastruktur:** ✅ Vorhanden und getestet (`audibility_targets` + §B2-Gate
  und normative Tests; HPI/Einladungs-Gate/Hörerlebnis).
- **Praktisch:** ⚠️→✅ **Die Kette ist jetzt geschlossen** (D-K3-68): Die
  §G10-Zusage ist ab 10.12.21 im Chunked-Pfad **vergebbar**; der reale
  Nachweis erfolgt im nächsten vollständigen Lauf (Beweis-Lauf-Kriterien s. o.).
- **Rest-Wahrheit:** Drei Ketten-Punkte (D-K3-60/64/67) und ein
  Bewertungs-Punkt (D-K3-65) stehen offen — sie sind im SOTA-Defizit-Register
  geführt und blockieren bis zur Schließung die vollständige, aussagbare
  Weltklasse-Garantie nur dort, wo sie messbar ist.

> Metriken sind Zeugen (Hörordnung §7) — diese Validierung trennt Norm,
> Infrastruktur und Wirkung und weist jede Ebene mit Pfad/Log/Test aus.
