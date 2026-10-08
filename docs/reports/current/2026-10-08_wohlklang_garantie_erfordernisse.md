# Wohlklang-Garantie — Erfordernis-Dokument (Tiefenanalyse 2026-10-08)

> **Auftrag:** Was ist erforderlich, damit Aurik mit weitem Abstand das
> garantiert weltweit beste Musikrestaurierungsprogramm für den Wohlklang des
> menschlichen Gehörs ist — bei allen erdenklichen Importsongs.
>
> **Methode:** Analyse der normativen Kette (copilot-instructions, VERBOTEN,
> Hörordnung, Specs 01–08 + v10.xx), des SOTA-Defizit-Registers (74 Einträge:
> 42 geschlossen / 14 offen K3, 9 offen K0, 1 offen K1, 5 offen K2), des
> Produktionslaufs „Trio Schweizer – 13 Tage" (10.12.12, 2026-10-08 02:38,
> RC=0) samt Export-Metadaten, und des laufenden Validierungslaufs 10.12.17.
>
> **Belegdisziplin:** Jede Aussage trägt Quelle (Datei, Register-ID, Lauf).
> Nicht-Belegbares ist als offen markiert, nie als Tatsache.

---

## 1. Was „garantiert bester Wohlklang" operational bedeutet

Fünf überprüfbare Kriterien (Abnahme-Definition):

| # | Kriterium | Messbar durch |
| --- | --- | --- |
| G1 | **Gleicher Input + gleiche Version ⇒ bit-identischer Output** (§G5) | Determinismus-Tests, Seed-Protokoll |
| G2 | **Jeder Song erhält die VOLLSTÄNDIGE Kette** — keine Stufe fällt aus Budget-/Performance-Gründen aus | `phases_executed` vs. Plan; 0 budgetbedingte Skips; Exzellenz-Stufe läuft |
| G3 | **Jedes [RELEASE_MUST]-Gate ist im ENDArtefakt nachweisbar** — nicht nur im Quelltext | Export-Metadaten des Chunked-Pfads; Wirkungs-Tests |
| G4 | **Jeder Defekt liegt unter der Maskierungsschwelle** (Hörordnung Ebene 2) | Defekt-Audibilitäts-Gate mit `gate_verified` (D-K3-30-Muster) |
| G5 | **Hör-Bestätigung** über Korpus, nicht nur Metriken | MUSHRA-Panel (Roadmap §3.6), C4-Hörstichproben |

**Ist-Stand:** G1 ist technisch verankert (§G5, Deterministizität gepinnt).
G2/G3 sind im Ist-Stand **nicht garantiert** (Belege unten). G4 ist methodisch
stark (Audibility statt Mess-Null), hängt aber an der Detektor-Kalibrierung.
G5 fehlt als installierte Praxis (Roadmap).

---

## 2. Die harten Lauf-Belege (13-Tage-Lauf, 10.12.12)

| Befund | Zahl | Quelle |
| --- | --- | --- |
| Laufzeit live | 53–58× RT (Norm 32×) | `/tmp/trio_lauf.log` „Performance EXCEEDED" |
| Restaurier-Pfad allein | 10 452 s für 360 s Audio (≈ 29× RT) | „RestaurierDenker überschritt RT-Budget" |
| **Exzellenz-Stufe [9/10] + Exzellenz-Optimierung** | **ÜBERSPRUNGEN** („RT-Budget ausgeschöpft") | AurikDenker-Log |
| Musical Goals am Ende | **8/15 bestanden**; „Verbesserungspotenzial erkannt" | Lauf-Log |
| Qualitäts-Score | 0,577 | Lauf-Ende |
| Export-Kern | Gate passed, LUFS erhalten, TP −7,36 dBTP, CD-Rauschprofil ✓ | `export_metrics.json` |
| **§8.6b [RELEASE_MUST] WCS-Gate im Export** | **LEER** (0.0/0.0/False/`profile=""`), ebenso HTEV `{}` und WCS-Evidenz | `export_metrics.json` (s. u. D-K3-60) |

---

## 3. Der Schlüsselbefund: Garantien brechen an Übergängen, nicht in Modulen

### D-K3-60 (neu, offen): §8.6b-WCS-Metadaten erreichen den Export im Chunked-Pfad nicht

**Fakten (alle verifiziert 2026-10-08):**

1. Das Export-JSON des Chunked-Laufs zeigt für `worldclass_composite_gate`
   die **Default-Leerwerte**; ein BERECHNETES Gate träte zwingend mit
   `profile ∈ {vocal, instrumental}` auf — **`profile=""` beweist: das Dict
   fehlte im Export-`meta`**, es wurde nie berechnet.
2. Die Payload-Kette (result → `build_export_quality_gate_payload`) wurde
   **isoliert getestet**: synthetische Metadaten mit `wcs=0.77` erreichen den
   Payload unverändert ⇒ **der Export-Builder ist nicht die Ursache**.
3. Die CLI-Seite (`cli/aurik_cli.py`) wurde geprüft: sie re-assembliert `meta`
   flach (`dict(meta)` + `update`) und ergänzt nur `export_gate_*`-Felder —
   kein WCS-Eingriff.
4. Der §8.6-Metadatenblock existiert in `restore()` (UV3, Z. 22773ff als
   unbedingte Top-Level-Items) — und `_restore_chunked` übernimmt die
   Metadaten des ERSTEN Chunks.
5. Nur **1×** „✅ Restoration vollstaendig" für 3 Chunks im Lauf-Log ⇒ die
   Chunk-`restore()`-Aufrufe nehmen einen anderen Ausstieg als den
   Haupt-Metadatenbau — die exakte Skip-Stelle ist noch zu pinnen.

**Erfordernis (P0):**

- (a) Exakte Skip-Stelle lokalisieren (Chunk-`restore()`-Ausstieg ODER
  Bridge-Durchreichung), **Ursache** fixen (§V7).
- (b) **Wirkungs-Test** statt Wortlaut-Test: Ein End-to-End-Test muss
  beweisen, dass ein CHUNKED-Export (≥ 2 Chunks) die §8.6b-Felder GEFÜLLT
  im finalen `export_metrics.json` trägt. Der bestehende normative Test
  (`test_worldclass_composite_score_gate`) prüft nur den Quelltext —
  dieselbe Lücke, die D-K3-9d/D-K3-34 schon einmal belegt haben.

**Meta-Lehre:** Dies ist der dritte belegte Fall der Klasse „Prüfung existiert,
sieht aber nichts" (D-K3-9d Vorlagen, D-K3-34 Wortlaut-Drift, D-K3-59
unbekannte Action). Die Garantie-Kette braucht eine systematische
**Wirkungs-Test-Schicht** an den Modul-Grenzen (Export-Artefakt, nicht Quelltext).

---

## 4. Die Erfordernisse im Einzelnen (priorisiert)

### P0 — Garantie-Kern

| ID | Erfordernis | Beleg | Abnahme |
| --- | --- | --- | --- |
| A1 | **Chunked-Metadaten-Integrität** (D-K3-60) — Root-Cause + Fix + End-to-End-Wirkungstest | s. o. | Chunked-Export trägt alle §8.6-Felder; Test fällt bei Regression |
| A2 | **Performance: Die Exzellenz-Stufe muss budgetfrei laufen.** Treiber laut Messung: Defekt-Scanner 140 s/min (Budget 4; D-K3-38/40: `_auto_detect_material` 38 % davon), `phase_12_wow_flutter_fix` (250 s bei Wow-Anteil 0,00; Kalman/RTS seit D-K3-39 bereits 2,5× schneller), `feedback_chain` 165 s/min | D-K3-37/38/40 | Kalibrierungslauf: 0 budgetbedingte Skips; Exzellenz-Stufe in 100 % der Läufe |
| A3 | **32×-Guard-Entscheidung** (Durchsetzen ODER ehrlich deklarieren) — heute: Norm = Anzeige | D-K3-37 | Entscheidung + konsistenter Normtext |
| A4 | **Budget-Divergenz StrategieDenker** (73,6× Zusage vs. 32× Guard) | D-K3-22 | Eine Norm (§G9) |

### P1 — Modell-Kapital aktivieren und schließen

| ID | Erfordernis | Beleg | Abnahme |
| --- | --- | --- | --- |
| A5 | **BigVGAN Musik-F3-Finetune exportieren + ausrollen** — aktuell läuft bytescharf bewiesen die BASIS (`onnx == bigvgan_v2.pth`; 469/783 Tensoren weichen vom Finetune ab) | D-K0-1/2 | ONNX-Export + Parität (§III.9) + Rollout; A/B |
| A6 | **SCNet 4-Stems freigeben** (+1,96…+3,59 dB SI-SDR fair gegen Demucs v4; `use_scnet_music=False` gesperrt) | D-K0-5 | **C4-Hörstichprobe + Hörordnungs-Sign-off (Nutzer)**; RT-Beitrag gegen Budget messen |
| A7 | **SGMSE+ Musik-Core freigeben** (3-Wege-A/B + Hörabnahme offen) | D-K0-3 | A/B-Ergebnis + Sign-off |
| A8 | **Miipher-Whisper-Denoiser entscheiden** (deprecated, trainiert auf MUSDB18-HQ) | D-K0-4 | Nutzen belegt ODER archiviert |
| A9 | **K2-Vocoder klären** (hifi_gan, vocos×2, nvsr, aero): Musik-Finetune ODER gesperrt dokumentieren | D-K2-1…5, §III.11 | Jeder Pfad hat einen belegten Zustand |
| A10 | **CLAP-Material-Klassifikator real nachtrainieren** (synthetisch trainiert) | D-K1-2 | Real-Musik-Korpus + A/B |
| A11 | **17 TorchScript-Stub-Module** implementieren ODER ehrlich deklarieren (§V6) — deklarierte ML-Pfade, die nie laufen | D-K3-3 | 0 stille Stubs |

### P1 — Systematik schließen

| ID | Erfordernis | Beleg | Abnahme |
| --- | --- | --- | --- |
| A12 | **Wirkungs-Test-Schicht** an allen RELEASE_MUST-Grenzen (Artefakt-basiert, Chunked-Pfad inklusive) | D-K3-34/35/59/60 (vier Fälle) | Jedes RELEASE_MUST hat ≥ 1 Wirkungs-Test am Endartefakt |
| A13 | **Verbleibende offene K3 abarbeiten** — Auswahl: D-K3-8 (1269 ungeprüfte Layout-Reduktionen), D-K3-20 (Rest `try/except: pass`-Tests), D-K3-23 (toter Denker-Code/Council-Duplikat), D-K3-21/31 (SNR-Adaption inaktiv), D-K3-54 (AM-Peak-Extremwert), D-K3-57 (pre_echo-Repro — Diagnostik seit 10.12.17 im Log) | Register | 0 offene K3 ohne Blocker-Notiz |

### P2 — Hör-Validierung installieren

| ID | Erfordernis | Beleg | Abnahme |
| --- | --- | --- | --- |
| A14 | **MUSHRA-Human-Panel** (§3.6 Roadmap): Ridge-Regression auf echten Hörtest-Daten | Roadmap | Kalibrierter MUSHRA-Proxy auf Korpus |
| A15 | **Hörprotokoll für alle Aktivierungs-Gates** (A5–A8): blinde A/B-Hörproben auf eingefrorener Baseline, dokumentiert | §III.11, `.github/WOHLKLANG_CLAIMS.md` | Jede Aktivierung hat ein Hörabnahme-Protokoll |
| A16 | **Specs nachziehen**: PMGG-Action-Liste (Spec 02, exemplarisch) um Reject-Klasse; Ära-Specs (v10_15/16/17) als historisch kennzeichnen | Spec-Sweep 2026-10-08 | `spec_drift_check` grün + Action-Liste vollständig |

---

## 5. Was NICHT geändert werden darf (Schutz der Norm)

- **32×-RT-Zielnorm bleibt** — nicht auf Ist (53–58×) senken; die Frist ist
  die Durchsetzungs-Entscheidung (A3).
- **Performance-Budgets bleiben** — Verletzungen werden dokumentiert
  (`budget_violations`), nicht weggeschrieben.
- **Hörordnung Ebene 1–2, §V1/§V2, Dither/Export-Verträge** — unantastbar.
- **Spec > Code** bleibt die Konfliktregel: Spec-Anpassungen nur dort, wo der
  Ist-Stand belegt ist (Muster: 64→71-Sweep, 2026-10-08).

## 6. Reihenfolge (Empfehlung)

1. **Heute/Morgen:** A1 (WCS-Loch — RELEASE_MUST-Verstoß im Export) →
   A12-Testebene für die Fälle, die er offenlegt.
2. **Parallel (läuft):** 10.12.17-Validierungslauf (D-K3-57-Repro,
   neue Zähler; Ergebnisanalyse A13-Input).
3. **Danach:** A2/A3 (Performance-Paket — der Hebel für „alle Songs volle
   Kette"), dann A5–A9 (Modell-Paket; A6–A8 brauchen DEINE Hörabnahmen).
4. **Fortlaufend:** A13, A16; **strategisch:** A14/A15.

---

_Stand: 2026-10-08 · Autor: Copilot-Session · Quellen: SOTA_DEFICIT_REGISTER.md
(D-K0/K1/K2/K3), `/tmp/trio_lauf.log`, `export_metrics.json` (13-Tage-Lauf),
`unified_restorer_v3.py`, `cli/aurik_cli.py`, `backend/api/bridge*.py`,
`.github/specs/02_pipeline_architecture.md`._
