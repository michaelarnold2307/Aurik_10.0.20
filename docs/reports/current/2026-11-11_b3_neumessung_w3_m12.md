# B3-Neumessung W-3 + M12-WF4 — erste echte Qualitätsurteile nach den Pfad-Fixes (2026-11-11)

> Sitzung 2026-11-11 (GPU-freies Mess-Fenster nach D3-Ketten). Beide Stufen wurden
> zum ersten Mal mit **wirksamem Pfad** gemessen — die Ablehnungen sind damit
> substanzielle Qualitätsurteile, keine Mess-Artefakte mehr. **Kein Flag-Flip**
> (§4-Verfahren, fail-closed; Flips bleiben AUS).

## 1. Kontext

- **W-3-Lehre (D-K3-97-Vorläufer, Report `2026-11-06_w3_pfadklaerung.md`):**
  Die W-3-Ablehnung vom 2026-11-05 (0/6, Δ ≈ 0) war ein **No-Op-Artefakt** —
  der Dauer-Achsen-Defekt in `phase_49_advanced_dereverb.py` machte den
  SGMSE+-Tier-0 unerreichbar. Fix + 3 Regressionstests (Rot→Grün) liegen seit
  Commit `c0ff1a26` vor; die Neumessung stand aus.
- **M12-Lehre (D-K3-85, Report `2026-11-09_m12_pfad_erreichbarkeit.md`):**
  WF-V4 lud ein nie existierendes ONNX, TP-V2 lief mit Seed-Gewichten — die
  B3-Ablehnungen vom 2026-11-08 waren Mess-Artefakte. Beide Ladepfade sind
  gefixt (fail-closed, Modell-Zeuge im Pfad-Gate); Neumessung stand aus.

## 2. Mess-Setup

- `scripts/b3_stufenmessung.py` (B3-Verfahren, Objektivmetrik-Beschluss):
  3 Clips × 2 Fenster × 2,5 s, Seed 42, blinde AN/AUS-Paarvergleiche,
  zertifizierte Metriken `musik_mos`/`defektlast` (Spearman 0,95),
  Pfad-Wirksamkeits-Gate (Muster C) vor jedem Urteil.
- Lauf-Logs (flüchtig, /tmp): `/tmp/w3_neu.log`, `/tmp/m12_neu.log`.
- Parallel liefen: Kompositions-Suche (`b3_komposition_suche.py --clips 3
  --windows 2`) — Ergebnis folgt gesondert; GPU war durch die drei Läufe
  geteilt (Messungen untereinander konkurrierend, ~100 % GPU).

## 3. Ergebnis W-3 (`use_sgmse_musik`) — ABGELEHNT auf Substanz

| Feld | Wert |
| --- | --- |
| pfad_wirksam | **true** (SGMSE+-Tier-0 erstmals erreicht — der Achsen-Fix wirkt) |
| never_worsen | true |
| Vorteile | 2/6 |
| p (Vorzeichen-Test) | 0,6875 — nicht signifikant |
| ML→DSP-Fallbacks (B1) | 9 ⇒ `b1_sauber=false` (fail-closed) |
| messung_ungueltig | false |
| Ergebnis | **ABGELEHNT** — Flag `use_sgmse_musik` bleibt AUS |

**Wertung:** Der 2026-11-05-Beschluss ist ersetzt: Der Messpfad erreicht
SGMSE+ jetzt wirklich, und der objektive Vergleich zeigt **keinen Vorteil**
gegen die Basis (p=0,6875) plus 9 Rückfälle — ein echtes Qualitätsurteil
über die Stufe, kein Artefakt. Die W-3-Lehre (Pfad-Wirksamkeit vor Urteil)
ist damit als Verfahren bestätigt: Ohne das Gate wäre diese Ablehnung erneut
ein No-Op-Urteil gewesen.

Belege: `docs/reports/current/b3_stufenmessung_W-3.json`,
`docs/reports/current/b3_objektivbeschluss_W-3.json`.

## 4. Ergebnis M12-WF4 (`use_wf4_warp_head`) — ABGELEHNT auf Substanz

| Feld | Wert |
| --- | --- |
| pfad_wirksam | **true** (trainiertes Checkpoint lädt — D-K3-85-Fix wirkt) |
| never_worsen | true |
| Vorteile | 1/6 |
| p (Vorzeichen-Test) | 0,21875 — nicht signifikant |
| ML→DSP-Fallbacks (B1) | 0 |
| messung_ungueltig | false |
| Ergebnis | **ABGELEHNT** — Flag `use_wf4_warp_head` bleibt AUS |

**Wertung:** Erster echter Modellvergleich des trainierten WF-V4-Heads
(50/50 Epochen, R1 echt fertig) gegen den Heuristik-Pfad: **kein objektiver
Vorteil** (1/6, p=0,21875), kein Artefakt. n=6 ist eine kleine Stichprobe;
die Aktivierungsentscheidung bleibt Hörpanel/menschlich (§4).

Belege: `docs/reports/current/b3_stufenmessung_M12-WF4.json`,
`docs/reports/current/b3_objektivbeschluss_M12-WF4.json`.

## 5. Buchung & offene Punkte

- Buchungen: Register D-K3-82 / D-K3-85 / D-K4-7 / D-K4-8 + Handover §2 Zeile 6 (W-3)
  und Zeile 7 (M12) aktualisiert; dieser Report.
- **M12-TPV2-Neumessung ABGESCHLOSSEN** — ABGELEHNT auf Substanz
  (pfad_wirksam ✓, never_worsen ✓, 2/6, p=0,6875, 0 Fallbacks); Buchung D-K4-8 erfolgt.
- **Kompositions-Suche** (`/tmp/komposition_suche.log`) läuft noch — Ergebnis
  (Kandidat für EINE Hör-Session) nach Abschluss buchen.
- **M5-vocos-B3-Stage** ist implementiert und verifiziert (20 Vertragstests,
  Ruff clean, Commit-Hygiene inkl. Ledger-Snapshot); Messung
  `python3 -B scripts/b3_stufenmessung.py --stages M5-vocos` im nächsten
  freien GPU-Fenster.
- **Flips bleiben AUS** bis §4/Hör-Sign-off (§V7 (copilot-instructions.md),
  kein stiller Flip).

## 6. Harte Regeln beachtet

- Keine Flag-Änderung; keine Code-Änderung während des Messzyklus
  (§V7 (copilot-instructions.md): eine Änderung je Messzyklus).
- Laufende Messungen nicht gestört; keine Trainings gestartet.
- Zitierdisziplin AGENTS.md §5: Register-IDs mit Quelle.
