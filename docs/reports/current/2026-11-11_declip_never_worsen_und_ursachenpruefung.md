# Declip-Never-worsen und Ursachenprüfung der Clipping-Verschlechterung

**Datum:** 2026-11-11 · **Register:** D-K3-97 (Korrektur zu D-K3-95) · **Status:** Fix implementiert und gemessen; Clipping-Lücke **offen**, aber neu verortet
**Kein Flag-Flip, kein Deploy, kein Eingriff in den UV3-Klangpfad.**

## 1. Anlass und Prüfauftrag

Die Vorsitzung hat §2 Zeile 13 („Sinn-Beweis", D-K3-95) bearbeitet und einen
unfertigen Patch an `phase_07_declipper.py` hinterlassen, mit der Behauptung:

> Der PCHIP-Declipper interpoliert über das Sättigungs-Plateau, „überschwingt"
> und clippt dann auf ±1,0 ⇒ neue Flat-Tops über 0,98 ⇒ `defektlast`
> 0,478 → 0,716.

Dazu wurde im Register notiert, der Fix sei „statisch lückenlos belegt". Diese
Sitzung hatte den Auftrag, den Patch zu verifizieren und die Lücke zu schließen.
**Ergebnis: die Kausalitätsbehauptung ist falsch.** Der Patch bleibt trotzdem —
aber als Never-worsen-Guard für den Pfad, in dem die Phase tatsächlich läuft.

## 2. Messung 1 — PCHIP gegen das saubere Original (ΔSDR)

10-s-Ausschnitt ab 60 s aus `Elke Best … (original).wav`, auf 0,99 normiert,
hart begrenzt, dann `_declip_pchip` mit selbstkalibrierter Schwelle.
ΔSDR = SDR(PCHIP) − SDR(Eingang), beide gegen das saubere Original.

| Decke | Schwelle | Clip-Anteil | SDR Eingang | SDR PCHIP | ΔSDR |
| --- | --- | --- | --- | --- | --- |
| 0,99 | 0,880 | 0,01 % | 162,39 dB | 46,42 dB | **−115,97 dB** |
| 0,95 | 0,880 | 0,01 % | 64,11 dB | 46,42 dB | **−17,70 dB** |
| 0,90 | 0,880 | 0,01 % | 53,25 dB | 46,41 dB | **−6,84 dB** |
| 0,60 | 0,880 | 0,00 % | 28,89 dB | 28,89 dB | 0,00 dB |
| 0,30 | 0,880 | 0,00 % | 12,64 dB | 12,64 dB | 0,00 dB |

Damit ist belegt:

- **Sparse Hard-Clips werden durch PCHIP schlechter**, nicht besser. Die
  Interpolation ersetzt wenige Einzel-Samples durch Werte, die weit daneben
  liegen — bei 0,01 % Clip-Anteil kostet das bis zu 18 dB SDR.
- **Der Detektor ist nach unten blind:** `_histogram_clip_threshold` klemmt die
  Schwelle auf **[0,85 … 0,999]**. Ein auf 0,60 oder 0,30 begrenzter Master wird
  als „kein Clipping" gemeldet (0,00 % Clip-Anteil) — dort passiert nichts,
  weder Gutes noch Schlechtes.

## 3. Messung 2 — Sättigung (das Schadens-Rezept „clipping")

Dieselben drei Schadens-Proben, die D-K3-95 bewertet hat:

| Probe | Schwelle | Clip-Anteil | ΔSDR durch PCHIP | max. Änderung |
| --- | --- | --- | --- | --- |
| Träumer · clipping | 0,898 | 0,50 % | 0,00 dB | 0,0113 |
| Boogie · clipping | 0,897 | 0,51 % | 0,00 dB | 0,0182 |
| Jets · clipping | 0,900 | 0,54 % | 0,00 dB | 0,0017 |

PCHIP ist auf diesem Material **wirkungslos** (Schwelle ≈ Decke). Es kann die
in D-K3-95 gemessene Verschlechterung (0,478 → 0,716) also nicht verursacht
haben.

## 4. Messung 3 — PCHIP ist formtreu, der ±1,0-Clamp griff nie

`_declip_pchip` ist (per Konstruktion) monotonie-erhaltend: die Ausgabe bleibt
im Wertebereich der Stützstellen, also **unter** dem Eingangs-Maximum. Auf allen
drei Proben gilt `pchip_neu == pchip_alt` (identische `defektlast`, identische
Proxy-Werte) — der `np.clip(…, -1.0, 1.0)` hat nie eingegriffen. Die Hypothese
„neue Flat-Tops bei 1,0" ist damit widerlegt; der Test
`test_pchip_ist_formtreu_unter_der_eingangs_decke` hält die Eigenschaft fest.

## 5. Messung 4 — die Phase läuft im Produktionspfad gar nicht

- `DeclipperPhase` hat genau **eine** Aufrufstelle im Repo:
  `backend/core/coordinated_repair.py:1361` (Repair-Planer/Ablations-Werkzeuge).
- Im vollständigen UV3-Lauf-Log der 12 Schadens-Paare
  (`/tmp/schadens_restore.log`, 3,1 MB, 13 Restaurationen) kommt das Wort
  `declipper` **0-mal** vor; die Phase-07-Logzeilen („Verarbeitungsschritt 07
  declip: …") fehlen vollständig.

⇒ Der Declipper ist **nicht** Teil des UV3-Produktionspfads. Jede
Qualitätsaussage über „Clipping im Produktionspfad" ist keine Aussage über
diesen Patch.

## 6. Nachbewertung der neu erzeugten Clipping-Proben (ohne Attribution)

Die drei `_restauriert.wav` der Clipping-Paare wurden am 2026-11-11 zwischen
08:43 und 10:36 neu geschrieben; die Bewertung vom 00:36 ist damit überholt.
Nachgemessen (`bewerte_schadens_proben.py`, pegelgematcht, unverändertes
Messverfahren):

| Probe | Δ defektlast | Δ musik_mos | Pegel-Gain |
| --- | --- | --- | --- |
| Träumer · clipping | −0,1188 | +0,1481 | 0,548 |
| Boogie · clipping | +0,0509 | +0,1214 | 0,824 |
| Jets · clipping | 0,0000 (bit-identisch) | 0,0000 | 1,000 |
| **Aggregat** | **−0,0226** (vorher +0,1667) | **+0,0898** (vorher −0,0935) | — |

**Diese Verbesserung ist dem Patch NICHT zuzuschreiben.** Im UV3-Pfad hat sich
kein Code geändert (der Declipper läuft dort nicht), und es gibt zu den neuen
WAVs kein Lauf-Log. Es ist eine Einzelbeobachtung ohne Wiederholung — als
Qualitätsurteil unbrauchbar, als Hinweis brauchbar: derselbe Input kann
offenbar in einen anderen Entscheidungszweig laufen (Jets ist jetzt
bit-identisch = „nichts tun" statt „Pegel 0,30 und verzerrt exportieren").

## 7. Was im alten Lauf tatsächlich auffällt

Im 12-Paare-Lauf (00:36) steht:

```
WARNING … Early-Silence-Gate phase_07_harmonic_restoration: Audio-RMS=-91.1 dBFS < -60 dBFS
WARNING … Verarbeitungsschritt-07-Silence-Guard: Ausgabe-RMS -91.1 dBFS → Rollback
```

also ein **Fast-Silence-Zustand (−91,1 dBFS)** in einem Goal-Compensation-
Sekundärlauf, den der Silence-Guard abgefangen hat. Die Pegel-Zeile der
Hauptkette bleibt über alle Phasen stabil (`pre=-14.4 … post=-14.9 dBFS`).
Der Pegel-Gain 0,30 des Jets-Exports ist damit **nicht** aus der Hauptkette
erklärt; er entsteht in der Export-/Entscheidungskette (Loudness-Ziel vs.
Qualitäts-Gate) — das ist der nächste, noch offene Ort.

## 8. Was geändert wurde (und was nicht)

`backend/core/phases/phase_07_declipper.py`:

1. **Never-worsen-Übernahme je Kanal:** PCHIP wird nur noch übernommen, wenn der
   Harmonik-Proxy gegen den Eingang sinkt (Muster des Sparse-Zweigs,
   delta-basiert). Motiv: Messung 1 (bis −18 dB SDR auf echten Hard-Clips).
2. **Wahrheits-Passthrough:** Wird kein Kandidat übernommen, ist das Signal
   unverändert — dann meldet die Phase `declip_applied: false`,
   `reason: never_worsen_proxy` und **kein** `resolved_defects["CLIPPING"]`.
   Vorher hätte sie eine Reparatur verbucht, die nicht stattgefunden hat.
3. **Keine Decken-Änderung.** Der ursprünglich mitgelieferte Clamp auf das
   Eingangs-Maximum ist messbar wirkungslos (Abschnitt 4) und wurde wieder
   entfernt — kein Code, der eine falsche Begründung trägt.

Tests: `tests/unit/test_phase_07_declipper.py` — **13/13 grün**, darunter neu
`test_phase07_senkt_den_harmonik_proxy_nie` (Never-worsen-Kontrakt),
`test_passthrough_meldet_keine_reparatur` (Metrik-Vertrag) und
`test_pchip_ist_formtreu_unter_der_eingangs_decke` (widerlegte Hypothese,
festgehalten). Ruff clean.

## 9. Grenzen

- n = 3 Schadens-Proben je Rezept, 10-s-Schnipsel, mono, 48 kHz — wie D-K3-95.
- `defektlast` bewertet nur die **ersten 16384 Samples** (0,34 s bei 48 kHz).
  Bei Stücken mit leisem Anfang ist die Metrik dort blind; das erklärt
  identische Werte über verschiedene Eingänge hinweg und ist eine **dritte**
  Grenze der Metrik neben der Sättigungs-Blindstelle.
- Die SDR-Zahlen für Decken 0,99/0,95 sind Extremfälle mit 0,01 % Clip-Anteil;
  sie zeigen die Richtung, nicht die Größe im Musikkontext.
- Reproduzierbarkeit: der UV3-Pfad ist nicht bit-reproduzierbar (bekannt seit
  D-K3-85, RMS-Rauschen ~6,5e-4). Vor jedem Qualitätsurteil muss derselbe Lauf
  zweimal gefahren und verglichen werden.

## 10. Nächste Schritte (nicht autonom)

1. **Reproduzierbarkeit zuerst:** Jets-Clipping zweimal durch den
   Produktionspfad, beide Logs behalten, WAV-Hashes vergleichen. Erst danach ist
   „repariert/nicht repariert" eine Aussage.
2. **Pegel-/Entscheidungskette:** warum exportiert die Pipeline bei Jets
   gelegentlich mit Gain 0,30 (und gelegentlich bit-identisch)? Kandidaten:
   Loudness-Ziel gegen Qualitäts-Gate, Goal-Compensation-Sekundärlauf
   (−91,1 dBFS).
3. **Detektor-Untergrenze 0,85:** Clipping unter −1,4 dBFS wird nicht erkannt
   (Abschnitt 2). Eine Erweiterung wäre klangwirksam ⇒ §4-Verfahren.
4. **`defektlast`-Grenzen** (Sättigung, 0,34-s-Fenster) als Zertifikats-Grenzen
   dokumentieren oder die Metrik erweitern — Entscheidung über die
   Metrik-Zertifizierung, nicht im Vorbeigehen.
