# Incident-Dokumentation: Pegelexplosion + L/R-Zeitversatz (25.04.2026)

## Kurzfassung

Im produktiven Lauf trat ein abrupter Qualitaetsabfall auf: ab etwa 18 % Fortschritt wurden gleichzeitig eine Pegelexplosion und ein Zeitversatz zwischen linkem und rechtem Kanal hoerbar.

Die Analyse zeigte einen gemeinsamen technischen Schwerpunkt in `phase_29_tape_hiss_reduction` mit Folge-Rollbacks durch den CIG-STFT-Guard.

## Beobachtetes Symptomprofil

- Frueher Laufabschnitt klang stabil.
- Am Kipppunkt traten parallel auf:
  - Pegelanstieg in eigentlich ruhigen Segmenten
  - deutliche L/R-Desynchronisation
- Frontend-Logs zeigten wiederholt:
  - PMGG best-effort um phase_29
  - CIG-Rollback wegen STFT group delay nach phase_29
  - in einigen Laeufen zusaetzlich spaeter Rollback nach phase_49

## Root-Cause-Hypothese und Befund

### 1) Stereo-Layout-Risiko in phase_29

`phase_29` nutzte den Stereo-Pfad mit Annahmen im channels-last-Format, ohne den Input vorher kanonisch zu normalisieren. Bei channel-first (`(2, N)`) kann das zu Fehlverarbeitung und damit zu L/R-Artefakten fuehren.

### 2) Loudness-Rescue-Risiko in phase_29

Der Loudness-Preservation-Pfad nutzte zuvor direkten globalen Gain mit Clipping. Dadurch konnten ruhige Tails/Fadeouts erneut hochgezogen werden (Reinflation), obwohl der musikalische Anteil bereits korrekt entrauscht war.

## Implementierte Korrekturen

Datei: `backend/core/phases/phase_29_tape_hiss_reduction.py`

1. Stereo-Layout normalisieren und wiederherstellen

- Eingang: `to_channels_last(audio)`
- Ausgang: `restore_layout(processed, was_transposed)`

2. Loudness-Rescue auf envelope-aware Gain umgestellt
statt globalem direct-clip nun:

- `apply_musical_gain_envelope(..., gate_dbfs=-36.0, crossfade_ms=10.0)`

3. Stereo-Lag-Sicherheitsguard in phase_29 hinzugefuegt

- misst Input- und Output-Lag zwischen L/R

- korrigiert neu eingefuehrten Lag > 1 ms lokal in der Phase
- schreibt diagnostische Metadaten:
  - `lag_input_samples`
  - `lag_output_samples`
  - `lag_corrected`
  - `lag_output_corrected_samples`

## Test- und Verifikationsstatus

Datei: `tests/unit/test_phases_mid_late.py`

Neue/erweiterte Regressionen in `TestPhase29TapeHissReduction`:

- channel-first Stereo bleibt zeitlich ausgerichtet
- quiet tail wird durch Loudness-Preservation nicht uebermaessig angehoben
- grosser kuenstlich eingefuehrter L/R-Lag wird durch den Guard zurueckgefuehrt

Ergebnis:
`TestPhase29TapeHissReduction`:

10 passed

## Offene Produktionsverifikation

Der Unit-Nachweis ist vorhanden. Fuer den finalen Produktionsnachweis ist ein frischer GUI-Lauf mit neuem Export erforderlich, damit bestaetigt wird, dass der 18%-Kipppunkt im echten Restore-Flow nicht mehr auftritt.

Empfohlener Ablauf:

1. Frontend neu starten (damit neuer Code sicher aktiv ist)
2. identisches Material erneut im gleichen Modus restaurieren
3. Output auf beide Kriterien pruefen:
   - keine Pegelexplosion in Intro/Fadeout
   - kein wahrnehmbarer L/R-Zeitversatz
