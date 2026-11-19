# Tutorial: Vinyl restaurieren

> §15.7: Schritt-für-Schritt-Anleitung für Vinyl-Restaurierung.

## Vorbereitung

1. Vinyl-Aufnahme digitalisiert (48 kHz, 24-bit WAV empfohlen)
2. Aurik installiert (siehe [Getting Started](../getting_started.md))

## Schritt 1: Automatische Restaurierung

```bash
python -m aurik restore meine_vinyl_aufnahme.wav --mode full
```

Aurik erkennt automatisch:
- Material: Vinyl
- Defekte: Klicks, Knackser, Rillenrauschen
- Ära: Anhand spektraler Signatur

## Schritt 2: Manuelle Feinabstimmung (optional)

```bash
python -m aurik restore meine_vinyl_aufnahme.wav \
    --material vinyl \
    --defects clicks,crackle,surface_noise \
    --era 1972 \
    --quality high
```

## Schritt 3: Ergebnis prüfen

```bash
python -m aurik compare original.wav restored.wav
```

## Typische Ergebnisse

| Defekt | Typische Verbesserung |
|--------|----------------------|
| Klicks | 95-99% entfernt |
| Knackser | 90-95% entfernt |
| Rillenrauschen | -12 bis -20 dB Reduktion |

[Weitere Details folgen...]
