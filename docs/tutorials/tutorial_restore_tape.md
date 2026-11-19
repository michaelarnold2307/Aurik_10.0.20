# Tutorial: Tonband restaurieren

> §15.7: Schritt-für-Schritt-Anleitung für Tonband-Restaurierung.

## Vorbereitung

1. Tonband digitalisiert (96 kHz, 24-bit WAV empfohlen für beste Ergebnisse)
2. Azimut kalibriert (falls Wiedergabegerät einstellbar)
3. Aurik installiert

## Schritt 1: Automatische Restaurierung

```bash
python -m aurik restore meine_bandaufnahme.wav --mode full
```

## Schritt 2: Typische Band-Defekte

```bash
python -m aurik restore meine_bandaufnahme.wav \
    --material tape \
    --defects hiss,dropouts,wow_flutter,print_through \
    --era 1965
```

## Schritt 3: Dolby-Dekodierung (falls zutreffend)

```bash
python -m aurik restore meine_bandaufnahme.wav \
    --dolby-type A \
    --mode full
```

[Weitere Details folgen...]
