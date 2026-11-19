# Aurik als lokale Desktop-App (Releasepfad)

## Kurzfassung

Aurik wird produktiv als reine Desktop-App betrieben.

- Linux: AppImage
- Windows 10/11: Installer (.exe)
- Kein Cloud-Betrieb, kein Server-Zwang, kein Docker-Zwang im Releasepfad

## Verbindlicher Ablauf

```text
Datei laden -> Bridge Import (get_load_audio_fn)
Voranalyse  -> run_pre_analysis genau einmal
Pipeline    -> AurikDenker.denke(audio, sr, mode)
Export      -> export_guard + validate_export_quality + AudioExporter
```

## Zulassige Modi

- `restoration`
- `studio2026`

Genau ein Nutzerentscheid pro Datei: Moduswahl.

## Deployment-Regeln (RELEASE_MUST)

- Desktop-only Distribution (AppImage/.exe)
- 100 % offline nach Installation
- Keine produktiven Legacy-Serverpfade als Standardkommunikation
- Keine Endnutzer-Pflicht zu `pip install`

## Empfohlene Nutzung

1. App starten (`run_aurik.sh` unter Linux oder Installer unter Windows).
2. Datei importieren.
3. Modus waehlen (`Restoration` oder `Studio 2026`).
4. Verarbeitung starten.
5. Ergebnis in `output/` pruefen.

## Technische Notizen fuer Entwickler

- Release-Einstiege muessen den Bridge/Denker/Exporter-Vertrag einhalten.
- Historische Server-, Docker- oder v2-Dokumentation gilt nur als `LEGACY_NON_RELEASE`.
- Feature-Arbeit darf keinen parallelen Produktpfad neben dem kanonischen Vertrag erzeugen.

## Legacy-Hinweis

Fruehere Beschreibungen mit FastAPI-/Docker-/Direktpipeline-Aufrufen sind fuer den
Desktop-Releasepfad nicht normativ und duerfen nicht als Standardanleitung verwendet werden.
