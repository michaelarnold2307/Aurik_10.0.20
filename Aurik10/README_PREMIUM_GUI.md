# AURIK 9.x.x Benutzeroberfläche 🎵

**Moderne Desktop-Anwendung für Audio-Restaurierung mit UnifiedRestorerV3**

Frameless Design, Real-Time Visualisierung und ML-Hybrid Processing

---

## ✨ Neue Features in Version 9.x.x

### 🚀 UnifiedRestorerV3 Integration

- **Defect-First Architecture** - Intelligente Defekt-Erkennung vor der Verarbeitung
- **Material-Adaptive Processing** - Automatische Erkennung von 14 Material-Typen
- **Performance-Guaranteed** - 3× RT Limit im Balanced Mode
- **56 Processing Phases** - Vollständige Pipeline mit Phase Skipping
- **8-Core Parallelization** - Optimale CPU-Auslastung

### ⚙️ System-Ressourcen-Monitor (NEU!)

- **Echtzeit CPU-Anzeige** - Live-Monitoring der Prozessor-Auslastung
- **Echtzeit Speicher-Anzeige** - Aktuelle RAM-Nutzung
- **Verarbeitungs-Modus-Anzeige** - FAST/BALANCED/QUALITY Status
- **ML-Plugin-Status** - Zeigt aktive ML-Plugins (DeepFilterNet, BANQUET, SGMSE+, MP-SENet, CREPE)
- **DSP-Modus-Indikator** - Unterscheidung zwischen ML-Hybrid und reinem DSP

### 🎨 Aussehen

- **Ohne Fensterrahmen** - Schlankes, modernes Design
- **Eigene Titelleiste** - Verschiebbar, mit Minimieren/Maximieren/Schließen
- **Dunkles Design** - Elegantes dunkles Erscheinungsbild mit Farbverläufen
- **Glas-Effekte** - Durchscheinende Karten mit Unschärfe
- **Weiche Übergänge** - Sanfte Einblendungen und Animationen
- **Edle Farben** - Hochwertige Farbverläufe (Purple/Pink Gradients)
- **Schatten-Effekte** - Räumliche Tiefe durch Schatten

### 🔬 Real-Time Visualisierungen

- **Stereo-Wellenform** - Dual-Channel-Anzeige mit Peak/RMS-Envelope
- **Spektrogramm** - Professional Inferno Colormap (20Hz-20kHz, dB-Skala)
- **Defekt-Counter** - Animierte Anzeige von 24 Defekt-Typen:
  - Klicks, Knistern, Pops, Übersteuerung
  - Brummen (50/60Hz), Rauschen, Sibilanzen
  - Aussetzer, Tonhöhenschwankungen (Wow/Flutter) und mehr
- **Phasen-Status** - Live-Updates der aktuellen Verarbeitungsphase

### 🚀 Funktionen

- **Keine Internetverbindung nötig** - Läuft komplett offline
- **Magic Button Interface** - Zwei prominente Verarbeitungs-Modi:
  - 💿 **RESTORATION** - Authentisch & behutsam, erhält Original-Charakter
  - 🎯 **STUDIO 2026** - Moderner Sound, streaming-optimiert
- **Mehrere Dateien gleichzeitig** - Batch-Verarbeitung ganzer Ordner
- **Drag & Drop** - Dateien per Drag & Drop hinzufügen
- **Warteschlange** - Queue-Management mit Status-Tracking
- **Adaptive Material-Erkennung** - Automatische Erkennung von Vinyl, Tape, CD, etc.

## 📦 Installation & Start

### Voraussetzungen

```bash
# Python 3.10+
python --version

# PyQt5 und Abhängigkeiten installieren
pip install PyQt5 numpy soundfile psutil
```

### Quick Start

```bash
# 1. Repository klonen oder entpacken
cd Aurik_Standalone

# 2. GUI starten
./run_aurik.sh

# Oder mit aktivierter venv:
source .venv_aurik/bin/activate  # Linux/Mac
.venv_aurik\Scripts\activate  # Windows
python start_aurik_90.py
```

### Alternative: Executable erstellen

```bash
# PyInstaller installieren
pip install pyinstaller

# Executable bauen
pyinstaller --onefile --windowed \
    --name "AURIK_90" \
    --icon resources/icon.ico \
    --add-data "core:core" \
    --add-data "dsp:dsp" \
    start_aurik_90.py

# Executable läuft dann ohne Python-Installation
./dist/AURIK_90
```

---

## 🎯 Bedienung

### Die 2 Magic Button Modi

AURIK 9.x.x bietet **2 prominente Verarbeitungs-Modi** mit einem Klick:

#### 💿 **RESTORATION Modus** (Balanced Quality)

- **Ziel:** Original-Charakter erhalten, behutsam verbessern
- **Quality Mode:** BALANCED (~2.4× RT)
- **Für wen:** Wer den authentischen Klang der Originalaufnahme bewahren möchte
- **Was passiert:**
  - Respektvolle Bearbeitung des Original-Klangs
  - Vorsichtige Entfernung von Störgeräuschen
  - Erhalt des Vintage-Charakters
  - Automatische Material-Erkennung (Vinyl, Tape, CD, etc.)
  - Defect-First Processing (nur notwendige Korrekturen)
- **Ideal für:** Schallplatten, Kassetten, historische Aufnahmen archivieren

#### 🎯 **STUDIO 2026 Modus** (Maximum Quality)

- **Ziel:** Moderner, klarer Sound für heutige Hörgewohnheiten
- **Quality Mode:** QUALITY (~9× RT, kein Limit)
- **Für wen:** Wer alte Aufnahmen modern und hochwertig klingen lassen möchte
- **Was passiert:**
  - Maximale Klarheit und Brillanz
  - Psychoacoustic Enhancement aktiviert
  - Optimiert für Spotify, YouTube, Streaming
  - Vollständige 42-Phasen-Pipeline
  - Premium ML-Hybrid Processing
- **Ideal für:** Digitale Veröffentlichung, modernes Hören, Studio-Produktionen

### GUI Starten

```bash
./run_aurik.sh
```

### So funktioniert's

1. **Datei öffnen** - Klick auf "📂 Datei öffnen" oder Dateien ins Fenster ziehen
2. **Live-Visualisierung** - Wellenform, Spektrogramm und Defekt-Analyse in Echtzeit
3. **Magic Button wählen** - RESTORATION oder STUDIO 2026 anklicken
4. **Verarbeitung läuft** - Automatische ML-Hybrid Processing mit Live-Status
5. **Fertig** - Restaurierte Datei wird automatisch gespeichert

---

## 🔬 Technische Details

### UnifiedRestorerV3 Architecture

- **Defect-First Approach** - Intelligente Defekt-Erkennung vor Processing
- **54 Processing Phases** - Vollständige Pipeline (Click, Hum, Denoise, Reverb, etc.)
- **Adaptive Phase Skipping** - Nur notwendige Phasen werden ausgeführt (20-40% Speedup)
- **Material Auto-Detection** - 12 Material-Typen (Shellac, Vinyl, Tape, CD, MP3, etc.)
- **Performance Guard** - 3× RT Enforcement im Balanced Mode
- **8-Core Parallelization** - Optimale CPU-Nutzung mit ThreadPoolExecutor

### ML-Hybrid Plugins (Optional, via Docker)

- **KIM Vocal/Inst Enhancer** - ML-Gesangs-/Instrumental-Klarheit (MDX23C-kompatibel)
- **DCCRN** - Deep Complex Convolution Recurrent Network für Dereverb
- **CREPE** - Pitch Detection für Wow/Flutter Correction
- **Fallback zu DSP** - Automatisch wenn ML-Plugins nicht verfügbar

### System-Ressourcen-Monitor

- **CPU-Monitoring** - Live-Anzeige der Prozessor-Auslastung
- **Memory-Monitoring** - Aktuelle RAM-Nutzung
- **Mode-Indikator** - Zeigt aktiven Quality Mode (FAST/BALANCED/QUALITY)
- **ML-Status** - Zeigt aktive ML-Plugins oder DSP-Fallback
- **Farbcodierung** - Grün (<70%), Gelb (70-90%), Rot (>90%)

---

- **🎯 Studio 2026** - Moderner, klarer Sound für heute
- **💿 Restoration** - Original-Charakter bewahren

3. **Tonträger auswählen** - AURIK optimiert automatisch basierend auf dem Quellmedium
4. **Starten** - Klick auf "🎵 Verarbeiten Starten"
5. **Speichern** - Nach der Bearbeitung mit "💾 Alle exportieren" sichern

**✨ Automatische Optimierung:** AURIK wendet automatisch alle erforderlichen Verbesserungen an:

- **Rauschunterdrückung** - Adaptiv an Ihr Material angepasst
- **Clipping-Reparatur** - Übersteuerungen werden intelligent rekonstruiert
- **Click/Crackle Removal** - Knackser und Knistern verschwinden
- **Normalisierung** - Optimaler Ausgangspegel für Ihr Zielformat

Keine manuellen Einstellungen nötig - AURIK wählt stets das Optimum für musikalische Exzellenz!

### Tastenkombinationen

- **Doppelklick auf Titelleiste** - Vollbild ein/aus
- **Titelleiste ziehen** - Fenster verschieben
- **ESC-Taste** - Fenster schließen (optional aktivierbar)

## 🎨 Bedienelemente

### Aufbau der Oberfläche

#### Titelleiste

- Logo/Symbol (links)
- App-Titel "AURIK 8.0"
- Status-Anzeige (Mitte)
- Fenster-Steuerung: Minimieren/Maximieren/Schließen (rechts)
- Verschiebbar zum Fenster bewegen

#### Schaltflächen

- Zwei Arten:
  - **Haupt-Schaltflächen:** Mit Farbverlauf (lila-blau)
  - **Neben-Schaltflächen:** Transparent mit Rahmen
- Reagieren auf Mauszeiger mit sanften Animationen

#### Informations-Karten

- Halbtransparente Kästen mit Glas-Effekt
- Automatische Schatten für räumliche Tiefe
- Weiche Unschärfe im Hintergrund

#### Fortschrittsbalken

- Zeigt Bearbeitungsfortschritt in Prozent
- Animiert mit Farbverlauf
- Aktualisiert sich während der Verarbeitung

### Fenster-Aufbau

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 🎵 AURIK 9.x.x - Musik Restauration       [Bereit]   [−][□][×]                    │ ← Titelleiste
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Musik Restauration - Reparatur - Rekonstruktion - Remastering                  │
│  Alte Musikaufnahmen fast wie am Tag der Aufnahme klingen lassen                │
│  [📂 Datei öffnen] [📁 Mehrere Dateien] [⚙️ Einstellungen]                      │ ← Kopfleiste
│                                                                                  │
├──────────────────┬───────────────────────────────────────┬───────────────────────┤
│ Steuerung        │ Live-Visualisierung & Analyse         │ Warteschlange         │
│ ┌──────────────┐ │ ┌───────────────────────────────────┐ │ ┌─────────────────┐ │
│ │ Bearbeitungs │ │ │ 🌊 Wellenform (Live)              │ │ │ 📄 datei1.wav   │ │
│ │ Modus:       │ │ │ ▁▃▅▇█▇▅▃▁▃▅▇█▇▅▃▁▃▅▇█▇▅▃▁        │ │ │ 📄 datei2.mp3   │ │
│ │ 🎯 Studio    │ │ │                                   │ │ │ 📄 datei3.flac  │ │
│ │ 💿 Restaur.  │ │ ├───────────────────────────────────┤ │ └─────────────────┘ │
│ │              │ │ │ 📊 Spektrogramm                   │ │                     │
│ │ Tonträger:   │ │ │ [Frequenz-Heatmap mit Defekten]   │ │ [██████░░░░░░░░]  │
│ │ [Vinyl 33⅓]  │ │ │ 20kHz ░░▓▓██▓▓░░                  │ │ 60,00%            │
│ │ [MiniDisc]   │ │ │ 10kHz ▓▓████████▓▓                │ │                     │
│ │ [Kassette]   │ │ │  5kHz ██████████████              │ │ [🗑️ Leeren]       │
│ │ [CD]         │ │ │  1kHz ████████████████            │ │ [💾 Speichern]    │
│ │ [...]        │ │ │   0Hz ▓▓████████▓▓                │ │                     │
│ │              │ │ └───────────────────────────────────┘ │                     │
│ │ [🎵 Starten] │ │ ┌───────────────────────────────────┐ │                     │
│ │              │ │ │ ⚠️ Erkannte Defekte & Korrekturen │ │                     │
│ └──────────────┘ │ │ ⚡ Knackser:           [1.247] ✓ BEREINIGT     │         │
│                  │ │ 🧻 Knistern:           [0.856] ✓ BEREINIGT     │         │
│                  │ │ 💥 Pops:               [0.034] ✓ BEREINIGT     │         │
│                  │ │ 🔊 Übersteuerung:      [0.023] ✓ KORRIGIERT    │         │
│                  │ │ 🔌 Brummen:            [50.2Hz] ✓ ENTFERNT     │         │
│                  │ │ 🌀 Rauschen:           [-12.8dB] ⚙️ REDUZIERT  │         │
│                  │ │ 🎤 Sibilanzen:         [0.142] ✓ REDUZIERT     │         │
│                  │ │ 📍 Aussetzer:          [0.003] ✓ REPARIERT     │         │
│                  │ │ 🎚️ Tonhöhenschwankung: [0.15%] ✓ STABILISIERT │         │
│                  │ └───────────────────────────────────┘ │                     │
├──────────────────┴───────────────────────────────────────┴───────────────────────┤
│ ⚙️ Verarbeitung läuft... | Phase: Defekt-Analyse | Zeit: 00:02:34 | Queue: 3   │ ← Statusleiste
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Legende der animierten Zähler (Zwei-Phasen-Animation):**

**Phase 1: Erkennung (🔍 ERKENNE - blau)**

- Zähler "rattern" **HOCH** von 0 zu den erkannten Werten
- Beispiel: Knackser [0] → [342] → [789] → [1.247]
- Status-Icon: 🔍 ERKENNE in blauer Farbe

**Phase 2: Korrektur (⚙️ BEARBEITE - orange)**

- Zähler "rattern" **RUNTER** von erkannten Werten auf 0
- Beispiel: Knackser [1.247] → [823] → [394] → [0]
- Status-Icon: ⚙️ BEARBEITE in oranger Farbe

**Phase 3: Abgeschlossen (✓ BEREINIGT - grün)**

- Alle Zähler bleiben bei [0]
- Status-Icon: ✓ BEREINIGT in grüner Farbe (fett)
- Signal: Alle Fehler wurden erfolgreich korrigiert

**Erkannte Defekte im Detail:**

- **⚡ Knackser** - Kleine, schnelle Transienten (< 0.2 Amplitude)
- **🧻 Knistern** - Kontinuierliche Knackser-Dichte (pro 10 Sekunden)
- **💥 Pops** - Große Transienten (> 0.2 Amplitude), tiefe Kratzer
- **🔊 Übersteuerung** - Clipping-Samples (> 0.99 Amplitude)
- **🔌 Brummen** - 50/60Hz Netzbrummen via FFT-Analyse
- **🌀 Rauschen** - Rauschpegel in dB (10. Perzentil)
- **🎤 Sibilanzen** - Überbetonung im 6-8kHz Bereich
- **📍 Aussetzer** - Dropouts/Nullstellen im Signal
- **🎚️ Tonhöhenschwankung** - Wow & Flutter (Gleichlaufschwankung)

## 🎨 Farben und Design

### Hintergrund-Farben

```
Hauptfenster:
  Dunkelblau → Lila → Dunkelviolett
  (Sanfter Farbverlauf)

Titelleiste:
  Dunkelgrau → Mitternachtsblau
  (Dezenter Verlauf)
```

### Schaltflächen-Farben

```
Normal:
  Lila → Blau (Farbverlauf)

Bei Mauszeiger darüber:
  Heller lila → Heller blau

Beim Klicken:
  Dunkler lila → Dunkler blau
```

### Text-Farben

```
Haupttext:        Weiß
Nebentext:        Hellgrau
Akzente:          Hellblau
Erfolg:           Grün
Warnung:          Orange
Fehler:           Rot
```

### Transparenzen

```
Karten-Hintergrund:    70% undurchsichtig
Neben-Schaltflächen:   5% sichtbar
Rahmen:                10% sichtbar
Mauszeiger-Effekt:     10% Aufhellung
```

## 🔧 Anpassungen

### Farben ändern

Sie können andere Farben verwenden:

- Grün-Blau statt Lila-Blau
- Orange-Pink für wärmeres Design
- Dunkelrot für elegantes Aussehen

### Ecken-Rundung anpassen

Sie können die Fensterecken mehr oder weniger rund machen.

### Schatten anpassen

Sie können die Schatten stärker oder schwächer machen.

## 📱 Fenster-Steuerung

### Minimieren

Klick auf "-" in der Titelleiste verkleinert das Fenster in die Taskleiste.

### Maximieren/Verkleinern

Klick auf "□" vergrößert das Fenster auf Vollbild oder verkleinert es wieder.

### Schließen

Klick auf "×" beendet das Programm.

### Fenster verschieben

Klicken und ziehen Sie die Titelleiste, um das Fenster zu bewegen.

## 🐛 Probleme beheben

### Problem: Programm startet nicht

**Lösung:** Stellen Sie sicher, dass PyQt5 installiert ist:

```bash
pip install PyQt5
```

### Problem: Fenster ist durchsichtig/fehlerhaft

**Lösung (nur Linux):** Sie benötigen einen Compositor wie `compton` oder `picom`.

### Problem: Fenster ist zu klein oder zu groß

**Lösung:** Beim ersten Start wird eine Standardgröße verwendet. Sie können das Fenster mit der Maus vergrößern/verkleinern.

### Problem: Fenster lässt sich nicht verschieben

**Lösung:** Versuchen Sie, die Titelleiste zu ziehen (nicht die Buttons).

### Problem: Farben werden nicht richtig angezeigt

**Lösung:** Starten Sie das Programm neu. Bei älteren Computern können manche Effekte fehlen.

## 📊 Performance

### Optimierungen

- **Hardware-Beschleunigung** - Qt verwendet GPU wenn verfügbar
- **Thread-basiert** - Prozessierung läuft in separatem Thread
- **Lazy Loading** - Widgets werden nur bei Bedarf geladen
- **Caching** - Styles werden gecacht

### Speicher

- **Idle:** ~50-80 MB
- **Mit Queue (10 Files):** ~100-150 MB
- **Bei Verarbeitung:** +200-500 MB (je nach Datei)

## 🚀 Deployment

### Standalone Executable

```bash
# Windows
pyinstaller aurik_professional.spec

# Mac
pyinstaller --windowed start_aurik_premium.py

# Linux
pyinstaller --onefile start_aurik_premium.py
```

### Nuitka (schneller)

```bash
pip install nuitka
python -m nuitka --standalone --onefile \
    --enable-plugin=pyqt5 \
    start_aurik_premium.py
```

## 📝 Lizenz

Siehe [LICENSE](../LICENSE) im Hauptverzeichnis.

## 🤝 Support

Bei Fragen oder Problemen:

- Issue erstellen auf GitHub
- Dokumentation lesen: [docs/](../docs/)
- Troubleshooting: [TROUBLESHOOTING.md](../docs/guides/TROUBLESHOOTING.md)

---

**AURIK Professional 8.0.3** - Premium Audio Restoration
