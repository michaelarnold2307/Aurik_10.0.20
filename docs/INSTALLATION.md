# Aurik 10.1.0 — Installation Guide

> Stand: 10.1.0 | Linux (Ubuntu 22.04/24.04 LTS, Zorin OS 17/18), macOS, Windows

---

## Linux (Ubuntu 22.04/24.04 LTS · Zorin OS 17/18)

Aurik liefert ein **komplettes Installationsprogramm** (`scripts/install_aurik.sh`),
das alle Schritte automatisiert: OS-Erkennung, System-Abhängigkeiten, virtuelle
Umgebung, PyTorch (CPU/GPU), Python-Pakete, Modell-Prüfung und Launcher-Integration.

### Schritt 1: Repository holen

```bash
git clone https://github.com/michaelarnold2307/Aurik_10.0.20.git
cd Aurik_10.0.20
```

### Schritt 2: Komplett-Installation

```bash
bash scripts/install_aurik.sh
```

Das Installationsprogramm erledigt automatisch:

| Schritt | Inhalt |
|---|---|
| OS-Prüfung | Ubuntu 22.04/24.04 LTS und Zorin OS 17/18 werden erkannt (`/etc/os-release`) |
| System-Pakete | `python3` (3.10–3.12), `python3-venv`, `python3-pip`, `python3-tk` (GUI), `git`, `curl`, `ffmpeg`, `libsndfile1`, `libportaudio2`, `portaudio19-dev`, `python3-dev` |
| Virtuelle Umgebung | `.venv_aurik` (Projekt-Standard) |
| PyTorch | CPU-only per Default (`torch==2.7.0+cpu`); `--cuda` für NVIDIA, `--rocm` für AMD |
| Python-Pakete | `requirements/requirements_aurik.txt` |
| Modell-Prüfung | Alle `models/manifest.json`-Einträge werden gegen die Platte geprüft |
| Launcher | CLI-Wrapper `aurik` + Desktop-Eintrag `Aurik.desktop` (per `--system` systemweit) |

### Schritt 3: ML-Modelle bereitstellen

Die ML-Modelle (`models/`, ca. 30 GB) sind **nicht im Git-Repository** enthalten.
Das Installationsprogramm weist am Ende aus, welche Modelle fehlen. Zwei Wege:

```bash
# Weg A: Lokales Modell-Bundle importieren (USB/Datenträger/Backup)
bash scripts/install_aurik.sh --models-src /media/michael/Modelle

# Weg B: Modell-Downloader nutzen (falls Quellen konfiguriert sind)
source .venv_aurik/bin/activate
python3 -c "from backend.core.model_downloader import get_model_downloader; get_model_downloader().download_all()"
```

Fehlende Modelle sind kein Blocker: Aurik fällt konform §V6
(copilot-instructions.md) sichtbar auf DSP-Pfade zurück.

### Schritt 4: Starten

```bash
./run_aurik.sh                # oder:
source .venv_aurik/bin/activate && python3 Aurik10/main.py
```

### Optional: GPU

```bash
bash scripts/install_aurik.sh --cuda    # NVIDIA
bash scripts/install_aurik.sh --rocm    # AMD ROCm
```

Hinweis (§III.9): ONNX-Execution-Provider außer CPU werden nur mit
Paritäts-Nachweis gegen ONNX-CPU verwendet — die ROCm-Kerne sind dedizierte,
paritätsverifizierte Torch-Pfade.

---

## macOS (12 Monterey+)

### Schritt 1: Homebrew installieren

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Schritt 2: Aurik installieren

```bash
git clone https://github.com/michaelarnold2307/Aurik_10.0.20.git
cd Aurik_10.0.20
bash scripts/install_macos.sh
```

### Schritt 3: Starten

```bash
source .venv_aurik/bin/activate
python3 Aurik10/main.py
```

> **Apple Silicon (M1/M2/M3):** Das Skript erkennt den Chip automatisch und installiert `onnxruntime-silicon` für GPU-Beschleunigung.

---

## Windows (10/11)

### Schritt 1: Python installieren

1. https://python.org → Download Python 3.10+
2. **Wichtig:** „Add Python to PATH" ankreuzen
3. Installation abschließen

### Schritt 2: Git installieren

1. https://git-scm.com → Download → Installation mit Defaults

### Schritt 3: Aurik installieren

```powershell
git clone https://github.com/michaelarnold2307/Aurik_10.0.20.git
cd Aurik_10.0.20
powershell -ExecutionPolicy Bypass -File scripts/install_windows.ps1
```

### Schritt 4: Starten

```powershell
.\.venv_aurik\Scripts\Activate.ps1
python Aurik10/main.py
```

> **NVIDIA GPU:** Treiber von nvidia.com installieren. Das Skript erkennt CUDA automatisch.

---

## Nach der Installation

### Installations-Selbsttest

```bash
bash scripts/install_aurik.sh --smoke
```

### Modelle prüfen

```bash
python3 -c "
from backend.core.model_downloader import get_model_downloader
dl = get_model_downloader()
print(dl.get_download_progress())
"
```

### GPU verifizieren

```bash
python3 scripts/detect_gpu_capabilities.py --json
```

---

## Deinstallation

```bash
# Linux/macOS
rm -rf Aurik_10.0.20
rm -rf ~/.aurik
rm -f ~/.local/share/applications/aurik.desktop ~/.local/bin/aurik

# Windows
# Ordner Aurik_10.0.20 löschen
# %USERPROFILE%\.aurik löschen
```

---

## Fehlerbehebung

| Problem | Lösung |
|---------|--------|
| `portaudio not found` | Linux: `sudo apt install libportaudio2` |
| `No module named 'PyQt5'` | `pip install PyQt5>=5.15.9` |
| `soundfile: OSError` | Linux: `sudo apt install libsndfile1` |
| GPU nicht erkannt | CPU-Fallback automatisch aktiv |
| Modelle fehlen | `bash scripts/install_aurik.sh --models-src PFAD` oder Model-Downloader |
| Nicht unterstütztes OS | Ubuntu 22.04/24.04 LTS oder Zorin OS 17/18 verwenden (Debian-Derivate laufen mit Warnung) |
