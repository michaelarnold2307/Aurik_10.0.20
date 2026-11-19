#!/usr/bin/env bash
# =============================================================================
# Aurik – Ein-Klick-Installation (Linux / macOS)
# Aurik – One-Click Installer (Linux / macOS)
# =============================================================================
# Was dieses Skript macht / What this script does:
#   1. Prüft Python 3.10+ / Checks for Python 3.10+
#   2. Erstellt eine virtuelle Umgebung / Creates a virtual environment
#   3. Installiert PyQt5 und alle Abhängigkeiten / Installs PyQt5 + dependencies
#   4. Erstellt einen Desktop-Eintrag / Creates a desktop menu entry
#
# Verwendung / Usage:
#   bash install_aurik.sh
#
# Du brauchst KEINE Terminal-Erfahrung – einfach ausführen!
# You do NOT need terminal experience – just run it!
# =============================================================================

set -euo pipefail

# --- Farbeinstellungen (falls Terminal Farben unterstützt) ---
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; NC=''
fi

info()    { echo -e "${BLUE}[Aurik]${NC} $*"; }
success() { echo -e "${GREEN}[Aurik]${NC} ✅ $*"; }
warn()    { echo -e "${YELLOW}[Aurik]${NC} ⚠️  $*"; }
err()     { echo -e "${RED}[Aurik]${NC} ❌ $*"; }
step()    { echo -e "\n${CYAN}${BOLD}▶ $*${NC}"; }

# --- Projektverzeichnis ermitteln ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv_aurik"

# =============================================================================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     ✨ Aurik – Intelligente Musik-Restaurierung  ✨          ║${NC}"
echo -e "${BOLD}║     ✨ Aurik – Intelligent Music Restoration     ✨          ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Willkommen zum Aurik-Installer! / Welcome to the Aurik installer!"
info "Dieser Assistent richtet alles Nötige für dich ein."
info "This wizard sets up everything you need."
echo ""

# =============================================================================
# SCHRITT 1: Python 3.10+ finden
# =============================================================================
step "Schritt 1/5: Python 3.10+ suchen... / Step 1/5: Finding Python 3.10+..."

PYTHON_BIN=""
for py in python3.13 python3.12 python3.11 python3.10 python3; do
    if cmd="$(command -v "$py" 2>/dev/null)"; then
        # Prüfe ob Version >= 3.10
        ver=$("$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null) && {
            PYTHON_BIN="$cmd"
            break
        } || true
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    err "Python 3.10 oder neuer wurde NICHT gefunden!"
    err "Python 3.10 or newer was NOT found!"
    echo ""
    echo -e "${YELLOW}👉 So installierst du Python / How to install Python:${NC}"
    echo ""
    echo "  Ubuntu / Debian:"
    echo "    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
    echo ""
    echo "  Fedora:"
    echo "    sudo dnf install -y python3 python3-pip"
    echo ""
    echo "  Arch / Manjaro:"
    echo "    sudo pacman -S python python-pip"
    echo ""
    echo "  macOS:"
    echo "    brew install python@3.12"
    echo ""
    echo -e "${YELLOW}Nach der Installation dieses Skript erneut ausführen.${NC}"
    echo -e "${YELLOW}After installing, re-run this script.${NC}"
    exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
success "Python $PY_VERSION → $PYTHON_BIN"

# =============================================================================
# SCHRITT 2: System-Abhängigkeiten prüfen (Linux-spezifisch)
# =============================================================================
step "Schritt 2/5: System-Pakete prüfen... / Step 2/5: Checking system packages..."

OS="$(uname -s)"
if [[ "$OS" == "Linux" ]]; then
    # Prüfe portaudio (für sounddevice — Mikrofon/Wiedergabe)
    if ldconfig -p 2>/dev/null | grep -q libportaudio; then
        success "libportaudio2 – OK"
    else
        warn "libportaudio2 fehlt / missing (für Mikrofon/Wiedergabe / for mic/playback)"
        if command -v apt-get &>/dev/null; then
            warn "  → sudo apt-get install -y libportaudio2 portaudio19-dev"
        elif command -v dnf &>/dev/null; then
            warn "  → sudo dnf install -y portaudio portaudio-devel"
        fi
        warn "  Aurik funktioniert auch ohne – Audio-Geräte sind dann ggf. eingeschränkt."
        warn "  Aurik works without it – audio device support may be limited."
    fi

    # Prüfe ffmpeg
    if command -v ffmpeg &>/dev/null; then
        success "ffmpeg – OK"
    else
        warn "ffmpeg fehlt / missing (für Audio-Format-Konvertierung / for audio format conversion)"
        if command -v apt-get &>/dev/null; then
            warn "  → sudo apt-get install -y ffmpeg"
        elif command -v dnf &>/dev/null; then
            warn "  → sudo dnf install -y ffmpeg"
        fi
        warn "  Aurik funktioniert auch ohne – MP3/FLAC-Import ist dann eingeschränkt."
        warn "  Aurik works without it – MP3/FLAC import may be limited."
    fi
elif [[ "$OS" == "Darwin" ]]; then
    # macOS: portaudio ist meist via Homebrew installiert
    if command -v brew &>/dev/null; then
        brew ls --versions portaudio &>/dev/null && success "portaudio (Homebrew) – OK" || {
            warn "portaudio fehlt (Homebrew). Optional: brew install portaudio"
        }
        brew ls --versions ffmpeg &>/dev/null && success "ffmpeg (Homebrew) – OK" || {
            warn "ffmpeg fehlt (Homebrew). Optional: brew install ffmpeg"
        }
    else
        warn "Homebrew nicht gefunden. Installiere es für beste Ergebnisse: https://brew.sh"
    fi
fi

# =============================================================================
# SCHRITT 3: Virtuelle Umgebung erstellen
# =============================================================================
step "Schritt 3/5: Python-Umgebung einrichten... / Step 3/5: Setting up Python environment..."

if [[ -d "$VENV_DIR" ]]; then
    warn ".venv_aurik existiert bereits – wird wiederverwendet."
    warn ".venv_aurik already exists – reusing."
else
    info "Erstelle virtuelle Umgebung / Creating virtual environment: .venv_aurik"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtuelle Umgebung erstellt / Virtual environment created"
fi

# Ensure we use the venv Python from now on
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Upgrade pip inside the venv
info "Aktualisiere pip / Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet 2>/dev/null || true
success "pip – aktuell / up to date"

# =============================================================================
# SCHRITT 4: Abhängigkeiten installieren (robust, forgive missing optional deps)
# =============================================================================
step "Schritt 4/5: Abhängigkeiten installieren... / Step 4/5: Installing dependencies..."
info "Das kann ein paar Minuten dauern. / This may take a few minutes."
info "Datei / File: requirements/requirements_aurik.txt"

# --- PyTorch CPU-only (wird vor requirements_aurik.txt installiert) ---
info "Installiere PyTorch (CPU-only)..."
"$VENV_PYTHON" -m pip install \
    torch==2.7.0+cpu \
    torchaudio==2.7.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --quiet 2>&1 | grep -v "^WARNING:" || true
# pip kann mit --quiet und redirect fehlschlagen, also Exit-Code ignorieren
_pip_rc=0
"$VENV_PYTHON" -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || _pip_rc=$?
if [[ $_pip_rc -eq 0 ]]; then
    success "PyTorch CPU-only – OK"
else
    warn "PyTorch-Installation schlug fehl. / PyTorch installation failed."
    warn "Versuche ohne extra-index... / Trying without extra-index..."
    "$VENV_PYTHON" -m pip install torch torchaudio --quiet 2>&1 | grep -v "^WARNING:" || true
    "$VENV_PYTHON" -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null && \
        success "PyTorch (Fallback) – OK" || \
        warn "PyTorch konnte nicht installiert werden. Aurik startet trotzdem, ML-Funktionen sind eingeschränkt. / PyTorch could not be installed. Aurik will still start, ML features limited."
fi

# --- Hauptabhängigkeiten ---
info "Installiere Aurik-Abhängigkeiten... / Installing Aurik dependencies..."
REQ_FILE="$SCRIPT_DIR/requirements/requirements_aurik.txt"
if [[ -f "$REQ_FILE" ]]; then
    "$VENV_PYTHON" -m pip install -r "$REQ_FILE" --quiet 2>&1 | grep -v "^WARNING:" || {
        warn "Einige Pakete konnten nicht installiert werden (optional)."
        warn "Some packages could not be installed (optional)."
        warn "Aurik funktioniert trotzdem – nicht-kritische Features sind deaktiviert."
        warn "Aurik will still work – non-critical features are disabled."
    }
    success "Aurik-Abhängigkeiten installiert / Dependencies installed"
else
    warn "requirements/requirements_aurik.txt nicht gefunden! / not found!"
    warn "Überspringe Abhängigkeiten. / Skipping dependencies."
fi

# =============================================================================
# SCHRITT 5: Desktop-Eintrag erstellen (Linux) / App-Ordner (macOS)
# =============================================================================
step "Schritt 5/5: Desktop-Verknüpfung erstellen... / Step 5/5: Creating desktop shortcut..."

if [[ "$OS" == "Linux" ]]; then
    # --- .desktop-Datei ---
    DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$DESKTOP_DIR"
    DESKTOP_FILE="$DESKTOP_DIR/aurik.desktop"

    # Symbol: Verwende ein eingebautes Audio-Symbol, oder generiere einen Pfad
    ICON_PATH="$SCRIPT_DIR/Aurik10/resources/icon.png"
    if [[ ! -f "$ICON_PATH" ]]; then
        # Fallback zu System-Icons
        ICON_PATH="audio-card"  # freedesktop icon name
    fi

    cat > "$DESKTOP_FILE" << DESKTOPEOF
[Desktop Entry]
Name=Aurik
Name[de]=Aurik
Comment=Intelligente Musik-Restaurierung
Comment[de]=Intelligente Musik-Restaurierung
Comment[en]=Intelligent Music Restoration
Exec=$SCRIPT_DIR/run_aurik.sh
Icon=$ICON_PATH
Terminal=false
Type=Application
Categories=Audio;AudioVideo;Music;
Keywords=audio;restoration;music;restaurierung;musik;
StartupNotify=true
DESKTOPEOF

    # Update desktop database (non-fatal)
    command -v update-desktop-database &>/dev/null && \
        update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

    success "Desktop-Eintrag erstellt: $DESKTOP_FILE"
    info "Finde Aurik jetzt in deinem Startmenü unter 'Audio'!"
    info "Find Aurik now in your start menu under 'Audio'!"

elif [[ "$OS" == "Darwin" ]]; then
    # macOS: Erstelle ein einfaches .command-Launcher-Skript im Projektordner
    # Ein echter .app-Bundle wäre overkill; .command ist per Doppelklick ausführbar
    LAUNCHER="$SCRIPT_DIR/Aurik.command"
    cat > "$LAUNCHER" << 'MACEOF'
#!/usr/bin/env bash
cd "$(dirname "$0")"
exec ./run_aurik.sh
MACEOF
    chmod +x "$LAUNCHER"
    success "Launcher erstellt / Launcher created: Aurik.command"
    info "Doppelklicke Aurik.command im Projektordner zum Starten."
    info "Double-click Aurik.command in the project folder to start."
fi

# =============================================================================
# ABSCHLUSS / FINISH
# =============================================================================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         ✨ Installation abgeschlossen! ✨                    ║${NC}"
echo -e "${BOLD}║         ✨ Installation complete!        ✨                    ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

success "Aurik ist bereit! / Aurik is ready!"
echo ""
info "So startest du Aurik / How to start Aurik:"
echo -e "  ${BOLD}bash run_aurik.sh${NC}     (Terminal)"
if [[ "$OS" == "Linux" ]]; then
    echo -e "  ${BOLD}Startmenü → Audio → Aurik${NC}   (Desktop)"
fi
echo ""
info "Nützliche Befehle / Useful commands:"
echo "  bash run_tests_safe.sh tests/unit    (Tests ausführen / run tests)"
echo ""
info "Optionale Pakete für erweiterte Features / Optional packages for advanced features:"
echo "  source .venv_aurik/bin/activate"
echo "  pip install madmom transformers flashsr"
echo ""
warn "Bei Problemen / If you encounter problems:"
echo "  -> docs/ oder GitHub Issues"
echo ""
