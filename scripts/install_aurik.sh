#!/usr/bin/env bash
# =============================================================================
# Aurik 10.1.0 — Komplettes Installationsprogramm für Linux
# Zielsysteme: Ubuntu 22.04 LTS / 24.04 LTS · Zorin OS 17 / 18
# =============================================================================
# Verwendung:
#   bash scripts/install_aurik.sh                    # Komplette Installation (CPU)
#   bash scripts/install_aurik.sh --cuda             # PyTorch mit CUDA (GPU)
#   bash scripts/install_aurik.sh --rocm             # PyTorch mit ROCm (AMD-GPU)
#   bash scripts/install_aurik.sh --models-src PFAD  # Modell-Bundle aus PFAD importieren
#   bash scripts/install_aurik.sh --system           # Launcher systemweit (/usr/local, sudo)
#   bash scripts/install_aurik.sh --smoke            # Nach Installation Import-Smoke-Test
#   bash scripts/install_aurik.sh --venv PFAD        # Eigenes venv-Verzeichnis
#   bash scripts/install_aurik.sh --no-venv          # Aktuelles Python verwenden
#
# Voraussetzungen: Python 3.10–3.12, apt-basierte Distribution.
# Die ML-Modelle (models/) sind NICHT im Repo gebündelt (§13.3): Sie werden
# aus einem lokalen Modell-Bundle übernommen (--models-src) oder müssen
# manuell nach models/ gelegt werden — der Installer prüft dies und meldet
# fehlende Modelle klar und unübersehbar.
# =============================================================================

set -euo pipefail

# --- Farben für Ausgabe ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[Aurik]${NC} $*"; }
success() { echo -e "${GREEN}[Aurik]${NC} ✓ $*"; }
warn()    { echo -e "${YELLOW}[Aurik]${NC} ! $*"; }
error()   { echo -e "${RED}[Aurik]${NC} FEHLER: $*"; exit 1; }

# --- Standardwerte ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv_aurik"
USE_EXISTING_VENV=false
PYTHON_BIN=""
TORCH_FLAVOR="cpu"
MODELS_SRC=""
SYSTEM_INSTALL=false
RUN_SMOKE=false
APT_DEPS=(python3 python3-pip python3-venv python3-tk python3-dev git curl ffmpeg libsndfile1 libportaudio2 portaudio19-dev)

# --- Argumente parsen ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)       VENV_DIR="$2"; shift 2 ;;
        --no-venv)    USE_EXISTING_VENV=true; shift ;;
        --cuda)       TORCH_FLAVOR="cuda"; shift ;;
        --rocm)       TORCH_FLAVOR="rocm"; shift ;;
        --models-src) MODELS_SRC="$2"; shift 2 ;;
        --system)     SYSTEM_INSTALL=true; shift ;;
        --smoke)      RUN_SMOKE=true; shift ;;
        -h|--help)
            echo "Verwendung: $0 [Optionen]"
            echo "  --venv PFAD        Venv in angegebenem Pfad (Standard: .venv_aurik)"
            echo "  --no-venv          Aktuell aktives Python verwenden"
            echo "  --cuda | --rocm    PyTorch-GPU-Variante (Standard: CPU)"
            echo "  --models-src PFAD  Lokales Modell-Bundle nach models/ importieren"
            echo "  --system           Launcher systemweit installieren (sudo)"
            echo "  --smoke            Import-Smoke-Test nach der Installation"
            exit 0 ;;
        *) warn "Unbekanntes Argument: $1 (ignoriert)"; shift ;;
    esac
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "   Aurik 10.1.0 — Komplettes Installationsprogramm (Linux)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# =============================================================================
# SCHRITT 0: Betriebssystem prüfen (Ubuntu 22.04/24.04 · Zorin OS 17/18)
# =============================================================================
info "Schritt 0/7: Betriebssystem prüfen…"
OS_ID=""; OS_VER=""; OS_PRETTY=""
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS_ID="${ID:-}"; OS_VER="${VERSION_ID:-}"; OS_PRETTY="${PRETTY_NAME:-}"
fi
OS_OK=false
case "$OS_ID" in
    ubuntu) [[ "$OS_VER" == "22.04" || "$OS_VER" == "24.04" ]] && OS_OK=true ;;
    zorin)  [[ "$OS_VER" == 17* || "$OS_VER" == 18* ]] && OS_OK=true ;;
esac
if [[ "$OS_OK" == "false" ]]; then
    if [[ "${ID_LIKE:-}" == *ubuntu* || "${ID_LIKE:-}" == *debian* ]]; then
        warn "Nicht offiziell freigegebene, aber Ubuntu/Debian-basierte Distribution: ${OS_PRETTY:-$OS_ID $OS_VER}"
        warn "Die Installation wird versucht — Support ist für Ubuntu 22.04/24.04 und Zorin OS 17/18 garantiert."
    else
        error "Nicht unterstütztes Betriebssystem: ${OS_PRETTY:-unbekannt}. Benötigt: Ubuntu 22.04/24.04 oder Zorin OS 17/18."
    fi
else
    success "Betriebssystem unterstützt: ${OS_PRETTY:-$OS_ID $OS_VER}"
fi

# =============================================================================
# SCHRITT 1: System-Abhängigkeiten installieren (apt, mit sudo)
# =============================================================================
info "Schritt 1/7: System-Abhängigkeiten prüfen/installieren…"
MISSING_APT=()
for pkg in "${APT_DEPS[@]}"; do
    if dpkg -s "$pkg" &>/dev/null; then
        :
    elif [[ "$pkg" == "ffmpeg" || "$pkg" == "curl" || "$pkg" == "git" ]] && command -v "${pkg}" &>/dev/null; then
        :
    else
        MISSING_APT+=("$pkg")
    fi
done
if [[ "${#MISSING_APT[@]}" -gt 0 ]]; then
    warn "Fehlende apt-Pakete: ${MISSING_APT[*]}"
    info "Installiere mit sudo (Passwort-Abfrage möglich)…"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING_APT[@]}"
    success "System-Abhängigkeiten installiert"
else
    success "Alle System-Abhängigkeiten vorhanden"
fi

# =============================================================================
# SCHRITT 2: Python 3.10–3.12 finden
# =============================================================================
info "Schritt 2/7: Python-Interpreter suchen…"
if $USE_EXISTING_VENV; then
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
else
    for py in python3.12 python3.11 python3.10 python3; do
        if command -v "$py" &>/dev/null; then
            VER="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
            case "$VER" in
                3.10|3.11|3.12) PYTHON_BIN="$(command -v "$py")"; break ;;
            esac
        fi
    done
fi
[[ -z "$PYTHON_BIN" ]] && error "Python 3.10–3.12 nicht gefunden. Bitte installieren (sudo apt-get install -y python3.12 python3.12-venv)."
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
success "Python $PY_VERSION gefunden: $PYTHON_BIN"

# =============================================================================
# SCHRITT 3: Virtuelle Umgebung
# =============================================================================
if ! $USE_EXISTING_VENV; then
    info "Schritt 3/7: Virtuelle Umgebung anlegen: $VENV_DIR"
    if [[ -d "$VENV_DIR/bin" ]]; then
        warn "venv existiert bereits — wird wiederverwendet"
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        success "Virtuelle Umgebung erstellt"
    fi
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    info "Schritt 3/7: Vorhandenes Python verwenden (--no-venv)"
fi
"$PYTHON_BIN" -m pip install --upgrade pip --quiet
success "pip aktuell"

# =============================================================================
# SCHRITT 4: PyTorch-Variante (§13.4 — CPU-Standard, GPU optional)
# =============================================================================
info "Schritt 4/7: PyTorch installieren (Variante: $TORCH_FLAVOR)…"
TORCH_INSTALLED="$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")"
case "$TORCH_FLAVOR" in
    cpu)
        TORCH_PIN="torch==2.7.0+cpu"
        TORCH_AUDIO_PIN="torchaudio==2.7.0+cpu"
        EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"
        ;;
    cuda)
        TORCH_PIN="torch==2.7.0+cu118"
        TORCH_AUDIO_PIN="torchaudio==2.7.0+cu118"
        EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cu118"
        ;;
    rocm)
        TORCH_PIN="torch==2.1.0+rocm5.7"
        TORCH_AUDIO_PIN="torchaudio==2.1.0+rocm5.7"
        EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/rocm5.7"
        ;;
esac
# Sicherheitsnetz (§13.4): torch>=2.6.0 behebt CVE-2025-32434 (kritisch).
if [[ "$TORCH_FLAVOR" == "rocm" ]]; then
    warn "ROCm-Pin 2.1.0 liegt unter dem Sicherheitsnetz 2.6.0 (CVE-2025-32434) — nur für AMD-GPUs mit ROCm 5.7 verwenden."
fi
if [[ "$TORCH_INSTALLED" == "${TORCH_PIN#torch==}" ]]; then
    success "PyTorch ${TORCH_PIN#torch==} bereits installiert"
else
    [[ -n "$TORCH_INSTALLED" ]] && warn "Andere torch-Version gefunden ($TORCH_INSTALLED) → wird ersetzt"
    # shellcheck disable=SC2086
    "$PYTHON_BIN" -m pip install "$TORCH_PIN" "$TORCH_AUDIO_PIN" $EXTRA_INDEX --quiet
    success "PyTorch installiert ($TORCH_PIN)"
fi

# =============================================================================
# SCHRITT 5: Aurik-Abhängigkeiten
# =============================================================================
info "Schritt 5/7: Aurik-Abhängigkeiten installieren…"
"$PYTHON_BIN" -m pip install -r "$PROJECT_DIR/requirements/requirements_aurik.txt" --quiet
success "Alle Aurik-Abhängigkeiten installiert"

# =============================================================================
# SCHRITT 6: ML-Modelle prüfen/importieren (§13.3)
# =============================================================================
info "Schritt 6/7: ML-Modelle prüfen…"
if [[ -n "$MODELS_SRC" && -d "$MODELS_SRC" ]]; then
    info "Importiere Modell-Bundle aus: $MODELS_SRC"
    mkdir -p "$PROJECT_DIR/models"
    cp -rn "$MODELS_SRC"/. "$PROJECT_DIR/models/"
    success "Modell-Bundle übernommen"
fi
MANIFEST="$PROJECT_DIR/models/manifest.json"
if [[ -f "$MANIFEST" ]]; then
    MODEL_REPORT="$("$PYTHON_BIN" - "$MANIFEST" "$PROJECT_DIR" <<'PYEOF'
import json, pathlib, sys
manifest_path, project = sys.argv[1], sys.argv[2]
manifest = json.loads(pathlib.Path(manifest_path).read_text())
entries = manifest.get("models", [])
present, missing = [], []
for m in entries:
    p = m.get("bundled_path") or m.get("path")
    if not p:
        continue
    full = pathlib.Path(project) / p
    (present if full.exists() else missing).append(m.get("name", p))
total = len(present) + len(missing)
print(f"{len(present)}/{total}")
if missing:
    print("FEHLEND:")
    for name in missing:
        print(f"  - {name}")
PYEOF
)"
    MODEL_COUNT="${MODEL_REPORT%%$'\n'*}"
    success "Lokale Modelle vorhanden: $MODEL_COUNT"
    if [[ "$MODEL_REPORT" == *"FEHLEND:"* ]]; then
        echo "$MODEL_REPORT" | tail -n +2
        warn "Es fehlen Modelle. Aurik startet trotzdem, fällt aber auf DSP zurück (§V6)."
        warn "Modelle aus einem lokalen Bundle übernehmen:"
        warn "  bash scripts/install_aurik.sh --models-src /pfad/zum/modell-bundle"
        warn "Oder das models/-Verzeichnis der Distribution manuell hierher kopieren."
    fi
else
    warn "models/manifest.json nicht gefunden — Modell-Prüfung übersprungen."
fi

# =============================================================================
# SCHRITT 7: Launcher installieren (Desktop + CLI)
# =============================================================================
info "Schritt 7/7: Launcher installieren…"
DESKTOP_CONTENT="[Desktop Entry]
Type=Application
Name=Aurik 10.1.0
GenericName=Musik-Restaurierung
Comment=Intelligente Musik- und Gesangs-Restaurierung (Weltklasse-Wohlklang)
Exec=$PROJECT_DIR/run_aurik.sh
Path=$PROJECT_DIR
Icon=audio-x-generic
Terminal=false
Categories=AudioVideo;Audio;"
if $SYSTEM_INSTALL; then
    echo "$DESKTOP_CONTENT" | sudo tee /usr/local/share/applications/aurik.desktop >/dev/null
    sudo ln -sf "$PROJECT_DIR/run_aurik.sh" /usr/local/bin/aurik
    success "Launcher systemweit installiert (/usr/local)"
else
    mkdir -p "$HOME/.local/share/applications" "$HOME/.local/bin"
    echo "$DESKTOP_CONTENT" > "$HOME/.local/share/applications/aurik.desktop"
    ln -sf "$PROJECT_DIR/run_aurik.sh" "$HOME/.local/bin/aurik"
    success "Launcher im Benutzerbereich installiert (~/.local)"
fi

# =============================================================================
# ABSCHLUSS
# =============================================================================
if $RUN_SMOKE; then
    info "Import-Smoke-Test…"
    "$PYTHON_BIN" -c "import numpy, soundfile; from backend.core.version import __version__; print('Aurik', __version__, 'Import OK')" \
        || warn "Smoke-Test fehlgeschlagen — Installation prüfen (siehe oben)."
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "            Installation abgeschlossen — Aurik 10.1.0"
echo "════════════════════════════════════════════════════════════════"
echo ""
success "Aurik ist einsatzbereit!"
echo ""
info "Starten:"
echo "  aurik                      # CLI-Wrapper (falls Launcher installiert)"
echo "  ./run_aurik.sh             # direkt aus dem Projektverzeichnis"
echo ""
info "Tests:"
echo "  ./run_tests_safe.sh tests/unit --maxfail=5 --tb=short"
echo ""
