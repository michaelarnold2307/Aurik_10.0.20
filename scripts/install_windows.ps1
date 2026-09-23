# Aurik 10 — Windows Installation Script
# Run: powershell -ExecutionPolicy Bypass -File scripts/install_windows.ps1

Write-Host "=== Aurik 10 — Windows Installation ===" -ForegroundColor Cyan
Write-Host ""

# Python check
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python nicht gefunden. Installiere exakt Python 3.10.12 von https://www.python.org/downloads/release/python-31012/" -ForegroundColor Red
    Write-Host "Wähle: Add Python to PATH"
    exit 1
}

# Baseline-Pin (Windows 10/11 x64): exakt Python 3.10.12 — identisch mit dem
# Linux-/CI-Pin (ci-cross-platform.yml). Andere Patch-Versionen können
# numba/librosa/onnxruntime-Kombinationen brechen (§15.4 Produktions-Pins).
$pyver = python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "Python $pyver"

if ($pyver.Trim() -ne "3.10.12") {
    Write-Host "Aurik benötigt exakt Python 3.10.12 (gefunden: $pyver)." -ForegroundColor Red
    Write-Host "Lade https://www.python.org/downloads/release/python-31012/ und wähle 'Add Python to PATH'." -ForegroundColor Yellow
    exit 1
}

# x64-Pflicht: Auriks Speicher-/Rechenbedarf setzt 64-Bit-Python voraus.
# 32-Bit (x86) wird nicht unterstützt.
python -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Aurik benötigt 64-Bit-Python (x64). 32-Bit wird nicht unterstützt." -ForegroundColor Red
    Write-Host "Lade den 'Windows installer (64-bit)' von python.org." -ForegroundColor Yellow
    exit 1
}
Write-Host "64-Bit-Interpreter bestätigt"

# Virtual environment
if (-not (Test-Path ".venv_aurik")) {
    Write-Host "Erstelle Virtual Environment..."
    python -m venv .venv_aurik
}
.\.venv_aurik\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installiere Abhängigkeiten..."
python -m pip install --upgrade pip
pip install -r requirements/requirements_aurik.txt

# GPU detection
Write-Host "GPU-Erkennung..."
$cuda = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($cuda) {
    Write-Host "  CUDA (NVIDIA) gefunden"
    pip install onnxruntime-gpu 2>$null
}

# PortAudio (bundled with sounddevice on Windows via pip)
pip install sounddevice 2>$null

Write-Host ""
Write-Host "=== Installation abgeschlossen ===" -ForegroundColor Green
Write-Host "Start: .\.venv_aurik\Scripts\Activate.ps1 && python Aurik10/main.py"
