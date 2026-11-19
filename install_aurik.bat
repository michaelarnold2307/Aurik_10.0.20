@echo off
REM Aurik - One-Click Installer (Windows)
echo.
echo ============================================
echo   Aurik - Audio Restoration Setup
echo ============================================
echo.

REM Check Python — exakter Baseline-Pin 3.10.12 (Windows 10/11 x64)
python -c "import sys; sys.exit(0 if sys.version_info[:3]==(3,10,12) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Aurik requires exactly Python 3.10.12.
    python --version 2>nul
    echo         Download: https://www.python.org/downloads/release/python-31012/
    pause
    exit /b 1
)
echo [OK] Python 3.10.12 found

REM Check 64-bit (x64) — 32-bit Python cannot handle Aurik's memory needs
python -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Aurik requires 64-bit Python ^(x64^). 32-bit is not supported.
    echo         Install the "Windows installer (64-bit)" from python.org.
    pause
    exit /b 1
)
echo [OK] 64-bit Python confirmed

REM Create venv
echo Creating virtual environment...
python -m venv .venv_aurik
echo [OK] Virtual environment created

REM Install dependencies
echo Installing dependencies...
.venv_aurik\Scripts\pip install --upgrade pip -q
.venv_aurik\Scripts\pip install PyQt5==5.15.11 numpy==1.26.4 soundfile==0.13.1 scipy==1.15.3 psutil -q
echo [OK] Dependencies installed

REM Create Start Menu shortcut
echo Creating Start Menu shortcut...
powershell -Command "=(New-Object -COM WScript.Shell).CreateShortcut('%APPDATA%\Microsoft\Windows\Start Menu\Programs\Aurik.lnk'); .TargetPath='%~dp0run_aurik.bat'; .WorkingDirectory='%~dp0'; .Save()"
echo [OK] Shortcut created

echo.
echo Aurik installation complete! Start from the Start Menu.
echo You can also double-click run_aurik.bat in this folder.
pause
