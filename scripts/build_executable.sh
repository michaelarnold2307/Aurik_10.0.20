#!/bin/bash
#
# AURIK Professional - Build Executable Script
# Creates standalone executable with PyInstaller
#

set -e  # Exit on error

echo "========================================="
echo "   AURIK Professional Build Script"
echo "========================================="
echo ""

# Activate virtual environment
if [ -d ".venv_aurik" ]; then
    echo "✓ Activating virtual environment..."
    source .venv_aurik/bin/activate
else
    echo "✗ Error: Virtual environment not found!"
    echo "  Please run this from the Aurik_Standalone directory"
    exit 1
fi

# Clean previous builds
echo "✓ Cleaning previous builds..."
rm -rf build/ dist/

# Check PyInstaller
if ! command -v pyinstaller &> /dev/null; then
    echo "✗ PyInstaller not found! Installing..."
    pip install pyinstaller
fi

# Build executable
echo "✓ Building executable..."
echo "  This may take several minutes..."
echo ""
pyinstaller aurik_professional.spec

# Check result
if [ -d "dist/AURIK_Professional" ]; then
    echo ""
    echo "========================================="
    echo "   ✓ Build Successful!"
    echo "========================================="
    echo ""
    echo "Executable location:"
    echo "  dist/AURIK_Professional/AURIK_Professional"
    echo ""
    echo "To run:"
    echo "  ./dist/AURIK_Professional/AURIK_Professional"
    echo ""

    # Get size
    SIZE=$(du -sh dist/AURIK_Professional | cut -f1)
    echo "Package size: $SIZE"
    echo ""
else
    echo ""
    echo "✗ Build failed! Check errors above."
    exit 1
fi
