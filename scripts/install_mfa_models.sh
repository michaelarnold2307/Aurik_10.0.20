#!/bin/bash
# Install MFA Models for Five Languages
# =======================================
#
# This script installs Montreal Forced Aligner models for
# English, German, French, Spanish, and Italian language support.
#
# Author: AURIK Development Team
# Date: 11. Februar 2026

set -e

echo "========================================"
echo "AURIK MFA Model Installation"
echo "========================================"
echo ""

# Check if MFA is installed
if ! command -v mfa &> /dev/null; then
    echo "❌ Montreal Forced Aligner (MFA) not found!"
    echo ""
    echo "Please install MFA first:"
    echo "  conda install -c conda-forge montreal-forced-aligner"
    echo ""
    exit 1
fi

echo "✅ MFA is installed"
echo ""

# Install English models
echo "📦 Installing English (US) models..."
echo "   Dictionary: english_us_arpa"
echo "   Acoustic:   english_us_arpa"
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
echo "✅ English models installed"
echo ""

# Install German models
echo "📦 Installing German models..."
echo "   Dictionary: german_mfa"
echo "   Acoustic:   german_mfa"
mfa model download dictionary german_mfa
mfa model download acoustic german_mfa
echo "✅ German models installed"
echo ""

# Install French models
echo "📦 Installing French models..."
echo "   Dictionary: french_mfa"
echo "   Acoustic:   french_mfa"
mfa model download dictionary french_mfa
mfa model download acoustic french_mfa
echo "✅ French models installed"
echo ""

# Install Spanish models
echo "📦 Installing Spanish models..."
echo "   Dictionary: spanish_mfa"
echo "   Acoustic:   spanish_mfa"
mfa model download dictionary spanish_mfa
mfa model download acoustic spanish_mfa
echo "✅ Spanish models installed"
echo ""

# Install Italian models
echo "📦 Installing Italian models..."
echo "   Dictionary: italian_mfa"
echo "   Acoustic:   italian_mfa"
mfa model download dictionary italian_mfa
mfa model download acoustic italian_mfa
echo "✅ Italian models installed"
echo ""

# List installed models
echo "========================================"
echo "Installed MFA Models:"
echo "========================================"
echo ""
echo "Dictionaries:"
mfa model list dictionary | grep -E "(english_us_arpa|german_mfa|french_mfa|spanish_mfa|italian_mfa)" || echo "  (none found)"
echo ""
echo "Acoustic Models:"
mfa model list acoustic | grep -E "(english_us_arpa|german_mfa|french_mfa|spanish_mfa|italian_mfa)" || echo "  (none found)"
echo ""

echo "========================================"
echo "✅ Installation complete!"
echo "========================================"
echo ""
echo "You can now use AURIK's Lyrics-Guided Vocal Enhancement with:"
echo "  🇬🇧 English (en)"
echo "  🇩🇪 German (de)"
echo "  🇫🇷 French (fr)"
echo "  🇪🇸 Spanish (es)"
echo "  🇮🇹 Italian (it)"
echo ""
echo "Example:"
echo "  from backend.lyrics_guided import create_integrated_vocal_timeline"
echo "  timeline = create_integrated_vocal_timeline("
echo "      audio, sr=48000, language='it'  # Italian"
echo "  )"
echo ""
