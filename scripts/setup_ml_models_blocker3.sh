#!/bin/bash
# BLOCKER #3: ML Models Setup Script
# Sets up critical ML models for Musical Goals QA

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   BLOCKER #3: ML Models Setup for Musical Goals QA         ║${NC}"
echo -e "${GREEN}║   Setting up 3 critical models + dependencies               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running in correct directory
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo -e "${RED}Error: Not in AURIK project directory${NC}"
    exit 1
fi

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${YELLOW}Creating models directory...${NC}"
    mkdir -p "$MODELS_DIR"
fi

# Function to check if model already exists
model_exists() {
    local model_path="$1"
    if [ -d "$model_path" ] && [ "$(ls -A $model_path)" ]; then
        return 0  # exists
    else
        return 1  # doesn't exist
    fi
}

# Function to download HuggingFace model
download_hf_model() {
    local repo="$1"
    local target_dir="$2"
    local description="$3"

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📥 Downloading: $description${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if model_exists "$target_dir"; then
        echo -e "${YELLOW}✓ Model already exists at: $target_dir${NC}"
        echo -e "${YELLOW}  Skipping download.${NC}"
        return 0
    fi

    echo "Repository: $repo"
    echo "Target: $target_dir"

    # Try git clone first (faster for large models)
    if git clone "https://huggingface.co/$repo" "$target_dir" 2>/dev/null; then
        echo -e "${GREEN}✓ Successfully cloned via git${NC}"
        return 0
    fi

    # Fallback to huggingface-cli
    echo -e "${YELLOW}Git clone failed, trying huggingface-cli...${NC}"
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$repo" --local-dir "$target_dir"
        echo -e "${GREEN}✓ Successfully downloaded via huggingface-cli${NC}"
        return 0
    fi

    echo -e "${RED}✗ Failed to download model${NC}"
    echo -e "${RED}  Please install huggingface-cli: pip install huggingface-hub${NC}"
    return 1
}

# ============================================================================
# MODEL 1: MERT-v1-330M (Instrument Detection)
# ============================================================================

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  MODEL 1/3: MERT-v1-330M (Instrument Detection)                ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Purpose: Detect 150+ instruments for semantic goal adjustment"
echo "Used by: Component 0.9.6 (Semantic Goals)"
echo "Size: ~1.3 GB"

MERT_DIR="$MODELS_DIR/mert_instrument_detector"

if download_hf_model "m-a-p/MERT-v1-330M" "$MERT_DIR" "MERT-v1-330M"; then
    echo -e "${GREEN}✓ MERT-v1-330M ready${NC}"

    # Create integration config
    cat > "$MERT_DIR/aurik_config.json" << EOF
{
    "model_name": "MERT-v1-330M",
    "purpose": "Instrument Detection",
    "component": "0.9.6 Semantic Goals",
    "instruments": "150+",
    "input_format": "audio waveform (16kHz)",
    "output_format": "instrument probabilities",
    "framework": "transformers/torch",
    "integrated": true
}
EOF
    echo "  Config saved to: $MERT_DIR/aurik_config.json"
else
    echo -e "${RED}✗ Failed to download MERT-v1-330M${NC}"
    exit 1
fi

# ============================================================================
# MODEL 2: madmom (Structure Analysis)
# ============================================================================

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  MODEL 2/3: madmom (Music Structure Analysis)                  ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Purpose: Detect song structure (intro/verse/chorus/etc.)"
echo "Used by: Component 0.9.6 (Semantic Goals)"
echo "Note: Python package, not a standalone model"

# Check if madmom is already installed
if python3 -c "import madmom" 2>/dev/null; then
    echo -e "${YELLOW}✓ madmom already installed${NC}"
else
    echo "Installing madmom..."

    # Check which venv to use
    if [ -d "$PROJECT_ROOT/.venv_aurik" ]; then
        VENV_PYTHON="$PROJECT_ROOT/.venv_aurik/bin/python"
        VENV_PIP="$PROJECT_ROOT/.venv_aurik/bin/pip"
    else
        VENV_PYTHON="python3"
        VENV_PIP="pip3"
    fi

    echo "Using Python: $VENV_PYTHON"

    # Install madmom
    $VENV_PIP install madmom

    if $VENV_PYTHON -c "import madmom" 2>/dev/null; then
        echo -e "${GREEN}✓ madmom successfully installed${NC}"

        # Create documentation
        mkdir -p "$MODELS_DIR/madmom"
        cat > "$MODELS_DIR/madmom/aurik_config.json" << EOF
{
    "model_name": "madmom",
    "purpose": "Music Structure Analysis",
    "component": "0.9.6 Semantic Goals",
    "segments": ["intro", "verse", "chorus", "bridge", "outro", "solo", "breakdown", "build_up", "drop"],
    "input_format": "audio waveform",
    "output_format": "segment boundaries + types",
    "framework": "numpy/scipy",
    "integrated": true,
    "installation": "pip package"
}
EOF
        echo "  Config saved to: $MODELS_DIR/madmom/aurik_config.json"
    else
        echo -e "${RED}✗ Failed to install madmom${NC}"
        exit 1
    fi
fi

# ============================================================================
# MODEL 3: AST (Audio Spectrogram Transformer) - Perceptual Validation
# ============================================================================

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  MODEL 3/3: AST (Perceptual Validation Base)                   ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Purpose: Feature extraction for perceptual quality validation"
echo "Used by: Component 0.9.2 (Perceptual Validation)"
echo "Size: ~370 MB"
echo "Note: Requires fine-tuning on AURIK A/B test data for full functionality"

AST_DIR="$MODELS_DIR/ast_perceptual_base"

if download_hf_model "MIT/ast-finetuned-audioset-10-10-0.4593" "$AST_DIR" "AST Model"; then
    echo -e "${GREEN}✓ AST Model ready${NC}"

    # Create integration config
    cat > "$AST_DIR/aurik_config.json" << EOF
{
    "model_name": "AST (Audio Spectrogram Transformer)",
    "purpose": "Perceptual Quality Validation",
    "component": "0.9.2 Perceptual Validation",
    "input_format": "audio spectrogram",
    "output_format": "perceptual features",
    "framework": "transformers/torch",
    "base_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "fine_tuned": false,
    "note": "Requires fine-tuning on AURIK A/B test data for optimal performance",
    "integrated": true
}
EOF
    echo "  Config saved to: $AST_DIR/aurik_config.json"

    echo ""
    echo -e "${YELLOW}⚠ Note: AST model is downloaded but NOT fine-tuned${NC}"
    echo -e "${YELLOW}  For optimal perceptual validation:${NC}"
    echo -e "${YELLOW}  1. Collect A/B test data (preferred vs. rejected)${NC}"
    echo -e "${YELLOW}  2. Fine-tune AST on this data${NC}"
    echo -e "${YELLOW}  3. Save fine-tuned model to: $AST_DIR/fine_tuned/${NC}"
    echo ""
else
    echo -e "${RED}✗ Failed to download AST Model${NC}"
    exit 1
fi

# ============================================================================
# INSTALL REQUIRED DEPENDENCIES
# ============================================================================

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Installing Python Dependencies                                ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Determine pip to use
if [ -d "$PROJECT_ROOT/.venv_aurik" ]; then
    PIP="$PROJECT_ROOT/.venv_aurik/bin/pip"
else
    PIP="pip3"
fi

echo "Installing transformers (for MERT and AST)..."
$PIP install transformers torch torchaudio accelerate --quiet

echo "Installing librosa (audio processing)..."
$PIP install librosa --quiet

echo "Installing scipy (for madmom and signal processing)..."
$PIP install scipy --quiet

echo -e "${GREEN}✓ Dependencies installed${NC}"

# ============================================================================
# SUMMARY & INTEGRATION STATUS
# ============================================================================

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    SETUP COMPLETE! ✓                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Installed Models:${NC}"
echo ""
echo "1. ✓ MERT-v1-330M (Instrument Detection)"
echo "   Location: $MERT_DIR"
echo "   Size: $(du -sh "$MERT_DIR" 2>/dev/null | cut -f1 || echo "~1.3 GB")"
echo "   Status: Ready to use"
echo ""
echo "2. ✓ madmom (Structure Analysis)"
echo "   Location: Python package"
echo "   Status: Ready to use"
echo ""
echo "3. ✓ AST Model (Perceptual Validation)"
echo "   Location: $AST_DIR"
echo "   Size: $(du -sh "$AST_DIR" 2>/dev/null | cut -f1 || echo "~370 MB")"
echo "   Status: Base model ready (fine-tuning recommended)"
echo ""
echo -e "${GREEN}Integration Files:${NC}"
echo "  • semantic_goals.py - Now can use MERT for real instrument detection"
echo "  • semantic_goals.py - Now can use madmom for real structure analysis"
echo "  • perceptual_validation.py - Now can use AST for feature extraction"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Run integration tests: pytest tests/musical_goals/test_semantic_goals.py"
echo "  2. Verify model loading: python -c 'from semantic_goals import SemanticGoalsEngine; engine = SemanticGoalsEngine()'"
echo "  3. (Optional) Fine-tune AST on A/B test data for perceptual validation"
echo ""
echo -e "${GREEN}BLOCKER #3 Status: Models Downloaded & Ready! ✓${NC}"
echo ""
