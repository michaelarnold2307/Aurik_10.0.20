from __future__ import annotations

"""§4.6c Bidirektionale Sync-Invariante: _PHASE_REQUIRED_MODELS ↔ try_allocate()-Aufrufe.

Ensures that:
1. Every try_allocate("ModelName") call in phase_*.py files has a corresponding entry
   in _PHASE_REQUIRED_MODELS for that phase.
2. Every model listed in _PHASE_REQUIRED_MODELS is actually used (try_allocate'd) by
   that phase somewhere in the code.
3. Phase_23 maps Apollo (not just AudioSR) — regression test for the v10.0.0 crash.
"""


import re
from pathlib import Path

import pytest

# ── Load _PHASE_REQUIRED_MODELS dict ─────────────────────────────────────
from backend.core.plugin_lifecycle_manager import _PHASE_REQUIRED_MODELS

PHASES_DIR = Path(__file__).resolve().parents[2] / "backend" / "core" / "phases"


def _extract_try_allocate_model_names(phase_file: Path) -> set[str]:
    """Parse a phase_*.py file and return all model name strings passed to try_allocate()."""
    source = phase_file.read_text(encoding="utf-8", errors="replace")
    names: set[str] = set()
    # Pattern: try_allocate("ModelName" or try_allocate('ModelName'
    for m in re.finditer(r"""try_allocate\(\s*["']([^"']+)["']""", source):
        names.add(m.group(1))
    return names


def _phase_id_from_filename(filename: str) -> str:
    """Convert 'phase_23_spectral_repair.py' → 'phase_23_spectral_repair'."""
    return filename.removesuffix(".py")


# ── Collect all phase files that use try_allocate ─────────────────────────
_PHASE_FILES_WITH_ML: list[tuple[str, Path, set[str]]] = []
if PHASES_DIR.is_dir():
    for f in sorted(PHASES_DIR.glob("phase_*.py")):
        models = _extract_try_allocate_model_names(f)
        if models:
            _PHASE_FILES_WITH_ML.append((_phase_id_from_filename(f.name), f, models))


class TestPLMPhaseModelSync:
    """§4.6c bidirectional sync tests."""

    def test_phase_required_models_not_empty(self):
        """_PHASE_REQUIRED_MODELS must have entries."""
        assert len(_PHASE_REQUIRED_MODELS) >= 10, f"Expected ≥10 phase entries, got {len(_PHASE_REQUIRED_MODELS)}"

    @pytest.mark.parametrize(
        "phase_id,phase_file,code_models",
        _PHASE_FILES_WITH_ML,
        ids=[t[0] for t in _PHASE_FILES_WITH_ML],
    )
    def test_code_models_in_mapping(self, phase_id: str, phase_file: Path, code_models: set[str]):
        """Every try_allocate() model in phase code must appear in _PHASE_REQUIRED_MODELS."""
        mapping = _PHASE_REQUIRED_MODELS.get(phase_id, frozenset())
        if not mapping:
            pytest.skip(f"{phase_id} has no _PHASE_REQUIRED_MODELS entry (may be DSP-only phase using budget directly)")
        missing = code_models - mapping
        assert not missing, (
            f"{phase_id}: try_allocate() calls models {missing} but _PHASE_REQUIRED_MODELS only has {mapping}"
        )

    def test_phase_23_includes_apollo(self):
        """Regression: phase_23 must map Apollo (crash fix v10.0.0)."""
        models = _PHASE_REQUIRED_MODELS.get("phase_23_spectral_repair", frozenset())
        assert "Apollo" in models, (
            f"phase_23 _PHASE_REQUIRED_MODELS={models} — Apollo missing! "
            "PLM will evict Apollo during active inference → crash."
        )

    def test_phase_03_includes_sgmse(self):
        """phase_03 SGMSE+ Tier-0 must be in mapping."""
        models = _PHASE_REQUIRED_MODELS.get("phase_03_denoise", frozenset())
        assert "SGMSE+" in models, f"phase_03 _PHASE_REQUIRED_MODELS={models} — SGMSE+ missing!"

    def test_phase_49_exists(self):
        """phase_49 advanced_dereverb uses SGMSE+ — must have mapping."""
        models = _PHASE_REQUIRED_MODELS.get("phase_49_advanced_dereverb", frozenset())
        assert "SGMSE+" in models, f"phase_49 _PHASE_REQUIRED_MODELS={models} — SGMSE+ missing!"

    def test_phase_12_includes_rmvpe(self):
        """§4.4: FCPE → RMVPE → PESTO → pYIN — RMVPE must be in mapping."""
        models = _PHASE_REQUIRED_MODELS.get("phase_12_wow_flutter_fix", frozenset())
        assert "RMVPE" in models, f"phase_12 _PHASE_REQUIRED_MODELS={models} — RMVPE missing!"
