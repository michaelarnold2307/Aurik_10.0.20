"""
AURIK v8.1 - Advanced Vocal Source Separation Module

Provides SOTA-level vocal/instrumental separation with HIPS compliance.

Models:
- Demucs v5 (Hybrid Transformer-based)
- MDX-Net (Spectral-domain separation)
- Hybrid Ensemble (combines both for optimal quality)

All models are HIPS-compliant:
- Kontextbewusstsein: ✅ Transformer-based (global context)
- Nebenwirkungen: ✅ Modeled (stereo width loss, phase issues)
- Reversibilität: ✅ Stems stored separately
- Auditierbarkeit: ✅ Pre/Post spectrograms logged
- Steuerbarkeit: ✅ Configurable separation aggressiveness
"""

from .demucs_v5_wrapper import DemucsV5Separator
from .hybrid_separation import HybridVocalSeparator
from .mdx_net_wrapper import MDXNetSeparator

__all__ = ["DemucsV5Separator", "HybridVocalSeparator", "MDXNetSeparator"]

__version__ = "9.15.0"
