from __future__ import annotations

"""Normative CI gate for §2.36 LyricsGuidedEnhancement invariants."""


from pathlib import Path

import pytest

from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_lge_singleton_and_public_api_available() -> None:
    """§2.36 must expose a singleton with the required public API."""
    lge = get_lyrics_guided_enhancement()

    assert lge is not None
    assert callable(getattr(lge, "enhance", None))
    assert callable(getattr(lge, "get_timeline", None))


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_lge_core_references_mandatory_onnx_models() -> None:
    """Core module must contain the mandatory Whisper + wav2vec2 model names."""
    content = Path("backend/core/lyrics_guided_enhancement.py").read_text(encoding="utf-8")

    assert "whisper_tiny.onnx" in content
    assert "wav2vec2_forced_alignment.onnx" in content
    assert "CPUExecutionProvider" in content


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_uv3_integrates_lge_in_restore_pipeline() -> None:
    """UnifiedRestorerV3 must integrate §2.36 in the restoration pipeline."""
    content = Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")

    assert "§2.36" in content
    assert "get_lyrics_guided_enhancement" in content


@pytest.mark.normative
@pytest.mark.timeout(10)
def test_bridge_exports_lge_function() -> None:
    """Bridge must expose the §2.36 access function for UI and batch usage."""
    content = Path("backend/api/bridge.py").read_text(encoding="utf-8")

    assert "def get_lyrics_guided_enhancement_fn" in content
    assert "from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement" in content
