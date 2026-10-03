"""TODO-P2 „Phase-Gap-Rest zerlegen" — Bit-Identitätstests (§G5 (copilot-instructions.md)).

Ziel (Roadmap P2): Die verbliebenen Phase-Gap-Reste (PMGG-Pre/Post via ``_measure_quick``,
u.a. Coalition-Bookkeeping via ``_fast_goal_snapshot``) nach dem R11/R12-Muster batchen/
cachen — **streng bit-identisch** zu heute.

Der Legacy-Pfad ``AURIK_P13_LEGACY=1`` (siehe ``_p13_legacy()`` in
``per_phase_musical_goals_gate.py``) erzwingt die historischen Python-Comprehension-
Rechenwege. Er ist ausschließlich für den gestuften Vergleich alt vs. neu gedacht und
wird in Produktion nie gesetzt. Der Test vergleicht beide Pfade Byte-für-Byte auf
deterministischen synthetischen Fällen (mono, stereo channels-first, mit/ohne Referenz,
mehrere Längen).

Referenz: Roadmap ``docs/TODOS_SOTA_ROADMAP.md`` — §PERF-R13, R11/R12, P2.
"""

import os

import numpy as np
import pytest

from backend.core import per_phase_musical_goals_gate as pmgg

_SR = 48000


def _load_pmgg(legacy: bool):
    """Lädt das PMGG-Modul mit gesetztem/unverändertem Legacy-Flag neu."""
    os.environ["AURIK_P13_LEGACY"] = "1" if legacy else ""
    try:
        import importlib

        importlib.reload(pmgg)
    finally:
        os.environ["AURIK_P13_LEGACY"] = ""
    return pmgg


def _cases() -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Deterministische synthetische Fälle (gestuft: Länge × Layout × Referenz)."""
    rng = np.random.default_rng(20261003)
    out: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for n in (4800, 48000, 48000 * 3):
        audio = (rng.standard_normal(n) * 0.25).astype(np.float32)
        ref = (rng.standard_normal(n) * 0.25).astype(np.float32)
        out.append((f"mono_{n}_ref", audio, ref))
        out.append((f"mono_{n}_noref", audio, None))
        # Stereo channels-first (2, N) — UV3-Layout (§Stereo-Layout-Invariante)
        stereo = np.stack([audio, np.roll(audio, 7)]).astype(np.float32)
        out.append((f"stereo2n_{n}", stereo, None))
    return out


def _assert_identical(new: dict[str, float], old: dict[str, float], label: str) -> None:
    """Gestufter Vergleich: exakte Gleichheit aller Keys (bit-identisch, §G5 (copilot-instructions.md))."""
    assert set(new) == set(old), f"{label}: Key-Menge unterschiedlich"
    for k in sorted(new):
        assert new[k] == old[k], f"{label}: {k} new={new[k]!r} != old={old[k]!r} (bit-identisch verletzt)"


class TestMeasureQuickBitIdentical:
    """_measure_quick (PMGG-Pre/Post) — neu (batched/cache) == Legacy (alt)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("label,audio,ref", _cases(), ids=lambda v: v if isinstance(v, str) else "")
    def test_bit_identical(self, label: str, audio: np.ndarray, ref: np.ndarray | None) -> None:
        m_new = _load_pmgg(False)
        res_new = m_new._measure_quick(
            audio.copy(), _SR, ref.copy() if ref is not None else None, precise_override=False
        )
        m_old = _load_pmgg(True)
        res_old = m_old._measure_quick(
            audio.copy(), _SR, ref.copy() if ref is not None else None, precise_override=False
        )
        _assert_identical(res_new, res_old, f"_measure_quick[{label}]")

    @pytest.mark.unit
    def test_repeatable(self) -> None:
        """§G5 (copilot-instructions.md) Determinismus: gleicher Input ⇒ gleicher Output, auch über Wiederholung."""
        m = _load_pmgg(False)
        audio, ref = _cases()[3][1], _cases()[3][2]
        r1 = m._measure_quick(audio.copy(), _SR, ref.copy() if ref is not None else None, precise_override=False)
        r2 = m._measure_quick(audio.copy(), _SR, ref.copy() if ref is not None else None, precise_override=False)
        _assert_identical(r1, r2, "_measure_quick[repeatability]")


class TestVocalGuardFeaturesBitIdentical:
    """_measure_vocal_guard_features — Vocal-Guard-Features des PMGG-Post-Pfads."""

    @pytest.mark.unit
    @pytest.mark.parametrize("label,audio,ref", _cases(), ids=lambda v: v if isinstance(v, str) else "")
    def test_bit_identical(self, label: str, audio: np.ndarray, ref: np.ndarray | None) -> None:
        m_new = _load_pmgg(False)
        f_new = m_new._measure_vocal_guard_features(audio.copy(), _SR)
        m_old = _load_pmgg(True)
        f_old = m_old._measure_vocal_guard_features(audio.copy(), _SR)
        _assert_identical(f_new, f_old, f"_measure_vocal_guard_features[{label}]")

    @pytest.mark.unit
    def test_repeatable(self) -> None:
        m = _load_pmgg(False)
        audio = _cases()[0][1]
        f1 = m._measure_vocal_guard_features(audio.copy(), _SR)
        f2 = m._measure_vocal_guard_features(audio.copy(), _SR)
        _assert_identical(f1, f2, "_measure_vocal_guard_features[repeatability]")
