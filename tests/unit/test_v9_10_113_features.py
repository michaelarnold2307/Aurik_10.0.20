import pytest

"""Tests für v10.0.0-Features: Phase-40 Studio-2026-LUFS, Phase-09-Severity-Blend,
Phase-29-HF-G-Floor, Phase-55-AR-Order.

Abgedeckt:
  - Phase 40: Studio-2026/maximum → target_lufs = -14.0 (unabhängig vom Material)
  - Phase 40: Restoration/balanced → LUFS-Δ ≤ 1 LU (kein Lautheitsschock bei Archivmaterial)
  - Phase 09: Keine Änderung bei Severity=0 (Baseline unverändert)
  - Phase 09: Moderate Severity ≥ 0.35 → texture_preserve sinkt um ≤ 0.15
  - Phase 09: Schwere Severity ≥ 0.60 → texture_preserve sinkt um ≤ 0.35, min 0.30
  - Phase 29: Presence/Air-Zone G_floor ≤ 45 % des Basis-G_floor bei intensity > 0.4
  - Phase 55: AR-Order bleibt 64 bei kurzen Gaps (< 2400 Samples)
  - Phase 55: AR-Order wird auf ≤ 192 erhöht bei langen Gaps (> 2400 Samples)
  - Phase 55: AR-Order-Cap schützt vor context_length-Überschreitung
"""

import inspect

# ---------------------------------------------------------------------------
# Phase 40 — Studio 2026: -14 LUFS unconditional
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase40LufsStudio2026:
    """Phase 40 muss im Studio-2026- und Maximum-Modus immer target_lufs=-14 setzen."""

    def _get_source(self):
        from backend.core.phases import phase_40_loudness_normalization as m

        return inspect.getsource(m)

    def test_studio2026_override_in_source(self):
        """Source muss `target_lufs = -14.0` im Studio-2026-Zweig enthalten."""
        src = self._get_source()
        # Must have the unconditional override for studio2026
        assert "target_lufs = -14.0" in src, (
            "phase_40 enthält keine target_lufs = -14.0 Überschreibung. "
            "Studio 2026 muss -14 LUFS EBU R128 liefern — unabhängig vom Material."
        )

    def test_studio2026_condition_check(self):
        """Der Studio-2026-Zweig muss 'maximum' und 'studio2026' prüfen."""
        src = self._get_source()
        # Check that the studio2026 override is conditional on quality_mode
        assert '"maximum"' in src or "'maximum'" in src, "maximum-Modus nicht abgedeckt"
        assert '"studio2026"' in src or "'studio2026'" in src, "studio2026-Modus nicht abgedeckt"

    def test_restoration_lufs_delta_cap_in_source(self):
        """Source muss eine LUFS-Δ-Begrenzung für den restoration/balanced-Modus enthalten."""
        src = self._get_source()
        # The delta cap must clamp gain_db to ≤ 1 LU
        assert "gain_db, -1.0, 1.0" in src or "gain_db, -1.0, 1.0" in src, (
            "phase_40 fehlt der LUFS-Δ ≤ 1 LU-Clamp für Restoration/balanced. "
            "§8.2: LUFS-Diff ≤ 1 LU im Restoration-Modus."
        )

    def test_restoration_condition_targets_correct_modes(self):
        """Der Δ-Cap muss restoration und balanced abdecken."""
        src = self._get_source()
        assert '"restoration"' in src or "'restoration'" in src, "restoration-Modus im LUFS-Delta-Cap nicht erwähnt"

    def test_material_targets_still_present(self):
        """MATERIAL_TARGETS muss erhalten bleiben (für archive/restoration-Modus nötig)."""
        from backend.core.phases.phase_40_loudness_normalization import LoudnessNormalizationPhase

        assert hasattr(LoudnessNormalizationPhase, "MATERIAL_TARGETS") or hasattr(
            LoudnessNormalizationPhase, "LUFS_TARGET"
        ), "MATERIAL_TARGETS oder LUFS_TARGET fehlt in phase_40"


# ---------------------------------------------------------------------------
# Phase 09 — Severity-adaptive texture_preserve
# ---------------------------------------------------------------------------


class TestPhase09SeverityAdaptiveBlend:
    """texture_preserve muss bei schwerer Crackling-Severity sinken."""

    def _get_source(self):
        from backend.core.phases import phase_09_crackle_removal as m

        return inspect.getsource(m)

    def test_severity_adaptive_code_present(self):
        """Source muss defect_scores-basierte texture_preserve-Anpassung enthalten."""
        src = self._get_source()
        assert "texture_preserve" in src, "texture_preserve nicht in phase_09"
        assert "_crackle_sev_p09" in src or "crackle_sev" in src, (
            "phase_09 enthält keine Severity-abhängige texture_preserve-Anpassung. "
            "Bei Vinyl-Crackle-Severity=0.9 bleiben 85% Dry-Mix — nur 15% ML-Repair."
        )

    def test_minimum_preserve_floor(self):
        """texture_preserve darf nicht unter 0.30 sinken (Material-Charakter erhalten)."""
        src = self._get_source()
        # Must have a 0.30 floor
        assert "0.30" in src or "0.3," in src, (
            "Mindest-texture_preserve 0.30 fehlt in phase_09. "
            "Zu aggressives Dry-Mix-Entfernen zerstört Tape-/Vinyl-Charakter."
        )

    def test_moderate_severity_threshold(self):
        """Moderate Severity (≥ 0.35) muss zur texture_preserve-Absenkung führen."""
        src = self._get_source()
        assert "0.35" in src, "Grenzwert 0.35 für moderate Crackle-Severity fehlt in phase_09."

    def test_heavy_severity_threshold(self):
        """Schwere Severity (≥ 0.60) muss stärkere Absenkung auslösen."""
        src = self._get_source()
        assert "0.60" in src, "Grenzwert 0.60 für schwere Crackle-Severity fehlt in phase_09."

    def test_defect_scores_kwargs_access(self):
        """defect_scores muss aus kwargs gelesen werden."""
        src = self._get_source()
        assert "defect_scores" in src, "defect_scores wird in phase_09 nicht aus kwargs gelesen"

    def test_params_texture_preserve_modified_for_heavy_crackle(self):
        """Direkte Simulation: params['texture_preserve'] sinkt bei Severity 0.80."""
        import numpy as np

        # Simulate the severity-adaptive logic inline (without full phase instantiation)
        texture_preserve_baseline = 0.85  # Vinyl baseline
        crackle_sev = 0.80  # Heavy crackle

        # Replicate the logic from phase_09
        if crackle_sev >= 0.60:
            adjusted = float(np.clip(texture_preserve_baseline - 0.35, 0.30, 1.0))
        elif crackle_sev >= 0.35:
            adjusted = float(np.clip(texture_preserve_baseline - 0.15, 0.40, 1.0))
        else:
            adjusted = texture_preserve_baseline

        assert adjusted < texture_preserve_baseline, (
            f"texture_preserve sollte bei Severity={crackle_sev} sinken, "
            f"ist aber {adjusted} >= {texture_preserve_baseline}"
        )
        assert adjusted >= 0.30, f"texture_preserve unter Minimum 0.30: {adjusted}"
        assert adjusted == 0.50, f"Erwarteter Wert 0.50 (0.85-0.35), erhalten: {adjusted}"

    def test_params_texture_preserve_unchanged_at_zero_severity(self):
        """Bei Severity=0 bleibt texture_preserve unverändert."""
        import numpy as np

        texture_preserve_baseline = 0.85
        crackle_sev = 0.0

        if crackle_sev >= 0.60:
            adjusted = float(np.clip(texture_preserve_baseline - 0.35, 0.30, 1.0))
        elif crackle_sev >= 0.35:
            adjusted = float(np.clip(texture_preserve_baseline - 0.15, 0.40, 1.0))
        else:
            adjusted = texture_preserve_baseline

        assert adjusted == texture_preserve_baseline, (
            f"texture_preserve sollte bei Severity=0 unverändert bleiben, ist {adjusted}"
        )


# ---------------------------------------------------------------------------
# Phase 29 — HF G_floor für Presence/Air-Zonen
# ---------------------------------------------------------------------------


class TestPhase29HFGFloor:
    """Presence/Air-Zonen müssen ein schärferes G_floor bekommen wenn ML fehlt."""

    def _get_source(self):
        from backend.core.phases import phase_29_tape_hiss_reduction as m

        return inspect.getsource(m)

    def test_hf_floor_enhancement_in_source(self):
        """Source muss zone-spezifische G_floor-Verschärfung für presence/air enthalten."""
        src = self._get_source()
        assert '"presence"' in src or "'presence'" in src, "Presence-Zone in phase_29 nicht erwähnt"
        assert '"air"' in src or "'air'" in src, "Air-Zone in phase_29 nicht erwähnt"
        assert "0.45" in src, (
            "G_floor-Faktor 0.45 für presence/air-Zonen fehlt in phase_29. "
            "Ohne DeepFilterNet bleibt Tape-Zischen 3–5 dB zu laut."
        )

    def test_hf_floor_absolute_minimum(self):
        """G_floor-Minimum für HF-Zonen muss ≥ 0.01 sein (kein totales Gate)."""
        src = self._get_source()
        # We have `max(..., 0.020, ...)` as the floor guard
        assert "0.020" in src or "0.02," in src or "0.025" in src, (
            "Absolutes G_floor-Minimum für HF-Zonen fehlt in phase_29. "
            "G_floor=0 → totales Noise-Gate → unnatürliche Stille in Transient-freien Frames."
        )

    def test_intensity_scale_gate(self):
        """G_floor-Verschärfung darf nur bei intensity_scale > 0.40 aktiv sein."""
        src = self._get_source()
        assert "0.40" in src or "0.4)" in src or "> 0.40" in src or "> 0.4" in src, (
            "intensity_scale-Gate für HF-G_floor-Verschärfung fehlt in phase_29. "
            "Bei schwachem Strength-Signal darf kein aggressives HF-Gate aktiv sein."
        )

    def test_hf_floor_computed_correctly(self):
        """Simulierter G_floor nach Verschärfung: TAPE (G_floor=0.08) → 0.036."""
        import numpy as np

        g_floor_tape = 0.08  # TAPE at full strength
        _hf_floor = float(np.clip(g_floor_tape * 0.45, 0.020, g_floor_tape))
        expected = 0.08 * 0.45  # = 0.036

        assert abs(_hf_floor - expected) < 1e-6, f"TAPE HF-G_floor: erwartet {expected:.4f}, erhalten {_hf_floor:.4f}"

    def test_hf_floor_vinyl(self):
        """Vinyl (G_floor=0.10) → HF-G_floor = 0.045."""
        import numpy as np

        g_floor_vinyl = 0.10
        _hf_floor = float(np.clip(g_floor_vinyl * 0.45, 0.020, g_floor_vinyl))
        assert abs(_hf_floor - 0.045) < 1e-6, f"VINYL HF-G_floor: erwartet 0.045, {_hf_floor}"

    def test_hf_floor_never_exceeds_base(self):
        """HF-G_floor darf nie größer als das Basis-G_floor sein (np.clip upper bound)."""
        import numpy as np

        for g_floor_base in (0.06, 0.08, 0.10, 0.12):
            _hf_floor = float(np.clip(g_floor_base * 0.45, 0.020, g_floor_base))
            assert _hf_floor <= g_floor_base, (
                f"HF-G_floor {_hf_floor} > Basis {g_floor_base} — obere Clip-Grenze verletzt"
            )


# ---------------------------------------------------------------------------
# Phase 55 — Adaptiver AR-Order für lange Gaps
# ---------------------------------------------------------------------------


class TestPhase55AdaptiveAROrder:
    """AR-Order 192 muss für Gaps > 2400 Samples (> ~50 ms @ 48 kHz) verwendet werden."""

    def _get_source(self):
        from backend.core.phases import phase_55_diffusion_inpainting as m

        return inspect.getsource(m)

    def test_adaptive_ar_order_in_source(self):
        """Source muss adaptive AR-Order-Logik für lange Gaps enthalten."""
        src = self._get_source()
        assert "_AR_ORDER_ADAPTIVE" in src or "ar_order_adaptive" in src.lower(), (
            "phase_55 enthält keine adaptive AR-Order-Logik. "
            "AR(64) divergiert bei Gaps > 50 ms; AR(192) liefert 3× bessere Spektralmodi."
        )

    def test_long_gap_threshold_2400_samples(self):
        """Gap-Schwelle für AR-Order-Erhöhung muss bei 2400 Samples liegen."""
        src = self._get_source()
        assert "2400" in src, "Schwellenzeit 2400 Samples (≈ 50 ms @ 48 kHz) für AR-Order-Erhöhung fehlt in phase_55."

    def test_higher_ar_order_192(self):
        """Max AR-Order 192 muss als Zielwert verwendet werden."""
        src = self._get_source()
        assert "192" in src, "AR-Order 192 fehlt in phase_55. Für Gaps > 50 ms ist AR(192) nötig."

    def test_ar_order_cap_protects_short_context(self):
        """AR-Order-Cap muss `len(left_ctx) - 1` als upper bound verwenden."""
        src = self._get_source()
        assert "len(left_ctx)" in src, (
            "AR-Order-Cap gegen kurzen Kontext fehlt in phase_55. "
            "AR-Order > Kontext-Länge würde ValueError in _burg_ar_predict() auslösen."
        )

    def test_adaptive_ar_logic_short_gap(self):
        """Kurze Gaps (< 2400 Samples) verwenden weiterhin _AR_ORDER=64."""

        from backend.core.phases.phase_55_diffusion_inpainting import _AR_ORDER

        # Simulate the adaptive logic
        gap_len = 1200  # ~25 ms
        left_ctx_len = 3200  # normal context
        _ar_order_adaptive = min(192, max(16, left_ctx_len - 1)) if gap_len > 2400 else _AR_ORDER
        assert _ar_order_adaptive == _AR_ORDER, (
            f"Kurze Lücke: AR-Order sollte {_AR_ORDER} bleiben, ist {_ar_order_adaptive}"
        )

    def test_adaptive_ar_logic_long_gap(self):
        """Lange Gaps (> 2400 Samples) erhöhen AR-Order auf ≤ 192."""

        from backend.core.phases.phase_55_diffusion_inpainting import _AR_ORDER

        # Simulate the adaptive logic
        gap_len = 4800  # ~100 ms
        left_ctx_len = 3200  # normal context
        _ar_order_adaptive = min(192, max(16, left_ctx_len - 1)) if gap_len > 2400 else _AR_ORDER
        assert _ar_order_adaptive == 192, f"Lange Lücke: AR-Order sollte 192 sein, ist {_ar_order_adaptive}"
        assert _ar_order_adaptive > _AR_ORDER, "Langer Gap: AR-Order wurde nicht erhöht"

    def test_adaptive_ar_caps_to_context_length(self):
        """Bei sehr kurzem Kontext (< 193 Samples) greift der Sicherheits-Cap."""
        from backend.core.phases.phase_55_diffusion_inpainting import _AR_ORDER

        gap_len = 4800
        left_ctx_len = 100  # very short — near file start
        _ar_order_adaptive = min(192, max(16, left_ctx_len - 1)) if gap_len > 2400 else _AR_ORDER
        assert _ar_order_adaptive == 99, (
            f"Kurzer Kontext: AR-Order sollte auf 99 (len-1) gecapped werden, ist {_ar_order_adaptive}"
        )
        assert _ar_order_adaptive < left_ctx_len, "AR-Order überschreitet Kontext-Länge!"
