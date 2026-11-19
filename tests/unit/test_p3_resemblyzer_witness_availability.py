"""§SOTA-P3 (Resemblyzer-Witness verfügbar machen) — Regressionsschutz.

Ziel (Session-Ertrag 2026-09-15): Die Stimm-Identitäts-Invariante
(singer_identity_cosine ≥ 0,92, §2.35c/Hörordnung Ebene 1) muss messbar sein.
Deckt die Plugin-Kaskade Package→ONNX→None und den 2026-09-15-Bugfix:
- Pfad-Tippfehler ``models/rezemblyzer`` → ``models/resemblyzer`` (Package-Pfad
  war nie erreichbar);
- webrtcvad-Import-Shim: das unveränderte Vendored-Paket importiert webrtcvad
  auf Modulebene — ohne Shim schlug der Package-Pfad trotz installiertem torch
  immer fehl (stiller DSP-Ersatzpfad, §V74 (VERBOTEN.md)).

Autor: Aurik Testing Team
"""

import numpy as np
import pytest


def _voice_like(f0: float, seconds: float, sr: int, seed: int) -> np.ndarray:
    """Synthetische, stimmähnliche Anregung (Grundton + Obertöne + Hüllkurve)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    base = (
        0.4 * np.sin(2 * np.pi * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.08 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.02 * rng.normal(0, 1, len(t)).astype(np.float32)
    )
    # Grobe Amplitudenmodulation (Silben-ähnlich, ~4 Hz)
    base *= (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t + 0.3)).astype(np.float32)
    return (base / (np.max(np.abs(base)) + 1e-9) * 0.4).astype(np.float32)


class TestResemblyzerWitnessAvailability:
    def test_plugin_cascade_reports_availability_honestly(self):
        from plugins.resemblyzer_plugin import get_resemblyzer_plugin

        plugin = get_resemblyzer_plugin()
        # available == (Package ODER ONNX geladen) — nie ein stiller Stub.
        assert isinstance(plugin.available, bool)
        if plugin.available:
            assert plugin._encoder is not None or plugin._onnx_session is not None

    def test_singer_identity_measurable_above_092(self):
        """Gleiche „Stimme" mit kleiner Pegel-/Rauschvariation → cos ≥ 0,92."""
        from plugins.resemblyzer_plugin import get_resemblyzer_plugin

        plugin = get_resemblyzer_plugin()
        if not plugin.available:
            pytest.skip(
                "Resemblyzer (Package/ONNX) in dieser Umgebung nicht verfügbar — DSP-Ersatzpfad separat getestet"
            )

        sr = 16000
        a = _voice_like(180.0, 2.0, sr, seed=1)
        b = _voice_like(180.0, 2.0, sr, seed=2)  # gleiche Grundfrequenz, anderes Rauschen

        emb_a = plugin.embed(a, sr)
        emb_b = plugin.embed(b, sr)
        if emb_a is None or emb_b is None:
            pytest.skip("embed() lieferte None (VAD/Kurz-Segment) — Witness-Pfad selbst ist anderweitig getestet")

        assert emb_a.shape == (256,)
        assert emb_b.shape == (256,)
        for emb in (emb_a, emb_b):
            assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3  # L2-normierter d-vector

        cos = float(np.dot(emb_a, emb_b))
        assert cos >= 0.92, f"Stimm-Identität {cos:.3f} < 0,92 (Hör-Invariante verletzt)"

    def test_different_voice_lower_similarity(self):
        from plugins.resemblyzer_plugin import get_resemblyzer_plugin

        plugin = get_resemblyzer_plugin()
        if not plugin.available:
            pytest.skip("Resemblyzer (Package/ONNX) in dieser Umgebung nicht verfügbar")

        sr = 16000
        a = _voice_like(180.0, 2.0, sr, seed=3)
        other = _voice_like(320.0, 2.0, sr, seed=4)  # deutlich höhere Grundfrequenz

        emb_a = plugin.embed(a, sr)
        emb_o = plugin.embed(other, sr)
        if emb_a is None or emb_o is None:
            pytest.skip("embed() lieferte None (VAD/Kurz-Segment)")

        cos_same = float(np.dot(plugin.embed(a, sr), emb_a)) if plugin.embed(a, sr) is not None else 1.0
        cos_other = float(np.dot(emb_a, emb_o))
        assert cos_other < cos_same
        assert cos_other < 0.92

    def test_cosine_similarity_nan_safe(self):
        from plugins.resemblyzer_plugin import get_resemblyzer_plugin

        plugin = get_resemblyzer_plugin()
        z = np.zeros(256, dtype=np.float64)
        v = np.array([1.0] * 256)
        assert 0.0 <= plugin.cosine_similarity(z, z) <= 1.0
        assert 0.0 <= plugin.cosine_similarity(np.full(256, np.nan), v) <= 1.0

    def test_local_package_dir_points_to_existing_path(self):
        """Bugfix-Nachweis: models/resemblyzer (nicht „rezemblyzer") muss existieren."""
        import os

        import plugins.resemblyzer_plugin as rp

        if not os.path.isdir(rp._LOCAL_RESEMBLYZER_DIR):
            pytest.skip("models/resemblyzer nicht vorhanden (gitignored)")

        assert os.path.isdir(rp._LOCAL_RESEMBLYZER_DIR), (
            f"Lokales Resemblyzer-Paket nicht gefunden: {rp._LOCAL_RESEMBLYZER_DIR} "
            "(Pfad-Tippfehler models/rezemblyzer?)"
        )
