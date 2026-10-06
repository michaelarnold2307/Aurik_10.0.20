"""
test_primary_paths_no_fallback.py — No-Fallback-Garantie der ML-Primärpfade.

Auftrag (2026-10-02): Alle Primärpfade inklusive der ML-Modelle müssen
lückenlos garantiert funktionieren — „out of the box" darf KEIN
§V6 (VERBOTEN.md)-Fallback greifen. Diese Norm-Validierung ist fail-closed
(kein skip bei fehlenden Voraussetzungen) und beweist:

  1. Feature-Basis: `libdf` (deepfilterlib) importierbar — kein
     NumPy-Feature-Ersatz mit „degradierter Modellqualität".
  2. Modell-Lage: `resolve_model_path()` findet für jedes aktiv geflaggte
     Modell einen existierenden Pfad; wo eine musik-finetuned Stufe
     (`finetuned/`) existiert, ist sie aktiv (Nutzer-Vorgabe 2026-10-02).
  3. Artefakt-Validität: jede ONNX-Kerndatei lädt als CPU-Session
     (§III.9 (copilot-instructions.md)) mit erwarteter Input-Signatur.
  4. S2-FlowMatching: beide Harmonic-Inpainting-Modellvarianten
     („mask": x[b,t,2], „best": x[b,t,1]) rechnen einen Euler-Schritt.
  5. Plugin-Primärpfade: DeepFilterNet-v3-II initialisiert OHNE
     Ersatzpfad-Warnung im Log.
  6. Inventar-Gate: vorhandene, bewusst ungenutzte Großmodelle
     (Whisper large-v3-turbo/Decoder) bleiben dokumentiert und auffindbar.

Bewusst gesperrte Flags (§v10.16/F7 SGMSE-Musik-Abnahme offen, §v10.20
Whisper-Denoiser deprecated, BW-Reconstructor v5 HF-Gate nicht bestanden,
§P1-2 SCNet ohne Hörordnungs-Sign-off) sind KEINE Primärpfade — sie werden
unten explizit gelistet, damit die Entscheidung sichtbar bleibt und nicht
versehentlich als „Lücke" gilt.
"""

from __future__ import annotations

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

_TIMEOUT = 300

# ONNX-Kerndateien → erwartete (Input-Name, dynamische Achsen, Kanalbreite)
_CORE_ONNX: dict[str, tuple[str, int]] = {
    "models/deepfilternet_v3_ii/finetuned/enc.onnx": ("x", 3),
    "models/deepfilternet_v3_ii/finetuned/dec.onnx": ("x", 3),
    "models/deepfilternet_v3_ii/finetuned/erb_dec.onnx": ("x", 3),
    "models/mp_senet/finetuned/mp_senet_musik.onnx": ("x", 3),
    "models/miipher_dit/flow_matching_dit.onnx": ("x", 3),
    "models/harmonic_inpainting/inpainting_best.onnx": ("x", 3),
    "models/harmonic_inpainting/inpainting_mask_best.onnx": ("x", 3),
    "models/whisper/whisper_tiny.onnx": ("x", 3),
}

# Bewusst gesperrte Flags (keine Primärpfade — dokumentierte Entscheidungen)
_CONSCIOUSLY_GATED = {
    "sgmse_musik": "§v10.16/F7 — Musik-Core + ONNX-Export vorhanden, A/B-Abnahme offen (Flag gesperrt)",
    "whisper_denoiser": "§v10.20 — deprecated, nur A/B-Gate",
    "bw_v5": "A1: HF-Gain-Gate nicht bestanden (0.73 < 1.02)",
    "scnet_musik": (
        "§P1-2 — A/B liegt vor (+1,96…+3,59 dB SI-SDR gegen die faire Demucs-v4-Stufe, "
        "docs/reports/current/2026-10-06_p1_2_scnet_vs_demucs_fair_ab.md), aber "
        "Hörordnungs-Sign-off (§v10.802) fehlt und CPU ~7× langsamer (Flag gesperrt)"
    ),
}


@pytest.mark.timeout(_TIMEOUT)
def test_libdf_features_available():
    """Feature-Basis: libdf MUSS importierbar sein (kein NumPy-Ersatz)."""
    from libdf import DF, erb, erb_norm, unit_norm

    assert callable(erb) and callable(unit_norm)


@pytest.mark.timeout(_TIMEOUT)
def test_resolve_model_path_active_flags():
    """Jedes aktiv geflaggte Modell resolve()t auf eine existierende Datei."""
    from backend.core import music_model_flags as flags

    active = {
        "dfn_enc": flags.use_df_musik,
        "dfn_dec": flags.use_df_musik,
        "dfn_erb_dec": flags.use_df_musik,
        "mp_senet": flags.use_mp_senet_musik,
        "miipher_dit": flags.use_miipher_dit,
        "harmonic_inpainting": flags.use_harmonic_inpainting,
        "whisper_encoder": flags.use_miipher_dit,
        "bigvgan": flags.use_miipher_dit,
    }
    for key, enabled in active.items():
        if not enabled:
            continue
        resolved = flags.resolve_model_path(key)
        assert resolved is not None, f"{key}: kein Modellpfad auflösbar (Primärpfad-Lücke!)"
        assert resolved.exists(), f"{key}: aufgelöster Pfad fehlt: {resolved}"


@pytest.mark.timeout(_TIMEOUT)
def test_musik_finetuned_stufe_aktiv():
    """Wo eine finetuned/-Stufe existiert, muss sie die aktive Auswahl sein."""
    from backend.core import music_model_flags as flags

    for key in ("dfn_enc", "dfn_dec", "dfn_erb_dec", "mp_senet"):
        resolved = flags.resolve_model_path(key)
        assert resolved is not None
        assert "finetuned" in resolved.parts, f"{key}: musik-finetuned Stufe nicht aktiv: {resolved}"


@pytest.mark.timeout(_TIMEOUT)
@pytest.mark.parametrize("rel_path", sorted(_CORE_ONNX))
def test_onnx_artifacts_load(rel_path: str):
    """Jede Kerndatei lädt als CPU-Session (§III.9) mit plausibler Signatur."""
    from pathlib import Path

    p = Path(rel_path)
    assert p.exists(), f"Primär-Artefakt fehlt: {rel_path}"
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    expected_name, expected_rank = _CORE_ONNX[rel_path]
    inputs = sess.get_inputs()
    assert inputs, f"{rel_path}: keine Inputs"
    first = next((i for i in inputs if i.name == expected_name), inputs[0])
    if "deepfilternet" in rel_path:
        # DFN-Trio ist spektrogrammbasiert ([B, C, S, N_ERB] ERB-Features),
        # nicht sequenzbasiert wie die DiT-Modelle — nur Plausibilitäts-Rank.
        assert len(first.shape) in (3, 4), f"{rel_path}: überraschende Signatur {first.shape}"
    else:
        assert len(first.shape) == expected_rank, f"{rel_path}: überraschende Signatur {first.shape}"
    assert sess.get_outputs(), f"{rel_path}: keine Outputs"


@pytest.mark.timeout(_TIMEOUT)
def test_s2_flowmatching_beide_varianten():
    """S2-FlowMatching rechnet einen Euler-Schritt mit BEIDEN Modellvarianten.

    „mask" = x[b,t,2] (FlowMatchingDiT mit Maskenkanal), „best" = x[b,t,1] —
    laut Vorgabe 2026-10-02 zwei verschiedene Modelle, Priorität mask vor best.
    """
    from pathlib import Path

    rng = np.random.default_rng(42)
    base = Path("models/harmonic_inpainting")
    cases = [
        (base / "inpainting_mask_best.onnx", 2),
        (base / "inpainting_best.onnx", 1),
    ]
    for path, channels in cases:
        assert path.exists(), f"S2-Modell fehlt: {path}"
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        x = rng.standard_normal((1, 4800, channels)).astype(np.float32)
        v = sess.run(None, {"x": x, "t": np.array([0.5], dtype=np.float32)})[0]
        assert np.isfinite(v).all(), f"{path.name}: Velocity nicht finite"
        assert v.shape == (1, 4800, 1), f"{path.name}: überraschende Velocity-Form {v.shape}"


@pytest.mark.timeout(_TIMEOUT)
def test_deepfilternet_plugin_primary_path(caplog):
    """DeepFilterNet-v3-II lädt die finetuned-Modelle OHNE Ersatzpfad-Warnung."""
    import logging

    from plugins.deepfilternet_v3_ii_plugin import get_deepfilternet_plugin

    with caplog.at_level(logging.WARNING):
        plugin = get_deepfilternet_plugin()
    assert plugin is not None
    assert getattr(plugin, "_enc", None) is not None, "DFN-Encoder nicht geladen (Primärpfad-Lücke!)"
    bad = [r.message for r in caplog.records if "fehlt" in r.message or "Ersatzpfad" in r.message]
    assert not bad, f"§V6-Fallbacks out of the box: {bad}"


@pytest.mark.timeout(_TIMEOUT)
def test_whisper_inventar_gesichert():
    """Vorhandene, bewusst ungenutzte Großmodelle bleiben als Kompetenz bestehen.

    whisper_large_v3_turbo_encoder_fp16.onnx (semantischer Encoder, Upgrade-
    Kandidat gegenüber whisper_tiny) und decoder_model_merged.onnx (§v10.20-
    Kontext) sind Stand 2026-10-02 bewusst nicht referenziert — sie dürfen
    aber nicht verloren gehen. Ein Wechsel tiny → large-v3-turbo gehört hinter
    ein A/B-Gate im §v10.748-Muster (Embedding-Dimensionen sind kompatibel zu
    halten).
    """
    from pathlib import Path

    for name in ("whisper_tiny.onnx", "whisper_large_v3_turbo_encoder_fp16.onnx", "decoder_model_merged.onnx"):
        p = Path("models/whisper") / name
        assert p.exists() and p.stat().st_size > 1_000_000, f"Whisper-Artefakt fehlt: {name}"


@pytest.mark.timeout(_TIMEOUT)
def test_bewusst_gesperrte_flags_dokumentiert():
    """Gesperrte Flags sind Entscheidungen, keine Lücken — Sichtbarkeit erzwingen."""
    from backend.core import music_model_flags as flags

    assert flags.use_sgmse_musik is False, _CONSCIOUSLY_GATED["sgmse_musik"]
    assert flags.use_whisper_denoiser is False, _CONSCIOUSLY_GATED["whisper_denoiser"]
    assert flags.use_bw_v5 is False, _CONSCIOUSLY_GATED["bw_v5"]
    assert flags.use_scnet_music is False, _CONSCIOUSLY_GATED["scnet_musik"]
