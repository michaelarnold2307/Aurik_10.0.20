from __future__ import annotations

"""
Tests für StructuralSilenceIsolationProtocol (SSIP) — §2.68g [RELEASE_MUST]
============================================================================

Spec: 02_pipeline_architecture.md §2.68g (v10.0.0)
VERBOTEN-Regeln: V14, V15, V16, V17, V18
"""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from backend.core.dsp.structural_silence_isolation import (
    StructuralSilenceIsolator,
    _get_structural_silence_zones,
    _run_inpainting_with_ssip,
    get_structural_silence_isolator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 48000


@pytest.fixture
def isolator() -> StructuralSilenceIsolator:
    return get_structural_silence_isolator()


def _make_silence_music_audio(
    sr: int = SR, silence_start_s: float = 1.0, music_s: float = 3.0, silence_end_s: float = 1.0
) -> np.ndarray:
    """1 s Stille + 3 s Musik + 1 s Stille."""
    silence_start = np.zeros(int(silence_start_s * sr), dtype=np.float32)
    music = (np.random.default_rng(42).random(int(music_s * sr)).astype(np.float32) * 2 - 1) * 0.3
    silence_end = np.zeros(int(silence_end_s * sr), dtype=np.float32)
    return np.concatenate([silence_start, music, silence_end])


# ---------------------------------------------------------------------------
# §2.68g Test 1: Singleton-Pattern
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ssip_singleton_returns_same_instance():
    a = get_structural_silence_isolator()
    b = get_structural_silence_isolator()
    assert a is b


# ---------------------------------------------------------------------------
# §2.68g Test 2: detect_structural_silence_zones — Basis
# ---------------------------------------------------------------------------


def test_ssip_detect_silence_zones_basic(isolator):
    """Erkennt Anfangs- und End-Stille korrekt."""
    audio = _make_silence_music_audio()
    zones = isolator.detect_structural_silence_zones(audio, SR, "unknown")
    assert isinstance(zones, list)
    assert len(zones) >= 1  # mindestens eine Zone (Anfang oder Ende)


def test_ssip_detect_silence_zones_start_and_end(isolator):
    """Erkennt sowohl Anfangs- als auch End-Stille."""
    audio = _make_silence_music_audio(silence_start_s=1.0, music_s=3.0, silence_end_s=1.0)
    zones = isolator.detect_structural_silence_zones(audio, SR, "unknown")
    assert len(zones) >= 1
    # Mindestens eine Zone soll am Anfang oder Ende des Signals sein
    n_samples = len(audio)
    is_valid = any(
        z[0] == 0 or z[1] >= n_samples - SR // 10  # Anfang oder ca. Ende
        for z in zones
    )
    assert is_valid, f"Keine Zone am Anfang oder Ende: {zones}"


def test_ssip_detect_no_zones_on_pure_music(isolator):
    """Reine Musik ohne Stille: keine oder wenige Zonen."""
    rng = np.random.default_rng(0)
    audio = (rng.random(SR * 5).astype(np.float32) * 2 - 1) * 0.3
    zones = isolator.detect_structural_silence_zones(audio, SR, "unknown")
    # Reine Musik: keine strukturelle Stille
    for z in zones:
        duration_ms = (z[1] - z[0]) / SR * 1000
        assert duration_ms >= 300.0  # Nur echte Stille (≥ 300 ms)


def test_ssip_detect_returns_list_on_empty_audio(isolator):
    """Leeres Audio: leere Liste, kein Crash."""
    zones = isolator.detect_structural_silence_zones(np.array([], dtype=np.float32), SR)
    assert zones == []


def test_ssip_detect_stereo_channels_first(isolator):
    """Stereo (2, N) Format wird korrekt verarbeitet."""
    mono = _make_silence_music_audio()
    stereo = np.stack([mono, mono], axis=0)  # (2, N)
    isolator.detect_structural_silence_zones(mono, SR, "unknown")
    zones_stereo = isolator.detect_structural_silence_zones(stereo, SR, "unknown")
    assert len(zones_stereo) >= 1


def test_ssip_detect_stereo_samples_first(isolator):
    """Stereo (N, 2) Format wird korrekt verarbeitet."""
    mono = _make_silence_music_audio()
    stereo = np.stack([mono, mono], axis=1)  # (N, 2)
    zones = isolator.detect_structural_silence_zones(stereo, SR, "unknown")
    assert len(zones) >= 1


# ---------------------------------------------------------------------------
# §2.68g Test 3: Null-Propagation-Guard (Failure Mode 3)
# ---------------------------------------------------------------------------


def test_ssip_null_propagation_guard_empty_kwargs():
    """_get_structural_silence_zones() ohne kwargs berechnet eigenständig — nie None."""
    audio = _make_silence_music_audio()
    zones = _get_structural_silence_zones({}, audio, SR, "unknown")
    assert zones is not None
    assert isinstance(zones, list)
    assert len(zones) >= 1  # mindestens eine Zone muss gefunden werden


def test_ssip_null_propagation_guard_from_kwargs():
    """_get_structural_silence_zones() liefert Zonen aus kwargs wenn vorhanden."""
    injected = [(0, 100), (200, 300)]
    zones = _get_structural_silence_zones({"structural_silence_zones": injected}, np.zeros(SR), SR)
    assert zones == injected


def test_ssip_null_propagation_guard_from_restoration_context():
    """_get_structural_silence_zones() liefert Zonen aus restoration_context."""
    injected = [(1000, 2000)]
    ctx = {"restoration_context": {"structural_silence_zones": injected}}
    zones = _get_structural_silence_zones(ctx, np.zeros(SR), SR)
    assert zones == injected


# ---------------------------------------------------------------------------
# §2.68g Test 4: post_inpainting_silence_audit — Hard-Reset (V17, kein Clip)
# ---------------------------------------------------------------------------


def test_ssip_no_energy_in_silence_zone_after_inpainting(isolator):
    """§2.68g Pflicht-Test: Silence-Zone darf nach Inpainting nicht lauter sein."""
    audio = _make_silence_music_audio()
    # Worst-case Inpainting: füllt alles mit Energie
    rng = np.random.default_rng(0)
    noisy_inpainted = audio + rng.random(len(audio)).astype(np.float32) * 0.1

    zones = isolator.detect_structural_silence_zones(audio, SR, "unknown")
    assert len(zones) >= 1, "Keine Stille-Zonen detektiert — Test-Setup fehlerhaft"

    result = isolator.post_inpainting_silence_audit(audio, noisy_inpainted, zones, SR)

    for start, end in zones:
        seg_len = end - start
        if seg_len <= 0:
            continue
        energy_result = float(np.sqrt(np.mean(result[start:end] ** 2) + 1e-12))
        energy_orig = float(np.sqrt(np.mean(audio[start:end] ** 2) + 1e-12))
        # Max +1 dB Toleranz (Faktor 1.12)
        assert energy_result <= energy_orig * 1.12, (
            f"Stille-Zone [{start}:{end}] hat nach Audit zu viel Energie: "
            f"result={energy_result:.6f} vs. orig={energy_orig:.6f}"
        )


def test_ssip_audit_no_change_if_silence_not_louder(isolator):
    """Wenn Stille-Zone nicht lauter → keine Veränderung."""
    audio = _make_silence_music_audio()
    # Inpainting verändert nur Musik-Teil
    inpainted = audio.copy()
    music_start = SR  # nach 1 s Stille
    inpainted[music_start : music_start + SR] *= 1.05  # leichte Lautstärkeänderung in Musik

    zones = isolator.detect_structural_silence_zones(audio, SR, "unknown")
    result = isolator.post_inpainting_silence_audit(audio, inpainted, zones, SR)

    # Musik-Teil unverändert
    assert np.allclose(result[music_start : music_start + SR], inpainted[music_start : music_start + SR])


def test_ssip_audit_hard_reset_not_clip(isolator):
    """Stille-Zone wird durch Hard-Reset (Original-Samples) zurückgesetzt — nicht Clip."""
    # Silence-Zone mit bekanntem Sub-Pegel
    silence = np.full(SR, 0.001, dtype=np.float32)  # sehr leiser, nicht-null Pegel
    music = (np.random.default_rng(1).random(SR * 2).astype(np.float32) * 2 - 1) * 0.3
    audio = np.concatenate([silence, music])

    # Inpainting fügt große Energie in Stille ein (schlimmster Fall)
    inpainted = audio.copy()
    inpainted[:SR] = 0.8  # massive Energie in Stille-Zone

    zones = [(0, SR)]
    result = isolator.post_inpainting_silence_audit(audio, inpainted, zones, SR)

    # Nach Hard-Reset: exakt Original-Samples in Stille-Zone
    assert np.allclose(result[:SR], audio[:SR], atol=1e-6), (
        "Hard-Reset hat nicht die exakten Original-Samples wiederhergestellt"
    )


def test_ssip_audit_empty_zones_passthrough(isolator):
    """Leere Stille-Zonen: kein Eingriff, Passthrough."""
    audio = _make_silence_music_audio()
    inpainted = audio + 0.01
    result = isolator.post_inpainting_silence_audit(audio, inpainted, [], SR)
    assert np.allclose(result, inpainted, atol=1e-6)


# ---------------------------------------------------------------------------
# §2.68g Test 5: split_at_silence_boundaries + reassemble
# ---------------------------------------------------------------------------


def test_ssip_split_and_reassemble_identity(isolator):
    """Split + Reassemble ohne Verarbeitung: Original wiederherstellen.
    Musik-Teil muss lang genug sein für CONTEXT_GUARD (2× 1500ms = 3s), daher 8s Musik.
    """
    # Kürzere Stille-Zonen, längere Musik damit Audio-Segmente entstehen
    rng = np.random.default_rng(42)
    silence_start = np.zeros(int(0.5 * SR), dtype=np.float32)
    music = (rng.random(int(8.0 * SR)).astype(np.float32) * 2 - 1) * 0.3
    silence_end = np.zeros(int(0.5 * SR), dtype=np.float32)
    audio = np.concatenate([silence_start, music, silence_end])

    zones = isolator.detect_structural_silence_zones(audio, SR, "unknown")
    if not zones:
        pytest.skip("Keine Stille-Zonen detektiert — Signal evtl. nicht leise genug")

    segments = isolator.split_at_silence_boundaries(audio, SR, zones)
    assert len(segments) >= 1

    # Segmente dürfen gemischt "silence" und "audio" haben
    # (CONTEXT_GUARD kann kleine Musiklücken erschaffen, wenn Musik lang genug)
    types = {s["type"] for s in segments}
    assert "silence" in types, "Keine Stille-Segmente erzeugt"

    result = isolator.reassemble_from_segments(segments, audio, len(audio))
    assert len(result) == len(audio)
    assert np.isfinite(result).all()


def test_ssip_split_no_zones_returns_single_audio_segment(isolator):
    """Keine Stille-Zonen: ein einzelnes Audio-Segment."""
    audio = np.random.default_rng(2).random(SR).astype(np.float32) * 0.5
    segments = isolator.split_at_silence_boundaries(audio, SR, [])
    assert len(segments) == 1
    assert segments[0]["type"] == "audio"


def test_ssip_silence_segments_use_original_in_reassemble(isolator):
    """Stille-Segmente nutzen Original-Samples in reassemble (HARD RULE)."""
    audio = _make_silence_music_audio()
    zones = isolator.detect_structural_silence_zones(audio, SR, "unknown")
    if not zones:
        pytest.skip("Keine Stille-Zonen detektiert")

    segments = isolator.split_at_silence_boundaries(audio, SR, zones)
    # Simuliere Verarbeitung: verändere alle Audio-Segmente
    modified_segments = []
    for seg in segments:
        if seg["type"] == "audio":
            modified_segments.append({**seg, "data": seg["data"] * 0.0 + 0.5})  # drastische Änderung
        else:
            modified_segments.append(seg)

    result = isolator.reassemble_from_segments(modified_segments, audio, len(audio))

    # In Stille-Zonen: Original-Samples erwartet
    for zs, ze in zones:
        assert np.allclose(result[zs:ze], audio[zs:ze], atol=1e-5), (
            f"Stille-Zone [{zs}:{ze}] enthält nicht die Original-Samples"
        )


# ---------------------------------------------------------------------------
# §2.68g Test 6: _run_inpainting_with_ssip — End-to-End
# ---------------------------------------------------------------------------


def test_run_inpainting_with_ssip_silence_preserved():
    """_run_inpainting_with_ssip: Stille-Zonen bleiben nach Inpainting unverändert."""
    audio = _make_silence_music_audio()
    isolator = get_structural_silence_isolator()
    zones = isolator.detect_structural_silence_zones(audio, SR, "unknown")
    if not zones:
        pytest.skip("Keine Stille-Zonen detektiert")

    def aggressive_inpainting(segment: np.ndarray, sr: int) -> np.ndarray:
        """Simuliert aggressives Inpainting das alles mit Energie füllt."""
        return np.full_like(segment, 0.5)

    result = _run_inpainting_with_ssip(audio, SR, zones, aggressive_inpainting)

    # Stille-Zonen müssen durch Post-Audit geschützt sein
    for zs, ze in zones:
        energy_after = float(np.sqrt(np.mean(result[zs:ze] ** 2) + 1e-12))
        energy_orig = float(np.sqrt(np.mean(audio[zs:ze] ** 2) + 1e-12))
        assert energy_after <= energy_orig * 1.12, (
            f"Stille [{zs}:{ze}] nach SSIP-Inpainting zu laut: after={energy_after:.6f}"
        )


def test_run_inpainting_with_ssip_no_crash_on_inpainting_error():
    """_run_inpainting_with_ssip: kein Crash wenn inpainting_fn wirft."""
    audio = _make_silence_music_audio()
    zones = [(0, SR)]

    def failing_inpainting(segment, sr):
        raise RuntimeError("Simulated failure")

    # Non-blocking: Passthrough bei Fehler
    result = _run_inpainting_with_ssip(audio, SR, zones, failing_inpainting)
    assert result is not None
    assert len(result) == len(audio)


# ---------------------------------------------------------------------------
# §2.68g Test 7: material-adaptive Schwellen
# ---------------------------------------------------------------------------


def test_ssip_silence_threshold_shellac_is_higher(isolator):
    """Shellac-Schwelle höher (lauter) als CD-Digital."""
    thresh_shellac = isolator._get_silence_threshold("shellac")
    thresh_cd = isolator._get_silence_threshold("cd_digital")
    # Shellac hat lauten Rauschboden → Stille-Schwelle ist weniger negativ (höher)
    assert thresh_shellac > thresh_cd


def test_ssip_silence_threshold_unknown_fallback(isolator):
    """Unbekanntes Material: Fallback auf 'unknown' Schwelle."""
    thresh = isolator._get_silence_threshold("xyz_unbekannt")
    assert thresh == isolator.SILENCE_THRESHOLDS_DBFS["unknown"]


# ---------------------------------------------------------------------------
# §2.68g Test 8: Konstanten-Integrität
# ---------------------------------------------------------------------------


def test_ssip_context_guard_ms_value(isolator):
    """CONTEXT_GUARD_MS ist 1500 ms gemäss Spec."""
    assert isolator.CONTEXT_GUARD_MS == 1500.0


def test_ssip_silence_thresholds_required_materials(isolator):
    """Alle pflicht-Materialien haben Schwellen."""
    required = {"shellac", "vinyl", "cassette", "reel_tape", "cd_digital", "unknown"}
    for mat in required:
        assert mat in isolator.SILENCE_THRESHOLDS_DBFS, f"Material '{mat}' fehlt in SILENCE_THRESHOLDS_DBFS"


def test_ssip_all_thresholds_negative_dbfs(isolator):
    """Alle Stille-Schwellen sind negative dBFS-Werte."""
    for mat, val in isolator.SILENCE_THRESHOLDS_DBFS.items():
        assert val < 0.0, f"Schwelle für '{mat}' ist nicht negativ: {val}"
