#!/usr/bin/env python3
"""Vertragstests für scripts/prepare_tfgrid_pairs.py (D-K4-2, SOTA-M9).

Determinismus §G5 (copilot-instructions.md), Paarungs-Integrität,
Manifest-Schema, SNR-Plausibilität. Synthetische Mini-Korpus-Struktur
(wie tests/unit/test_m1b_targeted_retry.py) — Test-Fixtures sind keine
Trainingsdaten (K1 gilt für den Trainingskorpus, nicht für Testsignale).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.prepare_tfgrid_pairs import (
    SCHEMA,
    TARGET_SR,
    _lade_wav_mono,
    _snr_skalieren,
    _zu_mono,
    erzeuge_paare,
)

_SEG_N = TARGET_SR  # 1 s Segmente in den Tests


def _messe_snr_db(clean: np.ndarray, rauschen: np.ndarray) -> float:
    p_clean = float(np.mean(np.asarray(clean, dtype=np.float64) ** 2))
    p_r = float(np.mean(np.asarray(rauschen, dtype=np.float64) ** 2))
    return 10.0 * np.log10(p_clean / p_r)


@pytest.fixture()
def mini_corpus(tmp_path: Path) -> Path:
    """Mini-MUSDB-Struktur: train/<Track>/mixture.wav (16 s, 48 kHz, 2 ch)."""
    rng = np.random.default_rng(7)
    for name in ("TrackA", "TrackB"):
        d = tmp_path / "musdb" / "train" / name
        d.mkdir(parents=True)
        t = np.arange(16 * TARGET_SR) / TARGET_SR
        links = 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.02 * rng.standard_normal(t.size)
        rechts = 0.3 * np.sin(2 * np.pi * 660.0 * t) + 0.02 * rng.standard_normal(t.size)
        sf.write(str(d / "mixture.wav"), np.stack([links, rechts], axis=1), TARGET_SR)
    return tmp_path / "musdb"


def _lauf(mini_corpus: Path, out: Path, seed: int = 42, anzahl: int = 4) -> dict:
    # noise_wavs=() ⇒ nur farbiges Rauschen — Tests unabhängig von Repo-Assets.
    return erzeuge_paare(
        mini_corpus,
        out,
        anzahl=anzahl,
        seg_s=1.0,
        seed=seed,
        noise_wavs=(),
    )


def test_determinismus_gleicher_seed_byte_identisch(mini_corpus: Path, tmp_path: Path) -> None:
    """§G5 (copilot-instructions.md): gleicher Seed ⇒ identische PCM-Samples.
    Vergleich über sha256_pcm — der WAV-Container ist durch den
    libsndfile-PEAK-Chunk byte-instabil (Determinismus-Befund Handover Punkt 7)."""
    m1 = _lauf(mini_corpus, tmp_path / "lauf1")
    m2 = _lauf(mini_corpus, tmp_path / "lauf2")
    assert [e["datei"] for e in m1["paare"]] == [e["datei"] for e in m2["paare"]]
    for a, b in zip(m1["paare"], m2["paare"]):
        assert a["sha256_pcm_clean"] == b["sha256_pcm_clean"]
        assert a["sha256_pcm_noisy"] == b["sha256_pcm_noisy"]


def test_determinismus_unterschiedlicher_seed_unterschiedlich(mini_corpus: Path, tmp_path: Path) -> None:
    _lauf(mini_corpus, tmp_path / "s1", seed=42)
    _lauf(mini_corpus, tmp_path / "s2", seed=43)
    f1 = sorted((tmp_path / "s1" / "clean").glob("*.wav"))
    f2 = sorted((tmp_path / "s2" / "clean").glob("*.wav"))
    assert any(a.read_bytes() != b.read_bytes() for a, b in zip(f1, f2))


def test_paarungs_integritaet(mini_corpus: Path, tmp_path: Path) -> None:
    """Jede clean-Datei hat ein noisy-Pendant mit GLEICHEM Dateinamen."""
    manifest = _lauf(mini_corpus, tmp_path / "paare")
    clean = {f.name for f in (tmp_path / "paare" / "clean").glob("*.wav")}
    noisy = {f.name for f in (tmp_path / "paare" / "noisy").glob("*.wav")}
    assert clean == noisy
    assert len(clean) == manifest["anzahl"] == 4


def test_manifest_schema(mini_corpus: Path, tmp_path: Path) -> None:
    manifest = _lauf(mini_corpus, tmp_path / "manifest")
    geladen = json.loads((tmp_path / "manifest" / "manifest.json").read_text(encoding="utf-8"))
    assert geladen == manifest
    assert manifest["schema"] == SCHEMA == "aurik.tfgrid_pairs/1"
    assert manifest["seed"] == 42
    assert manifest["anzahl"] == 4
    assert manifest["target_sr"] == TARGET_SR
    assert manifest["seg_s"] == 1.0
    assert sum(manifest["snr_verteilung"].values()) == 4
    assert set(manifest["snr_verteilung"]) <= {"0.0", "5.0", "10.0", "15.0", "20.0"}
    for eintrag in manifest["paare"]:
        for schluessel in (
            "datei",
            "quelle",
            "offset_s",
            "snr_db",
            "rauschquelle",
            "rauschquelle_art",
            "sha256_wav_clean",
            "sha256_wav_noisy",
            "sha256_pcm_clean",
            "sha256_pcm_noisy",
        ):
            assert schluessel in eintrag, f"Manifest-Feld fehlt: {schluessel}"


def test_snr_stufen_kontrolliert(mini_corpus: Path, tmp_path: Path) -> None:
    """Gemessene SNR jeder Datei ≈ Soll-SNR (±1,5 dB, Rundungs-/Format-Toleranz)."""
    _lauf(mini_corpus, tmp_path / "snr", anzahl=5)
    manifest = json.loads((tmp_path / "snr" / "manifest.json").read_text(encoding="utf-8"))
    for eintrag in manifest["paare"]:
        name = eintrag["datei"]
        clean, _ = sf.read(str(tmp_path / "snr" / "clean" / name))
        noisy, _ = sf.read(str(tmp_path / "snr" / "noisy" / name))
        gemessen = _messe_snr_db(clean, noisy - clean)
        assert abs(gemessen - eintrag["snr_db"]) <= 1.5, f"{name}: SNR {gemessen:.2f} dB statt {eintrag['snr_db']} dB"


def test_snr_skalierung_exakt() -> None:
    rng = np.random.default_rng(1)
    clean = rng.standard_normal(_SEG_N).astype(np.float32)
    rauschen = rng.standard_normal(_SEG_N).astype(np.float32)
    skaliert = _snr_skalieren(clean, rauschen, 10.0)
    assert abs(_messe_snr_db(clean, skaliert) - 10.0) < 0.01


def test_zu_mono_layout_invariante() -> None:
    """Stereo-Layout-Invariante (AGENTS.md §3): (N,2) und (2,N) liefern Mono (N,)."""
    rng = np.random.default_rng(3)
    nc = rng.standard_normal((_SEG_N, 2)).astype(np.float32)
    mono_aus_nc = _zu_mono(nc)
    assert mono_aus_nc.shape == (_SEG_N,)
    mono_aus_cn = _zu_mono(nc.T.copy())  # (2, N) channels-first
    assert mono_aus_cn.shape == (_SEG_N,)
    assert np.allclose(mono_aus_nc, mono_aus_cn, atol=1e-6)


def test_lade_wav_mono_resampling(tmp_path: Path) -> None:
    """Lader liefert Mono auf TARGET_SR unabhängig von Quell-SR/-Kanälen."""
    p = tmp_path / "mix44k.wav"
    t = np.arange(2 * 44100) / 44100
    stereo = np.stack([np.sin(2 * np.pi * 220.0 * t)] * 2, axis=1)
    sf.write(str(p), stereo, 44100)
    mono = _lade_wav_mono(p, TARGET_SR)
    assert mono.ndim == 1
    assert abs(len(mono) - 2 * TARGET_SR) <= TARGET_SR * 0.01


def test_stille_guard_ueberspringt_schweigende_fenster(tmp_path: Path) -> None:
    """Stille-Guard (M9-Datenexpansion 2026-11-14): ein Korpus mit einem
    stillen und einem lauten Track bricht nicht mehr mit „Null-Leistung“ ab —
    der Re-Draw landet deterministisch auf einem nicht-schweigenden Fenster."""
    root = tmp_path / "musdb"
    still = root / "train" / "Silent"
    laut = root / "train" / "Loud"
    for d in (still, laut):
        d.mkdir(parents=True)
    sf.write(str(still / "mixture.wav"), np.zeros((4 * TARGET_SR, 2), dtype=np.float32), TARGET_SR)
    t = np.arange(4 * TARGET_SR) / TARGET_SR
    ton = np.stack([0.3 * np.sin(2 * np.pi * 440.0 * t)] * 2, axis=1)
    sf.write(str(laut / "mixture.wav"), ton.astype(np.float32), TARGET_SR)

    manifest = erzeuge_paare(root, tmp_path / "out", anzahl=4, seg_s=1.0, seed=42, noise_wavs=())
    assert manifest["anzahl"] == 4
    for paar in manifest["paare"]:
        p = tmp_path / "out" / "clean" / paar["datei"]
        x = _lade_wav_mono(p, TARGET_SR)
        assert float(np.sqrt(np.mean(x**2))) > 1e-4, f"{paar['datei']} ist schweigend (Stille-Guard versagt)"


def test_stille_guard_determinismus(tmp_path: Path) -> None:
    """§G5 (copilot-instructions.md): der Re-Draw bleibt rein aus (seed, i) —
    zwei Läufe über denselben Stille-Korpus sind byte-identisch."""
    root = tmp_path / "musdb"
    still = root / "train" / "Silent"
    laut = root / "train" / "Loud"
    for d in (still, laut):
        d.mkdir(parents=True)
    sf.write(str(still / "mixture.wav"), np.zeros((4 * TARGET_SR, 2), dtype=np.float32), TARGET_SR)
    t = np.arange(4 * TARGET_SR) / TARGET_SR
    sf.write(
        str(laut / "mixture.wav"),
        np.stack([0.3 * np.sin(2 * np.pi * 440.0 * t)] * 2, axis=1).astype(np.float32),
        TARGET_SR,
    )
    ma = erzeuge_paare(root, tmp_path / "a", anzahl=4, seg_s=1.0, seed=42, noise_wavs=())
    mb = erzeuge_paare(root, tmp_path / "b", anzahl=4, seg_s=1.0, seed=42, noise_wavs=())
    # sha256_pcm statt WAV-Bytes: der Container ist durch den libsndfile-
    # PEAK-Chunk byte-instabil (Determinismus-Befund Handover Punkt 7).
    assert [e["datei"] for e in ma["paare"]] == [e["datei"] for e in mb["paare"]]
    for x, y in zip(ma["paare"], mb["paare"]):
        assert x["sha256_pcm_clean"] == y["sha256_pcm_clean"]
        assert x["sha256_pcm_noisy"] == y["sha256_pcm_noisy"]
