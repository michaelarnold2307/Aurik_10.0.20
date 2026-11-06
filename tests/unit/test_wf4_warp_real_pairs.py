#!/usr/bin/env python3
"""Vertragstests WF-V4-Echt-Pfad (D-K4-7, SOTA-M12).

Deckt die Lücke ab, die 2026-11-06 als Phantom-Verdrahtung aufgefallen ist:
``scripts/train_wf4_warp_head.py`` behauptete, die MUSDB-Paare über
``--real-dir`` zu konsumieren, hatte aber weder das Flag noch einen Lader —
und ``scripts/prepare_wf4_warp_pairs.py`` schrieb keine Tracker-Beobachtung.
Damit wäre R1 der GPU-Kette 4 still auf synthetischen Sinus-Signalen gelaufen
(§III.11 (copilot-instructions.md): Trainingsdaten müssen Musik sein).

Geprüft werden:
  * die F0-Beobachtung (pYIN clean ↔ degradiert) — Formen, Wertebereich,
    Erkennung einer bekannten Streckung;
  * der fail-closed-Lader (Manifest/Schema/Beobachtung/Dateien) — §V6 (copilot-instructions.md);
  * die Formen der Trainings-Tensoren (x_f0, x_conf, x_feat, y_warp, y_conf).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.dsp.wf4_warp_head import N_SPEC_FEAT
from scripts.prepare_wf4_warp_pairs import (
    SCHEMA,
    TARGET_FPS,
    TARGET_SR,
    beobachte_f0,
)
from scripts.train_wf4_warp_head import PAIRS_SCHEMA, lade_real_paare

_N_FRAMES = 500  # 2,5 s bei 200 fps — Vertrag der Paar-Dateien
_SEG_N = 2 * TARGET_SR // 2  # 1,0 s reicht für die Tracker-Tests


def _sine(freq_hz: float, n: int, sr: int = TARGET_SR, amp: float = 0.5) -> np.ndarray:
    """Harmonischer Ton (Grundton + 2 Obertöne).

    Ein reiner Sinus hat keine Obertonstruktur; pYIN markiert ihn teilweise als
    „unvoiced“. Musik-Material ist harmonisch, deshalb hier 3 Teiltöne.
    """
    t = np.arange(n, dtype=np.float64) / float(sr)
    x = (
        amp * np.sin(2 * np.pi * freq_hz * t)
        + 0.5 * amp * np.sin(2 * np.pi * 2.0 * freq_hz * t)
        + 0.33 * amp * np.sin(2 * np.pi * 3.0 * freq_hz * t)
    )
    return x.astype(np.float32)


def _strecke(audio: np.ndarray, oct_log2: float) -> np.ndarray:
    """Zeit-Transposition um ``oct_log2`` (positiv ⇒ höhere Tonhöhe)."""
    n = audio.size
    pos = np.arange(n, dtype=np.float64) * (2.0**oct_log2)
    pos = np.clip(pos, 0.0, float(n - 1))
    return np.interp(pos, np.arange(n, dtype=np.float64), audio.astype(np.float64)).astype(np.float32)


# ── 1. F0-Beobachtung ────────────────────────────────────────────────────────


def test_beobachte_f0_formen_und_wertebereich() -> None:
    """Ohne Warp ist die beobachtete Abweichung ≈ 0; Konfidenz in [0, 1]."""
    clean = _sine(220.0, _SEG_N)
    x_f0, x_conf = beobachte_f0(clean, clean.copy(), n_frames=_N_FRAMES)

    assert x_f0.shape == (_N_FRAMES,)
    assert x_conf.shape == (_N_FRAMES,)
    assert x_f0.dtype == np.float32 and x_conf.dtype == np.float32
    assert np.all(np.isfinite(x_f0)) and np.all(np.isfinite(x_conf))
    assert float(x_conf.min()) >= 0.0 and float(x_conf.max()) <= 1.0
    # Ein harmonischer Ton ist stimmhaft ⇒ es MUSS Beobachtungen geben (kein No-Op).
    assert float(np.mean(x_conf > 0.0)) > 0.2
    assert float(x_conf.max()) > 0.5
    # Keine Degradation ⇒ Abweichung nahe 0 (Tracker-Rundung erlaubt ±1 Cent).
    assert float(np.max(np.abs(x_f0))) < 0.02


def test_beobachte_f0_erkennt_bekannte_streckung() -> None:
    """Eine +0,05-log2-Streckung muss als ≈ +0,05 erkannt werden (Vorzeichen fest)."""
    clean = _sine(220.0, _SEG_N)
    degraded = _strecke(clean, 0.05)

    x_f0, x_conf = beobachte_f0(clean, degraded, n_frames=_N_FRAMES)

    beobachtet = x_f0[x_conf > 0.0]
    assert beobachtet.size > 0, "keine stimmhaften Frames — Tracker-Beobachtung fehlt"
    median = float(np.median(beobachtet))
    assert 0.02 < median < 0.08, f"erwartete ≈ +0,05 log2, gemessen {median:.4f}"
    # Vorzeichen: höher gestimmt ⇒ positive Abweichung.
    assert float(np.mean(beobachtet > 0.0)) > 0.9


def test_beobachte_f0_kurzes_signal_ist_fail_closed() -> None:
    """Zu kurzes Signal ⇒ Null-Beobachtung statt Exception (§0a/§V6 (copilot-instructions.md))."""
    kurz = _sine(220.0, 256)
    x_f0, x_conf = beobachte_f0(kurz, kurz.copy(), n_frames=32)
    assert x_f0.shape == (32,) and x_conf.shape == (32,)
    assert not np.any(x_f0) and not np.any(x_conf)


# ── 2. Fail-closed-Lader ────────────────────────────────────────────────────


def _schreibe_paar(root: Path, name: str, *, n: int = 64, mit_f0: bool = True) -> None:
    for sub in ("audio", "warp", "f0", "conf") if mit_f0 else ("audio", "warp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    audio = _sine(220.0, int(n * TARGET_SR / TARGET_FPS))
    sf.write(str(root / "audio" / f"{name}.wav"), audio, TARGET_SR, subtype="FLOAT")
    np.save(str(root / "warp" / f"{name}.npy"), np.zeros(n, dtype=np.float32))
    if mit_f0:
        np.save(str(root / "f0" / f"{name}.npy"), np.zeros(n, dtype=np.float32))
        np.save(str(root / "conf" / f"{name}.npy"), np.ones(n, dtype=np.float32))


def _schreibe_manifest(root: Path, paare: list[str], *, schema: str = SCHEMA, mit_f0: bool = True) -> None:
    manifest = {
        "schema": schema,
        "seed": 42,
        "target_sr": TARGET_SR,
        "target_fps": TARGET_FPS,
        "anzahl": len(paare),
        "f0_beobachtung": {"tracker": "pyin_fast"} if mit_f0 else None,
        "paare": [{"datei": p} for p in paare],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_lade_real_paare_formen(tmp_path: Path) -> None:
    """Geladene Paare liefern die Tensoren, die ``train()`` stapelt."""
    _schreibe_paar(tmp_path, "pair_0000")
    _schreibe_paar(tmp_path, "pair_0001")
    _schreibe_manifest(tmp_path, ["pair_0001", "pair_0000"])  # unsortiert ⇒ Lader sortiert

    paare = lade_real_paare(tmp_path)

    assert len(paare) == 2
    p = paare[0]
    assert p["x_f0"].shape == (64,)
    assert p["x_conf"].shape == (64,)
    assert p["x_feat"].shape == (64, N_SPEC_FEAT)
    assert p["y_warp"].shape == (64,)
    assert p["y_conf"].shape == (64,)
    assert float(p["y_conf"].min()) >= 0.0 and float(p["y_conf"].max()) <= 1.0
    for key in ("x_f0", "x_conf", "x_feat", "y_warp", "y_conf"):
        assert np.all(np.isfinite(p[key])), key


def test_lade_real_paare_ohne_manifest_ist_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest fehlt"):
        lade_real_paare(tmp_path)


def test_lade_real_paare_falsches_schema(tmp_path: Path) -> None:
    _schreibe_paar(tmp_path, "pair_0000")
    _schreibe_manifest(tmp_path, ["pair_0000"], schema="aurik.irgendwas/9")
    with pytest.raises(ValueError, match="Schema"):
        lade_real_paare(tmp_path)


def test_lade_real_paare_ohne_beobachtung_ist_fail_closed(tmp_path: Path) -> None:
    """Paare aus ``--ohne-f0`` dürfen NICHT als Echt-Training durchgehen (§III.11)."""
    _schreibe_paar(tmp_path, "pair_0000", mit_f0=False)
    _schreibe_manifest(tmp_path, ["pair_0000"], mit_f0=False)
    with pytest.raises(ValueError, match="Tracker-Beobachtung"):
        lade_real_paare(tmp_path)


def test_lade_real_paare_unvollstaendig_ist_fail_closed(tmp_path: Path) -> None:
    _schreibe_paar(tmp_path, "pair_0000")
    _schreibe_manifest(tmp_path, ["pair_0000"])
    (tmp_path / "conf" / "pair_0000.npy").unlink()
    with pytest.raises(FileNotFoundError, match="conf"):
        lade_real_paare(tmp_path)


def test_paar_schema_ist_synchron() -> None:
    """Datensatz- und Trainer-Seite müssen dasselbe Schema führen (§G9 (GEBOTE.md))."""
    assert PAIRS_SCHEMA == SCHEMA == "aurik.wf4_warp_pairs/1"


# ── 3. Determinismus der Datensatz-Erzeugung ───────────────────────────────


def test_erzeuge_paare_deterministisch_ueber_pcm(tmp_path: Path) -> None:
    """Zwei Läufe (Seed 42) müssen identische PCM-/Warp-/Beobachtungs-Hashes liefern.

    Hintergrund: der WAV-Container ist byte-instabil (libsndfile-``PEAK``-Chunk,
    gemessen 1 Byte Differenz bei bit-identischen Audiodaten). Der verbindliche
    Provenienzwert ist deshalb ``sha256_pcm`` (§G5 (GEBOTE.md)).
    """
    from scripts.prepare_wf4_warp_pairs import erzeuge_paare

    musdb = tmp_path / "musdb"
    (musdb / "train" / "T1").mkdir(parents=True)
    sf.write(
        str(musdb / "train" / "T1" / "mixture.wav"),
        _sine(220.0, int(3.0 * TARGET_SR)),
        TARGET_SR,
        subtype="FLOAT",
    )

    m1 = erzeuge_paare(musdb, tmp_path / "a", anzahl=2, seg_s=1.0, seed=42)
    m2 = erzeuge_paare(musdb, tmp_path / "b", anzahl=2, seg_s=1.0, seed=42)

    assert m1["f0_beobachtung"] is not None and m2["f0_beobachtung"] is not None
    assert len(m1["paare"]) == len(m2["paare"]) == 2
    for p1, p2 in zip(m1["paare"], m2["paare"], strict=True):
        assert p1["sha256_pcm"] == p2["sha256_pcm"], "PCM nicht deterministisch (§G5 (GEBOTE.md))"
        assert p1["sha256_npy"] == p2["sha256_npy"]
        assert p1["sha256_f0"] == p2["sha256_f0"]
        assert p1["sha256_conf"] == p2["sha256_conf"]
        assert 0.0 <= p1["f0_beobachtet_anteil"] <= 1.0
