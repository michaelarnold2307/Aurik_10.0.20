"""Tests für den EAR-VAE v1↔v2-Vergleichs-Harness (D-K0-9, LUECKEN-MATRIX D.1c).

Fokus: reine Messlogik ohne Modell-Artefakte - Fail-closed bei fehlenden
v2-Gewichten, Blind-Paarung (§G5 (copilot-instructions.md)), Never-worsen-
Auswertung mit JND-Toleranz, Vorzeichen-Test (scripts/ab_harness.py) und
Messdatei-Schema. Die Modell-Renderpfade selbst sind lazy und hier bewusst
nicht belegt (Gewichte fehlen - siehe Report 2026-11-03).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.ear_vae_v1_v2_vergleich import (
    JND_TOLERANZ,
    auswertung,
    fenster_plan,
    messung_schreiben,
    v2_checkpoints_finden,
    v2_verfuegbarkeit,
    vergleich_messen,
)

# ── Fail-closed: v2 ohne Gewichte ────────────────────────────────────────────


def test_v2_verfuegbarkeit_ohne_gewichte_fail_closed(tmp_path: Path) -> None:
    upstream = tmp_path / "models" / "ear_vae2_upstream"
    upstream.mkdir(parents=True)
    (upstream / "README.md").write_text("# ear-VAE2", encoding="utf-8")
    v2 = v2_verfuegbarkeit(tmp_path / "models")  # Parameter = models-Root (Modul-Dokstring)
    assert v2["verfuegbar"] is False
    assert v2["checkpoints"] == []
    assert "Gewicht" in v2["grund"] and "huggingface" in v2["grund"].lower()
    assert "ML_ARTIFACT_FINGERPRINTS" in v2["grund"], "Quelle im Grund zitieren (ID-/Quell-Disziplin)"
    assert v2["quelle"] == "scripts/ear_vae_v1_v2_vergleich.py v2_verfuegbarkeit"


def test_v2_checkpoints_finden_entdeckt_kandidaten(tmp_path: Path) -> None:
    upstream = tmp_path / "models" / "ear_vae2_upstream"
    (upstream / "checkpoints").mkdir(parents=True)
    ckpt = upstream / "checkpoints" / "ear_vae2.pt"
    ckpt.write_bytes(b"PT")  # Präsenz-Check; Laden ist Sache des Laufs
    gefunden = v2_checkpoints_finden(tmp_path / "models")  # Parameter = models-Root
    assert ckpt in gefunden
    v2 = v2_verfuegbarkeit(tmp_path / "models")
    assert v2["verfuegbar"] is True
    assert v2["grund"] is None


# ── Fenster-Plan (§G5 (copilot-instructions.md): deterministisch) ────────────


def test_fenster_plan_deterministisch() -> None:
    corpus = [{"path": "test_audio/tape/cassette_1980s_wow.wav", "material": "cassette"}]
    p1 = fenster_plan(corpus, starts=(0.0, 2.5, 5.0), seconds=2.5)
    p2 = fenster_plan(corpus, starts=(0.0, 2.5, 5.0), seconds=2.5)
    assert p1 == p2, "gleiche Eingaben ⇒ identischer Fenster-Plan (§G5 (copilot-instructions.md))"
    assert len(p1) == 3
    assert p1[0] == {
        "clip": "cassette_1980s_wow",
        "pfad": "test_audio/tape/cassette_1980s_wow.wav",
        "start_s": 0.0,
        "seconds": 2.5,
    }


# ── Blind messen + auswerten ─────────────────────────────────────────────────


def _fenster(n: int = 3) -> list[tuple[str, np.ndarray, int]]:
    rng = np.random.default_rng(0)
    return [(f"clip{i}", rng.standard_normal((64, 2)).astype(np.float32), 48000) for i in range(n)]


def test_vergleich_messen_blind_deterministisch() -> None:
    def v1(audio: np.ndarray, sr: int) -> np.ndarray:
        return audio * 0.9

    def v2(audio: np.ndarray, sr: int) -> np.ndarray:
        return audio * 1.1

    def bewerte(audio: np.ndarray, sr: int) -> tuple[float, float]:
        return 4.0, 0.5

    r1 = vergleich_messen(_fenster(), 42, v1_denoise=v1, v2_denoise=v2, bewerte=bewerte)
    r2 = vergleich_messen(_fenster(), 42, v1_denoise=v1, v2_denoise=v2, bewerte=bewerte)
    assert r1 == r2, "gleicher Seed ⇒ bit-identische Auswertung (§G5 (copilot-instructions.md))"
    assert [z["trial_id"] for z in r1["zeilen"]] == ["clip0__w0", "clip1__w1", "clip2__w2"]


def test_vergleich_messen_entblindung_korrekt() -> None:
    """Die Bewertung sieht nur „links"/„rechts" — die Entblindung muss die
    Messwerte exakt v1/v2 zuordnen (kein Kriterien-Tausch). Wertbasiert statt
    objekt-identisch: der Harness normiert/kopiert die Signale."""

    def v1(audio: np.ndarray, sr: int) -> np.ndarray:
        return np.full_like(audio, 0.5)

    def v2(audio: np.ndarray, sr: int) -> np.ndarray:
        return np.full_like(audio, -0.5)

    def bewerte(audio: np.ndarray, sr: int) -> tuple[float, float]:
        return (5.0, 0.1) if float(np.mean(audio)) > 0.0 else (1.0, 0.9)

    erg = vergleich_messen(_fenster(2), 7, v1_denoise=v1, v2_denoise=v2, bewerte=bewerte)
    assert len(erg["zeilen"]) == 2
    for z in erg["zeilen"]:
        assert (z["mos_v1"], z["last_v1"]) == (5.0, 0.1), "v1-Messwerte bleiben v1 zugeordnet"
        assert (z["mos_v2"], z["last_v2"]) == (1.0, 0.9), "v2-Messwerte bleiben v2 zugeordnet"
        assert set(z) == {"trial_id", "mos_v1", "mos_v2", "last_v1", "last_v2"}


def test_auswertung_never_worsen_und_vorzeichen_test() -> None:
    zeilen = [{"trial_id": f"t{i}", "mos_v1": 4.0, "mos_v2": 4.2, "last_v1": 0.5, "last_v2": 0.5} for i in range(6)]
    erg = auswertung(zeilen)
    assert erg["never_worsen"] is True
    assert erg["vorteile"] == "6/6"
    assert abs(erg["p_wert"] - 0.03125) < 1e-9, "exakter Vorzeichen-Test wie scripts/ab_harness.py"
    assert erg["signifikant"] is True
    assert erg["erfuellt"] is True


def test_auswertung_never_worsen_verletzung_schlaegt_vorteile() -> None:
    zeilen = [
        {"trial_id": "t0", "mos_v1": 4.0, "mos_v2": 4.5, "last_v1": 0.5, "last_v2": 0.4},
        {"trial_id": "t1", "mos_v1": 4.0, "mos_v2": 4.5, "last_v1": 0.5, "last_v2": 0.4},
        # Ein Fenster klar schlechter (-0.5 MOS ≫ JND) ⇒ Never-worsen verletzt.
        {"trial_id": "t2", "mos_v1": 4.5, "mos_v2": 4.0, "last_v1": 0.4, "last_v2": 0.9},
    ]
    erg = auswertung(zeilen)
    assert erg["never_worsen"] is False, "nie-schlechter-als (G-4) dominiert Einzelvorteile"
    assert erg["erfuellt"] is False


def test_auswertung_jnd_toleranz() -> None:
    zeilen = [
        # Innerhalb der Toleranz (JND_TOLERANZ = 0.05, Muster b3_stufenmessung.py).
        {"trial_id": "t0", "mos_v1": 4.0, "mos_v2": 3.97, "last_v1": 0.5, "last_v2": 0.54},
    ]
    erg = auswertung(zeilen, jnd=JND_TOLERANZ)
    assert erg["never_worsen"] is True
    zeilen[0]["mos_v2"] = 3.94  # -0.06 ⇒ außerhalb
    assert auswertung(zeilen, jnd=JND_TOLERANZ)["never_worsen"] is False


# ── Messdatei-Schema ─────────────────────────────────────────────────────────


def test_messung_schreiben_schema_ohne_erfundene_zahlen(tmp_path: Path) -> None:
    ziel = tmp_path / "messung.json"
    payload = messung_schreiben(
        {
            "seed": 42,
            "zeilen": [],
            "never_worsen": None,
            "vorteile": None,
            "p_wert": None,
            "signifikant": None,
            "status": "nicht_ausfuehrbar",
            "technischer_grund": "v2 ohne Gewichte",
        },
        ziel,
    )
    daten = json.loads(ziel.read_text(encoding="utf-8"))
    assert daten == payload
    for schluessel in ("seed", "zeilen", "never_worsen", "vorteile", "p_wert"):
        assert schluessel in daten, f"Schema analog b3-Messungen: {schluessel} fehlt"
    assert daten["status"] == "nicht_ausfuehrbar"
    assert daten["zeilen"] == [] and daten["p_wert"] is None, "keine erfundenen Messzahlen"
    assert daten["quelle"] == "scripts/ear_vae_v1_v2_vergleich.py"


def test_vergleich_messen_ohne_bewerte_laedt_keine_schweren_imports() -> None:
    """Sanity: die reine Logik bleibt numpy-only (Modell-Importe sind lazy) —
    im frischen Interpreter geprueft, weil pytest/conftest selbst torch laden kann."""
    pruef = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent.parent)!r});"
        "import scripts.ear_vae_v1_v2_vergleich;"
        "assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules"
    )
    ergebnis = subprocess.run([sys.executable, "-c", pruef], check=False, capture_output=True, text=True)
    assert ergebnis.returncode == 0, ergebnis.stderr
    assert callable(auswertung) and callable(vergleich_messen)
