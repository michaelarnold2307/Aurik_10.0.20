#!/usr/bin/env python3
"""F9-Identitäts-Gate + W-8-Nicht-Messbar-Markierung (Vertragstests, modell-frei)."""

from __future__ import annotations

import numpy as np

import scripts.verify_f9_identity_gate as f9
from scripts.b3_komposition_suche import KANDIDATEN, NICHT_MESSBAR


class TestF9Empfehlung:
    def test_klare_trennung_empfiehlt_schwelle(self) -> None:
        _gleiche = [0.9, 0.92, 0.88, 0.91, 0.9]
        _fremde = [0.4, 0.42, 0.38, 0.41, 0.4]
        _b = f9.empfehlung(_gleiche, _fremde)
        assert _b["margin"] >= f9._MARGIN_MIN
        assert _b["empfohlene_schwelle"] is not None
        assert _b["empfehlung"].startswith("tragfaehig")

    def test_ueberlappung_keine_schwelle(self) -> None:
        _gleiche = [0.7, 0.5, 0.9, 0.4, 0.8]
        _fremde = [0.6, 0.55, 0.45, 0.7, 0.5]
        _b = f9.empfehlung(_gleiche, _fremde)
        assert _b["empfohlene_schwelle"] is None
        assert "hoer-signierpflichtig" in _b["empfehlung"]

    def test_kosinus_grenzen_und_determinismus(self) -> None:
        _a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        _b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert abs(f9.kosinus(_a, _a) - 1.0) < 1e-6
        assert abs(f9.kosinus(_a, _b)) < 1e-6
        assert f9.kosinus(_a, _b) == f9.kosinus(_a, _b)


class TestW8NichtMessbar:
    def test_w8_ist_nicht_mehr_kandidat(self) -> None:
        assert "W-8" not in KANDIDATEN
        assert "W-8" in NICHT_MESSBAR
        assert "D-K3-92" in NICHT_MESSBAR["W-8"]

    def test_messbare_kandidaten_unveraendert(self) -> None:
        assert KANDIDATEN == {
            "W-2": "use_scnet_music",
            "W-3": "use_sgmse_musik",
            "W-9": "use_whisper_denoiser",
            "W-13": "use_bigvgan_f3_studio",
        }
