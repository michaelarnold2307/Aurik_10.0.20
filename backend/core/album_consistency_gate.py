"""Album-Konsistenz-Gate — berechnet Referenzwerte aus Track 1, kalibriert Tracks 2-N."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_TOLERANCE_LUFS: float = 1.0
_TOLERANCE_TILT: float = 1.5
_TOLERANCE_WIDTH: float = 0.10
_MAX_GAIN_DB: float = 4.0
_MAX_TILT_CORRECTION: float = 3.0


class AlbumConsistencyGate:
    """Selbstkalibrierung: Track 1 = Preset, Tracks 2-N = kalibriert."""

    def __init__(self):
        self._ref_lufs = None
        self._ref_tilt = None
        self._ref_width = None
        self._track_count = 0

    def set_reference(self, track1_lufs=-16.0, track1_tilt=-3.0, track1_width=0.70):
        self._ref_lufs = track1_lufs
        self._ref_tilt = track1_tilt
        self._ref_width = track1_width
        self._track_count = 1
        logger.info(
            "Album-Gate: Track 1 = Referenz (LUFS=%.1f, Tilt=%.1f, Width=%.2f)", track1_lufs, track1_tilt, track1_width
        )

    def calibrate_track(self, track_lufs=-18.0, track_tilt=-4.0, track_width=0.65):
        if self._ref_lufs is None:
            return {"gain_db": 0.0, "tilt_correction": 0.0, "width_adjust": 0.0}

        _gain_db = float(np.clip(self._ref_lufs - track_lufs, -_MAX_GAIN_DB, _MAX_GAIN_DB))
        _gain_db = 0.0 if abs(_gain_db) < _TOLERANCE_LUFS else _gain_db

        _tilt_corr = float(np.clip(self._ref_tilt - track_tilt, -_MAX_TILT_CORRECTION, _MAX_TILT_CORRECTION))
        _tilt_corr = 0.0 if abs(_tilt_corr) < _TOLERANCE_TILT else _tilt_corr

        _width_adj = float(np.clip(self._ref_width - track_width, -0.20, 0.20))
        _width_adj = 0.0 if abs(_width_adj) < _TOLERANCE_WIDTH else _width_adj

        self._track_count += 1

        if _gain_db or _tilt_corr or _width_adj:
            logger.info(
                "Album-Gate Track %d: Gain%+.1fdB Tilt%+.1f Width%+.2f",
                self._track_count,
                _gain_db,
                _tilt_corr,
                _width_adj,
            )

        return {"gain_db": _gain_db, "tilt_correction": _tilt_corr, "width_adjust": _width_adj}
