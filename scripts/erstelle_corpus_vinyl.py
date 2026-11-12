#!/usr/bin/env python3
"""D-K0-8-Daten (SOTA-M18, Welle D): synthetischer Vinyl-Untercorpus für den Material-Klassifikator.

Warum: WP-7-Bestand 2026-10-09: shellac 58 (Ziel erreicht), aber
**vinyl strukturell 0** — LP-Ära steht unter Schutz, echte Vinyl-Labels
sind ohne Rechteinhaber-Freigabe nicht beschaffbar. Das Register nennt als
Alternativweg ausdrücklich **„oder Synthetik"** (D-K0-8). Dieses Skript
erzeugt deterministisch ≥ 50 synthetische Vinyl-Proben (Label
``material: vinyl``) im Korpus-Schema (``corpus/MANIFEST_SCHEMA.yaml``).

Verfahren (alles lokal, deterministisch, Seed 42, §G5 (GEBOTE.md)):
- Clean-Quelle: MUSDB18-HQ-Mixturen (lizenziertes Trainingskorpus),
  10-s-Mittel-Ausschnitte, 48 kHz mono.
- Vinyl-Rezeptur (realistische Komponenten):
  * echte Vinyl-Rauschbank (``data/vinyl_noise_bank`` — Leerrillen und
    echt beschädigte Aufnahmen) als Träger-Rauschfläche bei geloggtem SNR;
  * deterministische Nadel-Klicker (echte Knistern-Statistik);
  * Vinyl-Bandbreite: mildes Tiefpass-/Hochpass-Band (nicht die
    Sättigungs-Blindheit der defektlast-Metrik berühren — Bandlimit ist
    hier Material-Eigenschaft, §V7 (copilot-instructions.md)-konform
    dokumentiert statt verschwiegen);
  * seltene „Scratch"-Einspieler aus der echten beschädigten Bank.
- Ehrlichkeit: ``source: Synthetic (Aurik Corpus Generator)``,
  ``license: CC0 (generated)``, ``era_year`` ist die **simulierte** Ära
  (LP-Ära 1975), keine echte Provenienz — im Manifest so markiert.

Ausgabe: ``corpus/vinyl/damaged/*.wav`` + ``corpus/vinyl/manifest.yaml``
(Schema ``corpus/MANIFEST_SCHEMA.yaml``, validiert durch
``tests/corpus/test_corpus_integrity.py``).

Usage:
    python3 -B scripts/erstelle_corpus_vinyl.py \
        --musdb-root data/musdb18hq --out corpus/vinyl \
        --count 55 --sekunden 10 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_SR = 48000
_ERA_YEAR = 1975  # simulierte LP-Ära — keine echte Provenienz (Ehrlichkeits-Vertrag)


def _decode_48k(pfad: Path) -> np.ndarray:
    import soundfile as sf

    _r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(pfad), "-ac", "1", "-ar", str(_SR), "-f", "wav", "-"],
        capture_output=True,
        check=True,
    )
    _data, _sr = sf.read(io.BytesIO(_r.stdout), always_2d=False)
    return np.asarray(_data, dtype=np.float32)


def _mittel_ausschnitt(audio: np.ndarray, sekunden: float) -> np.ndarray:
    arr = np.asarray(audio)
    want = int(round(sekunden * _SR))
    if arr.shape[-1] <= want or sekunden <= 0:
        return arr
    start = (arr.shape[-1] - want) // 2
    return arr[..., start : start + want]


def _pcm_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()


def _klicker(x: np.ndarray, rng: np.random.Generator, dichte: float = 3.0) -> np.ndarray:
    """Sparse Nadel-Klicks (Knistern-Komponente), deterministisch über rng."""
    y = np.asarray(x, dtype=np.float64).copy()
    n_klicks = max(1, int(y.size / _SR * dichte))
    for _ in range(n_klicks):
        pos = int(rng.integers(0, max(1, y.size - 20)))
        amp = float(rng.uniform(0.06, 0.30)) * (1.0 if rng.random() < 0.5 else -1.0)
        y[pos : pos + 20] += amp * np.hanning(20)
    return y.astype(np.float32)


def _vinyl_bandlimit(x: np.ndarray) -> np.ndarray:
    """Milde Vinyl-Bandbreite (Bandlimit als Material-Eigenschaft, dokumentiert)."""
    from scipy.signal import butter, sosfiltfilt

    _ny = _SR / 2
    _lp = butter(4, 12_000 / _ny, btype="low", output="sos")
    _hp = butter(2, 35 / _ny, btype="high", output="sos")
    y = sosfiltfilt(_lp, np.asarray(x, dtype=np.float64))
    y = sosfiltfilt(_hp, y)
    return y.astype(np.float32)


def _noise_flaeche(laenge: int, bank: list[Path], rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Echte Vinyl-Rauschfläche aus der Bank (zufälliger Abschnitt, geloggte Quelle)."""
    _quelle = bank[int(rng.integers(0, len(bank)))]
    _roh = _decode_48k(_quelle)
    if _roh.size <= laenge:
        _start = 0
        _flaeche = np.resize(_roh, laenge)
    else:
        _start = int(rng.integers(0, _roh.size - laenge))
        _flaeche = _roh[_start : _start + laenge]
    _snr_db = float(rng.uniform(6.0, 18.0))
    _snr_lin = 10.0 ** (_snr_db / 20.0)
    _ziel_rms = float(np.sqrt((_flaeche**2).mean()))
    if _ziel_rms <= 1e-12:
        _ziel_rms = 1e-6
    _signal_rms = 0.18  # Musik-Pegel-Ziel — deterministisch, keine Mess-Rückkopplung
    _flaeche = _flaeche * (_signal_rms / _snr_lin / _ziel_rms)
    return _flaeche.astype(np.float32), {
        "quelle": _quelle.name,
        "start_sample": _start,
        "snr_db": round(_snr_db, 3),
    }


def _vinyl_rezept(clean: np.ndarray, bank: list[Path], rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    _band = _vinyl_bandlimit(clean)
    _noise, _nparams = _noise_flaeche(_band.size, bank, rng)
    _y = np.asarray(_band, dtype=np.float64) + _noise.astype(np.float64)
    _y = _klicker(_y.astype(np.float32), rng)
    # seltener Scratch-Einspieler aus der echt-beschädigten Bank
    _scratch = bool(rng.random() < 0.25)
    if _scratch:
        _kand = [p for p in bank if "echt_beschaedigt" in p.name]
        if _kand:
            _s = _decode_48k(_kand[int(rng.integers(0, len(_kand)))])
            if _s.size > _SR:
                _off = int(rng.integers(0, _s.size - _SR))
                _y[:_SR] += 0.25 * np.asarray(_s[_off : _off + _SR], dtype=np.float64)
    _peak = float(np.max(np.abs(_y))) or 1.0
    _y = _y * (0.97 / _peak)
    return _y.astype(np.float32), {"klicks_dichte": 3.0, "scratch": _scratch, **_nparams}


def _musdb_mixturen(musdb_root: Path) -> list[Path]:
    _mixtures: list[Path] = []
    for _split in ("train", "test"):
        _split_dir = musdb_root / _split
        if not _split_dir.is_dir():
            continue
        for _track in sorted(_split_dir.iterdir()):
            _mix = _track / "mixture.wav"
            if _mix.is_file():
                _mixtures.append(_mix)
    return _mixtures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    ap.add_argument("--out", type=Path, default=Path("corpus/vinyl"))
    ap.add_argument("--noise-bank", type=Path, default=Path("data/vinyl_noise_bank"))
    ap.add_argument("--count", type=int, default=55)
    ap.add_argument("--sekunden", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import soundfile as sf
    import yaml

    _bank = sorted(p for p in args.noise_bank.glob("*.wav") if p.is_file())
    if not _bank:
        logger.error("keine Vinyl-Rauschbank unter %s", args.noise_bank)
        return 2
    _mixtures = _musdb_mixturen(args.musdb_root)
    if not _mixtures:
        logger.error("keine MUSDB-Mixturen unter %s", args.musdb_root)
        return 3

    _rng = np.random.default_rng(args.seed)
    _out_damaged = args.out / "damaged"
    _out_damaged.mkdir(parents=True, exist_ok=True)

    _entries: list[dict] = []
    _genre_pool = ("jazz", "rock", "pop", "blues", "soul")
    for _idx in range(args.count):
        _mix = _mixtures[_idx % len(_mixtures)]
        _clean = _mittel_ausschnitt(_decode_48k(_mix), args.sekunden)
        _damaged, _params = _vinyl_rezept(_clean, _bank, _rng)
        _stamm = f"vinyl_synth_{_idx:03d}_{_mix.parent.name[:28]}"
        _wav = _out_damaged / f"{_stamm}.wav"
        sf.write(_wav, _damaged, _SR, subtype="FLOAT")
        _entries.append(
            {
                "file": f"damaged/{_wav.name}",
                "duration_s": round(float(_damaged.size / _SR), 3),
                "sample_rate": _SR,
                "material": "vinyl",
                "era_year": _ERA_YEAR,
                "genre": _genre_pool[_idx % len(_genre_pool)],
                "defect_types": ["crackle", "surface_noise"],
                "source": "Synthetic (Aurik Corpus Generator, vinyl recipe)",
                "source_attribution": "Synthetic Corpus Generator (CC0); vinyl_noise_bank + MUSDB18-HQ",
                "license": "CC0 (generated)",
                "checksum_sha256": _pcm_sha256(_damaged),
                "synthetic": True,
                "recipe_params": _params,
            }
        )
        logger.info("%s (quelle=%s, snr=%.1f dB)", _wav.name, _params.get("quelle", "?"), _params.get("snr_db", 0.0))

    # Bestands-Manifest nicht zerstören (D-K3-80-Klasse): vorhandene
    # Einträge laden und die neuen anhängen (dedupe per file).
    _existing: list[dict] = []
    _manifest_pfad = args.out / "manifest.yaml"
    if _manifest_pfad.is_file():
        try:
            import yaml as _yaml

            _alt = _yaml.safe_load(_manifest_pfad.read_text(encoding="utf-8")) or {}
            _existing = list(_alt.get("entries", []))
        except Exception as _exc:  # §V6 (copilot-instructions.md): nicht still abbrechen
            logger.warning("Alt-Manifest nicht lesbar (%s) — Neuanlage", _exc)
    _bekannt = {str(_e.get("file")) for _e in _existing}
    _neu = [e for e in _entries if str(e.get("file")) not in _bekannt]
    if not _neu:
        logger.info("keine neuen Einträge (alle Dateien bereits manifestiert) — Manifest unverändert")
        return 0
    _entries = _existing + _neu
    _manifest = {
        "corpus_version": "1.0.0",
        "material": "vinyl",
        "description": (
            "Vinyl-Untercorpus: Bestands-Einträge (CC0-Synthetik/Validierung) + "
            f"{len(_neu)} deterministische LP-Rezept-Proben (D-K0-8, SOTA-M18, Seed 42) — "
            "echte Vinyl-Rauschbank + Klicker + Bandlimit. era_year ist die SIMULIERTE "
            "Ära (keine echte Provenienz, Lizenz-unbedenklich)."
        ),
        "entries": _entries,
    }
    with _manifest_pfad.open("w", encoding="utf-8") as _fh:
        yaml.safe_dump(_manifest, _fh, sort_keys=False, allow_unicode=True)

    logger.info("geschrieben: %d Einträge gesamt (%d neu) + %s", len(_entries), len(_neu), _manifest_pfad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
