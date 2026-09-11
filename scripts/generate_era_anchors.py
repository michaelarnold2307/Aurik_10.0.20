"""
Generiert synthetische ERA-Classifier-Ankerpunkte für Aurik 10.0.0.

Erzeugte Dateien:
    models/era_classifier/era_anchors.npy          — [15 × 512] CLAP-Embedding-Anker
    models/era_classifier/reference_anchors.npz    — 270 spektrale Envelopes (128-dim)
                                                      10 Dekaden × 9 Genres × 3 Materialien

Verwendung:
    python scripts/generate_era_anchors.py

Hinweis:
    Die Anker sind SYNTHETISCH (nicht aus echten Aufnahmen gelernt).
    Sie modellieren physikalisch plausible Merkmale pro Epoche/Genre/Material
    und dienen als Prior für Nearest-Neighbor-Suche wenn LAION-CLAP verfügbar ist.
    Die Qualität der Tier-2-DSP-Klassifikation ist davon unabhängig.

Referenz: §2.14, §2.25 Aurik-9-Spec v10.0.0
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

# Kanonischer Referenzwert: backend/core/reference_anchor_synthesizer.py:MATERIAL_EXPECTED_BW
# (lokal gespiegelt — Skript läuft standalone ohne backend-Import).
MATERIAL_EXPECTED_BW = 20000.0

# ============================================================
# Konfiguration
# ============================================================

DECADES_15 = [
    1890,
    1900,
    1910,
    1920,
    1930,
    1940,
    1950,
    1960,
    1970,
    1980,
    1990,
    2000,
    2010,
    2020,
    2025,
]

GENRES_9 = [
    "schlager",
    "jazz",
    "klassik",
    "oper",
    "rock",
    "folk",
    "elektronisch",
    "hip_hop",
    "unbekannt",
]

MATERIALS_3 = ["tape", "vinyl", "digital"]

# Spektrale Envelopes: 128 Bänder (log-spaced 20–20000 Hz)
N_SPEC_BINS = 128

# CLAP-Embedding-Dimension
CLAP_DIM = 512

# Ausgabe-Verzeichnis
OUT_DIR = pathlib.Path(__file__).parent.parent / "models" / "era_classifier"

# Seed für Reproduzierbarkeit
RNG_SEED = 42

# ============================================================
# Hilfsfunktionen
# ============================================================


def _decade_to_norm(decade: int) -> float:
    """Normalisiert Jahrzehnt auf [0, 1] für parametrisches Spektrum."""
    return (decade - 1890) / (2025 - 1890)


def _spectral_envelope_128(
    decade: int,
    genre: str,
    material: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetische 128-dim spektrale Envelope für eine Epoche/Genre/Material-Kombi.

    Modelliert empirisch beobachtete Charakteristika:
    - Historische Aufnahmen: starker HF-Abfall (Bandbreite begrenzt)
    - Jazz/Klassik: ausgeprägte Mitten, sanfte Bässe
    - Schlager: warme Bässe, moderate Präsenz
    - Elektronisch: flaches Spektrum, starker Sub-Bass
    - Tape: leichter Grundrauschen im HF-Bereich
    - Vinyl: Crackle-Rauschen in 5–12 kHz, Wärme in 200–500 Hz
    - Digital: flaches breitbandiges Spektrum
    """
    # Frequenzachse (log-spaced 20–20000 Hz)
    freqs = np.logspace(math.log10(20), math.log10(20000), N_SPEC_BINS)

    # Basis: Langzeitnäherung an rosa Rauschen (1/f)
    envelope = freqs ** (-0.5)

    t = _decade_to_norm(decade)  # 0 = 1890, 1 = 2025

    # ─── Bandbreiten-Begrenzung nach Epoche ─────────────────────────────────────
    # Ältere Aufnahmen: harter HF-Rolloff
    # Effektive Bandbreiten nach §2.14:
    bw_map = {
        1890: 4000,
        1900: 4500,
        1910: 5000,
        1920: 6500,
        1930: 8000,
        1940: 10000,
        1950: 12000,
        1960: 15000,
        1970: 18000,
        1980: 20000,
        1990: 20000,
        2000: 20000,
        2010: 22000,
        2020: 22000,
        2025: 22000,
    }
    bw_hz = bw_map.get(decade, MATERIAL_EXPECTED_BW)
    rolloff_mask = np.clip(1.0 - (freqs - bw_hz) / (bw_hz * 0.5), 0.0, 1.0)
    rolloff_mask = rolloff_mask**2
    envelope *= rolloff_mask

    # ─── Genre-Anpassungen ───────────────────────────────────────────────────────
    if genre == "schlager":
        # Wärme: Boost 200–500 Hz
        warm_mask = np.exp(-((np.log(freqs) - math.log(350)) ** 2) / (2 * 0.3**2))
        envelope *= 1.0 + 0.4 * warm_mask
    elif genre == "jazz":
        # Mitten: Boost 500–2000 Hz
        mid_mask = np.exp(-((np.log(freqs) - math.log(1000)) ** 2) / (2 * 0.5**2))
        envelope *= 1.0 + 0.3 * mid_mask
    elif genre in ("klassik", "oper"):
        # Gleichmäßiges Spektrum, leichter Hochmitten-Boost
        presence_mask = np.exp(-((np.log(freqs) - math.log(3000)) ** 2) / (2 * 0.4**2))
        envelope *= 1.0 + 0.2 * presence_mask
    elif genre == "rock":
        # Gitarre (3 kHz) und Sub-Bass stark
        sub_mask = np.exp(-((np.log(freqs) - math.log(80)) ** 2) / (2 * 0.3**2))
        guit_mask = np.exp(-((np.log(freqs) - math.log(3000)) ** 2) / (2 * 0.3**2))
        envelope *= 1.0 + 0.5 * sub_mask + 0.3 * guit_mask
    elif genre == "elektronisch":
        # Sub-Bass sehr stark, flattes Hochmittenband
        sub_mask = np.exp(-((np.log(freqs) - math.log(55)) ** 2) / (2 * 0.4**2))
        envelope *= 1.0 + 0.8 * sub_mask
    elif genre == "hip_hop":
        # Starker Sub + Präsenz
        sub_mask = np.exp(-((np.log(freqs) - math.log(70)) ** 2) / (2 * 0.3**2))
        pres_mask = np.exp(-((np.log(freqs) - math.log(4000)) ** 2) / (2 * 0.3**2))
        envelope *= 1.0 + 0.7 * sub_mask + 0.2 * pres_mask

    # ─── Material-Anpassungen ────────────────────────────────────────────────────
    if material == "tape":
        # Tape Hiss: leichtes Rauschen im HF
        tape_hiss = 0.03 * (freqs / 20000) ** 0.5
        envelope += tape_hiss
        # Leichte LF-Erwärmung
        lf_boost = np.exp(-((np.log(freqs) - math.log(200)) ** 2) / (2 * 0.5**2))
        envelope *= 1.0 + 0.1 * lf_boost
    elif material == "vinyl":
        # Vinyl: Wärme 200–500 Hz, Crackle 5–12 kHz, Rumpeln < 30 Hz
        vinyl_crackle = 0.05 * np.exp(-((np.log(freqs) - math.log(8000)) ** 2) / (2 * 0.5**2))
        envelope += vinyl_crackle
        vinyl_warm = np.exp(-((np.log(freqs) - math.log(350)) ** 2) / (2 * 0.4**2))
        envelope *= 1.0 + 0.15 * vinyl_warm
    elif material == "digital":
        # Digital: sauberes flaches Spektrum, kein extra Rauschen
        # Leichte Absenkung unter 30 Hz (Hochpassfilter typisch)
        lf_rolloff = np.clip((freqs - 20) / 30, 0.0, 1.0)
        envelope *= lf_rolloff

    # Normalisierung
    mx = envelope.max()
    if mx > 1e-12:
        envelope /= mx

    # Leichtes stochastisches Rauschen für Realismus (fest skaliert)
    noise_scale = 0.02 + 0.01 * t
    envelope = np.clip(envelope + rng.normal(0, noise_scale, N_SPEC_BINS).astype(np.float32), 0.0, 1.0)

    # Erneut normalisieren
    mx = envelope.max()
    if mx > 1e-12:
        envelope /= mx

    return envelope.astype(np.float32)


def _era_clap_embedding(
    decade: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetisches 512-dim L2-normalisiertes CLAP-Embedding für ein Jahrzehnt.

    Die Embeddings sind so konstruiert, dass:
    - Benachbarte Dekaden ähnliche Embeddings haben (hohe Cosinus-Ähnlichkeit)
    - Dekaden weit auseinander (z.B. 1920 vs. 2010) klar unterschiedlich sind
    - Alle Embeddings L2-normalisiert sind (‖v‖₂ = 1.0)

    Strategie: Linearer Unterraum pro Ära-Gruppe + Orthogonale Dekaden-Komponente.
    """
    t = _decade_to_norm(decade)  # 0.0 – 1.0

    # Basis-Vektor: langsame Drift entlang Unterraum (erste 64 Dims)
    base = np.zeros(CLAP_DIM, dtype=np.float64)
    for k in range(64):
        base[k] = math.sin(t * math.pi * (k + 1) / 64.0) * math.exp(-k / 32.0)

    # Dekaden-spezifische Sinusbasis (Dims 64–127, einzigartig pro Jahrzehnt)
    decade_idx = DECADES_15.index(decade)
    for k in range(64):
        base[64 + k] = math.cos(decade_idx * math.pi / 7.5 + k * math.pi / 32.0) * 0.5

    # Stochastischer Rest (Dims 128–511): geringe Zufallskomponente pro Jahrzehnt
    rng_dec = np.random.default_rng(RNG_SEED + decade_idx * 100)
    base[128:] = rng_dec.normal(0, 0.1, CLAP_DIM - 128)

    # L2-Normalisierung
    norm = np.linalg.norm(base)
    if norm < 1e-9:
        base = rng.normal(0, 1, CLAP_DIM)
        base /= np.linalg.norm(base)
    else:
        base /= norm

    return base.astype(np.float32)


# ============================================================
# Haupt-Generierungsfunktion
# ============================================================


def generate_era_anchors() -> None:
    """Generiert era_anchors.npy und reference_anchors.npz."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # ── 1. era_anchors.npy ─────────────────────────────────────────────────────
    print("Generiere era_anchors.npy …")
    anchors = np.stack([_era_clap_embedding(d, rng) for d in DECADES_15])  # shape: [15, 512]

    # Konsistenzprüfung: alle L2-normalisiert
    norms = np.linalg.norm(anchors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        print(f"  ⚠ Anchors neu normalisieren (max_abw={abs(norms - 1).max():.2e})")
        anchors /= norms[:, np.newaxis]

    # §2.47 Format-Pflicht: letztes Spalte = Dekaden-Label (float32) für EraClassifier NN-Suche.
    # era_classifier._clap_nearest_neighbor erwartet shape (n_anchors, embedding_dim + 1).
    decade_col = np.array(DECADES_15, dtype=np.float32).reshape(-1, 1)
    anchors_with_labels = np.concatenate([anchors, decade_col], axis=1)  # shape: [15, 513]

    era_path = OUT_DIR / "era_anchors.npy"
    np.save(str(era_path), anchors_with_labels)
    print(f"  ✓ era_anchors.npy: shape={anchors_with_labels.shape}, dtype={anchors_with_labels.dtype}")
    print(f"    Kosinus-Ähnlichkeit 1920↔1930: {float(anchors[3] @ anchors[4]):.3f}")
    print(f"    Kosinus-Ähnlichkeit 1920↔2010: {float(anchors[3] @ anchors[12]):.3f}")

    # ── 2. reference_anchors.npz ───────────────────────────────────────────────────
    print("\nGeneriere reference_anchors.npz …")
    # 10 Dekaden (1920–2010), 9 Genres, 3 Materialien = 270 Anker
    DECADES_10 = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]

    arrays: dict[str, np.ndarray] = {}
    total = 0
    for decade in DECADES_10:
        for genre in GENRES_9:
            for material in MATERIALS_3:
                key = f"d{decade}_g{genre}_m{material}"
                arrays[key] = _spectral_envelope_128(decade, genre, material, rng)
                total += 1

    # Metadaten als separate Arrays
    arrays["_decades"] = np.array(DECADES_10, dtype=np.int32)
    arrays["_genres"] = np.array(GENRES_9)
    arrays["_materials"] = np.array(MATERIALS_3)
    arrays["_n_spec_bins"] = np.array([N_SPEC_BINS], dtype=np.int32)
    arrays["_version"] = np.array(["aurik9_synthetic_v1"])

    ref_path = OUT_DIR / "reference_anchors.npz"
    np.savez_compressed(str(ref_path), **arrays)

    # Größencheck
    size_kb = ref_path.stat().st_size / 1024
    print(f"  ✓ reference_anchors.npz: {total} Anker, {size_kb:.1f} KB")
    print(f"    Schlüssel-Beispiel: 'd1950_gschlager_mtape' shape={arrays['d1950_gschlager_mtape'].shape}")

    print(f"\n✅ ERA-Anker erfolgreich generiert in: {OUT_DIR}")
    print(f"   era_anchors.npy     : {era_path.stat().st_size / 1024:.1f} KB  ({anchors_with_labels.shape})")
    print(f"   reference_anchors.npz: {size_kb:.1f} KB  ({total} Anker à {N_SPEC_BINS}-dim)")


if __name__ == "__main__":
    generate_era_anchors()
