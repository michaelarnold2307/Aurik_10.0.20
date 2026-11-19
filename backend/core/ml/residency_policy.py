"""§SOTA-P1-1 — Modell-Residency & Warm-up-Policy (normative Spezifikation).

Roadmap TODO-P1-1: „Spezifizieren und umsetzen, welche Modelle warmgehalten
werden (Residency/LRU je Session) und wie Warm-up einmalig je Modell
amortisiert wird; deterministisches Multi-Song-Batching desselben Modells
unter Wahrung von §G1 (GEBOTE.md, Seed-Isolation pro Song)."

Dieses Modul ist die normative Policy-Ebene:
- ``ResidencyTier`` + ``RESIDENCY_TABLE``: welche Modelle prozess-warm bleiben
  (ALWAYS), welche im LRU-Cache des ``InferenceSessionManager`` liegen
  (SESSION) und welche nach der Nutzung freigegeben werden (ONESHOT).
- Warm-up-Amortisierung: ``mark_warmed()``/``is_warmed()`` — jedes Modell wird
  je Prozess EINMAL warmgefahren; der zweite Song amortisiert die Kosten.
- §G1-Batching-Vertrag: ``song_seed()`` — deterministische, song-isolierte
  Seed-Ableitung (blake2b aus Song-Identität + Master-Seed). Multi-Song-Läufe
  verarbeiten Songs SEQUENTIELL auf denselben warmen Modellen; kein Song teilt
  veränderlichen RNG-Zustand mit einem anderen (§G1 (GEBOTE.md)).

Adoption: Plugins sind Singletons (einmal Laden je Prozess) und decken damit
bereits den Residency-Fall ab; dieses Modul formalisiert die Regeln, damit
neue Plugin-Modelle nach denselben Kriterien eingestuft werden.

Autor: Aurik Testing Team
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class ResidencyTier(Enum):
    """Residency-Stufe eines Modells je Prozess/Session."""

    ALWAYS = "always"  # klein/CPU-only: prozess-warm halten (Singleton, nie evicten)
    SESSION = "session"  # mittel: LRU im InferenceSessionManager (Default 4 Sessions/2 GB)
    ONESHOT = "oneshot"  # schwer: laden → nutzen → freigeben (VRAM-/RAM-Hygiene)


# ── Residency-Tabelle (kanonisch) ─────────────────────────────────────────────
# Kriterien ALWAYS: < 200 MB, CPU-only-Dauerläufer, je Song mindestens einmal
# benötigt (Resemblyzer-Witness, PANNs-Tagger, MuQ-Embedding).
# Kriterien ONESHOT: > 1 GB oder GPU-gebunden mit seltenem Einsatz.
_RESIDENCY_TABLE: dict[str, ResidencyTier] = {
    # ALWAYS — kleine CPU-Modelle, prozess-warm
    "resemblyzer": ResidencyTier.ALWAYS,
    "panns": ResidencyTier.ALWAYS,
    "beats": ResidencyTier.ALWAYS,
    "muq": ResidencyTier.ALWAYS,
    "fcpe": ResidencyTier.ALWAYS,
    "rmvpe": ResidencyTier.ALWAYS,
    "pesto": ResidencyTier.ALWAYS,
    "deepfilternet": ResidencyTier.ALWAYS,
    "hifigan": ResidencyTier.ALWAYS,
    # SESSION — mittelschwere Modelle im LRU-Cache
    "melbandroformer": ResidencyTier.SESSION,
    "bs_roformer": ResidencyTier.SESSION,
    "demucs": ResidencyTier.SESSION,
    "cqtdiff": ResidencyTier.SESSION,
    "gacela": ResidencyTier.SESSION,
    "diffwave": ResidencyTier.SESSION,
    "ear_vae": ResidencyTier.SESSION,
    # ONESHOT — schwere/GPU-Modelle: nach Nutzung freigeben
    "flashsr": ResidencyTier.ONESHOT,
    "bigvgan": ResidencyTier.ONESHOT,
    "audioldm2": ResidencyTier.ONESHOT,
    "miipher_dit": ResidencyTier.ONESHOT,
    "utmosv2": ResidencyTier.ONESHOT,
}


def _normalize_model_name(model_name: str) -> str:
    """Normalisiert Plugin-/Modellnamen auf den Tabellen-Key."""
    name = str(model_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    for suffix in ("_plugin", "_onnx", "_torch", "_v2", "_v3", "_v4", "_iter3"):
        if name.endswith(suffix):
            candidate = name[: -len(suffix)]
            if candidate in _RESIDENCY_TABLE:
                return candidate
    return name


def residency_tier_of(model_name: str) -> ResidencyTier:
    """Residency-Stufe eines Modells; unbekannte Modelle sind konservativ SESSION."""
    return _RESIDENCY_TABLE.get(_normalize_model_name(model_name), ResidencyTier.SESSION)


def should_keep_warm(model_name: str) -> bool:
    """True, wenn das Modell prozess-warm gehalten wird (ALWAYS)."""
    return residency_tier_of(model_name) is ResidencyTier.ALWAYS


# ── Warm-up-Amortisierung (einmal je Prozess) ─────────────────────────────────

_warmed: set[str] = set()
_warmed_lock = threading.Lock()


def is_warmed(model_name: str) -> bool:
    """True, wenn das Modell in diesem Prozess bereits warmgefahren wurde."""
    return _normalize_model_name(model_name) in _warmed


def mark_warmed(model_name: str) -> None:
    """Markiert ein Modell als warmgefahren (idempotent, thread-safe)."""
    with _warmed_lock:
        _warmed.add(_normalize_model_name(model_name))


def reset_warmup_registry() -> None:
    """Setzt die Warm-up-Registrierung zurück (nur für Tests)."""
    with _warmed_lock:
        _warmed.clear()


def warmup_once(model_name: str, warmup_fn) -> bool:
    """Führt ``warmup_fn()`` für ``model_name`` EINMAL je Prozess aus.

    Returns True, wenn der Warm-up DIESMAL ausgeführt wurde (False = bereits
    amortisiert). Fehler des Warm-ups werden geloggt und führen NICHT zu einem
    erneuten Versuch (fail-closed: das Modell bleibt funktional,
    §V6 (copilot-instructions.md)).
    """
    key = _normalize_model_name(model_name)
    if key in _warmed:
        return False
    with _warmed_lock:
        if key in _warmed:  # Double-Checked
            return False
        try:
            warmup_fn()
        except Exception as _exc:  # §V6 (copilot-instructions.md): nie still, nie blockierend
            logger.warning("Warm-up für %s fehlgeschlagen (%s) — Modell bleibt ohne Warm-up nutzbar", model_name, _exc)
        _warmed.add(key)
        return True


# ── §G1-Batching-Vertrag: song-isolierte Seed-Ableitung ───────────────────────


def song_seed(song_identity: str, master_seed: int = 42) -> int:
    """Deterministischer, song-isolierter Seed (§G1 (GEBOTE.md) Seed-Isolation).

    Ableitung: blake2b(song_identity || master_seed) → 31-bit int. Gleicher
    Song + gleicher Master-Seed ⇒ identischer Seed — unabhängig von der
    Verarbeitungsreihenfolge im Batch. Kein Song teilt veränderlichen
    RNG-Zustand mit einem anderen (jeder Seed ist rein abgeleitet).
    """
    payload = f"{song_identity}\u0000{int(master_seed)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def master_seed_from_env() -> int:
    """Master-Seed aus ``AURIK_MASTER_SEED`` (Default 42, deterministisch)."""
    raw = os.environ.get("AURIK_MASTER_SEED", "").strip()
    if not raw:
        return 42
    try:
        return int(raw)
    except ValueError:
        logger.warning("AURIK_MASTER_SEED=%r ist keine Ganzzahl — Default 42", raw)
        return 42
