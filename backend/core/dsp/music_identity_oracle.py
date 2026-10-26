"""Musik-nativer Vokal-Identitäts-Zeuge (KP-3, Pfad A — KEIN Training).

Ersetzt rollentechnisch das sprachtrainierte Resemblyzer-Embedding
(LibriSpeech, `voice_encoder.onnx`) als **Vokal-Identitäts-Witness** durch
musiknatives **MuQ-MuLan** (`plugins/muq_plugin.extract_embedding`, 1024-d,
musiktrainiert, deterministisch §G5 (copilot-instructions.md)). Empfehlung
„Ersetzung statt Finetune" aus `docs/SOTA_TRAINING_KONZEPT.md` §3/KP-3.

Rolle: **report-only Zeuge** für Ebene-1-Guard und phase_65. Dieses Modul
misst den Kosinus der MuQ-MuLan-Embeddings von pre/post und liefert ihn
zusätzlich zur bestehenden Messung zurück — es übt NIE ein Veto-Recht aus,
flippt keine Flags und ändert keine bestehende Schwelle (Hörordnung §8:
Metriken sind Zeugen, die Hör-Instanz entscheidet; §III.11 (copilot-instructions.md):
ein sprachtrainiertes Embedding darf Musik/Gesang nicht richten, ein
musiktrainiertes Embedding ist der zulässige Kandidat). Die Resemblyzer-Sperre
(`music_model_flags.use_resemblyzer_music=False`, Vertrag W-12) bleibt
unverändert; dieses Modul hängt NUR als zusätzlicher Zeuge an.

Determinismus (§G5 (copilot-instructions.md)): keine Zeitquellen in der
Entscheidungslogik; die MuQ-Inferenz ist modell-deterministisch.
Silent-Failure (§V6 (copilot-instructions.md)): Jedes fehlende Modell/Embedding
liefert ``None`` NUR mit ``logger.warning`` + Begründung — ein fehlender
Messwert darf nie bestrafen und nie eine Verbesserung vortäuschen.
Stereo-Layout-Invariante (AGENTS.md §3): channels-first (C, N) und (N, C)
werden beide explizit auf Mono normalisiert; nie ``audio[:, 0]`` ohne
Layout-Check (Produktionsbefund: Export ``(2,)``).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# MuQ-MuLan-Embedding-Dimension (plugins/muq_plugin._EMBED_DIM = 1024).
_EMBED_DIM = 1024


def _downmix(audio: np.ndarray) -> np.ndarray:
    """Stereo-Layout-Invariante (§11 Spec 02): (C, N) und (N, C) → Mono float32.

    ``audio[:, 0]``/``mean(axis=0)`` ohne Layout-Check kollabiert auf
    C Samples (Produktionsbefund: Export ``(2,)``) — deshalb beide Layouts
    explizit bedienen und auf float32 normalisieren (NaN/Inf neutralisiert,
    §0a (copilot-instructions.md)).
    """
    arr: np.ndarray = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    res: np.ndarray
    if arr.ndim <= 1:
        res = arr.astype(np.float32)
    elif arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
        # kanonisch channels-first (C, N); (N, C) ebenfalls bedienen (wie muq_plugin).
        res = arr.mean(axis=0).astype(np.float32)
    else:
        res = arr.mean(axis=1).astype(np.float32)
    return res


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """Kosinus-Ähnlichkeit zweier Vektoren in [-1, 1]; None bei Null-Norm (§V6 (copilot-instructions.md))."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    denom = na * nb
    if denom < 1e-12:
        return None
    sim = float(np.dot(a, b) / denom)
    if not np.isfinite(sim):
        return None
    return float(np.clip(sim, 0.0, 1.0))


def _muq_embedding(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """MuQ-MuLan-Einbettung (1024-d float32); None wenn Modell/GPU fehlt (§V6 (copilot-instructions.md))."""
    try:
        from plugins.muq_plugin import extract_embedding  # pylint: disable=import-outside-toplevel

        emb = extract_embedding(audio, sr)
    except Exception as _exc:  # pylint: disable=broad-except
        logger.warning(
            "MuQ-MuLan-Einbettung nicht ladbar (%s) — Vokal-Identitäts-Zeuge übersprungen "
            "(§V6 (copilot-instructions.md))",
            _exc,
        )
        return None
    if emb is None:
        logger.warning(
            "MuQ-MuLan-Einbettung nicht verfügbar (Modell/GPU/Checkpoint fehlt) — "
            "Vokal-Identitäts-Zeuge übersprungen (§V6 (copilot-instructions.md))"
        )
        return None
    arr: np.ndarray = np.asarray(emb, dtype=np.float32).reshape(-1)
    if arr.size < 1:
        logger.warning("MuQ-MuLan-Einbettung leer — Zeuge übersprungen (§V6 (copilot-instructions.md))")
        return None
    return arr


def music_identity_cosine(pre: np.ndarray, post: np.ndarray, sr: int) -> tuple[float | None, list[str]]:
    """Musiknativer Vokal-Identitäts-Kosinus über MuQ-MuLan-Embeddings.

    Misst die Kosinus-Ähnlichkeit der MuQ-MuLan-Song-Einbettungen von ``pre``
    und ``post`` (identischer Sänger ⇒ Kosinus hoch trotz Verarbeitung;
    verschiedener Sänger ⇒ niedriger). Deterministisch (§G5 (copilot-instructions.md)):
    gleicher Input ⇒ gleicher Kosinus.

    Args:
        pre: Audio vor dem Eingriff (beliebiges Layout, z. B. [N] oder [2, N]).
        post: Audio nach dem Eingriff.
        sr: Abtastrate in Hz.

    Returns:
        ``(cosine, reasons)`` — ``cosine`` in [0, 1] oder ``None``, wenn keine
        Messung möglich ist (Modell/Embedding fehlt oder Null-Norm). ``reasons``
        ist eine (ggf. leere) Liste begründender Meldungen. ``None`` ist kein
        „schlecht": Ohne Messwert darf nichts bestraft werden (Zeuge-Prinzip,
        §V6 (copilot-instructions.md)).
    """
    reasons: list[str] = []
    mono_pre = _downmix(pre)
    mono_post = _downmix(post)

    # Der MuQ-Pfad braucht ein messbares, nicht-konstantes Signal.
    if mono_pre.size < 256 or mono_post.size < 256:
        reasons.append("Signal zu kurz (< 256 Samples) — kein MuQ-Messwert (§V6 (copilot-instructions.md))")
        return None, reasons
    if float(np.std(mono_pre)) < 1e-6 or float(np.std(mono_post)) < 1e-6:
        reasons.append("konstantes/stummes Signal — kein MuQ-Messwert (§V6 (copilot-instructions.md))")
        return None, reasons

    emb_pre = _muq_embedding(mono_pre, sr)
    if emb_pre is None:
        reasons.append("pre-Embedding fehlt — MuQ-Zeuge übersprungen (§V6 (copilot-instructions.md))")
        return None, reasons
    emb_post = _muq_embedding(mono_post, sr)
    if emb_post is None:
        reasons.append("post-Embedding fehlt — MuQ-Zeuge übersprungen (§V6 (copilot-instructions.md))")
        return None, reasons

    # MuQ liefert deterministische Songs-Einbettungen; für die Identität wird
    # die Kosinus-Ähnlichkeit der Zeit-gemittelten last-hidden-States genutzt.
    cosine = _cosine(emb_pre, emb_post)
    if cosine is None:
        reasons.append("Null-Norm/kein endlicher Kosinus — MuQ-Zeuge übersprungen (§V6 (copilot-instructions.md))")
        return None, reasons
    return cosine, reasons


def measure_and_log_music_identity(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    stage: str = "level1_guard",
) -> tuple[float | None, list[str]]:
    """Report-only Convenience: misst UND protokolliert den MuQ-Identitäts-Kosinus.

    Kein Veto, kein Flag-Flip (§III.11 (copilot-instructions.md)). Rückgabe ist
    ``(cosine, reasons)`` für die Metadaten-Anreicherung durch den Aufrufer.
    """
    cosine, reasons = music_identity_cosine(pre, post, sr)
    cos_s = f"{cosine:.4f}" if cosine is not None else "n/a"
    reason_s = "; ".join(reasons) if reasons else "—"
    logger.info("MuQ-Identitäts-Zeuge (report-only, Stufe=%s): Kosinus=%s, Gründe=[%s]", stage, cos_s, reason_s)
    return cosine, reasons
