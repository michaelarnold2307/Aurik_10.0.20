"""
Longitudinales Künstler-Signaturmodell — Aurik 10.0.0 (§2.13)

Bei mehreren Dateien derselben Session / desselben Künstlers speichert Aurik 10.0.0
den Klang-Fingerabdruck des Künstlers und nutzt ihn als Prior für stark degradierte
Stellen zur authentischeren Rekonstruktion.

ArtistSignature enthält:
    - Stimmtyp (VoiceGender), Altersgruppe (VoiceAgeGroup)
    - Formant-Profil F1–F4 (Median ± Std)
    - Vibrato-Rate & -Tiefe
    - Breathiness-Ratio
    - Spektral-Envelope (128-dim)
    - PANNs-Instrument-Tags

Persistenz: ~/.aurik/artist_signatures/<artist_id>.json

Invarianten (§2.13):
    - confidence < 0.3 (< 2 Dateien): Signatur nur als schwacher Prior
    - confidence ≥ 0.7 (≥ 5 Dateien): Signatur als starker Prior
    - Signatur-Update NIEMALS rückwirkend (bestehende Dateien unberührt)
    - Datenschutz: Signaturen nur lokal, niemals übertragen

Aktivierung: automatisch wenn Nutzer ≥ 2 Dateien aus gleichem Ordner verarbeitet
             (Session-Kennung via SHA256[:8] des Ordnerpfads)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grundkonfiguration (§2.13)
# ---------------------------------------------------------------------------

SIGNATURES_DIR: Path = Path.home() / ".aurik" / "artist_signatures"
CONFIDENCE_WEAK: float = 0.3  # < Weak Prior
CONFIDENCE_STRONG: float = 0.7  # ≥ Strong Prior
SPECTRAL_ENVELOPE_DIM: int = 128  # Dimension des Spektral-Profils

# Confidence-Mapping: N Dateien → confidence-Wert


def _confidence_from_n(n: int) -> float:
    """Confidence steigt mit der Anzahl analysierter Dateien (§2.13).

    n=0 → 0.0, n=1 → 0.15, n=2 → 0.30, n=5 → 0.70, n≥10 → 1.0
    """
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.15
    # Logarithmische Sättigungskurve
    val = 0.15 + 0.85 * (1.0 - math.exp(-0.4 * (n - 1)))
    return float(min(1.0, val))


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class ArtistSignature:
    """Persistierter Klang-Fingerabdruck eines Künstlers / einer Session (§2.13).

    Attribute:
        artist_id:          SHA256[:8] aus Ordnerpfad-Hash
        voice_gender:       VoiceGender-Wert als String (MALE/FEMALE/CHILD/ANDROGYNOUS/UNKNOWN)
        voice_age_group:    VoiceAgeGroup-Wert als String
        formant_profile:    F1–F4 Median ± Std {"F1_median": ..., "F1_std": ..., ...}
        vibrato_rate_hz:    Typische Vibrato-Rate ∈ [4, 8] Hz
        vibrato_depth_cent: Typische Vibrato-Tiefe ∈ [10, 80] Cent
        breathiness_ratio:  Atemgeräusch-Anteil ∈ [0, 0.3]
        spectral_envelope:  128-dim mittleres Spektral-Profil (float32)
        instrument_tags:    PANNs-Tags der Session
        confidence:         Qualität des Fingerabdrucks ∈ [0, 1]
        n_files_analyzed:   Anzahl analysierter Dateien
        last_updated:       ISO 8601 Timestamp
    """

    artist_id: str
    voice_gender: str = "UNKNOWN"
    voice_age_group: str = "ADULT"
    formant_profile: dict[str, float] = field(default_factory=dict)
    vibrato_rate_hz: float = 5.5
    vibrato_depth_cent: float = 30.0
    breathiness_ratio: float = 0.05
    spectral_envelope: np.ndarray = field(
        default_factory=lambda: np.ones(SPECTRAL_ENVELOPE_DIM, dtype=np.float32) / SPECTRAL_ENVELOPE_DIM
    )
    instrument_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    n_files_analyzed: int = 0
    last_updated: str = ""

    def __post_init__(self) -> None:
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()
        # Sicherheitsguard
        self.spectral_envelope = np.asarray(self.spectral_envelope, dtype=np.float32)
        if self.spectral_envelope.shape != (SPECTRAL_ENVELOPE_DIM,):
            self.spectral_envelope = np.ones(SPECTRAL_ENVELOPE_DIM, dtype=np.float32) / SPECTRAL_ENVELOPE_DIM

    def as_dict(self) -> dict:
        """Serialisierungsformat für Persistenz."""
        return {
            "artist_id": self.artist_id,
            "voice_gender": self.voice_gender,
            "voice_age_group": self.voice_age_group,
            "formant_profile": self.formant_profile,
            "vibrato_rate_hz": self.vibrato_rate_hz,
            "vibrato_depth_cent": self.vibrato_depth_cent,
            "breathiness_ratio": self.breathiness_ratio,
            "spectral_envelope": self.spectral_envelope.tolist(),
            "instrument_tags": self.instrument_tags,
            "confidence": self.confidence,
            "n_files_analyzed": self.n_files_analyzed,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArtistSignature:
        """Deserialisiert aus JSON-Dict."""
        envelope = np.asarray(
            data.get("spectral_envelope", [1.0 / SPECTRAL_ENVELOPE_DIM] * SPECTRAL_ENVELOPE_DIM),
            dtype=np.float32,
        )
        return cls(
            artist_id=str(data.get("artist_id", "")),
            voice_gender=str(data.get("voice_gender", "UNKNOWN")),
            voice_age_group=str(data.get("voice_age_group", "ADULT")),
            formant_profile=dict(data.get("formant_profile", {})),
            vibrato_rate_hz=float(data.get("vibrato_rate_hz", 5.5)),
            vibrato_depth_cent=float(data.get("vibrato_depth_cent", 30.0)),
            breathiness_ratio=float(data.get("breathiness_ratio", 0.05)),
            spectral_envelope=envelope,
            instrument_tags=list(data.get("instrument_tags", [])),
            confidence=float(data.get("confidence", 0.0)),
            n_files_analyzed=int(data.get("n_files_analyzed", 0)),
            last_updated=str(data.get("last_updated", "")),
        )


@dataclass
class VoiceCharacteristics:
    """Basis-Stimmmerkmale einer einzelnen Datei-Analyse.

    Wird von VocalAIEnhancement / GenderDetector geliefert.
    Hier als Minimalstruktur für die Signatur-Integration.
    """

    voice_gender: str = "UNKNOWN"  # "MALE" / "FEMALE" / "CHILD" / "ANDROGYNOUS"
    voice_age_group: str = "ADULT"
    f1_hz: float = 600.0
    f2_hz: float = 1200.0
    f3_hz: float = 2500.0
    f4_hz: float = 3500.0
    vibrato_rate_hz: float = 5.5
    vibrato_depth_cent: float = 30.0
    breathiness_ratio: float = 0.05
    spectral_envelope: np.ndarray | None = None
    instrument_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.spectral_envelope is None:
            self.spectral_envelope = np.ones(SPECTRAL_ENVELOPE_DIM, dtype=np.float32) / SPECTRAL_ENVELOPE_DIM


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: ArtistSignatureStore | None = None
_lock = threading.Lock()


class ArtistSignatureStore:
    """Verwaltet persistierte Künstler-Signaturen (§2.13).

    Persistenz:  ~/.aurik/artist_signatures/<artist_id>.json
    Aktivierung: automatisch wenn Nutzer ≥ 2 Dateien aus gleichem Ordner verarbeitet
                 (Session-Kennung via SHA256[:8] des Ordnerpfads)

    Hauptmethoden::

        store = get_signature_store()
        artist_id = store.detect_session([Path("song1.flac"), Path("song2.flac")])
        sig = store.load(artist_id)       # Optional[ArtistSignature]
        sig = store.update_from_analysis(artist_id, voice_characteristics)
        store.save(sig)

    Anwendung im Restoration-Workflow:
        1. Neue Datei → Signatur aus Session laden (wenn confidence ≥ 0.3)
        2. Formant-Priors aus Signatur → VocalAIEnhancement initialisieren
        3. Vibrato-Referenz → Pitch-Vocoder-Stabilisierung
        4. Spectral Envelope → NMF-Initialisierungsmatrix W₀ für Inpainting
        5. Nach Restaurierung: Signatur mit neuen Beobachtungen updaten
    """

    def __init__(self) -> None:
        SIGNATURES_DIR.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ArtistSignature] = {}
        self._cache_lock = threading.Lock()
        logger.debug("ArtistSignatureStore initialisiert (Pfad: %s)", SIGNATURES_DIR)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def detect_session(self, file_paths: list[Path]) -> str:
        """Erzeugt Session-ID aus dem gemeinsamen Ordnerpfad.

        Session = SHA256[:8] des Ordnerpfads (des ersten Files).
        Alle Dateien aus demselben Ordner teilen die Session-ID.

        Args:
            file_paths: Liste analysierter Audio-Dateien

        Returns:
            8-stellige Hex-Session-ID (z.B. "a3f92b1c")
        """
        if not file_paths:
            return "00000000"
        folder = file_paths[0].parent.resolve()
        h = hashlib.sha256(str(folder).encode("utf-8")).hexdigest()[:8]
        return h

    def load(self, artist_id: str) -> ArtistSignature | None:
        """Lädt Signatur aus Cache oder Disk.

        Args:
            artist_id: 8-stellige Hex-Session-ID

        Returns:
            ArtistSignature oder None falls nicht vorhanden
        """
        with self._cache_lock:
            if artist_id in self._cache:
                return self._cache[artist_id]

        path = SIGNATURES_DIR / f"{artist_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sig = ArtistSignature.from_dict(data)
            with self._cache_lock:
                self._cache[artist_id] = sig
            logger.debug(
                "Signatur geladen: artist_id=%s, confidence=%.2f, n_files=%d",
                artist_id,
                sig.confidence,
                sig.n_files_analyzed,
            )
            return sig
        except Exception as exc:
            logger.warning("Signatur-Laden fehlgeschlagen (%s): %s", path, exc)
            return None

    def save(self, sig: ArtistSignature) -> None:
        """Persistiert Signatur auf Disk.

        Invariante (§2.13): Nur vorwärts (niemals rückwirkend überschreiben).

        Args:
            sig: ArtistSignature (wird in <artist_id>.json gespeichert)
        """
        sig.last_updated = datetime.now(timezone.utc).isoformat()
        with self._cache_lock:
            self._cache[sig.artist_id] = sig
        path = SIGNATURES_DIR / f"{sig.artist_id}.json"
        try:
            path.write_text(
                json.dumps(sig.as_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(
                "Signatur gespeichert: artist_id=%s, confidence=%.2f",
                sig.artist_id,
                sig.confidence,
            )
        except Exception as exc:
            logger.warning("Signatur-Speichern fehlgeschlagen: %s", exc)

    def update_from_analysis(
        self,
        artist_id: str,
        analysis: VoiceCharacteristics,
    ) -> ArtistSignature:
        """Aktualisiert Signatur mit neuen Voice-Characteristics (EMA).

        Exponentieller gleitender Mittelwert (α = 0.3 neue Beobachtung,
        0.7 bestehende Signatur) für alle kontinuierlichen Merkmale.

        Invariante (§2.13): Neue Datei erhöht n_files_analyzed um 1.
        Confidence steigt gemäß _confidence_from_n.

        Args:
            artist_id: Session-ID
            analysis:  VoiceCharacteristics der neuen Datei

        Returns:
            Aktualisierte ArtistSignature (noch NICHT gespeichert — explizit save() aufrufen)
        """
        existing = self.load(artist_id)

        if existing is None:
            # Neue Signatur anlegen
            existing = ArtistSignature(artist_id=artist_id)

        n_new = existing.n_files_analyzed + 1
        alpha = 0.3  # EMA-Gewicht für neue Beobachtung

        # --- Formant-Update ---
        for key, val_new in {
            "F1_median": analysis.f1_hz,
            "F2_median": analysis.f2_hz,
            "F3_median": analysis.f3_hz,
            "F4_median": analysis.f4_hz,
        }.items():
            old_val = existing.formant_profile.get(key, val_new)
            existing.formant_profile[key] = float(alpha * val_new + (1.0 - alpha) * old_val)

        # --- Vibrato-Rate ---
        vr_new = float(np.clip(analysis.vibrato_rate_hz, 4.0, 8.0))
        existing.vibrato_rate_hz = float(alpha * vr_new + (1.0 - alpha) * existing.vibrato_rate_hz)

        # --- Vibrato-Tiefe ---
        vd_new = float(np.clip(analysis.vibrato_depth_cent, 10.0, 80.0))
        existing.vibrato_depth_cent = float(alpha * vd_new + (1.0 - alpha) * existing.vibrato_depth_cent)

        # --- Breathiness ---
        br_new = float(np.clip(analysis.breathiness_ratio, 0.0, 0.3))
        existing.breathiness_ratio = float(alpha * br_new + (1.0 - alpha) * existing.breathiness_ratio)

        # --- Spektral-Envelope ---
        if analysis.spectral_envelope is not None:
            env_new = np.asarray(analysis.spectral_envelope, dtype=np.float32)
            if env_new.shape == (SPECTRAL_ENVELOPE_DIM,):
                existing.spectral_envelope = (alpha * env_new + (1.0 - alpha) * existing.spectral_envelope).astype(
                    np.float32
                )
                # Normieren
                norm = float(np.linalg.norm(existing.spectral_envelope))
                if norm > 0:
                    existing.spectral_envelope /= norm

        # --- Gender / Age (Mehrheitsabstimmung vereinfacht: nur ersetzen wenn confidence gering) ---
        if existing.confidence < CONFIDENCE_WEAK:
            existing.voice_gender = analysis.voice_gender
            existing.voice_age_group = analysis.voice_age_group

        # --- Instrument-Tags ---
        for tag in analysis.instrument_tags:
            if tag not in existing.instrument_tags:
                existing.instrument_tags.append(tag)

        # --- Confidence-Update ---
        existing.n_files_analyzed = n_new
        existing.confidence = _confidence_from_n(n_new)

        # Cache aktualisieren (wichtig für aufeinanderfolgende Aufrufe ohne save())
        with self._cache_lock:
            self._cache[artist_id] = existing

        logger.info(
            "🎙️ Künstler-Signatur aktualisiert: artist_id=%s | n=%d | confidence=%.2f | gender=%s",
            artist_id,
            n_new,
            existing.confidence,
            existing.voice_gender,
        )
        return existing

    def delete(self, artist_id: str) -> bool:
        """Löscht Signatur (für Datenschutz-UI §2.13).

        Returns:
            True falls Datei erfolgreich gelöscht, False wenn nicht vorhanden.
        """
        with self._cache_lock:
            self._cache.pop(artist_id, None)
        path = SIGNATURES_DIR / f"{artist_id}.json"
        if path.exists():
            try:
                path.unlink()
                logger.info("Signatur gelöscht: artist_id=%s", artist_id)
                return True
            except Exception as exc:
                logger.warning("Signatur-Löschen fehlgeschlagen: %s", exc)
        return False

    def list_all(self) -> list[str]:
        """Listet alle vorhandenen Signatur-IDs auf.

        Returns:
            Liste von artist_id Strings
        """
        return [p.stem for p in SIGNATURES_DIR.glob("*.json")]

    def get_prior_strength(self, artist_id: str) -> str:
        """Gibt die Prior-Stärke als verständlichen Text zurück.

        Returns:
            "kein Prior" / "schwacher Prior" / "starker Prior"
        """
        sig = self.load(artist_id)
        if sig is None:
            return "kein Prior"
        if sig.confidence < CONFIDENCE_WEAK:
            return "kein Prior"
        if sig.confidence < CONFIDENCE_STRONG:
            return "schwacher Prior"
        return "starker Prior"


# ---------------------------------------------------------------------------
# Singleton-Accessor + Convenience
# ---------------------------------------------------------------------------


def get_signature_store() -> ArtistSignatureStore:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ArtistSignatureStore()
    return _instance


def load_artist_signature(artist_id: str) -> ArtistSignature | None:
    """Convenience: Künstler-Signatur laden.

    Args:
        artist_id: SHA256[:8] des Ordnerpfads

    Returns:
        ArtistSignature oder None
    """
    return get_signature_store().load(artist_id)


def update_artist_signature(
    artist_id: str,
    analysis: VoiceCharacteristics,
    auto_save: bool = True,
) -> ArtistSignature:
    """Convenience: Signatur aus neuer Analyse aktualisieren und speichern.

    Args:
        artist_id:  Session-ID
        analysis:   VoiceCharacteristics der neuen Datei
        auto_save:  Falls True, automatisch persistieren

    Returns:
        Aktualisierte ArtistSignature
    """
    store = get_signature_store()
    sig = store.update_from_analysis(artist_id, analysis)
    if auto_save:
        store.save(sig)
    return sig
