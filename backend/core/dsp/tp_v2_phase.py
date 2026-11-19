"""SOTA-M12 (D-K4-8): TP-V2-Klassen-Phasenschätzung für Transienten-Phasenkohärenz.

Challenger zum bestehenden Transienten-/Phasen-Rekonstruktionspfad (§v10.303
``restore_phase_coherence`` in ``backend/core/dsp/phase_coherent_stft.py``).
Flag: ``backend.core.music_model_flags.use_tp_v2_phase`` (SOTA-M12 (D-K4-8),
Default False) — ausschließlich hinter ``True`` aktiv; False ⇒ keine
Verhaltensänderung (bit-identisch zum Bestand, gemessen in
``tests/unit/test_tp_v2_phase.py``).

KAMMFILTER-FREIHEIT (Kernvertrag, D-K4-8)
----------------------------------------
Ein statischer Phasen-Präfix (z. B. konstante Phasenverschiebung je Bin oder
eine feste Verzögerungs-Phase ``e^{-jωτ}`` über alle Fenster) erzeugt im
Zeitbereich periodische Echo-/Kammfilter-Strukturen (Interferenz fest
versetzter Kopien). Dieses Modell ist DESHALB kammfilter-frei konstruiert:

1. **Relativ-Phase statt Absolut-Phase:** Das Netz lernt eine Korrektur
   ``Δφ(f, t)`` RELATIV zur instantanen Phase des Eingangs-Spektrums, nie eine
   absolute Phasenvorgabe. Ohne Korrekturbedarf ⇒ Δφ = 0 ⇒ Identität.
2. **Null-Anfangs-Initialisierung der Ausgangsschicht:** Das Residual
   ``out_proj`` ist am Trainingsstart exakt 0 ⇒ ein untrainiertes Modell ist
   der Identitäts-Passthrough, nicht eine beliebige Phasenverzerrung.
3. **Keine (f, t)-unabhängigen Präfixe:** Es gibt keine globale Phase,
   keine Verzögerungs-Parameter und keine Frequenz-Embeddings, die als
   statischer Präfix wirken könnten (Gegentest:
   ``tests/unit/test_tp_v2_phase.py::test_kein_phasen_praefix_regelmuster``).

WARUM DIESE ARCHITEKTUR (ehrliche Begründung)
---------------------------------------------
- **U-Net aus Conv2d-Blöcken (kein Transformer, kein RNN):** Die Trainings-
  datenlage (synthetische Percussion-Transienten, siehe
  ``scripts/train_tp_v2_phase.py``) ist klein; ein kompaktes faltungs-
  basiertes U-Net ist auf kleinen Daten robust, deterministisch (keine
  Sampling-Schritte, kein Dropout zur Inferenz), schnell auf CPU und ohne
  ONNX-Paritätsrisiko reine Torch-Inferenz. Ein Transformer wäre für dieses
  Datenvolumen überdimensioniert (§III.9 (copilot-instructions.md): kleine
  lokale Modelle bevorzugen).
- **Relative Sinus/Cosinus-Kodierung der Zielphase:** Statt Winkel-Regression
  (2π-Sprung-Problematik) lernt das Netz ``(cos Δφ, sin Δφ)``; die Rück-
  transformation ``atan2`` ist eindeutig und NaN/Inf-sicher begrenzt.
- **Kontextfenster über die Zeitachse:** Transienten-Phasenkohärenz ist eine
  lokale Zeit-Frequenz-Eigenschaft; die U-Net-Flaschenhalsstruktur sieht
  ±T/2 Kontextfenster und glättet so phasen-inkohärente Ein-Bin-Sprünge, ohne
  globale Statistik zu erzwingen.
- **Determinismus (§G5 (copilot-instructions.md)):** feste Gewichts-
  Initialisierung über ``build_model(seed=...)`` in ``torch.random.fork_rng``,
  Inferenz unter ``torch.inference_mode()`` ohne Zufall, kein ``time.time()``
  in der Logik — gleicher Input ⇒ bit-identischer Output (getestet).

VERTRÄGE
--------
- **Eingabe:** log-Magnituden-Spektrogramm ``(F, T)`` float32 (oder ``(1, F, T)``
  mit führender Batch-Dimension) — z. B. ``np.log1p(|STFT|)``.
- **Ausgabe:** Phasen-Korrektur ``Δφ(f, t)`` in Radiant, Shape ``(F, T)``
  float32, immer endlich und in ``[-π, π]`` (§0a (copilot-instructions.md)).
- **Anwendung durch den Aufrufer:**
  ``Z_neu = |Z| · exp(i·(angle(Z) + Δφ))`` — die Magnitude bleibt unangetastet.
- **NaN/Inf-Schutz (§0a (copilot-instructions.md)):** Eingang wird vor der
  Inferenz sanitisiert (Warnung), Ausgabe wird geclippt/kontrolliert; nie
  nicht-finite Werte nach außen.
- **§V6 (copilot-instructions.md) — Silent-Failure-Verbot:** fehlendes
  Artefakt, nicht-ladbares Artefakt, nicht-finites Modell-Ergebnis ⇒
  ``logger.warning`` + Begründung; die Fassade ``estimate_transient_phase``
  liefert ``None`` und der Aufrufer fällt auf den bestehenden Pfad
  (``restore_phase_coherence``) zurück.
- **Modell-Domäne:** Musik (§III.11 (copilot-instructions.md)) — Transienten-
  Phasenkohärenz ist eine Signalpfad-Eigenschaft von Percussion/Musik;
  Training auf synthetischen transientenreichen Signalen
  (``scripts/train_tp_v2_phase.py``), reale Waveform-Option dokumentiert dort.

Kein automatischer Flag-Flip (§v10.802 (copilot-instructions.md)): Aktivierung
erst nach A/B auf transient-reichem Material (Percussion ohne
Kammfilter-Artefakte) + Hörordnungs-Sign-off.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)  # §III.7 (copilot-instructions.md)

# ── Konstanten (Vertrag mit scripts/train_tp_v2_phase.py) ─────────────────────

TARGET_SR = 48_000
N_FFT = 1024
N_FREQ = N_FFT // 2 + 1  # 513
HOP = 256
CONTEXT_FRAMES = 16  # Kontextfenster über die Zeitachse (±T/2 siehe U-Net)
DEFAULT_HIDDEN = 16
DEFAULT_BLOCKS = 2
DEFAULT_SEED = 42

# Produktionsartefakt des Trainings (scripts/train_tp_v2_phase.py).
DEFAULT_CHECKPOINT_SUBPATH = "models/tp_v2/tp_v2_phase_v1.pt"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = _PROJECT_ROOT / DEFAULT_CHECKPOINT_SUBPATH


# ── Hilfsfunktionen (deterministisch, NaN/Inf-sicher) ────────────────────────


def _sanitize(arr: np.ndarray, context: str) -> np.ndarray:
    """NaN/Inf → 0 (§0a (copilot-instructions.md)), mit Warnung (§V6)."""
    out: np.ndarray = np.asarray(arr, dtype=np.float32)
    if not bool(np.isfinite(out).all()):
        logger.warning(
            "TP-V2-Phasenschätzer: NaN/Inf in %s — sanitisiert auf 0 (§0a/§V6 (copilot-instructions.md))",
            context,
        )
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _to_freq_time(log_mag: np.ndarray) -> np.ndarray:
    """Normalisiert die Eingabe auf ``(F, T)`` float32.

    Akzeptiert ``(F, T)`` und ``(1, F, T)`` (Batch=1). Andere Shapes werfen
    ``ValueError`` (laut, kein stiller Fallback).
    """
    arr: np.ndarray = np.asarray(log_mag, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Shape {arr.shape} nicht unterstützt — erwartet (F, T) oder (1, F, T) (Spektrogramm-Vertrag)")
    return arr


# ── Torch-Architektur (lazy Import wie aspade_declipper.py) ──────────────────


def build_model(
    *,
    n_freq: int = N_FREQ,
    context_frames: int = CONTEXT_FRAMES,
    hidden: int = DEFAULT_HIDDEN,
    blocks: int = DEFAULT_BLOCKS,
    seed: int = DEFAULT_SEED,
):
    """TP-V2-Phasenschätz-Netz (kompaktes U-Net über (F, T)) als Torch-Modell.

    Forward-Vertrag: ``forward(log_mag (B, F, T)) → delta_phi (B, F, T)`` in
    Radiant, relativ zur instantanen Eingangsphase (siehe Modul-Docstring:
    Kammfilter-Freiheit). Deterministische Initialisierung (§G5 (copilot-instructions.md)):
    gleicher Seed ⇒ bit-identische Gewichte, der
    globale RNG-Zustand bleibt unangetastet (``torch.random.fork_rng``).
    """
    import torch
    import torch.nn as nn

    if n_freq < 2 or context_frames < 2 or hidden < 2 or blocks < 1:
        raise ValueError(
            f"Ungültige Architektur-Parameter: n_freq={n_freq} context_frames={context_frames} "
            f"hidden={hidden} blocks={blocks}"
        )

    class _ConvBlock(nn.Module):
        """2-D-Conv-Block (Längentreue über beide Achsen, ONNX-fähig)."""

        def __init__(self, channels: int, down: bool) -> None:
            super().__init__()
            self.op: nn.Module
            if down:
                self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
            else:
                self.op = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(self.op(x))

    class TPv2PhaseNet(nn.Module):
        """Kompaktes U-Net: log_mag (B, F, T) → Δφ (B, F, T) in Radiant.

        Residual-Parameterisierung mit NULL-initialisierter Ausgangsschicht ⇒
        untrainiert = Identität (Δφ ≡ 0), nie ein Kammfilter-Präfix.
        """

        def __init__(self, freq_bins: int, hidden_dim: int, n_blocks: int) -> None:
            super().__init__()
            self.in_proj = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
            self.downs = nn.ModuleList([_ConvBlock(hidden_dim, down=True) for _ in range(n_blocks)])
            self.ups = nn.ModuleList([_ConvBlock(hidden_dim, down=False) for _ in range(n_blocks)])
            self.out_proj = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
            # Null-Initialisierung: kein statischer Phasen-Präfix (Kammfilter-
            # Vertrag, D-K4-8) — der Ausgang startet exakt bei 0.
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

        def forward(self, log_mag):
            x = log_mag.unsqueeze(1) if log_mag.dim() == 3 else log_mag
            h = torch.nn.functional.gelu(self.in_proj(x))
            skips = []
            for blk in self.downs:
                skips.append(h)
                h = blk(h)
            for blk in self.ups:
                h = blk(h)
                skip = skips.pop()
                # Shape-Restaurierung nach ungeraden F/T: zentrierter Crop.
                if h.shape[-2:] != skip.shape[-2:]:
                    h = h[..., : skip.shape[-2], : skip.shape[-1]]
                h = h + skip
            delta = self.out_proj(h).squeeze(1)
            # Radiant-Begrenzung (§0a): endlich und in [-π, π].
            return torch.clamp(delta, -np.pi, np.pi)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)  # §G5 (copilot-instructions.md)
        model = TPv2PhaseNet(n_freq, hidden, blocks)
    model.eval()
    return model


# ── Inferenz-Hülle (Layout-sicher, deterministisch) ──────────────────────────


class TPv2PhaseEstimator:
    """Inferenz-Hülle um ``TPv2PhaseNet`` — deterministisch, NaN/Inf-sicher.

    Die Instanz hält nur unveränderliche Gewichte (kein Song-State,
    §V8/§G1 (copilot-instructions.md)).
    """

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        *,
        n_freq: int = N_FREQ,
        context_frames: int = CONTEXT_FRAMES,
        hidden: int = DEFAULT_HIDDEN,
        blocks: int = DEFAULT_BLOCKS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        import torch

        self._n_freq = int(n_freq)
        self._checkpoint: Path | None = Path(checkpoint) if checkpoint is not None else None
        self._source: str

        if self._checkpoint is not None and self._checkpoint.is_file():
            try:
                payload = torch.load(self._checkpoint, map_location="cpu", weights_only=True)
                cfg = dict(payload.get("tp_v2_config", {}))
                self._net = build_model(
                    n_freq=int(cfg.get("n_freq", n_freq)),
                    context_frames=int(cfg.get("context_frames", context_frames)),
                    hidden=int(cfg.get("hidden", hidden)),
                    blocks=int(cfg.get("blocks", blocks)),
                    seed=seed,
                )
                self._net.load_state_dict(payload["model_state_dict"], strict=True)
                self._net.eval()
                self._source = f"checkpoint:{self._checkpoint.name}"
                logger.info("TP-V2-Phasenschätzer: Artefakt geladen: %s", self._checkpoint)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "TP-V2-Phasenschätzer: Artefakt %s nicht ladbar (%s) — deterministische "
                    "Seed-Gewichte (seed=%d) nur für Test-/Challenger-Betrieb (§V6 (copilot-instructions.md))",
                    self._checkpoint,
                    exc,
                    seed,
                )
                self._net = build_model(
                    n_freq=n_freq, context_frames=context_frames, hidden=hidden, blocks=blocks, seed=seed
                )
                self._source = "seed_fallback"
        else:
            self._net = build_model(
                n_freq=n_freq, context_frames=context_frames, hidden=hidden, blocks=blocks, seed=seed
            )
            self._source = "seed"
            if self._checkpoint is not None:
                logger.warning(
                    "TP-V2-Phasenschätzer ohne trainiertes Artefakt (%s) — deterministische "
                    "Seed-Gewichte nur für Test-/Challenger-Betrieb (§V6 (copilot-instructions.md))",
                    self._checkpoint,
                )

    @property
    def source(self) -> str:
        """Herkunft der Gewichte (``checkpoint:…``/``seed``/``seed_fallback``)."""
        return self._source

    # ------------------------------------------------------------------

    def estimate_delta_phase(self, log_mag: np.ndarray) -> np.ndarray:
        """Phasen-Korrektur ``Δφ(f, t)`` in Radiant aus log-Magnituden.

        Args:
            log_mag: ``(F, T)`` oder ``(1, F, T)`` float — z. B. ``np.log1p(|STFT|)``.

        Returns:
            ``(F, T)`` float32 in ``[-π, π]``, immer endlich (§0a).
            Ungültige Argumente werfen ``ValueError`` (laut, kein stiller Fallback).
        """
        import torch

        arr = _to_freq_time(log_mag)
        if arr.shape[0] != self._n_freq:
            raise ValueError(f"Frequenz-Bins {arr.shape[0]} ≠ Vertrag {self._n_freq} (N_FFT={N_FFT} erwartet)")
        if arr.size == 0:
            raise ValueError("Leeres Spektrogramm — kein Δφ definierbar")
        arr = _sanitize(arr, "Eingangs-Log-Magnituden")

        feed = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None]
        with torch.inference_mode():  # deterministisch (§G5 (copilot-instructions.md))
            out = self._net(feed)
        delta = np.asarray(out, dtype=np.float32)[0]
        if delta.shape != arr.shape:
            raise ValueError(f"Modell-Ausgabe {delta.shape} ≠ Eingang {arr.shape}")
        # §0a: nie nicht-finite Werte nach außen; Radiant-Begrenzung.
        delta = np.nan_to_num(delta, nan=0.0, posinf=float(np.pi), neginf=float(-np.pi))
        res: np.ndarray = np.clip(delta, -float(np.pi), float(np.pi)).astype(np.float32)
        return res


# ── Fassade (lazy Singleton, Muster rt60_regressor/aspade_declipper) ─────────


_estimator_singleton: TPv2PhaseEstimator | None = None


def get_tp_v2_phase_estimator() -> TPv2PhaseEstimator:
    """Prozess-weite Singleton-Instanz (lazy) — ohne Artefakt mit Seed-Gewichten.

    Die Instanz hält nur unveränderliche Gewichte (kein Song-State,
    §V8/§G1 (copilot-instructions.md)).
    """
    global _estimator_singleton  # pylint: disable=global-statement
    if _estimator_singleton is None:
        _estimator_singleton = TPv2PhaseEstimator(checkpoint=DEFAULT_CHECKPOINT)
    return _estimator_singleton


def estimate_transient_phase(log_mag: np.ndarray) -> np.ndarray | None:
    """Challenger-Fassade für den Transienten-/Phasen-Rekonstruktionspfad.

    ``None`` ⇒ nicht verfügbar oder Fehler (§V6-Warnung erfolgte im Inneren bzw.
    hier); der Aufrufer fällt dann auf den bestehenden Pfad
    (``restore_phase_coherence``) zurück. Fehler laufen als Exception zum
    Aufrufer nur bei ungültigen Argumenten (``ValueError``, laut) — Modell-
    Fehler sind abgefangen, protokolliert und führen zu ``None`` (kein stiller
    Fallback).
    """
    try:
        est = get_tp_v2_phase_estimator()
        return est.estimate_delta_phase(log_mag)
    except ValueError:
        raise  # ungültige Argumente bleiben laut (Vertragsbruch des Aufrufers)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "TP-V2-Phasenschätzung fehlgeschlagen (%s) — bestehender Phasen-Pfad bleibt "
            "(§V6 (copilot-instructions.md))",
            exc,
        )
        return None
