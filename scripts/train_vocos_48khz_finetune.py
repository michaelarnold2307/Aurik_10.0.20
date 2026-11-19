"""D-K2-3 (SOTA-M6): Musik-Finetune des Vocos-48kHz-Vocoders auf MUSDB18-HQ-Gesang.

Rolle und Grenzen (Register .github/SOTA_DEFICIT_REGISTER.md, D-K2-3): Der
Vocos-48kHz-Vocoder (upstream kittn/vocos-mel-48khz-alpha1, Sprach-Korpora)
rekonstruiert Musik-Gesang nur mittelmaessig — Kandidatenpfad im
Studio-Finisher. Dieser Job ist der TRAININGSWEG (Kette 4 R5,
scripts/gpu_kette4_modellkoepfe.sh); die PRIMAER-ROLLEN-Entscheidung
(Studio-Finisher: Vocos vs. BigVGAN-F3) bleibt MENSCHLICH (C4-Hoerstichprobe,
Hoerordnung Ebene 1). KEIN Deploy, KEIN Flag-Flip (§V8 (copilot-instructions.md)/G1-Grenze: nur
Checkpoint + Manifest nach models/vocos_48khz/finetune/).

Architektur-Paritaet (§III.9 (copilot-instructions.md)): Training nutzt das
Upstream-Paket `vocos==0.1.0` (MIT, Siuzdak 2023 — wie beim ONNX-Export,
scripts/export_vocos_48khz_onnx.py Zeile 54) und laedt das deployte Checkpoint
models/vocos_48khz/{config.yaml,pytorch_model.bin} via
vocos.pretrained.Vocos.from_pretrained — identische Architektur wie das
deployte vocos_48khz.onnx. Die Eingangs-Mel-Extraktion nutzt exakt
`VocosPlugin._compute_mel` aus dem Produktionspfad (48 kHz, n_fft 2048,
hop 256, 128 Bänder); der Modell-Feature-Extractor wird nur für den
Rekonstruktions-Verlust verwendet. Der Flugschein pinnt diese Trennung.

Determinismus (§G5 (copilot-instructions.md)): Seed 42, deterministische
Track-Auswahl und Segment-Fenster, kein time.time in Entscheidungslogik.

Finetune-Rezept: reiner Rekonstruktions-Finetune (Multi-Resolution-Spektral:
L1-Mel + L1-STFT-Magnituden bei 2 Aufloesungen). Der adversarialen
Diskriminator-Block des Upstream-Trainers wird bewusst NICHT nachgebaut
(bewertende Instanz waere ohne die Upstream-Discriminator-Checkpoints ein
ad-hoc-Neubau, §V7 (copilot-instructions.md)); Rekonstruktion-only ist die
Hausroutine (DiffWave-F1: models/ear_vae_upstream/finetune_music.py,
Commits 98276a32/4b783d18). Never-worsen-Gate: Validierungs-Verlust des
Finetune-Checkpoints muss <= Basis-Checkpoint sein (Toleranz 1 %), sonst
bleibt der Basis-Checkpoint und das Manifest traegt Status "degraded"
(§V6 (copilot-instructions.md): Warnung + Begruendung, kein Silent-Failure).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_SR = 48000
_WIN_S = 4.0
_DEFAULT_BATCH = 8
_DEFAULT_LR = 2e-5
_EPOCH_GATE_N = 2
GATE_HINT = (
    "Eingangs-Mel ueber plugins.vocos_plugin.VocosPlugin._compute_mel; "
    "Rekonstruktionsverlust darf den Produktions-Preprozessor nicht ersetzen."
)

# Manifest-Schema (analog aurik.wf4_warp_pairs/1 und aurik.train_job/1):
_SCHEMA = "aurik.vocos48k_finetune/1"


def parse_args(argv=None) -> argparse.Namespace:
    """CLI-Vertrag gemaess scripts/gpu_kette4_modellkoepfe.sh R5."""
    ap = argparse.ArgumentParser(description="D-K2-3: Vocos-48kHz Musik-Finetune auf MUSDB18-HQ-Gesang")
    ap.add_argument("--musdb-root", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "models" / "vocos_48khz" / "finetune")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=_DEFAULT_LR)
    ap.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    ap.add_argument("--smoke", action="store_true", help="1 Epoch, 2 Tracks, Batch 2")
    return ap.parse_args(argv)


def _stable_track_list(musdb_root: Path, seed: int) -> list[Path]:
    """Deterministische, seed-feste Track-Auswahl aus dem MUSDB18-HQ-Layout."""
    train_dir = Path(musdb_root) / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"MUSDB18-HQ nicht gefunden unter {train_dir} — --musdb-root pruefen")
    rng = np.random.default_rng(seed)
    tracks = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not tracks:
        raise SystemExit(f"Keine Trainings-Tracks unter {train_dir}")
    order = rng.permutation(len(tracks))
    return [tracks[i] for i in order]


def _windows(x: np.ndarray, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """n deterministische, nicht-ueberlappende 4-s-Mono-Fenster aus einem Vocal."""
    if x.ndim == 2:
        # Layout-Invariante (AGENTS.md §3): Stereo (C, N) oder (N, C) -> Mono-Mix.
        x = x.mean(axis=1) if x.shape[0] < x.shape[1] else x.mean(axis=0)
    win = int(_WIN_S * _SR)
    if x.shape[-1] < win + _SR:
        x = np.pad(x, (0, win + _SR - x.shape[-1]))
    max_start = x.shape[-1] - win
    starts = sorted(rng.integers(0, max_start, size=n))
    return [np.ascontiguousarray(x[s : s + win], dtype=np.float32) for s in starts]


def _mel_from_wave(wav: np.ndarray) -> np.ndarray:
    """Produktions-Mel [128, T]; dieselbe Implementierung wie der Vocos-Plugin-Pfad."""
    from plugins.vocos_plugin import _HOP_48K, _N_FFT_48K, _N_MELS_48K, VocosPlugin

    mono = np.asarray(wav, dtype=np.float32).reshape(-1)
    return VocosPlugin._compute_mel(
        mono,
        _SR,
        n_mels=_N_MELS_48K,
        n_fft=_N_FFT_48K,
        hop=_HOP_48K,
    )


def _spectral_loss(y_pred, y_true, model, device) -> torch.Tensor:
    """Multi-Resolution: L1-Mel + L1-STFT (2048/512, hop 1/4) — Vocos-Stil."""
    import torch
    import torch.nn.functional as F

    eps = 1e-8
    yt = y_true.to(device)
    total = 0.0
    try:
        with torch.no_grad():
            mel_t = model.feature_extractor(yt)
        mel_p = model.feature_extractor(y_pred)
        total = total + F.l1_loss(mel_p, mel_t)
    except Exception as exc:  # pragma: no cover - defensiv, §V6 (copilot-instructions.md)
        logger.warning("Mel-Verlust nicht berechenbar (%s) — STFT-only", exc)
    for n_fft, hop in ((2048, 512), (512, 128)):
        st = torch.stft(
            y_pred.squeeze(1),
            n_fft=n_fft,
            hop_length=hop,
            return_complex=True,
            window=torch.hann_window(n_fft, device=device),
        )
        st_t = torch.stft(
            yt.squeeze(1),
            n_fft=n_fft,
            hop_length=hop,
            return_complex=True,
            window=torch.hann_window(n_fft, device=device),
        )
        total = total + F.l1_loss(st.abs(), st_t.abs().detach()) + eps
    return total


def _validation_loss(model, tracks: list[Path], device) -> float:
    """Deterministischer Holdout-Verlust über eine Produktions-Mel-Probe je Track."""
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    if not tracks:
        raise ValueError("Vocos-Never-worsen-Gate benötigt mindestens einen Holdout-Track")

    model_was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        with torch.no_grad():
            for track in tracks:
                vocal = Path(track) / "vocals.wav"
                if not vocal.is_file():
                    raise ValueError(f"Vocos-Validierung: Vocal fehlt: {vocal}")
                data, sr = sf.read(vocal, dtype="float32", always_2d=True)
                if sr != _SR:
                    from scipy.signal import resample_poly

                    n_samples = int(round(len(data) * _SR / sr))
                    data = resample_poly(data, _SR, sr, axis=0)[:n_samples].astype(np.float32)
                mono = data.mean(axis=1)
                window = _windows(mono, np.random.default_rng(42), 1)[0]
                target = torch.from_numpy(window[None, :]).to(device)
                mel = torch.from_numpy(_mel_from_wave(window)[None, ...]).to(device)
                restored = _decode_mel(model, mel, device)
                if restored is None or not hasattr(restored, "shape"):
                    raise ValueError(f"Vocos-Validierung: ungültige Modellausgabe für {vocal}")
                if restored.shape[-1] < target.shape[-1]:
                    restored = F.pad(restored, (0, target.shape[-1] - restored.shape[-1]))
                else:
                    restored = restored[..., : target.shape[-1]]
                if restored.dim() == 3 and restored.shape[1] == 1:
                    restored = restored.squeeze(1)
                loss = _spectral_loss(restored, target, model, device)
                loss_value = float(loss.item())
                if not np.isfinite(loss_value) or loss_value < 0:
                    raise ValueError(f"Vocos-Validierung: ungültiger Verlust für {vocal}: {loss_value}")
                losses.append(loss_value)
    finally:
        model.train(model_was_training)
    return float(np.mean(losses))


def _baseline_validation_loss(model, base_state: dict, tracks: list[Path], device) -> float:
    """Misst den echten Basis-Checkpoint und stellt danach einen Resume-Kandidaten wieder her."""
    candidate_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    try:
        model.load_state_dict(base_state)
        return _validation_loss(model, tracks, device)
    finally:
        model.load_state_dict(candidate_state)


def _epoch_never_worsen_gate(model, base_state: dict, baseline_loss: float, tracks: list[Path], device) -> dict:
    """Prüft einen Epochen-Checkpoint gegen die unveränderte Basis; Fehler verwerfen den Kandidaten."""
    try:
        candidate_loss = _validation_loss(model, tracks, device)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Vocos-Epoch-Gate nicht messbar: %s — Baseline wird wiederhergestellt (§V6 (copilot-instructions.md))",
            exc,
        )
        candidate_loss = float("inf")
    passed = _never_worsen_gate(model, base_state, baseline_loss, candidate_loss)
    return {
        "candidate_loss": candidate_loss if np.isfinite(candidate_loss) else None,
        "never_worsen": passed,
    }


def _epoch_gate_due(epoch_number: int, total_epochs: int, smoke: bool = False) -> bool:
    """Gate alle zwei Epochen, am Laufende und bei Smoke-Läufen."""
    return smoke or epoch_number % _EPOCH_GATE_N == 0 or epoch_number == total_epochs


def _never_worsen_gate(
    model, base_state: dict, base_loss: float, candidate_loss: float, tolerance: float = 0.01
) -> bool:
    """Verwirft nichtfinite oder mehr als 1 % schlechtere Finetunes (§V6 (copilot-instructions.md))."""
    passed = (
        np.isfinite(base_loss)
        and np.isfinite(candidate_loss)
        and base_loss >= 0.0
        and candidate_loss >= 0.0
        and candidate_loss <= base_loss * (1.0 + tolerance)
    )
    if not passed:
        model.load_state_dict(base_state)
    return bool(passed)


def _decode_mel(model, mel, device):
    """Mel -> Waveform ueber den kanonischen Vocos-Pfad (vocos.pretrained.Vocos.decode)."""
    import torch

    m = mel.to(device)
    if hasattr(model, "decode"):
        return model.decode(m)
    if hasattr(model, "backbone") and hasattr(model, "head"):
        # Fallback: expliziter backbone->head-Pfad (wie scripts/export_vocos_48khz_onnx.py)
        return model.head(model.backbone(m))
    raise SystemExit("Modell-API unerwartet (kein decode/backbone+head) — §V7 (copilot-instructions.md): nicht raten")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_vocos(device: str):
    """Fail-closed: Upstream-Paket + deploytes Checkpoint laden (§V6 (copilot-instructions.md))."""
    try:
        from vocos.pretrained import Vocos  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "vocos-Paket fehlt — .venv_aurik/bin/pip install vocos==0.1.0 "
            f"(MIT, Siuzdak 2023; ONNX-Export nutzt dasselbe Paket). Fehler: {exc}"
        ) from exc
    ckpt_dir = _ROOT / "models" / "vocos_48khz"
    for needed in ("config.yaml", "pytorch_model.bin"):
        if not (ckpt_dir / needed).is_file():
            raise SystemExit(f"Deploytes Checkpoint unvollstaendig: {ckpt_dir / needed}")
    # Lokales Checkpoint: Vocos.from_pretrained() reicht den Pfad an
    # huggingface_hub.hf_hub_download() weiter und akzeptiert nur Repo-IDs
    # ("Repo id must be in the form 'repo_name' oder 'namespace/repo_name'") —
    # Befund 2026-11-08 (Kette 4 R5, Exit 1 nach 2 s). Deshalb der Offline-Pfad:
    # from_hparams() + load_state_dict() auf die deployten Dateien (§G5 (copilot-instructions.md): kein
    # Netzwerk, deterministisch derselbe Stand wie das ONNX-Artefakt).
    try:
        import torch as _torch

        model = Vocos.from_hparams(str(ckpt_dir / "config.yaml"))
        state = _torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Vocos-Checkpoint nicht ladbar: {exc}") from exc
    model = model.to(device)
    model.eval()
    return model


def _run(args: argparse.Namespace) -> dict:
    """Traingslauf (nur Aufruf via Kette 4 R5 / manuell; kein Auto-Start)."""
    import torch
    import torch.optim as optim

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = _load_vocos(str(device))
    model.train()
    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    # Nur die Rekonstruktions-Komponenten optimieren (kein Discriminator vorhanden):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    # Resume-Anker (D-K3-72-Klasse, 2026-11-08): checkpoint_latest.pt je Epoche
    # + Load beim Neustart — sonst verliert ein 6-h-Timeout den ganzen Lauf.
    _anchor_v = Path(args.out_dir) / "checkpoint_latest.pt"
    _anchor_v.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if _anchor_v.exists():
        import torch as _torch_v

        try:
            _ckv = _torch_v.load(_anchor_v, map_location="cpu", weights_only=False)
            model.load_state_dict(_ckv["state_dict"])
            start_epoch = int(_ckv.get("epoch", 0))
            logger.info("Resume ab Epoche %d aus %s", start_epoch, _anchor_v)
        except Exception as _exc:  # fail-closed: nie still bei Epoche 0 neu starten
            logger.error(
                "Resume-Anker korrupt (%s): %s — Abbruch Exit 2 (§V6 (copilot-instructions.md))", _anchor_v, _exc
            )
            raise SystemExit(2) from _exc

    tracks = _stable_track_list(args.musdb_root, args.seed)
    if args.smoke:
        tracks, epochs, batch = tracks[:2], 1, 2
    else:
        epochs, batch = args.epochs, args.batch

    val_tracks = tracks[::5]
    train_tracks = [track for index, track in enumerate(tracks) if index % 5 != 0]
    if not train_tracks:
        raise SystemExit("Vocos-Finetune benötigt mindestens zwei Tracks für Train/Holdout (fail-closed)")
    try:
        baseline_loss = _baseline_validation_loss(model, base_state, val_tracks, device)
    except (OSError, ValueError) as exc:
        logger.error("Vocos-Baseline-Validierung fehlgeschlagen: %s — Abbruch (fail-closed)", exc)
        raise SystemExit(3) from exc
    if not np.isfinite(baseline_loss):
        logger.error("Vocos-Baseline-Verlust ist nicht endlich — Abbruch (fail-closed)")
        raise SystemExit(3)

    rng = np.random.default_rng(args.seed)
    history: list[float] = []
    epoch_gate_failed = False
    candidate_validated = False
    candidate_loss = baseline_loss
    for epoch in range(start_epoch, epochs):
        epoch_losses: list[float] = []
        for track in train_tracks:
            vocal = Path(track) / "vocals.wav"
            if not vocal.is_file():
                logger.warning("Vocal fehlt: %s — uebersprungen (§V6 (copilot-instructions.md))", vocal)
                continue
            import soundfile as sf

            data, sr = sf.read(vocal, dtype="float32", always_2d=True)
            if sr != _SR:
                from scipy.signal import resample_poly

                num = int(round(len(data) * _SR / sr))
                data = resample_poly(data, _SR, sr, axis=0)[:num].astype(np.float32)
            x = data.mean(axis=1)
            wins = _windows(x, rng, 4 if args.smoke else 8)
            for i in range(0, len(wins), batch):
                chunk = np.stack(wins[i : i + batch])
                wt = torch.from_numpy(chunk).to(device)  # [B, T]
                mel = np.stack([_mel_from_wave(sample) for sample in chunk], axis=0)
                mel = torch.from_numpy(mel).to(device)
                y = _decode_mel(model, mel, device)
                if y is None or not hasattr(y, "shape"):
                    raise SystemExit(
                        "Modell-API unerwartet (kein backbone) — §V7 (copilot-instructions.md): nicht raten"
                    )
                if y.shape[-1] < wt.shape[-1]:
                    import torch.nn.functional as F

                    y = F.pad(y, (0, wt.shape[-1] - y.shape[-1]))
                else:
                    y = y[..., : wt.shape[-1]]
                if y.dim() == 3 and y.shape[1] == 1:
                    y = y.squeeze(1)  # [B, T] konsistent halten (Layout-Invariante AGENTS.md §3)
                loss = _spectral_loss(y, wt, model, device)
                optimizer.zero_grad(set_to_none=True)
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    raise RuntimeError(
                        "Vocos-Trainingsverlust ist nicht differenzierbar (§V6 (copilot-instructions.md))"
                    )
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        history.append(mean_loss)
        logger.info("Epoch %d/%d — Verlust %.5f", epoch + 1, epochs, mean_loss)
        gate_due = _epoch_gate_due(epoch + 1, epochs, args.smoke)
        if gate_due:
            gate_result = _epoch_never_worsen_gate(model, base_state, baseline_loss, val_tracks, device)
            candidate_validated = True
            gate_loss = gate_result["candidate_loss"]
            candidate_loss = float(gate_loss) if gate_loss is not None else float("inf")
            logger.info(
                "Epoche %d Epoch-Gate: Basis %.6f, Kandidat %s — %s",
                epoch + 1,
                baseline_loss,
                f"{candidate_loss:.6f}" if np.isfinite(candidate_loss) else "nicht messbar",
                "BESTANDEN" if gate_result["never_worsen"] else "ABGELEHNT",
            )
            if not gate_result["never_worsen"]:
                epoch_gate_failed = True
                from backend.core.training_artifacts import save_guarded as save_guarded_anchor

                save_guarded_anchor(
                    _anchor_v,
                    {"state_dict": base_state, "epoch": 0, "loss": baseline_loss, "status": "degraded"},
                )
                break

            # Resume-Anker enthält ausschließlich Epochen mit bestandenem Gate.
            from backend.core.training_artifacts import save_guarded as _sg_vocos

            _sg_vocos(
                _anchor_v,
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "loss": mean_loss,
                    "validation_loss": candidate_loss,
                    "never_worsen": True,
                },
            )
        if args.smoke:
            break

    never_worsen = not epoch_gate_failed
    if never_worsen and not candidate_validated:
        final_gate = _epoch_never_worsen_gate(model, base_state, baseline_loss, val_tracks, device)
        candidate_loss = (
            float(final_gate["candidate_loss"]) if final_gate["candidate_loss"] is not None else float("inf")
        )
        never_worsen = bool(final_gate["never_worsen"])
    if not never_worsen and not epoch_gate_failed:
        logger.warning(
            "Vocos-Never-worsen-Gate verfehlt: Baseline %.6f, Finetune %.6f; Baseline wird gespeichert "
            "(§V6 (copilot-instructions.md))",
            baseline_loss,
            candidate_loss,
        )
        from backend.core.training_artifacts import save_guarded as save_guarded_anchor

        save_guarded_anchor(
            _anchor_v,
            {"state_dict": base_state, "epoch": 0, "loss": baseline_loss, "status": "degraded"},
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "vocos_48khz_music_ft.pt"
    from backend.core.training_artifacts import save_guarded

    save_guarded(
        ckpt_path,
        {"state_dict": model.state_dict(), "epochs": epochs, "seed": args.seed},
    )
    manifest = {
        "schema": _SCHEMA,
        "artifact": "vocos_48khz_music_ft",
        "base_checkpoint": str(_ROOT / "models" / "vocos_48khz"),
        "epochs": epochs,
        "seed": args.seed,
        "lr": args.lr,
        "batch": batch,
        "tracks": len(tracks),
        "final_train_loss": history[-1] if history else None,
        "validation": {
            "baseline_loss": baseline_loss,
            "candidate_loss": candidate_loss if np.isfinite(candidate_loss) else None,
            "tolerance": 0.01,
            "never_worsen": never_worsen,
        },
        "sha256_ckpt": _sha256(ckpt_path),
        "gate_hint": GATE_HINT,
        "status": "ok" if (never_worsen and history and np.isfinite(history[-1])) else "degraded",
        "hinweis": (
            "Rekonstruktions-only-Finetune; PRIMAER-ROLLEN-Entscheidung "
            "(Studio-Finisher Vocos vs. BigVGAN-F3) bleibt menschlich (C4). "
            "Nie eigenstaendig deployen oder Flag-Flips vornehmen."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(
        "D-K2-3 Finetune: musdb=%s epochs=%s device=%s out=%s", args.musdb_root, args.epochs, args.device, args.out_dir
    )
    manifest = _run(args)
    logger.info("Manifest: %s", manifest["status"])
    return 0 if manifest["status"] == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
