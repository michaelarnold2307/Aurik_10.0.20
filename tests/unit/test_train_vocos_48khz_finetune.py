"""Vertragstests D-K2-3 (scripts/train_vocos_48khz_finetune.py, SOTA-M6).

Ohne Modell-Laden: Daten-/CLI-Pfade testbar, Modell-Teile nur fail-closed
(§G5 (GEBOTE.md): keine umgebungsabhaengigen Kipp-Tests). Zitierdisziplin
AGENTS.md §5.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.io import wavfile

import scripts.train_vocos_48khz_finetune as tvf

_SRC = pathlib.Path(tvf.__file__).read_text(encoding="utf-8")


def _musdb(tmp_path: pathlib.Path, n_tracks: int = 3) -> pathlib.Path:
    root = tmp_path / "musdb18hq"
    for i in range(n_tracks):
        d = root / "train" / f"track{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(i)
        t = np.arange(48000 * 6) / 48000.0
        x = 0.2 * np.sin(2 * np.pi * (80.0 + 30.0 * i) * t) + 0.01 * rng.standard_normal(t.size)
        wavfile.write(d / "vocals.wav", 48000, (x * 32767).astype(np.int16))
    return root


def test_cli_vertrag_r5() -> None:
    """Kette-4-Vertrag R5: --musdb-root/--epochs/--device/--out-dir exakt bedient."""
    args = tvf.parse_args(
        [
            "--musdb-root",
            "data/musdb18hq",
            "--epochs",
            "30",
            "--device",
            "cuda",
            "--out-dir",
            "models/vocos_48khz/finetune",
        ]
    )
    assert args.epochs == 30
    assert args.device == "cuda"
    assert args.seed == 42
    assert args.lr == 2e-5
    assert args.batch == 8


def test_stable_track_list_determinismus(tmp_path: pathlib.Path) -> None:
    root = _musdb(tmp_path)
    a = tvf._stable_track_list(root, 42)
    b = tvf._stable_track_list(root, 42)
    assert [str(p) for p in a] == [str(p) for p in b]
    assert len(a) == 3
    assert all((p / "vocals.wav").is_file() for p in a)


def test_stable_track_list_fail_closed(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        tvf._stable_track_list(tmp_path / "fehlt", 42)


def test_windows_layout_und_determinismus() -> None:
    x_stereo = np.stack([np.ones(96000), np.zeros(96000)], axis=0)  # (C, N)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    w1 = tvf._windows(x_stereo, rng1, 2)
    w2 = tvf._windows(x_stereo, rng2, 2)
    assert len(w1) == len(w2) == 2
    for a, b in zip(w1, w2):
        assert np.array_equal(a, b)
        assert a.shape == (192000,)
        assert np.all(np.isfinite(a))
    # Layout-Invariante: (N, C) liefert denselben Mono-Mix wie (C, N)
    x_nc = x_stereo.T
    rng3 = np.random.default_rng(42)
    w3 = tvf._windows(x_nc, rng3, 2)
    for a, b in zip(w1, w3):
        assert np.array_equal(a, b)


def test_load_vocos_fail_closed_missing_checkpoint(monkeypatch, tmp_path: pathlib.Path) -> None:
    import sys
    import types

    fake = types.ModuleType("vocos")
    fake.pretrained = types.ModuleType("vocos.pretrained")
    fake.pretrained.Vocos = type("Vocos", (), {})  # nur fuer den Import-Pfad noetig
    monkeypatch.setitem(sys.modules, "vocos", fake)
    monkeypatch.setitem(sys.modules, "vocos.pretrained", fake.pretrained)
    monkeypatch.setattr(tvf, "_ROOT", tmp_path)
    (tmp_path / "models" / "vocos_48khz").mkdir(parents=True)
    with pytest.raises(SystemExit, match="unvollstaendig"):
        tvf._load_vocos("cpu")


def test_load_vocos_fail_closed_missing_package(monkeypatch, tmp_path: pathlib.Path) -> None:
    (tmp_path / "models" / "vocos_48khz").mkdir(parents=True)
    monkeypatch.setattr(tvf, "_ROOT", tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "vocos", None)
    with pytest.raises(SystemExit, match="vocos-Paket fehlt"):
        tvf._load_vocos("cpu")


def test_keine_flag_flips() -> None:
    assert "flag_freigabe" not in _SRC
    assert ".flags" not in _SRC


def test_never_worsen_gate_erhaelt_baseline_bei_degradation() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    baseline = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)

    assert not tvf._never_worsen_gate(model, baseline, 1.0, 1.02)
    for key, value in baseline.items():
        assert torch.equal(model.state_dict()[key], value)


def test_never_worsen_gate_akzeptiert_innerhalb_toleranz() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    baseline = {key: value.detach().clone() for key, value in model.state_dict().items()}

    assert tvf._never_worsen_gate(model, baseline, 1.0, 1.01)


def test_epoch_gate_laeuft_alle_zwei_epochen_und_am_ende() -> None:
    assert not tvf._epoch_gate_due(1, 5)
    assert tvf._epoch_gate_due(2, 5)
    assert not tvf._epoch_gate_due(3, 5)
    assert tvf._epoch_gate_due(5, 5)
    assert tvf._epoch_gate_due(1, 1, smoke=True)


def test_epoch_never_worsen_gate_verwirft_degradierte_epoche(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    baseline = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.fill_(2.0)
    monkeypatch.setattr(tvf, "_validation_loss", lambda *_args: 1.02)

    result = tvf._epoch_never_worsen_gate(model, baseline, 1.0, [], torch.device("cpu"))

    assert result == {"candidate_loss": 1.02, "never_worsen": False}
    assert torch.equal(model.weight, baseline["weight"])


def test_epoch_never_worsen_gate_schlaegt_bei_messfehler_fail_closed(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    baseline = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.fill_(2.0)

    def _messfehler(*_args):
        raise ValueError("kein Holdout")

    monkeypatch.setattr(tvf, "_validation_loss", _messfehler)
    result = tvf._epoch_never_worsen_gate(model, baseline, 1.0, [], torch.device("cpu"))

    assert result == {"candidate_loss": None, "never_worsen": False}
    assert torch.equal(model.weight, baseline["weight"])


def test_run_verwirft_resume_anker_nach_fehlgeschlagenem_epoch_gate(monkeypatch, tmp_path: pathlib.Path) -> None:
    import sys
    import types
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    class TinyVocos(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))

    model = TinyVocos()
    baseline = model.scale.detach().clone()
    tracks = [pathlib.Path("holdout"), pathlib.Path("train")]
    monkeypatch.setattr(tvf, "_load_vocos", lambda _device: model)
    monkeypatch.setattr(tvf, "_stable_track_list", lambda *_args: tracks)
    monkeypatch.setattr(tvf, "_windows", lambda *_args: [np.zeros(16, dtype=np.float32)])
    monkeypatch.setattr(tvf, "_mel_from_wave", lambda _wave: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(
        tvf,
        "_decode_mel",
        lambda current, mel, _device: current.scale.expand(mel.shape[0], 16),
    )
    monkeypatch.setattr(
        tvf,
        "_spectral_loss",
        lambda prediction, target, _current, _device: torch.mean((prediction - target) ** 2),
    )

    soundfile = types.ModuleType("soundfile")
    soundfile.read = lambda *_args, **_kwargs: (np.zeros((16, 1), dtype=np.float32), 48000)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)

    validation_losses = iter((1.0, 1.0, 1.1))
    monkeypatch.setattr(tvf, "_validation_loss", lambda *_args: next(validation_losses))
    gate_results = []
    epoch_gate = tvf._epoch_never_worsen_gate

    def _record_gate(*args, **kwargs):
        result = epoch_gate(*args, **kwargs)
        gate_results.append(result)
        return result

    monkeypatch.setattr(tvf, "_epoch_never_worsen_gate", _record_gate)
    out_dir = tmp_path / "vocos-out"
    args = SimpleNamespace(
        seed=42,
        device="cpu",
        musdb_root=tmp_path,
        out_dir=out_dir,
        smoke=False,
        epochs=5,
        batch=1,
        lr=1e-3,
    )

    manifest = tvf._run(args)

    assert len(gate_results) == 2, "Bei fünf Epochen muss das Gate in Epoche 2 und 4 laufen"
    assert [result["never_worsen"] for result in gate_results] == [True, False]
    assert manifest["status"] == "degraded"
    assert manifest["validation"]["never_worsen"] is False
    assert manifest["validation"]["candidate_loss"] == pytest.approx(1.1)

    anchor = torch.load(out_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert anchor["epoch"] == 0
    assert anchor["status"] == "degraded"
    assert torch.equal(anchor["state_dict"]["scale"], baseline)
    assert torch.equal(model.scale.detach(), baseline)
    backups = list((out_dir / ".checkpoint_backups").glob("checkpoint_latest.pt.*.bak"))
    assert len(backups) == 1
    accepted_anchor = torch.load(backups[0], map_location="cpu", weights_only=False)
    assert accepted_anchor["epoch"] == 2
    assert accepted_anchor["never_worsen"] is True


def test_resume_baseline_gate_vergleicht_gegen_originalcheckpoint(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    baseline = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.fill_(1.5)

    monkeypatch.setattr(tvf, "_validation_loss", lambda current, _tracks, _device: float(current.weight.item()))
    baseline_loss = tvf._baseline_validation_loss(model, baseline, [], torch.device("cpu"))

    assert baseline_loss == 1.0
    assert model.weight.item() == 1.5, "Resume-Kandidat muss nach Baseline-Messung wieder aktiv sein"
    assert not tvf._never_worsen_gate(model, baseline, baseline_loss, float(model.weight.item()))
    assert model.weight.item() == 1.0, "Never-worsen muss bei kumulierter Resume-Degradation auf Basis zurückfallen"


def test_never_worsen_gate_lehnt_nichtfinite_metriken_ab() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    baseline = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)

    assert not tvf._never_worsen_gate(model, baseline, 1.0, float("nan"))
    for key, value in baseline.items():
        assert torch.equal(model.state_dict()[key], value)


def test_spectral_loss_erhaelt_gradienten_bis_zum_modell() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((1, 4096), dtype=torch.float32)
    prediction = torch.full((1, 4096), 0.1, dtype=torch.float32, requires_grad=True)
    model = type("FeatureModel", (), {"feature_extractor": staticmethod(lambda audio: audio)})()

    loss = tvf._spectral_loss(prediction, target, model, torch.device("cpu"))

    assert torch.is_tensor(loss)
    assert loss.requires_grad
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad).item() > 0


# ── Regressionen 2026-11-08 (Kette 4 R5, D-K2-3) ────────────────────────────
def test_kein_hub_download_fuer_lokales_checkpoint() -> None:
    """Regression: `Vocos.from_pretrained(<lokaler Pfad>)` starb an der Repo-ID-Prüfung.

    Produktionsbefund: Exit 1 nach 2,1 s, „Repo id must be in the form …". Ein
    lokales Checkpoint muss offline über `from_hparams` + `load_state_dict` laden.
    """
    assert "model = Vocos.from_pretrained" not in _SRC
    assert "Vocos.from_hparams(" in _SRC


def test_load_vocos_offline(monkeypatch, tmp_path: pathlib.Path) -> None:
    import sys
    import types

    torch = pytest.importorskip("torch")
    calls: dict[str, object] = {}

    class FakeVocos:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise AssertionError("from_pretrained darf fuer ein lokales Checkpoint nicht genutzt werden")

        @classmethod
        def from_hparams(cls, config_path: str):
            calls["hparams"] = config_path
            obj = cls()
            obj.load_state_dict = lambda sd: calls.setdefault("state", sd)  # type: ignore[attr-defined]
            obj.to = lambda dev: obj  # type: ignore[attr-defined]
            obj.eval = lambda: obj  # type: ignore[attr-defined]
            return obj

    fake = types.ModuleType("vocos")
    fake.pretrained = types.ModuleType("vocos.pretrained")
    fake.pretrained.Vocos = FakeVocos
    monkeypatch.setitem(sys.modules, "vocos", fake)
    monkeypatch.setitem(sys.modules, "vocos.pretrained", fake.pretrained)
    monkeypatch.setattr(tvf, "_ROOT", tmp_path)
    ckpt = tmp_path / "models" / "vocos_48khz"
    ckpt.mkdir(parents=True)
    (ckpt / "config.yaml").write_text("backbone: {}\n", encoding="utf-8")
    torch.save({"w": torch.zeros(1)}, ckpt / "pytorch_model.bin")

    tvf._load_vocos("cpu")
    assert str(calls["hparams"]).endswith("config.yaml")
    assert "w" in calls["state"]  # type: ignore[operator]
