#!/usr/bin/env python3
"""Vergleichsmessung EAR-VAE v1 vs v2 — LUECKEN-MATRIX D.1c (Defizit D-K0-9).

Zweck: Vorbereitung der **menschlichen** Entscheidung, ob der v2-Upstream
(``models/ear_vae2_upstream/``) das deployte v1-MUSDB-Finetune
(``models/ear_vae/{encoder,decoder}.onnx``) ersetzt. **Keine Aktivierung,
kein Flag-Flip** (§v10.802 (copilot-instructions.md)) — das Urteil fällt die
menschliche Hör-Instanz (Hörordnung, Konfliktregel §7 in
``.github/instructions/hoerordnung.instructions.md``).

Methodik (Muster ``scripts/b3_stufenmessung.py``):

* gleiche Corpus-Fenster wie die Release-Pipeline
  (``scripts/release_candidate_pipeline._existing_corpus``-MUSTER und
  dessen ``_load_clip``);
* deterministische Seeds (§G5 (copilot-instructions.md)) — kein
  ``time.time()`` in der Entscheidungslogik;
* Blind-Paarung: je Fenster v1-Ausgabe gegen v2-Ausgabe, Seiten per Seed
  verlost, Bewertung sieht nur „links"/„rechts";
* Zielmetriken **nur** richtungszertifiziert (``musik_mos`` + ``defektlast``;
  Zertifikat ``docs/reports/current/qualitaetsmetrik_richtungszertifikat.json``);
* Never-worsen (G-4) mit JND-Toleranz + exakter Vorzeichen-Test
  (``scripts/ab_harness.sign_test_two_sided``), p < 0,05;
* Hör-Stimuli: verblindete Paare via ``scripts/ab_harness.prepare``
  (Versuchsblatt + getrennter Schlüssel) für die Hör-Instanz.

Fail-closed (§V7 (copilot-instructions.md)): Ist v2 technisch nicht
lauffähig (kein Modell-Gewicht vorhanden), schreibt dieses Skript die
Messdatei mit ``status = "nicht_ausfuehrbar"`` und dem technischen Grund —
**ohne** erfundene Zahlen (Beleg-Prinzip, ``.github/WOHLKLANG_CLAIMS.md``).

Aufruf (aus dem Bestands-Root, in dem ``models/`` und ``test_audio/`` liegen):
    nice -n 19 python3 -B scripts/ear_vae_v1_v2_vergleich.py --seed 42
    nice -n 19 python3 -B scripts/ear_vae_v1_v2_vergleich.py --preflight-only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

JND_TOLERANZ = 0.05  # MOS-Punkte — Never-worsen-Toleranz (Audibility-Stufe, wie b3)
SEED_VORGABE = 42  # Schritt 1d (docs/reports/current/2026-10-06_steps_0-5_preparation.md)
FENSTER_STARTS = (0.0, 2.5, 5.0, 7.5)  # Fenster-Schedule wie scripts/b3_stufenmessung.py
SCHLUESSEL_ID = "ear_vae_v1_v2"  # Namensraum für den Blind-RNG (§G5 (copilot-instructions.md))

MESSDATEI_VORGABE = "docs/reports/current/2026-11-03_ear_vae_v1_v2_vergleich.json"
STIMULI_VORGABE = "output/ear_vae_v1_v2/hoer_stimuli"

HOER_FRAGE = "Welcher Kandidat klingt besser/authentischer — A oder B? (blind, Reihenfolge verlost)"

# Kandidaten-Positionen des v2-Checkpoints (Upstream-README:
# "huggingface-cli download earlab/EAR_VAE2 --local-dir checkpoints/").
_V2_UPSTREAM_DIR = "models/ear_vae2_upstream"
_V2_CHECKPOINT_KANDIDATEN = (
    "models/ear_vae2_upstream/ear_vae2.pt",
    "models/ear_vae2_upstream/checkpoints/ear_vae2.pt",
    "models/ear_vae2_upstream/checkpoints/ear_vae2.pth",
)
_V2_CHECKPOINT_SUFFIXE = (".pt", ".pth", ".ckpt", ".safetensors")

_TECHNISCHER_GRUND_V2_FEHLT = (
    "models/ear_vae2_upstream/ enthaelt kein Modell-Gewicht: der Upstream-Klon "
    "umfasst nur Code/Docs/LICENSE (inference.py, ear_vae2/, configs/ear_vae2.json); "
    "das benoetigte Checkpoint ear_vae2.pt fehlt (Upstream-README: "
    "'huggingface-cli download earlab/EAR_VAE2 --local-dir checkpoints/'). "
    "Gemessener Artefakt-Bestand 0 Dateien/0 B (.github/ML_ARTIFACT_FINGERPRINTS.md). "
    "Zusaetzlich Format-/Architektur-Bruch: v2 = monolithischer Torch-Checkpoint "
    "(EarVAE2, C0=64/D=128/use_vae, 48 kHz) vs. v1 = ONNX-Paar "
    "(encoder/decoder, Latent [batch,64,latent_time], 44,1 kHz) — v2 hat weder "
    "ONNX-Export noch Adapter im Produktionspfad. Ohne Gewichte ist v2 technisch "
    "nicht lauffaehig — ehrlich dokumentiert statt konstruiert (§V7 (copilot-instructions.md))."
)


# ── V2-Verfuegbarkeit (fail-closed) ─────────────────────────────────────────


def v2_checkpoints_finden(modell_root: Path) -> list[Path]:
    """Sucht v2-Checkpoint-Kandidaten unter ``modell_root`` (Bestand 2026-11-03)."""
    gefunden: list[Path] = []
    upstream = modell_root / "ear_vae2_upstream"
    for relativ in (
        "ear_vae2_upstream/ear_vae2.pt",
        "ear_vae2_upstream/checkpoints/ear_vae2.pt",
        "ear_vae2_upstream/checkpoints/ear_vae2.pth",
    ):
        kandidat = modell_root / relativ
        if kandidat.is_file():
            gefunden.append(kandidat)
    if upstream.is_dir():
        for suffix in _V2_CHECKPOINT_SUFFIXE:
            for kandidat in sorted(upstream.glob(f"*{suffix}")):
                if kandidat not in gefunden:
                    gefunden.append(kandidat)
            for kandidat in (
                sorted((upstream / "checkpoints").glob(f"*{suffix}")) if (upstream / "checkpoints").is_dir() else []
            ):
                if kandidat not in gefunden:
                    gefunden.append(kandidat)
    return gefunden


def v2_verfuegbarkeit(modell_root: Path) -> dict[str, Any]:
    """Fail-closed-Pruefung: Ist v2 technisch lauffaehig?

    Nur **Vorhandensein** eines Checkpoints wird geprueft (ohne torch.load) —
    der Ladeweg liegt in ``v2_denoise_fenster``. Bei Fehlbestand enthaelt
    ``grund`` den dokumentierten technischen Grund (D-K0-9).
    """
    checkpoints = v2_checkpoints_finden(modell_root)
    verfuegbar = bool(checkpoints)
    return {
        "verfuegbar": verfuegbar,
        "checkpoints": [str(p) for p in checkpoints],
        "grund": None if verfuegbar else _TECHNISCHER_GRUND_V2_FEHLT,
        "quelle": "scripts/ear_vae_v1_v2_vergleich.py v2_verfuegbarkeit",
    }


# ── Corpus-Fenster (Muster release_candidate_pipeline) ──────────────────────


def fenster_plan(
    corpus_eintraege: list[dict[str, str]],
    *,
    starts: tuple[float, ...] = FENSTER_STARTS,
    seconds: float = 2.5,
) -> list[dict[str, Any]]:
    """Deterministischer Fenster-Plan je Corpus-Clip (§G5 (copilot-instructions.md)).

    Fenster werden zur Laufzeit verworfen, wenn ``_load_clip`` wegen Restlaufzeit
    < 1 s wirft (ValueError) — dieselbe Regel wie in ``scripts/b3_stufenmessung.py``.
    """
    plan: list[dict[str, Any]] = []
    for eintrag in corpus_eintraege:
        for start_s in starts:
            plan.append(
                {
                    "clip": Path(eintrag["path"]).stem,
                    "pfad": eintrag["path"],
                    "start_s": start_s,
                    "seconds": seconds,
                }
            )
    return plan


# ── Blind-Paarung + Auswertung (Muster b3_stufenmessung) ────────────────────


def blind_seiten_ziehen(rng: random.Random) -> bool:
    """True = v2 liegt links; False = v2 rechts. Blindheit: Bewertung sieht nur
    „links"/„rechts\" — die Zuordnung bleibt im Schlüssel (getrennt)."""
    return rng.random() < 0.5


def _zu_zeit_zuerst(x: np.ndarray) -> np.ndarray:
    """Stereo-Layout-Invariante (AGENTS.md §3): (C, N) → (N, C)."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] <= 8 and a.shape[-1] > 8:
        a = a.T
    return a


def _mono(x: np.ndarray) -> np.ndarray:
    a = _zu_zeit_zuerst(x)
    if a.ndim == 2:
        return a.mean(axis=1)
    return a


def auswertung(zeilen: list[dict[str, float]], *, jnd: float = JND_TOLERANZ) -> dict[str, Any]:
    """Never-worsen + Vorzeichen-Test (exakt) ueber die Fenster-Zeilen.

    Zeile: {"mos_v1", "mos_v2", "last_v1", "last_v2"} — Deltas sind v2−v1.
    Never-worsen: keine Metrik je Fenster schlechter als die JND-Toleranz
    (musik_mos hoeher = besser ⇒ delta ≥ −jnd; defektlast hoeher = schlechter
    ⇒ delta ≤ +jnd).
    """
    from scripts.ab_harness import sign_test_two_sided

    if not zeilen:
        return {
            "never_worsen": None,
            "vorteile": None,
            "p_wert": None,
            "signifikant": None,
            "erfuellt": False,
        }
    mos_deltas = [z["mos_v2"] - z["mos_v1"] for z in zeilen]
    last_deltas = [z["last_v2"] - z["last_v1"] for z in zeilen]
    never_worsen = all(d >= -jnd for d in mos_deltas) and all(d <= jnd for d in last_deltas)
    vorteile = sum(1 for d in mos_deltas if d > 0)
    p_wert = sign_test_two_sided(vorteile, len(mos_deltas))
    signifikant = bool(p_wert < 0.05 and sum(mos_deltas) > 0)
    return {
        "never_worsen": bool(never_worsen),
        "vorteile": f"{vorteile}/{len(mos_deltas)}",
        "p_wert": round(p_wert, 6),
        "signifikant": signifikant,
        "erfuellt": bool(never_worsen and signifikant),
    }


def vergleich_messen(
    fenster: list[tuple[str, np.ndarray, int]],
    seed: int,
    *,
    v1_denoise: Callable[[np.ndarray, int], np.ndarray] | None = None,
    v2_denoise: Callable[[np.ndarray, int], np.ndarray] | None = None,
    bewerte: Callable[[np.ndarray, int], tuple[float, float]] | None = None,
    ausgaben: dict[str, dict[str, Any]] | None = None,
    jnd: float = JND_TOLERANZ,
) -> dict[str, Any]:
    """Misst v1 vs. v2 blind je Fenster; Auswertung erst nach Entblindung.

    ``v1_denoise``/``v2_denoise`` liefern je Fenster das restaurierte Signal
    (beliebiges Stereo-Layout, wird normiert); ``bewerte(audio, sr)`` liefert
    ``(musik_mos, defektlast)``. Injektion erlaubt Tests ohne Modell-Artefakte.
    ``ausgaben`` (optional) wird mit den gerenderten Signalen befuellt — Grundlage
    der Hoer-Stimuli (``stimuli_vorbereiten``).
    """
    if v1_denoise is None:
        v1_denoise = v1_denoise_fenster
    if v2_denoise is None:
        v2_denoise = v2_denoise_fenster
    if bewerte is None:
        bewerte = _bewerte

    rng = random.Random(f"{seed}:{SCHLUESSEL_ID}")
    schluessel: dict[str, bool] = {}
    paare: list[dict[str, Any]] = []

    for i, (clip, audio, sr) in enumerate(fenster):
        out_v1 = _zu_zeit_zuerst(v1_denoise(audio, sr))
        out_v2 = _zu_zeit_zuerst(v2_denoise(audio, sr))
        tid = f"{clip}__w{i}"
        v2_links = blind_seiten_ziehen(rng)
        schluessel[tid] = v2_links
        werte: dict[str, dict[str, float]] = {}
        for seite, sig in (
            ("links", out_v2 if v2_links else out_v1),
            ("rechts", out_v1 if v2_links else out_v2),
        ):
            mos, last = bewerte(sig, sr)
            werte[seite] = {"musik_mos": float(mos), "defektlast": float(last)}
        paare.append({"trial_id": tid, "blindwerte": werte})
        if ausgaben is not None:
            ausgaben[tid] = {
                "v1": out_v1,
                "v2": out_v2,
                "input": _zu_zeit_zuerst(audio),
                "sr": int(sr),
            }

    # Entblindung + Auswertung — erst hier sind Zuordnungen sichtbar.
    zeilen: list[dict[str, Any]] = []
    for p in paare:
        v2_seite = "links" if schluessel[p["trial_id"]] else "rechts"
        v1_seite = "rechts" if v2_seite == "links" else "links"
        v1, v2 = p["blindwerte"][v1_seite], p["blindwerte"][v2_seite]
        zeilen.append(
            {
                "trial_id": p["trial_id"],
                "mos_v1": v1["musik_mos"],
                "mos_v2": v2["musik_mos"],
                "last_v1": v1["defektlast"],
                "last_v2": v2["defektlast"],
            }
        )

    ergebnis = auswertung(zeilen, jnd=jnd)
    ergebnis["zeilen"] = zeilen
    return ergebnis


# ── Render-Pfade (nur bei lauffaehigen Modellen genutzt) ────────────────────


def _bewerte(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(musik_mos, defektlast) — beide Metriken sind richtungszertifiziert
    (docs/reports/current/qualitaetsmetrik_richtungszertifikat.json)."""
    from plugins.musik_mos_plugin import get_musik_mos
    from scripts.quality_metric_direction_gate import _defektlast

    mono = _mono(audio)
    return float(get_musik_mos().estimate_mos(mono, sr).mos), float(_defektlast(mono, sr))


def v1_denoise_fenster(audio: np.ndarray, sr: int) -> np.ndarray:
    """v1 = deploytes MUSDB-Finetune ueber das Produktions-Plugin
    (plugins/ear_vae_denoiser.py, ONNX-Paar models/ear_vae/)."""
    from plugins.ear_vae_denoiser import get_ear_vae_denoiser

    out = get_ear_vae_denoiser().denoise(np.asarray(audio, dtype=np.float32), int(sr))
    if out is None:
        raise RuntimeError("v1-Denoise nicht verfuegbar (§V6 (copilot-instructions.md))")
    return _zu_zeit_zuerst(out)


_V2_MODELL_CACHE: dict[tuple[str, str], Any] = {}


def v2_denoise_fenster(
    audio: np.ndarray,
    sr: int,
    *,
    modell_root: Path | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """v2 = Upstream-EarVAE2 ueber dessen eigenes inference.py (Torch).

    Spiegelt ``process_file`` (Chunked Encode/Decode, samples_per_latent-Padding,
    deterministic=True) — **ohne** Lautheits-Normalisierung (upstream ``--no_norm``),
    damit Pegel v1/v2 vergleichbar bleiben. Fail-closed: ohne Checkpoint wirft
    die Funktion eine RuntimeError mit dem dokumentierten technischen Grund.
    """
    root = Path(modell_root) if modell_root is not None else Path("models")
    verf = v2_verfuegbarkeit(root)
    if not verf["verfuegbar"]:
        raise RuntimeError(verf["grund"])

    import importlib.util

    from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

    upstream = root / "ear_vae2_upstream"
    ckpt_pfad = verf["checkpoints"][0]
    cache_key = (ckpt_pfad, device)
    if cache_key not in _V2_MODELL_CACHE:
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        spec = importlib.util.spec_from_file_location("ear_vae2_upstream_inference", upstream / "inference.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Upstream-inference.py nicht ladbar: {upstream / 'inference.py'}")
        inf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inf)
        inf.set_deterministic()  # §G5 (copilot-instructions.md) — upstream: torch.manual_seed(42)
        config = inf.load_config(str(upstream / "configs" / "ear_vae2.json"), ckpt_pfad)
        _V2_MODELL_CACHE[cache_key] = (inf, inf.load_model(ckpt_pfad, config, device))
    inf, model = _V2_MODELL_CACHE[cache_key]

    x = _zu_zeit_zuerst(audio)  # (N, 2)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    n_orig = x.shape[0]
    # Upstream-Kontrakt: (1, 2, T) @ 48 kHz.
    sig = x.T[None]
    if int(sr) != int(inf.SAMPLE_RATE):
        sig = resample_poly(sig, inf.SAMPLE_RATE, int(sr), axis=-1).astype(np.float32)
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as torch_f  # pylint: disable=import-outside-toplevel

    tensor = torch.from_numpy(np.ascontiguousarray(sig))
    spl = int(model.samples_per_latent)
    pad_len = (-tensor.shape[-1]) % spl
    if pad_len:
        tensor = torch_f.pad(tensor, (0, pad_len))
    with torch.no_grad():
        lat = model.encode_audio(tensor, chunked=True, chunk_size=512, overlap=16, deterministic=True)
        rec = model.decode_audio(lat, chunked=True, chunk_size=512, overlap=16)
    out = rec[..., : tensor.shape[-1]].squeeze(0).cpu().numpy()  # (2, T)
    out = np.clip(out, -1.0, 1.0)
    if int(sr) != int(inf.SAMPLE_RATE):
        out = resample_poly(out, int(sr), inf.SAMPLE_RATE, axis=-1).astype(np.float32)
    out = out.T  # → (N, 2)
    n_out = min(n_orig, out.shape[0])
    return np.ascontiguousarray(out[:n_out])


# ── Hoer-Stimuli (verblindete Paare, Schlüssel getrennt) ────────────────────


def stimuli_vorbereiten(
    ausgaben: dict[str, dict[str, Any]],
    out_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Schreibt v1/v2-Paare als WAV und erzeugt das verblindete Versuchsblatt
    ueber ``scripts/ab_harness.prepare`` (Stimuli byte-identisch unter neutralen
    Namen, Schlüssel getrennt — nicht an die Hör-Instanz)."""
    import soundfile as sf  # pylint: disable=import-outside-toplevel

    from scripts.ab_harness import prepare

    out_dir.mkdir(parents=True, exist_ok=True)
    roh = out_dir / "roh"
    roh.mkdir(exist_ok=True)
    items: list[dict[str, str]] = []
    for tid, daten in sorted(ausgaben.items()):
        p_v1 = roh / f"{tid}_v1.wav"
        p_v2 = roh / f"{tid}_v2.wav"
        p_in = roh / f"{tid}_input.wav"
        sf.write(str(p_v1), _zu_zeit_zuerst(daten["v1"]), int(daten["sr"]))
        sf.write(str(p_v2), _zu_zeit_zuerst(daten["v2"]), int(daten["sr"]))
        sf.write(str(p_in), _zu_zeit_zuerst(daten["input"]), int(daten["sr"]))
        items.append({"id": tid, "a": str(p_v1), "b": str(p_v2), "ref": str(p_in)})
    sheet = prepare(items, out_dir, seed=seed, label_a="Kandidat A", label_b="Kandidat B")
    log.info(
        "Hoer-Stimuli vorbereitet: %d Paare → %s (Schlüssel: key.json, VERTRAULICH)", len(sheet["trials"]), out_dir
    )
    return sheet


# ── Messdatei ───────────────────────────────────────────────────────────────


def messung_schreiben(payload: dict[str, Any], ziel: Path) -> dict[str, Any]:
    """Schreibt die Messdatei (Schema analog b3-Messungen) und liefert den
    geschriebenen Payload — inkl. ``quelle``-Default (Selbstbeschreibung,
    Muster ``scripts/b3_stufenmessung.py`` ``beschluss_schreiben``)."""
    daten = dict(payload)
    daten.setdefault("quelle", "scripts/ear_vae_v1_v2_vergleich.py")
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(daten, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return daten


def _quellen(modell_root: Path, corpus_root: Path, corpus_eintraege: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "v1_deployt": "models/ear_vae/{encoder,decoder}.onnx (byte-identisch ear_vae_ft_*_inline.onnx, SHA-belegt 2026-10-06; docs/reports/current/2026-10-06_steps_0-5_preparation.md Schritt 1d)",
        "v1_encoder_sha256": "8f97ef673d4120ed1d92e762f8687b072de45dc666a166f38209cf9cf629c387",
        "v1_decoder_sha256": "ed64a1f12b685bb1e55eb27f69683430692c3570f6812c54a898109df5d911ab",
        "v2_upstream": f"{modell_root / 'ear_vae2_upstream'} (Klon earlab/EAR_VAE2 — README, LICENSE; Artefakt-Fingerabdruck: 0 Gewichte, .github/ML_ARTIFACT_FINGERPRINTS.md)",
        "corpus": [f"{corpus_root / e['path']} ({e.get('material', '?')})" for e in corpus_eintraege],
        "metriken": "musik_mos + defektlast (richtungszertifiziert, docs/reports/current/qualitaetsmetrik_richtungszertifikat.json)",
        "statistik": "scripts/ab_harness.py sign_test_two_sided (exakter Vorzeichen-Test)",
    }


_HINWEIS_URTEIL = (
    "Urteil = menschliche Hoer-Instanz (Hörordnung, Konfliktregel §7 in "
    ".github/instructions/hoerordnung.instructions.md; im Auftrag D.1c als "
    '"§4 Hoerordnung" bezeichnet). Metriken sind Zeugen. KEINE Aktivierung, '
    "KEIN Flag-Flip (§v10.802 (copilot-instructions.md)) — D-K0-9 bleibt "
    "menschlich-gated (.github/SOTA_DEFICIT_REGISTER.md)."
)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="EAR-VAE v1<->v2 Vergleichsmessung (D-K0-9) — Vorbereitung, keine Aktivierung"
    )
    ap.add_argument("--seed", type=int, default=SEED_VORGABE)
    ap.add_argument("--clips", type=int, default=8, help="Anzahl Corpus-Clips (Reihenfolge wie CORPUS)")
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--root", default=".", help="Bestands-Root mit models/ und test_audio/")
    ap.add_argument("--out", default=MESSDATEI_VORGABE)
    ap.add_argument("--stimuli-dir", default=STIMULI_VORGABE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight-only", action="store_true", help="nur v2-Verfuegbarkeit pruefen")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from scripts.release_candidate_pipeline import CORPUS, _load_clip

    root = Path(args.root).resolve()
    modell_root = root / "models"
    verf = v2_verfuegbarkeit(modell_root)
    corpus_eintraege = [c for c in CORPUS if (root / c["path"]).is_file()][: args.clips]
    plan = fenster_plan(corpus_eintraege, seconds=args.seconds)

    payload: dict[str, Any] = {
        "schema": "aurik.ear_vae_v1_v2_messung/1",
        "verfahren": "EAR-VAE v1<->v2 Vergleichsmessung (LUECKEN-MATRIX D.1c, Defizit D-K0-9)",
        "seed": args.seed,
        "zeilen": [],
        "never_worsen": None,
        "vorteile": None,
        "p_wert": None,
        "signifikant": None,
        "zielmetriken": ["musik_mos", "defektlast"],
        "delta_sdr": {
            "status": "nicht_korpusfaehig",
            "grund": "Fester Referenz-Korpus (test_audio/) hat keine sauberen Referenzen — Delta-SDR nur bei Referenz-Paarung aussagefaehig; fuer D-K0-9 tragen musik_mos + defektlast (beide richtungszertifiziert).",
        },
        "geplante_fenster": plan,
        "harness": "scripts/ear_vae_v1_v2_vergleich.py",
        "hoer_stimuli": {
            "status": "vorbereitet",
            "verfahren": "scripts/ab_harness.py prepare (Blind-Paare A/B, Schlüssel getrennt)",
            "ziel": args.stimuli_dir,
            "frage": HOER_FRAGE,
        },
        "quellen": _quellen(modell_root, root, corpus_eintraege),
        "hinweis": _HINWEIS_URTEIL,
    }

    if args.preflight_only or not verf["verfuegbar"]:
        payload["status"] = "nicht_ausfuehrbar"
        payload["technischer_grund"] = verf["grund"]
        if verf["verfuegbar"]:
            payload["status"] = "preflight_ok"
            payload["technischer_grund"] = None
        messung_schreiben(payload, root / args.out if not Path(args.out).is_absolute() else Path(args.out))
        if verf["verfuegbar"]:
            log.info("Preflight OK: v2-Checkpoint vorhanden (%s)", verf["checkpoints"][0])
            return 0
        log.error("v2 technisch nicht lauffaehig — Messung abgebrochen, Grund dokumentiert (D-K0-9)")
        return 1

    # Messlauf: nur erreichbar, sobald v2-Gewichte beschafft sind.
    fenster: list[tuple[str, np.ndarray, int]] = []
    for eintrag in corpus_eintraege:
        pfad = root / eintrag["path"]
        for start_s in FENSTER_STARTS:
            try:
                audio, sr = _load_clip(pfad, start_s=start_s, seconds=args.seconds)
            except ValueError:
                continue
            fenster.append((pfad.stem, np.asarray(audio, dtype=np.float32), int(sr)))
    if not fenster:
        log.error("kein Referenzmaterial unter %s", root)
        return 2

    ausgaben: dict[str, dict[str, Any]] = {}

    def _v2_wrapped(audio: np.ndarray, sr: int) -> np.ndarray:
        return v2_denoise_fenster(audio, sr, modell_root=modell_root, device=args.device)

    messung = vergleich_messen(
        fenster,
        args.seed,
        v2_denoise=_v2_wrapped,
        ausgaben=ausgaben,
    )
    payload["status"] = "gemessen"
    payload["zeilen"] = messung["zeilen"]
    for k in ("never_worsen", "vorteile", "p_wert", "signifikant"):
        payload[k] = messung[k]
    payload["erfuellt_objektiv"] = messung["erfuellt"]
    messung_schreiben(payload, root / args.out if not Path(args.out).is_absolute() else Path(args.out))
    stimuli_vorbereiten(ausgaben, root / args.stimuli_dir, seed=args.seed)
    log.info(
        "Messung geschrieben: never_worsen=%s vorteile=%s p=%s → %s",
        messung["never_worsen"],
        messung["vorteile"],
        messung["p_wert"],
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
