#!/usr/bin/env python3
"""scripts/model_artifact_probe.py — Artefakt-Probe für ``models/`` (§III.13).

**Zweck (§III.13 (copilot-instructions.md) — Evidenzpflicht):** Die Identität und
die Architektur jedes Modell-Artefakts werden **am Artefakt selbst** gemessen —
nicht aus Registry, Manifest oder Skript-Kommentaren übernommen. Die Probe ist
die maschinelle Grundlage für `.github/ML_ARTIFACT_FINGERPRINTS.md` und für die
Domänen-Spalten in `.github/ML_MODEL_DOMAIN_REGISTRY.md` sowie
`.github/SOTA_DEFICIT_REGISTER.md`.

**Was gemessen wird (je Format):**

* `.onnx` (+ `.onnx.data`): Opset/Producer/IR, Ein-/Ausgangsnamen mit Shape und
  Dtype, Knoten- und Initializer-Zahl, **Parameterzahl**, Operator-Histogramm,
  External-Data-Verweis, `metadata_props`. Die ONNX-Datei wird **ohne**
  External Data geladen (Kopfdaten) — kein 500-MB-Vollzugriff.
* `.pth`/`.pt`/`.ckpt`/`.pyt` (Torch-Zip): Eintragsnamen, `data.pkl` **ohne
  Gewichte gelesen** (restricted Unpickler — Storages werden nie materialisiert),
  daraus: Tensornamen, **Parameterzahl aus Storage-`numel`**, Dtype-Histogramm
  und die **skalaren Hparams** (`config`/`hparams`/`args`/`hyper_parameters`).
* `.safetensors`: Header-JSON (Tensornamen, Dtypes, `__metadata__`).
* `.ts` (TorchScript): Zip-Einträge.
* `.joblib`: Größe + Ladeversuch (nur < 32 MB), sonst ehrlich „nicht gelesen".
* `.npy`: Header (Shape/Dtype) über `numpy.lib.format`.
* Begleitende **lokale Doku-Evidenz** je Verzeichnis: `config.json`,
  `README*`/`*.md`, `LICENSE` — inkl. der für die Domäne entscheidenden Felder
  (`sampling_rate`, `n_mels`/`num_mels`, `architectures`, `model_type`,
  `_name_or_path`, `upsample_rates`, …).

**Grenze der Aussage (§G8 (copilot-instructions.md)):** Aus Bytes ist die
**Architektur-Identität** belegbar, **nicht** der Trainingskorpus — eine
Modell-Datei trägt kein Trainingsmaterial. Die Probe behauptet deshalb keine
Domäne; sie liefert die Fingerabdrücke, gegen die eine Domänen-Aussage geprüft
werden muss.

Aufruf::

    python scripts/model_artifact_probe.py                  # JSON + Markdown
    python scripts/model_artifact_probe.py --list-only      # nur Bestandsliste
    python scripts/model_artifact_probe.py --dirs ear_vae   # Teilmenge

Rein lesend; schreibt ausschließlich die beiden Ausgabedateien.
"""

from __future__ import annotations

import argparse
import builtins
import collections
import hashlib
import io
import json
import logging
import pickle
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_JSON = ROOT / "reports" / "model_artifact_fingerprints.json"
DEFAULT_MD = ROOT / ".github" / "ML_ARTIFACT_FINGERPRINTS.md"

ARTIFACT_SUFFIXES: tuple[str, ...] = (
    ".onnx",
    ".pth",
    ".pt",
    ".pyt",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".ts",
    ".joblib",
    ".npy",
    ".tflite",
    ".mlmodel",
    ".h5",
    ".ort",
    ".engine",
    ".pb",
)
SKIP_DIRS: frozenset[str] = frozenset({".git", ".venv", "venv", ".cache", "__pycache__", "node_modules", ".mypy_cache"})
DOC_SUFFIXES: tuple[str, ...] = (".json", ".md", ".yaml", ".yml", ".txt")

# Für die Domänen-Frage entscheidende Konfigurationsfelder (Architektur-Fingerprint).
CONFIG_KEYS_OF_INTEREST: tuple[str, ...] = (
    "sampling_rate",
    "sample_rate",
    "sr",
    "num_mels",
    "n_mels",
    "n_mel_channels",
    "mel_bins",
    "hop_size",
    "hop_length",
    "win_size",
    "n_fft",
    "upsample_rates",
    "upsample_kernel_sizes",
    "upsample_initial_channel",
    "resblock",
    "activation",
    "model_type",
    "architectures",
    "backbone",
    "hidden_size",
    "num_hidden_layers",
    "vocab_size",
    "n_vocab",
    "text_cleaners",
    "speech_encoder",
    "version",
    "f0",
    "target_sr",
    "model_name",
    "_name_or_path",
)

HPARAM_KEYS: tuple[str, ...] = (
    "config",
    "hparams",
    "hyper_parameters",
    "args",
    "cfg",
    "ckpt_cfg",
    "model_config",
)

# `torch.save`-Legacy-Format schreibt VOR das Objekt: Magic (0x1950A86A20F9469CFC6C)
# und die Protokollversion. Ohne Überspringen liest man nur eine große Zahl.
TORCH_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


# ── Hilfsfunktionen ─────────────────────────────────────────────────────────


# Fremd-READMEs enthalten rohes HTML (Logos, Badges, `<img>` ohne Alt-Text).
# Die Fingerabdruck-Doku wird von markdownlint geprüft (MD033/MD045 sind NICHT
# auto-fixbar) — beides wird deshalb an der Quelle entfernt
# (§V7 (copilot-instructions.md) — Ursache statt Symptom).
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _sanitize_inline(text: str) -> str:
    """HTML-/Bild-Syntax aus fremden Karten entfernen (markdownlint-fest)."""
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", text)
    cleaned = _MD_IMAGE_RE.sub(" ", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace("`", "'")
    return " ".join(cleaned.split())


# Fremde Modell-Karten enthalten rohes HTML (Logos, Badges, `img` ohne Alt-Text).
# Die Fingerabdruck-Doku wird von markdownlint geprüft; MD033/MD045 sind NICHT
# auto-fixbar und blockierten den Commit — deshalb Entfernung an der Quelle
# (§V7 (copilot-instructions.md): Ursache statt Symptom).
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _sanitize_inline(text: str) -> str:
    """HTML-/Bild-/Link-Syntax aus fremden Karten entfernen (markdownlint-fest)."""
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", text)
    cleaned = _MD_IMAGE_RE.sub(" ", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace("`", "'")
    return " ".join(cleaned.split())


def _sha256(path: Path) -> str:
    """SHA-256 (streaming, speicherschonend)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _magic(path: Path, n: int = 8) -> str:
    """Erste Bytes als Hex (Format-Fingerprint)."""
    try:
        with path.open("rb") as fh:
            return fh.read(n).hex()
    except OSError:
        return ""


def _scalars_of(obj: Any, depth: int = 0, max_depth: int = 2) -> dict[str, Any]:
    """Skalare Felder eines (verschachtelten) Dicts — Tensoren werden ignoriert."""
    out: dict[str, Any] = {}
    if depth > max_depth or not isinstance(obj, dict):
        return out
    for key in sorted(obj.keys(), key=str):
        val = obj[key]
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[str(key)] = val
        elif (
            isinstance(val, (list, tuple))
            and len(val) <= 12
            and all(isinstance(x, (str, int, float, bool)) for x in val)
        ):
            out[str(key)] = list(val)
        elif isinstance(val, dict):
            for sub_key, sub_val in _scalars_of(val, depth + 1, max_depth).items():
                out[f"{key}.{sub_key}"] = sub_val
    return out


# ── Torch-Zip ohne Gewichte lesen (restricted Unpickler) ────────────────────


class _Stub:
    """Platzhalter für Klassen/Tensoren — Gewichte werden NIE materialisiert.

    Wichtig: `torch`-Checkpoints aus `fairseq`-Zeiten werden mit `NEWOBJ`
    serialisiert — dabei läuft **kein `__init__`**. Die Attribute müssen daher
    als Klassen-Defaults existieren, sonst bricht die Auswertung ab.
    """

    __stub_name__ = "stub"
    size: tuple[int, ...] | None = None
    storage: _Stub | None = None
    args: tuple[Any, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.size = None
        self.storage = None
        for idx, arg in enumerate(args):
            if isinstance(arg, _Stub) and arg.__stub_name__.endswith("Storage"):
                if self.storage is None:
                    self.storage = arg
            elif (
                self.size is None
                and idx >= 1
                and isinstance(arg, tuple)
                and arg
                and all(isinstance(x, int) for x in arg)
            ):
                # `_rebuild_tensor_v2(storage, offset, size, stride, …)` — die
                # ERSTE Int-Tupel nach dem Storage ist die Tensor-Form (`size`),
                # die zweite ist `stride` und darf sie nicht überschreiben.
                self.size = tuple(int(x) for x in arg)

    def __setstate__(self, state: Any) -> None:
        """NEWOBJ-Zustand bewusst ignorieren (keine Gewichte, keine Kanten)."""

    def __repr__(self) -> str:  # pragma: no cover - Debug-Hilfe
        return f"<stub {self.__stub_name__} size={self.size}>"


def _named_stub(module: str, name: str) -> type[_Stub]:
    """Klasse mit erhaltenem Symbolnamen (Dtype-Name aus `<X>Storage`)."""
    return type(name, (_Stub,), {"__module__": module, "__stub_name__": name})


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler, der nur die Struktur liest — keine Gewichte, kein Code."""

    def find_class(self, module: str, name: str) -> Any:
        if module in ("builtins", "__builtin__"):
            return getattr(builtins, name, _Stub)
        if module == "collections" and name in ("OrderedDict", "defaultdict"):
            return getattr(collections, name)
        return _named_stub(module, name)

    def persistent_load(self, pid: Any) -> Any:
        # Legacy-Format (`torch.save` alt / fairseq): pid = ('storage', <Klasse>,
        # key, location, numel). Der Dtype steckt im Klassennamen — der
        # Platzhalter muss ihn tragen, sonst bleibt die Storage untypisiert.
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            storage_cls = _named_stub("torch", str(getattr(pid[1], "__stub_name__", "Storage")))
            return storage_cls(*pid)
        return _Stub(*pid) if isinstance(pid, tuple) else _Stub()


def _walk_stats(obj: Any, out: dict[str, Any]) -> None:
    """Sammelt Tensor-/Parameter-Statistik aus dem Stub-Graphen (iterativ)."""
    stack: list[Any] = [obj]
    seen = 0
    while stack and seen < 200_000:
        seen += 1
        node = stack.pop()
        if isinstance(node, _Stub):
            if node.__stub_name__.endswith("Storage"):
                # Nur Dtype-Typisierung + Storage-`numel` als belegte
                # Rückfallquelle (sonst würde doppelt gezählt).
                dtype = node.__stub_name__.replace("Storage", "").lower()
                hist = out.setdefault("dtype_histogram", {})
                hist[dtype] = hist.get(dtype, 0) + 1
                out["storage_count"] = out.get("storage_count", 0) + 1
                if len(node.args) > 4 and isinstance(node.args[4], int):
                    out["storage_numel"] = out.get("storage_numel", 0) + int(node.args[4])
            elif node.size is not None:
                out["tensor_count"] = out.get("tensor_count", 0) + 1
                numel = 1
                for dim in node.size:
                    numel *= int(dim)
                out["param_count"] = out.get("param_count", 0) + numel
            else:
                out["unresolved_tensor_stubs"] = out.get("unresolved_tensor_stubs", 0) + 1
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple, set)):
            stack.extend(node)
    out["walk_nodes"] = seen


def probe_torch_zip(path: Path) -> dict[str, Any]:
    """Torch-Zip-Checkpoint: Struktur, Parameterzahl, skalare Hparams."""
    info: dict[str, Any] = {"format": "torch-zip"}
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            info["zip_entries"] = len(names)
            info["zip_entry_sample"] = sorted(names)[:6]
            pkl_names = [n for n in names if n.endswith("data.pkl")]
            if not pkl_names:
                info["note"] = "keine data.pkl gefunden"
                return info
            data = zf.read(pkl_names[0])
    except (zipfile.BadZipFile, OSError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    try:
        obj = _RestrictedUnpickler(io.BytesIO(data)).load()
    except Exception as exc:
        info["error"] = f"Unpickle: {type(exc).__name__}: {exc}"
        return info

    stats: dict[str, Any] = {}
    _walk_stats(obj, stats)
    _finalize_torch_info(info, obj, stats)
    return info


def _finalize_torch_info(info: dict[str, Any], obj: Any, stats: dict[str, Any]) -> None:
    """Gemeinsamer Abschluss für Zip- und Legacy-Checkpoints."""
    # Belegte Größenquelle benennen: Tensor-Formen sind exakt, `storage_numel`
    # ist der Rückfall für NEWOBJ-Checkpoints (fairseq-Altbestände).
    if not stats.get("param_count") and stats.get("storage_numel"):
        stats["param_count"] = stats["storage_numel"]
        stats["param_source"] = "storage_numel"
    elif stats.get("param_count"):
        stats["param_source"] = "tensor_shapes"
    info.update(stats)
    if isinstance(obj, dict):
        info["top_level_keys"] = [str(k) for k in list(obj.keys())[:24]]
        hparams: dict[str, Any] = {}
        for key in list(obj.keys()):
            if str(key).lower() in HPARAM_KEYS or "config" in str(key).lower():
                hparams.update(_scalars_of(obj[key]))
        if hparams:
            info["hparams"] = {k: hparams[k] for k in sorted(hparams)[:60]}


def probe_torch_legacy(path: Path, limit_mb: int = 1024) -> dict[str, Any]:
    """Legacy-Checkpoint (kein Zip, `fairseq`/`torch.save` alt): Struktur ohne Gewichte.

    Der restricted Unpickler liefert für jede Storage nur einen Platzhalter —
    Gewichte werden nie materialisiert. Inline-Numpy-Daten werden über die
    Dateigrößen-Grenze abgefangen (ehrlich gemeldet statt Speicher zu sprengen).
    """
    info: dict[str, Any] = {"format": "torch-legacy-pickle"}
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > limit_mb:
        info["note"] = f"nicht gelesen ({size_mb:.0f} MB > {limit_mb} MB)"
        return info
    try:
        with path.open("rb") as fh:
            unpickler = _RestrictedUnpickler(fh)
            obj = unpickler.load()
            if isinstance(obj, int) and obj == TORCH_LEGACY_MAGIC:
                # Legacy-Header von `torch.save`: Magic, Protokollversion,
                # `sys_info`-Dict, dann erst das eigentliche Objekt.
                info["torch_legacy_protocol_version"] = unpickler.load()
                candidate = unpickler.load()
                sys_info_marker = {"protocol_version", "little_endian", "type_sizes"}
                if isinstance(candidate, dict) and sys_info_marker & set(candidate.keys()):
                    info["torch_legacy_sys_info"] = _scalars_of(candidate, max_depth=0)
                    obj = unpickler.load()
                else:
                    obj = candidate
    except Exception as exc:
        info["error"] = f"Unpickle: {type(exc).__name__}: {exc}"
        return info
    stats: dict[str, Any] = {}
    _walk_stats(obj, stats)
    _finalize_torch_info(info, obj, stats)
    return info


def probe_onnx(path: Path) -> dict[str, Any]:
    """ONNX-Kopf: Graph-I/O, Parameterzahl, Operator-Mix (ohne External Data)."""
    info: dict[str, Any] = {"format": "onnx"}
    try:
        import onnx  # lokal importiert: nur wenn eine .onnx geprüft wird
    except ImportError:  # pragma: no cover
        info["error"] = "onnx nicht installiert"
        return info
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    graph = model.graph
    info["producer"] = f"{model.producer_name} {model.producer_version}".strip()
    info["ir_version"] = int(model.ir_version)
    info["opset"] = [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import]
    info["graph_name"] = graph.name
    meta = {p.key: p.value for p in model.metadata_props}
    info["metadata_props"] = meta

    inputs: list[dict[str, Any]] = []
    for ti in graph.input:
        dims = [d.dim_value if d.dim_value else (d.dim_param or "?") for d in ti.type.tensor_type.shape.dim]
        if dims:
            inputs.append({"name": ti.name, "shape": dims})
    info["inputs"] = inputs[:6]
    outputs: list[dict[str, Any]] = []
    for ti in graph.output:
        dims = [d.dim_value if d.dim_value else (d.dim_param or "?") for d in ti.type.tensor_type.shape.dim]
        outputs.append({"name": ti.name, "shape": dims})
    info["outputs"] = outputs[:6]

    params = 0
    for init in graph.initializer:
        prod = 1
        for d in init.dims:
            prod *= int(d)
        params += prod
    info["initializer_count"] = len(graph.initializer)
    info["param_count"] = params
    info["node_count"] = len(graph.node)
    ops = collections.Counter(n.op_type for n in graph.node)
    info["op_mix"] = dict(ops.most_common(12))
    info["has_external_data"] = any(init.data_location == 1 for init in graph.initializer)
    data_file = path.with_suffix(path.suffix + ".data")
    info["external_data_file"] = data_file.name if data_file.exists() else None
    return info


def probe_safetensors(path: Path) -> dict[str, Any]:
    """safetensors: Header-JSON (Tensornamen, Dtypes, Metadaten)."""
    info: dict[str, Any] = {"format": "safetensors"}
    try:
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                info["error"] = "Header zu kurz"
                return info
            header_len = int.from_bytes(raw, "little")
            header = json.loads(fh.read(header_len).decode("utf-8"))
    except (OSError, ValueError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    meta = header.pop("__metadata__", {}) if isinstance(header.get("__metadata__"), dict) else {}
    dtypes = collections.Counter(str(v.get("dtype")) for v in header.values() if isinstance(v, dict))
    info["tensor_count"] = len(header)
    info["dtype_histogram"] = dict(dtypes)
    info["tensor_name_sample"] = sorted(header.keys())[:8]
    if meta:
        info["metadata"] = {
            str(k): (v if isinstance(v, (str, int, float)) else str(v)) for k, v in list(meta.items())[:12]
        }
    return info


def probe_joblib(path: Path, limit_mb: int = 32) -> dict[str, Any]:
    """joblib: nur kleine Dateien werden probeweise geladen."""
    info: dict[str, Any] = {"format": "joblib"}
    if path.stat().st_size > limit_mb * 1024 * 1024:
        info["note"] = f"nicht geladen (> {limit_mb} MB)"
        return info
    try:
        import joblib

        obj = joblib.load(path)
        info["object_type"] = type(obj).__name__
        info["object_repr"] = repr(obj)[:200]
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_numpy(path: Path) -> dict[str, Any]:
    """`.npy`: Shape/Dtype aus dem Header."""
    info: dict[str, Any] = {"format": "npy"}
    try:
        import numpy as np

        with path.open("rb") as fh:
            version = np.lib.format.read_magic(fh)
            shape, fortran, dtype = np.lib.format._read_array_header(fh, version)  # type: ignore[attr-defined]
        info["shape"] = [int(s) for s in shape]
        info["dtype"] = str(dtype)
        info["fortran_order"] = bool(fortran)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_torchscript(path: Path) -> dict[str, Any]:
    """TorchScript `.ts`: Zip-Einträge (Struktur, keine Gewichte)."""
    info: dict[str, Any] = {"format": "torchscript"}
    try:
        with zipfile.ZipFile(path) as zf:
            info["zip_entries"] = len(zf.namelist())
            info["zip_entry_sample"] = sorted(zf.namelist())[:8]
            info["uncompressed_bytes"] = sum(i.file_size for i in zf.infolist())
    except (zipfile.BadZipFile, OSError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_file(path: Path, hash_limit_mb: int) -> dict[str, Any]:
    """Ein Artefakt vollständig probieren (Format über Magic-Bytes erkannt)."""
    entry: dict[str, Any] = {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "magic_hex": _magic(path),
    }
    if entry["size_bytes"] <= hash_limit_mb * 1024 * 1024:
        entry["sha256"] = _sha256(path)
    else:
        entry["sha256"] = None
        entry["hash_note"] = f"übersprungen (> {hash_limit_mb} MB)"

    suffix = path.suffix.lower()
    magic = bytes.fromhex(str(entry["magic_hex"])) if entry["magic_hex"] else b""
    is_zip = magic.startswith(b"PK\x03\x04")
    is_pickle = magic[:1] in (b"\x80", b"(", b"]", b"}")

    try:
        if suffix == ".onnx":
            entry.update(probe_onnx(path))
        elif suffix in (".pth", ".pt", ".pyt", ".ckpt"):
            if is_zip:
                entry.update(probe_torch_zip(path))
            elif is_pickle:
                entry.update(probe_torch_legacy(path))
            else:
                entry.update(_describe_non_model_file(path, suffix))
        elif suffix == ".bin":
            entry.update(probe_torch_zip(path) if is_zip else probe_torch_legacy(path))
        elif suffix == ".safetensors":
            entry.update(probe_safetensors(path))
        elif suffix == ".joblib":
            entry.update(probe_joblib(path))
        elif suffix == ".npy":
            entry.update(probe_numpy(path))
        elif suffix == ".ts":
            entry.update(probe_torchscript(path) if is_zip else _describe_non_model_file(path, suffix))
        else:
            entry["format"] = suffix.lstrip(".") or "unbekannt"
            entry["note"] = "kein Prober für dieses Format"
    except Exception as exc:
        entry["probe_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _describe_non_model_file(path: Path, suffix: str) -> dict[str, Any]:
    """Datei mit Modell-Endung, aber ohne Modell-Magic (z. B. pip-`.pth`).

    Ehrlich beschreiben statt als Fehler melden: `*.pth` kollidiert mit
    `site-packages`-Pfaddateien (Produktionsbefund: `distutils-precedence.pth`
    in einem Archivverzeichnis).
    """
    info: dict[str, Any] = {
        "format": f"text/{suffix.lstrip('.') or 'unbekannt'} (kein Modellartefakt)",
    }
    try:
        preview = path.read_text(encoding="utf-8", errors="replace")[:160]
        info["text_preview"] = " ".join(preview.split())[:160]
    except OSError as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def collect_doc_evidence(directory: Path) -> dict[str, Any]:
    """Lokale Doku-Evidenz im Verzeichnis (Karte/Konfiguration/Lizenz)."""
    evidence: dict[str, Any] = {"docs": [], "configs": {}}
    for item in sorted(directory.iterdir(), key=lambda p: p.name):
        if not item.is_file():
            continue
        name = item.name
        lower = name.lower()
        if lower.startswith("readme") or (item.suffix.lower() == ".md" and not evidence["docs"]):
            try:
                head = item.read_text(encoding="utf-8", errors="replace").splitlines()[:6]
            except OSError:
                head = []
            joined = _sanitize_inline(" / ".join(h.strip() for h in head if h.strip()))
            evidence["docs"].append(
                {
                    "file": _sanitize_inline(name),
                    "size_bytes": item.stat().st_size,
                    "head": joined[:300],
                }
            )
        elif lower.startswith("license") or lower.startswith("notice"):
            evidence["docs"].append({"file": name, "size_bytes": item.stat().st_size, "head": "(Lizenz)"})
        elif item.suffix == ".json" and item.stat().st_size < 4 * 1024 * 1024:
            try:
                payload = json.loads(item.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict):
                scalars = _scalars_of(payload, max_depth=1)
                interesting = {k: v for k, v in scalars.items() if k.split(".")[-1] in CONFIG_KEYS_OF_INTEREST}
                if interesting:
                    evidence["configs"][name] = interesting
    return evidence


def probe_directory(directory: Path, hash_limit_mb: int) -> dict[str, Any]:
    """Alle Artefakte eines Modellverzeichnisses probieren + Doku-Evidenz."""
    artifacts = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in ARTIFACT_SUFFIXES and not any(part in SKIP_DIRS for part in p.parts)
    )
    probed: list[dict[str, Any]] = []
    for path in artifacts:
        entry = probe_file(path, hash_limit_mb)
        # Relativer Pfad nach dem Regelwerk: gleichnamige Artefakte in
        # Unterordnern bleiben unterscheidbar.
        entry["path"] = str(path.relative_to(directory))
        probed.append(entry)
    record: dict[str, Any] = {
        "dir": directory.name,
        "artifact_count": len(probed),
        "doc_evidence": collect_doc_evidence(directory),
        "artifacts": probed,
    }
    record["total_bytes"] = sum(int(a["size_bytes"]) for a in record["artifacts"])
    return record


def _provenance_relations(
    records: list[dict[str, Any]], archive_name: str = "_archive_20260920"
) -> list[dict[str, Any]]:
    """Herkunftsbeziehungen Archiv ↔ deploytes Artefakt (bytescharf, §III.13).

    Drei belegte Relationen — in dieser Reihenfolge geprüft:

    * ``byte-identisch`` — gleicher SHA-256 (gleiche Gewichte, nur anderer Pfad).
    * ``architektur-gleich`` — gleiche Parameterzahl wie ein deploytes Artefakt,
      aber anderer SHA ⇒ Finetune-/Generationsbeziehung (Gewichte geändert).
    * ``ohne-gegenstueck`` — kein Partner im deployten Bestand (ehrlich offen).

    Das ist die maschinelle Form des EAR-VAE-Beweises (Basis ↔ Finetune) über
    **alle** archivierten Artefakte; sie ersetzt jede Doku-Annahme.
    """
    deployed = [r for r in records if r["dir"] != archive_name]
    archive = next((r for r in records if r["dir"] == archive_name), None)
    if archive is None:
        return []
    by_sha: dict[str, list[str]] = {}
    by_params: dict[int, list[str]] = {}
    for record in deployed:
        for art in record["artifacts"]:
            tag = f"{record['dir']}/{art['path']}"
            if art.get("sha256"):
                by_sha.setdefault(str(art["sha256"]), []).append(tag)
            if art.get("param_count"):
                by_params.setdefault(int(art["param_count"]), []).append(tag)
    relations: list[dict[str, Any]] = []
    for art in archive["artifacts"]:
        sha = art.get("sha256")
        params = art.get("param_count")
        entry: dict[str, Any] = {
            "archive_path": f"{archive_name}/{art['path']}",
            "param_count": params,
            "size_bytes": art.get("size_bytes"),
        }
        if sha and str(sha) in by_sha:
            entry["relation"] = "byte-identisch"
            entry["partners"] = by_sha[str(sha)]
        elif params and int(params) in by_params:
            entry["relation"] = "architektur-gleich"
            entry["partners"] = by_params[int(params)]
        else:
            entry["relation"] = "ohne-gegenstueck"
            entry["partners"] = []
        relations.append(entry)
    return relations


def build_report(models_dir: Path, only: list[str] | None, hash_limit_mb: int) -> dict[str, Any]:
    """Report über alle (oder ausgewählte) Modellverzeichnisse bauen."""
    dirs = sorted(p for p in models_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS)
    if only:
        wanted = {name.lower() for name in only}
        dirs = [d for d in dirs if d.name.lower() in wanted]
    records: list[dict[str, Any]] = []
    loose: list[dict[str, Any]] = []
    for directory in dirs:
        logger.info("Probe: %s", directory.name)
        records.append(probe_directory(directory, hash_limit_mb))
    for item in sorted(models_dir.iterdir(), key=lambda p: p.name):
        if item.is_file() and item.suffix.lower() in ARTIFACT_SUFFIXES:
            loose.append(probe_file(item, hash_limit_mb))
    return {
        "root": "models",
        "directory_count": len(records),
        "artifact_count": sum(r["artifact_count"] for r in records) + len(loose),
        "total_bytes": sum(r["total_bytes"] for r in records) + sum(a["size_bytes"] for a in loose),
        "directories": records,
        "loose_artifacts": loose,
        "provenance": _provenance_relations(records),
    }


def _fmt_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _artifact_fingerprint(entry: dict[str, Any]) -> str:
    """Kurz-Fingerabdruck je Artefakt (die entscheidenden Messwerte)."""
    parts: list[str] = []
    if entry.get("param_count"):
        source = entry.get("param_source")
        suffix = "" if source in (None, "tensor_shapes") else f" ({source})"
        parts.append(f"{int(entry['param_count']) / 1e6:.2f} M Params{suffix}")
    if entry.get("unresolved_tensor_stubs"):
        parts.append(f"{entry['unresolved_tensor_stubs']} Stubs ohne Formangabe")
    if entry.get("inputs"):
        shapes = ", ".join(str(i.get("shape")) for i in entry["inputs"][:2])
        parts.append(f"IN {shapes}")
    if entry.get("outputs"):
        parts.append(f"OUT {entry['outputs'][0].get('shape')}")
    if entry.get("hparams"):
        hp = entry["hparams"]
        keep = [
            k for k in ("sr", "sampling_rate", "sample_rate", "version", "f0", "model_type", "architecture") if k in hp
        ]
        if keep:
            parts.append("hparams: " + ", ".join(f"{k}={hp[k]}" for k in keep))
        else:
            parts.append(f"hparams: {len(hp)} Felder")
    if entry.get("tensor_count") and not entry.get("param_count"):
        parts.append(f"{entry['tensor_count']} Tensoren")
    if entry.get("dtype_histogram"):
        parts.append("dtypes: " + ", ".join(f"{k}×{v}" for k, v in sorted(entry["dtype_histogram"].items())[:4]))
    if entry.get("shape"):
        parts.append(f"npy {entry['shape']} {entry.get('dtype')}")
    if entry.get("object_type"):
        parts.append(f"joblib: {entry['object_type']}")
    if entry.get("text_preview"):
        parts.append(f"Textdatei: {entry['text_preview'][:80]}")
    if entry.get("op_mix"):
        top = list(entry["op_mix"].items())[:4]
        parts.append("ops: " + ", ".join(f"{k}×{v}" for k, v in top))
    if entry.get("zip_entries") and not parts:
        parts.append(f"{entry['zip_entries']} Zip-Einträge")
    for flag in ("error", "probe_error", "note", "hash_note"):
        if entry.get(flag):
            parts.append(f"[{flag}: {entry[flag]}]")
    return "; ".join(parts) if parts else "—"


def write_markdown(report: dict[str, Any], md_path: Path) -> None:
    """Markdown-Dokumentation aus dem Report erzeugen (deterministisch)."""
    lines: list[str] = [
        "# ML-Artefakt-Fingerabdrücke (`models/`) — am Artefakt gemessen",
        "",
        "> **Erzeugt von** `scripts/model_artifact_probe.py` (§III.13 (copilot-instructions.md) — Evidenzpflicht).",
        "> Die Tabelle nennt **gemessene** Architektur-Fingerabdrücke (Mel-Bänder, "
        "Upsample-Faktor, Kanalbreiten, Parameterzahl, Hparams, Dtype-Mix) — "
        "**keine** Domänen-Behauptung. Ob ein Artefakt Musik- oder sprach-trainiert "
        "ist, entscheidet sich an Trainings-Skript, Modell-Karte, SHA-Identität "
        "oder einem widerspruchsfreien Plugin-Header; fehlt das, gilt die Domäne "
        "als `unbekannt`.",
        "",
        f"**Bestand:** {report['directory_count']} Verzeichnisse · "
        f"{report['artifact_count']} Artefakte · {_fmt_bytes(int(report['total_bytes']))}.",
        "",
        "## Verzeichnis-Übersicht",
        "",
        "| Verzeichnis | Artefakte | Größe | Lokale Doku-Evidenz |",
        "| --- | --- | --- | --- |",
    ]
    for record in report["directories"]:
        docs = record["doc_evidence"]["docs"]
        configs = record["doc_evidence"]["configs"]
        doc_cell = ", ".join(d["file"] for d in docs[:3]) or "—"
        if configs:
            doc_cell += ("; " if doc_cell != "—" else "") + ", ".join(sorted(configs)[:3])
        lines.append(
            f"| `{record['dir']}` | {record['artifact_count']} | "
            f"{_fmt_bytes(int(record['total_bytes']))} | {doc_cell} |"
        )
    lines.append("")

    lines.append("## Messungen je Verzeichnis")
    lines.append("")
    for record in report["directories"]:
        lines.append(f"### `{record['dir']}`")
        lines.append("")
        configs = record["doc_evidence"]["configs"]
        if configs:
            lines.append("**Lokale Konfigurations-Evidenz:**")
            lines.append("")
            for name, values in sorted(configs.items()):
                pairs = ", ".join(f"`{k}`={v}" for k, v in sorted(values.items())[:10])
                lines.append(f"- `{name}`: {pairs}")
            lines.append("")
        docs = record["doc_evidence"]["docs"]
        if docs:
            lines.append("**Lokale Karten/Lizenzen:**")
            lines.append("")
            for doc in docs[:4]:
                lines.append(f"- `{doc['file']}` — `{doc['head'][:200]}`")
            lines.append("")
        lines.append("| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |")
        lines.append("| --- | --- | --- | --- |")
        for art in record["artifacts"]:
            sha = art.get("sha256")
            sha_cell = f"`{sha[:16]}…`" if sha else "—"
            fp = _sanitize_inline(_artifact_fingerprint(art)).replace("|", "/")
            lines.append(f"| `{art['path']}` | {_fmt_bytes(int(art['size_bytes']))} | {sha_cell} | {fp} |")
        lines.append("")

    if report["loose_artifacts"]:
        lines.append("## Lose Artefakte in `models/`")
        lines.append("")
        lines.append("| Artefakt | Größe | SHA-256 | Gemessener Fingerabdruck |")
        lines.append("| --- | --- | --- | --- |")
        for art in report["loose_artifacts"]:
            sha = art.get("sha256")
            sha_cell = f"`{sha[:16]}…`" if sha else "—"
            fp = _sanitize_inline(_artifact_fingerprint(art)).replace("|", "/")
            lines.append(f"| `{art['path']}` | {_fmt_bytes(int(art['size_bytes']))} | {sha_cell} | {fp} |")
        lines.append("")

    relations = report.get("provenance") or []
    if relations:
        groups = (
            ("byte-identisch", "Byte-identisch (gleiche Gewichte, nur anderer Pfad)"),
            ("architektur-gleich", "Architektur-gleich, Gewichte verschieden (Finetune/Generation)"),
            ("ohne-gegenstueck", "Ohne Gegenstück im deployten Bestand"),
        )
        lines.append("## Herkunftsbelege (`_archive_20260920` ↔ deployt)")
        lines.append("")
        lines.append(
            "> Maschinell gemessen (SHA-256 bzw. Parameterzahl) — Muster des "
            "EAR-VAE-Nachweises, hier für **alle** archivierten Artefakte. Das "
            "Archiv enthält die Generationen VOR dem jeweiligen Finetune; die "
            "Relation ist damit ein Evidenz-Beleg nach §III.13 "
            "(copilot-instructions.md)."
        )
        lines.append("")
        for key, title in groups:
            subset = [r for r in relations if r["relation"] == key]
            if not subset:
                continue
            lines.append(f"### {title} — {len(subset)}")
            lines.append("")
            lines.append("| Archiv-Artefakt | Parameter | Partner (deployt) |")
            lines.append("| --- | --- | --- |")
            for rel in subset:
                params = f"{int(rel['param_count']) / 1e6:.2f} M" if rel.get("param_count") else "—"
                partners = ", ".join(f"`{p}`" for p in rel["partners"][:4]) or "—"
                lines.append(f"| `{rel['archive_path']}` | {params} | {partners} |")
            lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI-Einstieg."""
    parser = argparse.ArgumentParser(description="Artefakt-Probe für models/ (§III.13)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--hash-limit-mb", type=int, default=1024, help="SHA-256 nur bis zu dieser Dateigröße")
    parser.add_argument("--dirs", nargs="*", default=None, help="nur diese Verzeichnisse probieren")
    parser.add_argument("--no-md", action="store_true", help="kein Markdown schreiben")
    parser.add_argument("--list-only", action="store_true", help="nur Bestandsliste, keine Probe")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.quiet:
        logger.setLevel(logging.WARNING)

    if args.list_only:
        for directory in sorted(p for p in args.models_dir.iterdir() if p.is_dir()):
            count = sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in ARTIFACT_SUFFIXES)
            sys.stdout.write(f"{directory.name}\t{count}\n")
        return 0

    report = build_report(args.models_dir, args.dirs, args.hash_limit_mb)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Report: %s (%d Verzeichnisse, %d Artefakte)", args.out, report["directory_count"], report["artifact_count"]
    )
    if not args.no_md:
        write_markdown(report, args.md)
        logger.info("Dokumentation: %s", args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
