"""
update_yaml_with_dsp_modules.py

Dieses Skript liest die sota_dsp_module_list.yaml ein, durchsucht das DSP-Modulverzeichnis nach Python-Klassen,
und ergänzt für jeden Eintrag die Felder modul_file und modul_class, sofern ein passendes Modul gefunden wird.

Voraussetzung: pyyaml ist installiert (pip install pyyaml)
"""

import logging
import os
import re

import yaml

logger = logging.getLogger(__name__)

YAML_PATH = "tests/sota/sota_dsp_module_list.yaml"
DSP_PATH = "Aurik_Standalone/dsp/"

# Lade YAML
with open(YAML_PATH, encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Erzeuge Mapping: Name -> (Datei, Klasse)
module_map = {}
for fname in os.listdir(DSP_PATH):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(DSP_PATH, fname)
    with open(fpath, encoding="utf-8") as f:
        content = f.read()
    # Suche nach Klassen
    for match in re.finditer(r"class (\w+)", content):
        cls = match.group(1)
        # Mapping nach Namensähnlichkeit (vereinfachte Heuristik)
        key = cls.replace("_", " ")
        replacements = {
            "AI": "Ai",
            "SOTA": "Sota",
            "Adaptive": "Adaptive ",
            "Remover": " Remover",
            "Enhancer": " Enhancer",
            "Separator": " Separator",
            "Exciter": " Exciter",
            "Limiter": " Limiter",
            "Compressor": " Compressor",
            "Gate": " Gate",
            "Shaper": " Shaper",
            "Preservation": " Preservation",
            "Detection": " Detection",
            "Correction": " Correction",
            "Matrix": " Matrix",
            "Widener": " Widener",
            "Expander": " Expander",
            "Declipper": " Declipper",
            "Denoiser": " Denoiser",
            "Equalizer": " Equalizer",
            "Analyzer": " Analyzer",
            "Evaluator": " Evaluator",
            "Estimator": " Estimator",
            "Profile": " Profile",
            "Filter": " Filter",
            "Synthesizer": " Synthesizer",
            "Regenerator": " Regenerator",
            "Filler": " Filler",
            "Inpainting": " Inpainting",
            "SuperRes": " SuperRes",
            "SuperResolution": " SuperResolution",
            "Normalizer": " Normalizer",
            "Balancer": " Balancer",
            "Ducker": " Ducker",
            "Panner": " Panner",
            "Artifact": " Artifact",
            "TruePeak": " True Peak",
            "Oversampler": " Oversampler",
            "AGC": " AGC",
            "Leveler": " Leveler",
            "Formant": " Formant",
            "Preserver": " Preserver",
            "Isolator": " Isolator",
            "Confidence": " Confidence",
            "Weighting": " Weighting",
            "Hole": " Hole",
            "Segment": " Segment",
            "Genre": " Genre",
            "Key": " Key",
            "Beat": " Beat",
            "Onset": " Onset",
            "Sibilance": " Sibilance",
            "Breath": " Breath",
            "Dropout": " Dropout",
            "Crackle": " Crackle",
            "Bias": " Bias",
            "NAB": " NAB",
            "RIAA": " RIAA",
            "PrintThrough": " PrintThrough",
            "Azimuth": " Azimuth",
            "Bark": " Bark",
            "ISO": " ISO",
            "BS": " BS",
            "Loudness": " Loudness",
            "Range": " Range",
            "Fade": " Fade",
            "SampleRate": " Sample Rate",
            "Dithering": " Dithering",
            "NoiseShaping": " Noise Shaping",
            "Intersample": " Intersample",
            "LRA": " LRA",
            "Multiband": " Multiband",
            "MCRA": " MCRA",
            "IMCRA": " IMCRA",
            "MMSE": " MMSE",
            "OMLSA": " OMLSA",
            "PSOLA": " PSOLA",
            "WSOLA": " WSOLA",
            "CQT": " CQT",
            "STFT": " STFT",
            "ISTFT": " ISTFT",
            "RMS": " RMS",
            "SNR": " SNR",
            "LSD": " LSD",
            "MOS": " MOS",
            "SDR": " SDR",
            "SI": " SI",
            "DNSMOS": " DNSMOS",
            "NISQA": " NISQA",
            "POLQA": " POLQA",
            "PESQ": " PESQ",
            "ViSQOL": " ViSQOL",
            "STOI": " STOI",
        }
        for old, new in replacements.items():
            key = key.replace(old, new)
        module_map[key.lower()] = (fname, cls)

# Ergänze YAML-Einträge
for entry in data["dsp_modules"]:
    key = entry["name"].lower()
    if key in module_map:
        entry["modul_file"] = module_map[key][0]
        entry["modul_class"] = module_map[key][1]
    else:
        entry["modul_file"] = None
        entry["modul_class"] = None

# Schreibe YAML zurück
with open(YAML_PATH, "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False)

logger.info("YAML-Liste wurde mit modul_file und modul_class ergänzt.")
