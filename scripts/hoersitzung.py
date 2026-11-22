#!/usr/bin/env python3
"""hoersitzung — EINE unterbrechungsfreie, verblindete Hörsitzung über alle Entscheidungsblöcke.

Zweck (Menschenzeit): Alle wartenden Hör-Entscheidungen (Declip, 12 Witness-Kanäle, Stereo,
C4, M5, Dereverb) laufen in einer Sitzung mit EINEM Befehl; zwischen den Paketen liegt keine
Handarbeit mehr. Der Mensch hört und drückt je Probe EINE Taste.

    python3 -B scripts/hoersitzung.py --hoerer NAME        # bauen (falls nötig) → hören → auswerten

Was das Werkzeug dem Menschen abnimmt (und warum es nicht „nur ein Player“ ist):

* **Machbarkeits-Vorprüfung:** Der exakte Vorzeichentest erreicht bei n Paaren höchstens
  p = 2/2^n; erst n ≥ 6 kann signifikant werden. Blöcke mit zu wenigen Paaren (heute Declip 3,
  Stereo 3, Jitter 1, Transport 1) werden VOR dem Hören zurückgestellt — Hörzeit dort wäre
  garantiert verschwendet. ``--auch-nicht-entscheidbare`` hört sie trotzdem (nur beschreibend).
* **Verblindung:** Die Sitzung zeigt nur „Probe i/N · Runde r/R“ — nie Block, Bedingung oder
  Dateinamen. Die Zuordnung liegt getrennt in ``versiegelt/schluessel.json``.
* **Ausgewogenheit:** Erste/zweite Position je Block exakt halb/halb (seed-deterministisch).
* **Pegel:** p25-Body-Pegelabgleich (Hörordnung §7); nur Abschwächung, nie Verstärkung.
* **Kontroll-Trials:** identische Paare (erwartet „=“) und 3,5-kHz-Tiefpass-Anker (erwartet:
  das Original). Die Anker-Trefferquote entscheidet über die Gültigkeit des Hörers.
* **Richtungs-explizite Belege:** ``aurik.ab_stats/2`` trägt ``siege_kandidat``/``siege_bestand``;
  die Auflösung A/B → Kandidat/Bestand erfolgt hier, nicht von Hand (§G9 (copilot-instructions.md)).
* **Kein erfundener Beleg:** Simulierte Läufe tragen ``simuliert=True`` und sind für
  ``flag_freigabe.py`` nie buchbar; das Werkzeug flippt nie ein Flag und schreibt den
  Maintainer-Sign-off nie selbst (``AUSSTEHEND`` im Entwurf).

Determinismus (§G5 (copilot-instructions.md)): Seed 42, keine Zeit in Entscheidungslogik;
Zeitstempel stehen nur in den Antwortprotokollen. Fail-closed (§V6 (copilot-instructions.md)):
unlesbare/inkompatible Paare werden mit Grund ausgewiesen, nie still übergangen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROBEN = ROOT / "study_stimuli" / "hoerproben"
LAUF_BASIS = ROOT / "output_audio" / "hoersitzung"

SCHEMA_PLAN = "aurik.hoersitzung_plan/1"
SCHEMA_ANTWORT = "aurik.hoersitzung_antwort/1"
SCHEMA_STATS = "aurik.ab_stats/2"

SEED = 42
GAP_S = 0.8  # Stille zwischen den beiden Proben eines Versuchs
ANTWORTZEIT_S = 6.0  # Planungsannahme je Versuch (nur Zeitschätzung, keine Entscheidung)
PEGEL_TOL_DB = 0.5  # p25-Pegelunterschied, ab dem abgeschwächt wird (Vorbelegung, Hörpanel offen)
ANKER_CUTOFF_HZ = 3500.0  # ITU-R BS.1534: 3,5-kHz-Tiefpass-Anker
ANKER_MIN_RICHTIG = 0.75  # Gültigkeit: Anteil richtig erkannter Anker (Vorbelegung, Hörpanel offen)
MIN_HOER_DAUER_S = 2.0  # Je Hälfte: kürzere Schnipsel (z. B. 0,1-s-Defektregionen) tragen kein Wohlklang-Urteil (Vorbelegung, Hörpanel offen)
ANKER_MIN_DAUER_S = 5.0  # Anker nur auf ausreichend langen Proben (1-s-Clicks tragen keinen Tiefpass)
KONTROLL_ANTEIL = 0.05
KONTROLL_MIN = 3
ALPHA = 0.05
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

# Reihenfolge der Blöcke in der Sitzung (empfohlen im Hör-Batch-Brief).
BLOCK_REIHENFOLGE = ("declip", "witness_", "stereo", "c4_vocos", "m5_rekonstruktion", "dereverb")


# ── Datenmodell ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Paar:
    """Ein A/B-Paar, bereits aufgelöst auf Bestand (Referenz) und Kandidat (Challenger)."""

    block: str
    paar_id: str
    titel: str
    bestand: Path
    kandidat: Path


@dataclass
class Block:
    """Eine Hör-Entscheidung (ein Flag oder ein Befund) mit ihren Paaren."""

    id: str
    name: str
    flag: str | None
    paare: list[Paar]
    nachbau: str = ""


# ── Statistik ────────────────────────────────────────────────────────────────


def vorzeichentest_p(siege: int, n: int) -> float:
    """Zweiseitiger exakter Vorzeichentest gegen H0 = 1/2 (identisch zu hoer_entscheidung)."""
    if n <= 0:
        return 1.0
    siege = max(0, min(siege, n))
    tail = sum(math.comb(n, k) for k in range(0, min(siege, n - siege) + 1))
    return min(1.0, 2.0 * tail / float(2**n))


def p_minimal(n: int) -> float:
    """Kleinstmögliches p bei n entscheidenden Paaren (alle für denselben Kandidaten)."""
    return 1.0 if n <= 0 else min(1.0, 2.0 / float(2**n))


def n_fuer_signifikanz(alpha: float = ALPHA) -> int:
    """Kleinste Paarzahl, bei der Signifikanz überhaupt möglich ist (ohne Gleichstände)."""
    n = 1
    while p_minimal(n) >= alpha:
        n += 1
    return n


# ── Pakete laden ─────────────────────────────────────────────────────────────


def _pfad(wert: str, proben: Path) -> Path:
    p = Path(wert)
    if p.is_absolute():
        return p
    return ROOT / p if (ROOT / p).exists() else proben / p


def _lade_manifest_sets(proben: Path) -> list[Block]:
    manifest = proben / "manifest.json"
    if not manifest.is_file():
        return []
    daten = json.loads(manifest.read_text(encoding="utf-8"))
    bloecke = []
    for set_name, s in (daten.get("sets") or {}).items():
        paare = [
            Paar(set_name, f"{set_name}__{i}", str(p.get("stamm", i)), _pfad(p["a"], proben), _pfad(p["b"], proben))
            for i, p in enumerate(s.get("proben") or [], start=1)
        ]
        bloecke.append(Block(set_name, str(s.get("entscheidung", set_name)), None, paare))
    return bloecke


def _lade_declip(proben: Path) -> Block | None:
    ordner = proben / "declip_produktionspfad"
    schluessel = ordner / "schluessel.json"
    if not schluessel.is_file():
        return None
    key = json.loads(schluessel.read_text(encoding="utf-8"))
    paare = []
    for stamm, zuordnung in sorted(key.items()):
        a, b = ordner / f"{stamm}_A.wav", ordner / f"{stamm}_B.wav"
        a_ist_kandidat = str(zuordnung).strip() == "A=flag_an"
        if str(zuordnung).strip() not in ("A=flag_an", "A=flag_aus"):
            raise ValueError(f"Declip-Schlüssel '{stamm}': unbekannte Zuordnung {zuordnung!r}")
        paare.append(
            Paar(
                "declip",
                f"declip__{stamm}",
                stamm,
                bestand=b if a_ist_kandidat else a,
                kandidat=a if a_ist_kandidat else b,
            )
        )
    return Block(
        "declip",
        "Musik-Declip-Kopf: Flag-AN (Kandidat) vs. Flag-AUS (Bestand)",
        "use_aspade_declipper",
        paare,
        "weitere Titel mit Clipping-Schaden erzeugen (scripts/erstelle_schadens_proben.py), dann "
        "AURIK_FORCE_CPU=1 .venv_aurik/bin/python -B scripts/erstelle_aspade_hoerpaare.py --produktionspfad",
    )


def _lade_datei_schluessel(proben: Path, ordner_name: str, block_id: str, name: str, nachbau: str) -> Block | None:
    ordner = proben / ordner_name
    schluessel = ordner / "schluessel.json"
    if not schluessel.is_file():
        return None
    key = json.loads(schluessel.read_text(encoding="utf-8"))
    paare = []
    for a in sorted(ordner.glob("*_A.wav")):
        stamm = a.name[: -len("_A.wav")]
        b = ordner / f"{stamm}_B.wav"
        ka, kb = key.get(a.name), key.get(b.name)
        if {ka, kb} != {"restauriert", "schaden"}:
            raise ValueError(f"{ordner_name} '{stamm}': Schlüssel nicht eindeutig ({ka!r}/{kb!r})")
        paare.append(
            Paar(
                block_id,
                f"{block_id}__{stamm}",
                stamm,
                bestand=a if ka == "schaden" else b,
                kandidat=a if ka == "restauriert" else b,
            )
        )
    return Block(block_id, name, None, paare, nachbau)


def _lade_witness(proben: Path) -> list[Block]:
    wurzel = proben / "witness_kopf"
    bloecke = []
    for ordner in sorted(p for p in wurzel.glob("*") if p.is_dir()) if wurzel.is_dir() else []:
        manifest = ordner / "manifest.jsonl"
        if not manifest.is_file():
            continue
        paare = []
        for zeile in manifest.read_text(encoding="utf-8").splitlines():
            if not zeile.strip():
                continue
            d = json.loads(zeile)
            # Erzeuger-Vertrag (erstelle_witness_kopf_hoerproben.py): A = Schaden, B = Witness-Kopf-Ergebnis.
            paare.append(
                Paar(
                    f"witness_{ordner.name}",
                    f"witness_{ordner.name}__{d['probe']}",
                    str(d.get("titel", d["probe"])),
                    bestand=_pfad(d["a"], proben),
                    kandidat=_pfad(d["b"], proben),
                )
            )
        bloecke.append(
            Block(
                f"witness_{ordner.name}",
                f"Witness-Kopf {ordner.name}: repariert (Kandidat) vs. Schaden (Bestand)",
                f"use_witness_kopf_{ordner.name}",
                paare,
                "weitere wirksame Regionen erzeugen: .venv_aurik/bin/python -B "
                f"scripts/erstelle_witness_kopf_hoerproben.py --klassen {ordner.name} --paare 6",
            )
        )
    return bloecke


def lade_bloecke(proben: Path = PROBEN) -> list[Block]:
    """Alle vorhandenen Pakete, aufgelöst auf Bestand/Kandidat, in Sitzungsreihenfolge."""
    bloecke: list[Block] = []
    declip = _lade_declip(proben)
    if declip:
        bloecke.append(declip)
    bloecke.extend(_lade_witness(proben))
    stereo = _lade_datei_schluessel(
        proben,
        "stereo_sinnbeweis",
        "stereo",
        "Stereo-Balance-Richtungs-Fix: restauriert (Kandidat) vs. Schaden (Bestand)",
        "weitere Stereo-Schadensproben: scripts/erstelle_sinnbeweis_hoerpaare.py --rezepte stereo_imbalance --paare 6",
    )
    if stereo:
        bloecke.append(stereo)
    bloecke.extend(b for b in _lade_manifest_sets(proben) if b.id in ("c4_vocos", "m5_rekonstruktion"))
    dereverb = _lade_datei_schluessel(
        proben,
        "dereverb_sinnbeweis",
        "dereverb",
        "Dereverb-Sinn-Beweis: restauriert (Kandidat) vs. Schaden (Bestand)",
        "",
    )
    if dereverb:
        bloecke.append(dereverb)
    rang = {b: i for i, b in enumerate(BLOCK_REIHENFOLGE)}

    def _schluessel(block: Block) -> tuple[int, str]:
        for prefix, i in rang.items():
            if block.id == prefix or (prefix.endswith("_") and block.id.startswith(prefix)):
                return i, block.id
        return len(rang), block.id

    return sorted(bloecke, key=_schluessel)


# ── Audio ────────────────────────────────────────────────────────────────────


def _lade_audio(pfad: Path) -> tuple[np.ndarray, int]:
    daten, sr = sf.read(str(pfad), dtype="float32", always_2d=True)
    return np.nan_to_num(daten, nan=0.0, posinf=0.0, neginf=0.0), int(sr)


def _p25(x: np.ndarray) -> float:
    return float(np.percentile(np.abs(x), 25))


def pegel_gains(bestand: np.ndarray, kandidat: np.ndarray, tol_db: float = PEGEL_TOL_DB) -> tuple[float, float, str]:
    """p25-Body-Pegelabgleich: (gain_bestand_dB, gain_kandidat_dB, Status) — nur Abschwächung."""
    pb, pk = _p25(bestand), _p25(kandidat)
    if pb < 1e-6 or pk < 1e-6:
        return 0.0, 0.0, "pegel_nicht_bestimmbar"
    delta = 20.0 * math.log10(pk / pb)
    if abs(delta) <= tol_db:
        return 0.0, 0.0, "ausgeglichen"
    return (0.0, -delta, "kandidat_abgeschwaecht") if delta > 0 else (delta, 0.0, "bestand_abgeschwaecht")


def anker_tiefpass(x: np.ndarray, sr: int, cutoff_hz: float = ANKER_CUTOFF_HZ) -> np.ndarray:
    """3,5-kHz-Tiefpass-Anker (8. Ordnung); gleiche Form wie die Eingabe."""
    if cutoff_hz >= sr / 2.0:
        raise ValueError(f"Abtastrate {sr} Hz zu niedrig für einen {cutoff_hz:.0f}-Hz-Anker")
    sos = signal.butter(8, cutoff_hz / (sr / 2.0), btype="low", output="sos")
    return np.asarray(signal.sosfilt(sos, x, axis=0), dtype=np.float32)


def _db_zu_faktor(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _sha256(pfad: Path, cache: dict[Path, str]) -> str:
    if pfad not in cache:
        h = hashlib.sha256()
        with pfad.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[pfad] = h.hexdigest()
    return cache[pfad]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _schreibe_wav(pfad: Path, erst: np.ndarray, zweit: np.ndarray, sr: int) -> float:
    luecke = np.zeros((int(round(GAP_S * sr)), erst.shape[1]), dtype=np.float32)
    voll = np.clip(np.concatenate([erst, luecke, zweit], axis=0), -1.0, 1.0)
    sf.write(str(pfad), voll, sr, subtype="PCM_24")
    return float(voll.shape[0]) / sr


# ── Plan bauen ───────────────────────────────────────────────────────────────


def _pruefe_paar(paar: Paar) -> tuple[tuple[np.ndarray, np.ndarray, int] | None, str]:
    """Lädt beide Dateien; liefert (Audio, '') oder (None, Grund)."""
    for p in (paar.bestand, paar.kandidat):
        if not p.is_file():
            return None, f"Datei fehlt: {_rel(p)}"
    try:
        b, sr_b = _lade_audio(paar.bestand)
        k, sr_k = _lade_audio(paar.kandidat)
    except Exception as exc:  # §V6 (copilot-instructions.md): Grund ausweisen
        return None, f"nicht lesbar ({type(exc).__name__}: {exc})"
    if sr_b != sr_k or b.shape[1] != k.shape[1]:
        return (
            None,
            f"Format verschieden (Bestand {sr_b} Hz/{b.shape[1]} Kanäle, Kandidat {sr_k} Hz/{k.shape[1]} Kanäle)",
        )
    if b.shape[0] == 0 or k.shape[0] == 0:
        return None, "leere Datei"
    if np.array_equal(b, k):
        return None, "Bestand und Kandidat sind identisch (A=B) — kein hörbarer Unterschied zu prüfen"
    return (b, k, sr_b), ""


def _balanciert(n: int, rng: np.random.Generator) -> list[bool]:
    """Exakt ausgewogene Wahrheitswerte (bei ungerader Zahl entscheidet der Seed)."""
    wahr = n // 2 + (int(rng.integers(0, 2)) if n % 2 else 0)
    werte = [True] * wahr + [False] * (n - wahr)
    return [werte[i] for i in rng.permutation(n)]


def _runden(dauern: list[float], runde_min: float) -> list[int]:
    grenze = runde_min * 60.0
    runde, summe, ergebnis = 1, 0.0, []
    for d in dauern:
        aufwand = d + ANTWORTZEIT_S
        if summe + aufwand > grenze and summe > 0:
            runde, summe = runde + 1, 0.0
        summe += aufwand
        ergebnis.append(runde)
    return ergebnis


def baue_plan(
    proben: Path,
    ausgang: Path,
    *,
    seed: int = SEED,
    runde_min: float = 20.0,
    mit_nicht_entscheidbaren: bool = False,
    simuliert: bool = False,
    bloecke: list[Block] | None = None,
) -> dict[str, Any]:
    """Prüft alle Pakete, stellt Unentscheidbares zurück und rendert die verblindete Sitzung."""
    rng = np.random.default_rng(seed)
    quellen = bloecke if bloecke is not None else lade_bloecke(proben)
    n_min = n_fuer_signifikanz()
    ausgeschlossen: list[dict[str, str]] = []
    zurueckgestellt: list[dict[str, Any]] = []
    aktiv: list[tuple[Block, list[tuple[Paar, np.ndarray, np.ndarray, int]]]] = []
    machbarkeit: dict[str, dict[str, Any]] = {}
    for block in quellen:
        gueltig = []
        for paar in block.paare:
            ergebnis, grund = _pruefe_paar(paar)
            if ergebnis is None:
                ausgeschlossen.append({"block": block.id, "paar": paar.paar_id, "grund": grund})
                continue
            gueltig.append((paar, *ergebnis))
        n = len(gueltig)
        median_s = float(np.median([min(len(b), len(k)) / sr for _p, b, k, sr in gueltig])) if gueltig else 0.0
        if gueltig and median_s < MIN_HOER_DAUER_S:
            machbarkeit[block.id] = {"n_paare": n, "hoerbar": False, "median_halbdauer_s": round(median_s, 2)}
            zurueckgestellt.append(
                {
                    "block": block.id,
                    "name": block.name,
                    "n_paare": n,
                    "p_minimal": round(p_minimal(n), 6),
                    "grund": f"Proben zu kurz für ein Hörurteil (Median {median_s:.2f} s < {MIN_HOER_DAUER_S:.0f} s je Hälfte, ohne musikalischen Kontext)",
                    "nachbau": block.nachbau
                    or "Paare mit musikalischem Kontext (Defekt eingebettet in ≥ 4 s Musik) neu erzeugen",
                }
            )
            logger.warning("hoersitzung: Block %s zurückgestellt — Proben zu kurz (Median %.2f s)", block.id, median_s)
            continue
        entscheidbar = p_minimal(n) < ALPHA
        machbarkeit[block.id] = {
            "n_paare": n,
            "p_minimal": round(p_minimal(n), 6),
            "entscheidbar": entscheidbar,
            "n_fuer_signifikanz": n_min,
            "fehlende_paare": max(0, n_min - n),
            "nachbau": block.nachbau,
        }
        if not entscheidbar and not mit_nicht_entscheidbaren:
            zurueckgestellt.append(
                {
                    "block": block.id,
                    "name": block.name,
                    "n_paare": n,
                    "p_minimal": round(p_minimal(n), 6),
                    "nachbau": block.nachbau,
                }
            )
            continue
        if gueltig:
            aktiv.append((block, gueltig))
    if not aktiv:
        raise ValueError(
            "Keine hörbaren Blöcke: alle Pakete fehlen, sind zu klein (n < "
            f"{n_min}) oder unlesbar — siehe zurückgestellt/ausgeschlossen."
        )

    cache = ausgang / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (ausgang / "versiegelt").mkdir(parents=True, exist_ok=True)
    (ausgang / "antworten").mkdir(parents=True, exist_ok=True)
    (ausgang / "ergebnis").mkdir(parents=True, exist_ok=True)

    echte: list[dict[str, Any]] = []
    for block, gueltig in aktiv:
        reihenfolge = [gueltig[i] for i in rng.permutation(len(gueltig))]
        seiten = _balanciert(len(reihenfolge), rng)
        for (paar, b, k, sr), kandidat_zuerst in zip(reihenfolge, seiten):
            g_b, g_k, pegel = pegel_gains(b, k)
            echte.append(
                {
                    "art": "paar",
                    "block": block.id,
                    "paar_id": paar.paar_id,
                    "titel": paar.titel,
                    "erstes": "kandidat" if kandidat_zuerst else "bestand",
                    "bestand": paar.bestand,
                    "kandidat": paar.kandidat,
                    "gain_bestand_db": round(g_b, 3),
                    "gain_kandidat_db": round(g_k, 3),
                    "pegel": pegel,
                    "_b": b if g_b == 0.0 else b * _db_zu_faktor(g_b),
                    "_k": k if g_k == 0.0 else k * _db_zu_faktor(g_k),
                    "_sr": sr,
                }
            )

    kontrollen = _baue_kontrollen(echte, rng)
    versuche = _mische_kontrollen(echte, kontrollen)
    dauern = []
    sha_cache: dict[Path, str] = {}
    schluessel_eintraege: dict[str, dict[str, Any]] = {}
    oeffentlich: list[dict[str, Any]] = []
    for nr, v in enumerate(versuche, start=1):
        vid = f"t{nr:03d}"
        if v["art"] == "paar":
            erst, zweit = (v["_k"], v["_b"]) if v["erstes"] == "kandidat" else (v["_b"], v["_k"])
        else:
            erst, zweit = v["_erst"], v["_zweit"]
        dauer = _schreibe_wav(cache / f"{vid}.wav", erst, zweit, v["_sr"])
        dauern.append(dauer)
        eintrag = {k: val for k, val in v.items() if not k.startswith("_") and k not in ("bestand", "kandidat")}
        if v["art"] == "paar":
            eintrag["bestand"] = _rel(v["bestand"])
            eintrag["kandidat"] = _rel(v["kandidat"])
            eintrag["sha256_bestand"] = _sha256(v["bestand"], sha_cache)
            eintrag["sha256_kandidat"] = _sha256(v["kandidat"], sha_cache)
        else:
            eintrag["quelle"] = _rel(v["quelle"])
            eintrag["sha256_quelle"] = _sha256(v["quelle"], sha_cache)
        schluessel_eintraege[vid] = eintrag
        oeffentlich.append({"nr": nr, "id": vid, "datei": f"cache/{vid}.wav", "dauer_s": round(dauer, 2)})
    for eintrag_o, runde in zip(oeffentlich, _runden(dauern, runde_min)):
        eintrag_o["runde"] = runde
        schluessel_eintraege[eintrag_o["id"]]["runde"] = runde

    kal = next((v for v in versuche if len(_erste_probe(v)) / v["_sr"] >= ANKER_MIN_DAUER_S), versuche[0])
    sf.write(
        str(cache / "kalibrierung.wav"),
        np.clip(_erste_probe(kal)[: int(8 * kal["_sr"])], -1.0, 1.0),
        kal["_sr"],
        subtype="PCM_24",
    )

    schluessel = {
        "schema": SCHEMA_PLAN + "/schluessel",
        "seed": seed,
        "simuliert": simuliert,
        "versuche": schluessel_eintraege,
        "bloecke": {b.id: {"name": b.name, "flag": b.flag, "n_paare": len(g)} for b, g in aktiv},
        "machbarkeit": machbarkeit,
        "zurueckgestellt": zurueckgestellt,
        "ausgeschlossen": ausgeschlossen,
    }
    kanonisch = json.dumps(schluessel, sort_keys=True, ensure_ascii=False)
    plan_sha = hashlib.sha256(kanonisch.encode("utf-8")).hexdigest()
    sitzung_id = plan_sha[:12]
    schluessel["plan_sha256"] = plan_sha
    schluessel["sitzung_id"] = sitzung_id
    anzahl_runden = max(r["runde"] for r in oeffentlich)
    minuten = round(sum(dauern) / 60.0 + len(dauern) * ANTWORTZEIT_S / 60.0, 1)
    plan = {
        "schema": SCHEMA_PLAN,
        "sitzung_id": sitzung_id,
        "plan_sha256": plan_sha,
        "seed": seed,
        "simuliert": simuliert,
        "versuche": oeffentlich,
        "n_versuche": len(oeffentlich),
        "n_runden": anzahl_runden,
        "geschaetzte_minuten": minuten,
        "kalibrierung": "cache/kalibrierung.wav",
    }
    (ausgang / "versiegelt" / "schluessel.json").write_text(
        json.dumps(schluessel, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ausgang / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    (ausgang / "LIESMICH.txt").write_text(
        "Hörsitzung starten:\n  python3 -B scripts/hoersitzung.py hoeren --sitzung <dieser Ordner> --hoerer NAME\n"
        "Taste je Probe: 1 = erste besser, 2 = zweite besser, = = kein Unterschied (w wiederholen, z zurück, p Pause, q beenden).\n"
        "Ordner 'versiegelt' erst NACH der Sitzung öffnen.\n",
        encoding="utf-8",
    )
    return {
        "ausgang": str(ausgang),
        "sitzung_id": sitzung_id,
        "n_versuche": len(oeffentlich),
        "n_runden": anzahl_runden,
        "geschaetzte_minuten": minuten,
        "bloecke": {b.id: len(g) for b, g in aktiv},
        "zurueckgestellt": zurueckgestellt,
        "ausgeschlossen": ausgeschlossen,
        "machbarkeit": machbarkeit,
        "n_kontrollen": len(kontrollen),
    }


def _erste_probe(v: dict[str, Any]) -> np.ndarray:
    """Hörprobe zum Einstellen der Lautstärke (die ohnehin in der Sitzung vorkommt)."""
    if v["art"] == "paar":
        return v["_k"] if v["erstes"] == "kandidat" else v["_b"]
    return v["_erst"]


def _baue_kontrollen(echte: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    """Anker- und Identisch-Kontrollen aus langen Bestands-Proben (seed-deterministisch)."""
    lange = [v for v in echte if len(v["_b"]) / v["_sr"] >= ANKER_MIN_DAUER_S]
    if not lange:
        # Nur Kurzproben (z. B. 1-s-Witness-Clips): der Anker bleibt hörbar, ist aber weniger trennscharf.
        logger.warning("hoersitzung: keine Probe ≥ %.0f s — Anker-Kontrollen auf Kurzproben", ANKER_MIN_DAUER_S)
        lange = list(echte)
    if not lange:
        return []
    n = max(KONTROLL_MIN, math.ceil(KONTROLL_ANTEIL * len(echte)))
    kontrollen: list[dict[str, Any]] = []
    for art in ("anker", "identisch"):
        wahl = rng.choice(len(lange), size=n, replace=n > len(lange))
        seiten = _balanciert(n, rng)
        for idx, original_zuerst in zip(wahl, seiten):
            v = lange[int(idx)]
            x, sr = v["_b"], v["_sr"]
            if art == "anker":
                anker = anker_tiefpass(x, sr)
                erst, zweit = (x, anker) if original_zuerst else (anker, x)
                erstes = "original" if original_zuerst else "anker"
            else:
                erst, zweit, erstes = x, x, "identisch"
            kontrollen.append(
                {
                    "art": art,
                    "block": None,
                    "paar_id": v["paar_id"],
                    "erstes": erstes,
                    "quelle": v["bestand"],
                    "_erst": erst,
                    "_zweit": zweit,
                    "_sr": sr,
                }
            )
    return kontrollen


def _mische_kontrollen(echte: list[dict[str, Any]], kontrollen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kontrollen gleichmäßig über die Sitzung verteilen (deterministisch, nie als Erstes)."""
    gesamt = len(echte) + len(kontrollen)
    if not kontrollen:
        return list(echte)
    stellen = sorted({max(1, int(round((j + 1) * gesamt / (len(kontrollen) + 1)))) for j in range(len(kontrollen))})
    while len(stellen) < len(kontrollen):  # Kollisionen auffüllen
        stellen.append(next(i for i in range(1, gesamt) if i not in stellen))
        stellen.sort()
    ergebnis: list[dict[str, Any]] = []
    e_iter, k_iter = iter(echte), iter(kontrollen)
    for pos in range(gesamt):
        ergebnis.append(next(k_iter) if pos in stellen else next(e_iter))
    return ergebnis


# ── Hören ────────────────────────────────────────────────────────────────────


def finde_player() -> str | None:
    import shutil

    for kandidat in ("ffplay", "pw-play", "aplay", "paplay"):
        if shutil.which(kandidat):
            return kandidat
    return None


def spiele(pfad: Path, player: str | None = None) -> bool:
    """Spielt eine Datei vollständig ab (blockierend); stdin bleibt für Tasten frei."""
    player = player or finde_player()
    if not player:
        return False
    befehle = {
        "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(pfad)],
        "pw-play": ["pw-play", str(pfad)],
        "aplay": ["aplay", "-q", str(pfad)],
        "paplay": ["paplay", str(pfad)],
    }
    try:
        subprocess.run(befehle[player], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        return True
    except Exception as exc:
        logger.warning("Abspielen fehlgeschlagen (%s): %s", player, exc)
        return False


def lese_taste() -> str:
    """Eine Taste ohne Enter (TTY) bzw. das erste Zeichen der Zeile (Pipe)."""
    if sys.stdin.isatty():
        import termios
        import tty

        fd = sys.stdin.fileno()
        alt = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            zeichen = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, alt)
        return zeichen
    zeile = sys.stdin.readline()
    return zeile.strip()[:1] if zeile else "q"


def _lade_antworten(pfad: Path) -> dict[str, dict[str, Any]]:
    """Letzter gültiger Eintrag je Versuch; ein Widerruf entfernt den Versuch."""
    antworten: dict[str, dict[str, Any]] = {}
    if not pfad.is_file():
        return antworten
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        if not zeile.strip():
            continue
        try:
            d = json.loads(zeile)
        except json.JSONDecodeError:
            logger.warning("Antwortprotokoll %s: defekte Zeile ignoriert", pfad.name)
            continue
        if d.get("widerrufen"):
            antworten.pop(d.get("id"), None)
        elif d.get("id") and d.get("antwort") in ("1", "2", "="):
            antworten[d["id"]] = d
    return antworten


def _anhaengen(pfad: Path, eintrag: dict[str, Any]) -> None:
    """Append mit fsync: ein Absturz verliert höchstens den laufenden Versuch."""
    with pfad.open("a", encoding="utf-8") as f:
        f.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _jetzt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hoeren(
    ausgang: Path,
    hoerer: str,
    *,
    spiele_fn: Callable[[Path], bool] | None = None,
    taste_fn: Callable[[], str] | None = None,
    ausgabe: Callable[[str], None] = print,
    uhr: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Unterbrechungsfreie Sitzung: spielen → eine Taste → nächste Probe. Wiederaufnahmefähig."""
    import time as _time

    if not NAME_RE.match(hoerer):
        raise ValueError("Hörer-Kennung: 1–40 Zeichen aus A–Z, a–z, 0–9, _ und -")
    plan = json.loads((ausgang / "plan.json").read_text(encoding="utf-8"))
    versuche = plan["versuche"]
    spiele_fn = spiele_fn or (lambda p: spiele(p))
    taste_fn = taste_fn or lese_taste
    uhr = uhr or _time.monotonic
    protokoll = ausgang / "antworten" / f"{hoerer}.jsonl"
    antworten = _lade_antworten(protokoll)
    offen = [i for i, v in enumerate(versuche) if v["id"] not in antworten]
    if not offen:
        ausgabe("Alle Proben sind bereits beantwortet.")
        return {"beantwortet": len(antworten), "gesamt": len(versuche), "abgeschlossen": True, "status": "fertig"}

    if not antworten:
        ausgabe(
            "\nHörsitzung — bitte jetzt Kopfhörer aufsetzen und die LAUTSTÄRKE EINMAL einstellen (danach nicht mehr ändern).\n"
            "Tasten je Probe:  1 = erste besser   2 = zweite besser   = = kein Unterschied\n"
            "                  w = wiederholen    z = zurück         p = Pause    q = speichern und beenden\n"
            f"{len(versuche)} Proben in {plan['n_runden']} Runde(n), etwa {plan['geschaetzte_minuten']:.0f} Minuten.\n"
            "Kalibrierungston folgt — danach beliebige Taste."
        )
        if not spiele_fn(ausgang / plan["kalibrierung"]):
            return {"beantwortet": 0, "gesamt": len(versuche), "abgeschlossen": False, "status": "player_fehler"}
        taste_fn()

    pos = offen[0]
    verlauf: list[str] = [v["id"] for v in versuche[:pos] if v["id"] in antworten]
    letzte_runde = versuche[pos - 1]["runde"] if pos > 0 else None
    while pos < len(versuche):
        v = versuche[pos]
        if v["id"] in antworten:
            pos += 1
            continue
        if letzte_runde is not None and v["runde"] != letzte_runde:
            ausgabe(
                f"\n── Runde {letzte_runde} von {plan['n_runden']} geschafft. Kurze Pause tut den Ohren gut. Weiter: beliebige Taste (q = beenden)"
            )
            if taste_fn() == "q":
                return {
                    "beantwortet": len(antworten),
                    "gesamt": len(versuche),
                    "abgeschlossen": False,
                    "status": "abgebrochen",
                }
        letzte_runde = v["runde"]
        wiederholungen = 0
        aktion: str | None = None
        while aktion is None:
            ausgabe(f"Probe {v['nr']}/{len(versuche)} · Runde {v['runde']}/{plan['n_runden']}  ▶")
            if not spiele_fn(ausgang / v["datei"]):
                ausgabe("Abspielen fehlgeschlagen — Sitzung gespeichert, bitte Audioausgabe prüfen und neu starten.")
                return {
                    "beantwortet": len(antworten),
                    "gesamt": len(versuche),
                    "abgeschlossen": False,
                    "status": "player_fehler",
                }
            start = uhr()
            while True:
                taste = taste_fn().lower()
                if taste == "0":
                    taste = "="
                if taste in ("1", "2", "="):
                    aktion = taste
                    break
                if taste in ("w", " "):
                    wiederholungen += 1
                    break
                if taste == "p":
                    ausgabe("Pause — beliebige Taste zum Fortsetzen.")
                    taste_fn()
                    break
                if taste == "z":
                    if verlauf:
                        zurueck = verlauf.pop()
                        _anhaengen(protokoll, {"id": zurueck, "widerrufen": True, "zeit": _jetzt()})
                        antworten.pop(zurueck, None)
                        pos = next(i for i, x in enumerate(versuche) if x["id"] == zurueck)
                        aktion = "zurueck"
                    else:
                        ausgabe("Nichts zum Zurücknehmen.")
                    break
                if taste == "q":
                    ausgabe("Gespeichert. Später mit demselben Befehl fortsetzen.")
                    return {
                        "beantwortet": len(antworten),
                        "gesamt": len(versuche),
                        "abgeschlossen": False,
                        "status": "abgebrochen",
                    }
            if aktion == "zurueck":
                break
        if aktion == "zurueck":
            letzte_runde = versuche[pos]["runde"]  # kein erneuter Runden-Hinweis nach „zurück“
            continue
        eintrag = {
            "schema": SCHEMA_ANTWORT,
            "hoerer": hoerer,
            "sitzung_id": plan["sitzung_id"],
            "id": v["id"],
            "antwort": aktion,
            "wiederholungen": wiederholungen,
            "antwortzeit_s": round(uhr() - start, 2),
            "zeit": _jetzt(),
        }
        _anhaengen(protokoll, eintrag)
        antworten[v["id"]] = eintrag
        verlauf.append(v["id"])
        pos += 1
    ausgabe("\nAlle Proben beantwortet — vielen Dank.")
    return {"beantwortet": len(antworten), "gesamt": len(versuche), "abgeschlossen": True, "status": "fertig"}


# ── Auswerten ────────────────────────────────────────────────────────────────


def _gewinner(eintrag: dict[str, Any], antwort: str) -> str | None:
    """'kandidat' | 'bestand' | None (Gleichstand) für einen Paar-Versuch."""
    if antwort == "=":
        return None
    erstes_ist_kandidat = eintrag["erstes"] == "kandidat"
    gewaehlt_erstes = antwort == "1"
    return "kandidat" if gewaehlt_erstes == erstes_ist_kandidat else "bestand"


def _hoerer_validitaet(
    schluessel: dict[str, Any], antworten: dict[str, dict[str, Any]], min_anker: float
) -> dict[str, Any]:
    anker = [(vid, e) for vid, e in schluessel["versuche"].items() if e["art"] == "anker"]
    ident = [(vid, e) for vid, e in schluessel["versuche"].items() if e["art"] == "identisch"]
    richtig = sum(1 for vid, e in anker if vid in antworten and _anker_richtig(e, antworten[vid]["antwort"]))
    beantwortet_anker = sum(1 for vid, _ in anker if vid in antworten)
    gleich = sum(1 for vid, _ in ident if vid in antworten and antworten[vid]["antwort"] == "=")
    beantwortet_ident = sum(1 for vid, _ in ident if vid in antworten)
    quote = (richtig / beantwortet_anker) if beantwortet_anker else None
    return {
        "anker_richtig": richtig,
        "anker_beantwortet": beantwortet_anker,
        "anker_quote": round(quote, 3) if quote is not None else None,
        "identisch_gleich": gleich,
        "identisch_beantwortet": beantwortet_ident,
        "gueltig": bool(quote is not None and quote >= min_anker),
        "grund": (
            "keine Anker beantwortet"
            if quote is None
            else ("" if quote >= min_anker else f"Anker-Trefferquote {quote:.2f} < {min_anker:.2f}")
        ),
    }


def _anker_richtig(eintrag: dict[str, Any], antwort: str) -> bool:
    if antwort == "=":
        return False
    original_zuerst = eintrag["erstes"] == "original"
    return (antwort == "1") == original_zuerst


def auswerten(ausgang: Path, *, min_anker: float = ANKER_MIN_RICHTIG) -> dict[str, Any]:
    """Löst A/B auf Kandidat/Bestand auf, prüft Validität und schreibt richtungs-explizite Belege."""
    plan = json.loads((ausgang / "plan.json").read_text(encoding="utf-8"))
    schluessel = json.loads((ausgang / "versiegelt" / "schluessel.json").read_text(encoding="utf-8"))
    if plan["plan_sha256"] != schluessel["plan_sha256"]:
        raise ValueError("Plan und Schlüssel passen nicht zusammen (plan_sha256) — Sitzung nicht auswertbar")
    simuliert = bool(plan.get("simuliert"))
    protokolle = sorted((ausgang / "antworten").glob("*.jsonl"))
    hoerer: dict[str, dict[str, dict[str, Any]]] = {p.stem: _lade_antworten(p) for p in protokolle}
    validitaet = {h: _hoerer_validitaet(schluessel, a, min_anker) for h, a in hoerer.items()}
    gueltige = sorted(h for h, v in validitaet.items() if v["gueltig"])
    ausgeschlossen = {h: validitaet[h]["grund"] for h in sorted(hoerer) if h not in gueltige}

    ergebnis_dir = ausgang / "ergebnis"
    ergebnis_dir.mkdir(exist_ok=True)
    verdikte: dict[str, dict[str, Any]] = {}
    for block_id, info in schluessel["bloecke"].items():
        paar_versuche = {
            vid: e for vid, e in schluessel["versuche"].items() if e["art"] == "paar" and e["block"] == block_id
        }
        stimmen_je_paar: dict[str, dict[str, list[str | None]]] = {}
        vollstaendig = bool(gueltige)
        for vid, e in paar_versuche.items():
            stimmen_je_paar[e["paar_id"]] = {}
            for h in gueltige:
                if vid in hoerer[h]:
                    stimmen_je_paar[e["paar_id"]][h] = [_gewinner(e, hoerer[h][vid]["antwort"])]
                else:
                    vollstaendig = False
        urteile: dict[str, str | None] = {}
        for paar_id, je_hoerer in stimmen_je_paar.items():
            votes = [v[0] for v in je_hoerer.values()]
            k, b = votes.count("kandidat"), votes.count("bestand")
            urteile[paar_id] = "kandidat" if k > b else ("bestand" if b > k else None)
        sk = sum(1 for u in urteile.values() if u == "kandidat")
        sb = sum(1 for u in urteile.values() if u == "bestand")
        gleich = len(urteile) - sk - sb
        n_ent = sk + sb
        p = vorzeichentest_p(sk, n_ent)
        signifikant = bool(n_ent > 0 and p < ALPHA)
        richtung = "kandidat" if sk > sb else ("bestand" if sb > sk else "unentschieden")
        verdikt = {
            "schema": SCHEMA_STATS,
            "block": block_id,
            "entscheidung": info["name"],
            "flag": info["flag"],
            "kandidat": "Challenger",
            "bestand": "Bestand/Referenz",
            "siege_kandidat": sk,
            "siege_bestand": sb,
            "gleichstaende": gleich,
            "n_paare": info["n_paare"],
            "n_entscheidend": n_ent,
            "p_wert_vorzeichen_test": round(p, 6),
            "signifikant": signifikant,
            "richtung": richtung,
            "gueltig": bool(gueltige),
            "vollstaendig": bool(vollstaendig and gueltige),
            "hoerer": gueltige,
            "ausgeschlossene_hoerer": ausgeschlossen,
            "simuliert": simuliert,
            "quelle": "scripts/hoersitzung.py",
            "sitzung_id": plan["sitzung_id"],
            "plan_sha256": plan["plan_sha256"],
            "paar_urteile": urteile,
        }
        verdikt["buchbar"] = bool(
            signifikant
            and richtung == "kandidat"
            and verdikt["gueltig"]
            and verdikt["vollstaendig"]
            and not simuliert
            and info["flag"]
        )
        verdikte[block_id] = verdikt
        (ergebnis_dir / f"hoer_verdikt_{block_id}.json").write_text(
            json.dumps(verdikt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if verdikt["buchbar"]:
            (ergebnis_dir / f"signoff_entwurf_{block_id}.md").write_text(
                _signoff_entwurf(verdikt, validitaet, ergebnis_dir), encoding="utf-8"
            )

    zusammenfassung = {
        "sitzung_id": plan["sitzung_id"],
        "simuliert": simuliert,
        "hoerer": validitaet,
        "gueltige_hoerer": gueltige,
        "ausgeschlossene_hoerer": ausgeschlossen,
        "verdikte": {
            b: {
                k: v[k]
                for k in (
                    "richtung",
                    "siege_kandidat",
                    "siege_bestand",
                    "gleichstaende",
                    "p_wert_vorzeichen_test",
                    "signifikant",
                    "gueltig",
                    "vollstaendig",
                    "buchbar",
                    "hoerer",
                    "simuliert",
                )
            }
            for b, v in verdikte.items()
        },
        "zurueckgestellt": schluessel["zurueckgestellt"],
        "ausgeschlossen_paare": schluessel["ausgeschlossen"],
    }
    (ergebnis_dir / "ergebnis.json").write_text(
        json.dumps(zusammenfassung, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ergebnis_dir / "bericht.md").write_text(
        _bericht(plan, schluessel, verdikte, validitaet, ergebnis_dir), encoding="utf-8"
    )
    return zusammenfassung


def _signoff_entwurf(verdikt: dict[str, Any], validitaet: dict[str, Any], ergebnis_dir: Path) -> str:
    pfad = _rel(ergebnis_dir / f"hoer_verdikt_{verdikt['block']}.json")
    anker = ", ".join(
        f"{h}: {validitaet[h]['anker_richtig']}/{validitaet[h]['anker_beantwortet']}" for h in verdikt["hoerer"]
    )
    return (
        f"# Hör-Sign-off-Entwurf — {verdikt['entscheidung']}\n\n"
        f"- Beleg: `{pfad}` (`aurik.ab_stats/2`, Sitzung `{verdikt['sitzung_id']}`)\n"
        f"- Ergebnis: Kandidat {verdikt['siege_kandidat']} : Bestand {verdikt['siege_bestand']} "
        f"(Gleichstände {verdikt['gleichstaende']}, n={verdikt['n_entscheidend']}), p={verdikt['p_wert_vorzeichen_test']}, signifikant\n"
        f"- Hörer: {', '.join(verdikt['hoerer'])}; Anker richtig: {anker}\n"
        "- Verfahren: verblindet, Position ausgewogen, p25-Pegelabgleich, Kontroll-Trials (Anker + identische Paare)\n"
        "- Grenzen: Der Beleg gilt für diese Stichprobe von Paaren; er ist kein Mehr-Hörer-Panel und keine Aussage über andere Genres.\n"
        "- Maintainer Sign-off: AUSSTEHEND — vom Hörer nach eigener inhaltlicher Prüfung mit Name und Datum zu ergänzen.\n\n"
        "Buchung (erst nach dem Sign-off):\n\n"
        "```bash\n"
        f".venv_aurik/bin/python -B scripts/flag_freigabe.py --flag {verdikt['flag']} \\\n"
        f"  --verdikt {pfad} --signoff <diese Datei>\n"
        "```\n"
    )


def _bericht(
    plan: dict[str, Any],
    schluessel: dict[str, Any],
    verdikte: dict[str, dict[str, Any]],
    validitaet: dict[str, Any],
    ergebnis_dir: Path,
) -> str:
    z = [f"# Hörsitzung {plan['sitzung_id']} — Ergebnis", ""]
    if plan.get("simuliert"):
        z += ["> **SIMULIERTER LAUF — kein Hörbeleg, nicht buchbar.**", ""]
    z += [
        "## Hörer und Gültigkeit",
        "",
        "| Hörer | Anker richtig | Identisch → „=“ | gültig |",
        "| --- | --- | --- | --- |",
    ]
    for h, v in validitaet.items():
        z.append(
            f"| {h} | {v['anker_richtig']}/{v['anker_beantwortet']} | {v['identisch_gleich']}/{v['identisch_beantwortet']} | "
            f"{'ja' if v['gueltig'] else 'NEIN — ' + v['grund']} |"
        )
    z += [
        "",
        "## Entscheidungen",
        "",
        "| Block | Kandidat : Bestand | Gleich | p | Richtung | vollständig | buchbar |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b, v in verdikte.items():
        z.append(
            f"| {b} | {v['siege_kandidat']} : {v['siege_bestand']} | {v['gleichstaende']} | {v['p_wert_vorzeichen_test']} | "
            f"{v['richtung']} | {'ja' if v['vollstaendig'] else 'nein'} | {'**ja**' if v['buchbar'] else 'nein'} |"
        )
    z += [
        "",
        "Buchbar heißt: signifikant **für den Kandidaten**, gültiger Hörer, vollständig, nicht simuliert, Flag vorhanden.",
    ]
    if schluessel["zurueckgestellt"]:
        z += ["", "## Zurückgestellt — statistisch nicht entscheidbar (zu wenige Paare)", ""]
        for r in schluessel["zurueckgestellt"]:
            z.append(
                f"- **{r['block']}**: n={r['n_paare']}, {r.get('grund') or ('kleinstmögliches p=' + str(r['p_minimal']) + ' (≥ 0,05)')}. {('Nächster Schritt: ' + r['nachbau']) if r['nachbau'] else ''}"
            )
    if schluessel["ausgeschlossen"]:
        z += ["", "## Ausgeschlossene Paare", ""]
        z += [f"- {a['block']} / {a['paar']}: {a['grund']}" for a in schluessel["ausgeschlossen"]]
    nicht_buchbar = [
        b for b, v in verdikte.items() if not v["buchbar"] and v["signifikant"] and v["richtung"] == "bestand"
    ]
    if nicht_buchbar:
        z += [
            "",
            "## Hinweis",
            "",
            f"Signifikant **für den Bestand**: {', '.join(nicht_buchbar)} — kein Flag-Flip; der Bestandspfad bleibt bestätigt.",
        ]
    return "\n".join(z) + "\n"


# ── Status / CLI ─────────────────────────────────────────────────────────────


def status(ausgang: Path) -> str:
    plan = json.loads((ausgang / "plan.json").read_text(encoding="utf-8"))
    zeilen = [
        f"Sitzung {plan['sitzung_id']}: {plan['n_versuche']} Proben, {plan['n_runden']} Runde(n), ≈ {plan['geschaetzte_minuten']:.0f} min"
        + ("  [SIMULIERT]" if plan.get("simuliert") else "")
    ]
    for p in sorted((ausgang / "antworten").glob("*.jsonl")):
        zeilen.append(f"  Hörer {p.stem}: {len(_lade_antworten(p))}/{plan['n_versuche']} beantwortet")
    return "\n".join(zeilen)


def _neueste_sitzung() -> Path | None:
    if not LAUF_BASIS.is_dir():
        return None
    kandidaten = sorted(
        (p for p in LAUF_BASIS.iterdir() if (p / "plan.json").is_file()), key=lambda p: p.stat().st_mtime
    )
    return kandidaten[-1] if kandidaten else None


def _drucke_plan(r: dict[str, Any]) -> None:
    print(
        f"\nSitzung {r['sitzung_id']} gebaut: {r['n_versuche']} Proben (davon {r['n_kontrollen']} Kontrollen), "
        f"{r['n_runden']} Runde(n), ≈ {r['geschaetzte_minuten']:.0f} Minuten Hörzeit."
    )
    print("Blöcke (Paare):", ", ".join(f"{b}={n}" for b, n in r["bloecke"].items()))
    if r["zurueckgestellt"]:
        print(
            f"\nZurückgestellt (nicht hörbar oder statistisch nicht entscheidbar; Signifikanz braucht ≥ {n_fuer_signifikanz()} Paare):"
        )
        for z in r["zurueckgestellt"]:
            print(
                f"  - {z['block']}: n={z['n_paare']}, {z.get('grund') or 'kleinstmögliches p=' + str(z['p_minimal'])}"
            )
    if r["ausgeschlossen"]:
        print(f"\nAusgeschlossene Paare: {len(r['ausgeschlossen'])} (Gründe im Plan/Bericht)")


def _drucke_ergebnis(erg: dict[str, Any], ausgang: Path) -> None:
    print("\n── Ergebnis ─────────────────────────────────────────")
    if erg["simuliert"]:
        print("SIMULIERTER LAUF — kein Hörbeleg.")
    for h, v in erg["hoerer"].items():
        print(
            f"  Hörer {h}: Anker {v['anker_richtig']}/{v['anker_beantwortet']} — {'gültig' if v['gueltig'] else 'UNGÜLTIG: ' + v['grund']}"
        )
    for b, v in erg["verdikte"].items():
        marke = (
            "BUCHBAR"
            if v["buchbar"]
            else ("signifikant für Bestand" if v["signifikant"] and v["richtung"] == "bestand" else "—")
        )
        print(
            f"  {b:40} Kandidat {v['siege_kandidat']} : Bestand {v['siege_bestand']} (p={v['p_wert_vorzeichen_test']}) {marke}"
        )
    print(f"\nBericht: {_rel(ausgang / 'ergebnis' / 'bericht.md')}")
    buchbar = [b for b, v in erg["verdikte"].items() if v["buchbar"]]
    if buchbar:
        print("Sign-off-Entwürfe liegen bereit (Maintainer-Sign-off AUSSTEHEND) — Buchung erst nach Ihrem Sign-off:")
        for b in buchbar:
            print(f"  {_rel(ausgang / 'ergebnis' / f'signoff_entwurf_{b}.md')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unterbrechungsfreie verblindete Hörsitzung")
    sub = ap.add_subparsers(dest="befehl")
    for name in ("bauen", "hoeren", "auswerten", "status", "los"):
        sp = sub.add_parser(name)
        sp.add_argument("--sitzung", type=Path, default=None, help="Sitzungsordner (Standard: neueste bzw. neu)")
        sp.add_argument("--hoerer", default=os.environ.get("USER", "hoerer"))
        sp.add_argument("--proben", type=Path, default=PROBEN)
        sp.add_argument("--seed", type=int, default=SEED)
        sp.add_argument("--runde-min", type=float, default=20.0)
        sp.add_argument("--auch-nicht-entscheidbare", action="store_true")
        sp.add_argument("--min-anker", type=float, default=ANKER_MIN_RICHTIG)
        sp.add_argument("--neu", action="store_true", help="Plan neu bauen (verwirft den alten)")
    rohargs = list(sys.argv[1:] if argv is None else argv)
    if not rohargs or rohargs[0].startswith("-"):
        rohargs = ["los", *rohargs]  # ohne Unterbefehl: bauen (falls nötig) → hören → auswerten
    args = ap.parse_args(rohargs)
    befehl = args.befehl
    ausgang = args.sitzung or _neueste_sitzung() or (LAUF_BASIS / f"sitzung_s{args.seed}")

    def _bauen() -> None:
        r = baue_plan(
            args.proben,
            ausgang,
            seed=args.seed,
            runde_min=args.runde_min,
            mit_nicht_entscheidbaren=args.auch_nicht_entscheidbare,
        )
        _drucke_plan(r)

    if befehl == "status":
        print(status(ausgang))
        return 0
    if befehl == "bauen" or (befehl in ("los", "hoeren") and (args.neu or not (ausgang / "plan.json").is_file())):
        _bauen()
        if befehl == "bauen":
            return 0
    if befehl in ("los", "hoeren"):
        ergebnis = hoeren(ausgang, args.hoerer)
        if ergebnis["status"] in ("abgebrochen", "player_fehler"):
            return 1 if ergebnis["status"] == "player_fehler" else 0
        if befehl == "hoeren":
            return 0
    erg = auswerten(ausgang, min_anker=args.min_anker)
    _drucke_ergebnis(erg, ausgang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
