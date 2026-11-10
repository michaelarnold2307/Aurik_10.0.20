#!/usr/bin/env python3
"""ohr_panel — „Mein Ohr" ohne manuelle Umstände (M6-Komfort, 2026-11-09).

EIN Kommando für die komplette Ohr-Kalibrierung:
  1. erkennt A/B-Kandidaten-Paare im angegebenen Verzeichnis — ODER erzeugt
     sie selbst (--erzeuge-paare: Original vs. kontrollierte Defekt-Variante
     je Musikstück — Sie brauchen NUR einen Ordner mit Musik),
  2. baut das verblindete Versuchsblatt (scripts/ab_harness.py prepare),
  3. fragt interaktiv ab („links oder rechts besser? [l/r/=]"),
  4. erzeugt die Urteils-Session (ab_harness ohr), fittet den Ohr-Prior
     (scripts/ohr_kalibrieren.py) und schreibt models/ohr/ohr_prior.json,
  5. zeigt den Reife-Status (n, getragen, Stufe).

Sie hören in IHREM Player und tippen nur l / r / = (Enter für Wiederholung
des aktuellen Trials). Alles andere läuft automatisch. Kein Audio-Playback
im Skript — die Dateipfade werden angezeigt.

Paar-Layouts (automatisch erkannt, deterministisch sortiert):
  * <verzeichnis>/A/*.wav + <verzeichnis>/B/*.wav   (gleiche Dateinamen)
  * <verzeichnis>/*.a.wav + <verzeichnis>/*.b.wav   (gleicher Stammname)
  * <verzeichnis>/<paar>/{a,b}.(wav|flac)           (je Paar ein Unterordner)
  * sonst: --items <json>  (Format wie ab_harness prepare)

Aufruf (komplett ohne manuelle Paar-Erzeugung):
  python3 -B scripts/ohr_panel.py --erzeuge-paare --verzeichnis /pfad/zu/musik

Aufruf (fertige Paare):
  python3 -B scripts/ohr_panel.py --verzeichnis /pfad/zu/paaren
  python3 -B scripts/ohr_panel.py --verzeichnis /pfad --seed 7 --wiederholen 2

Determinismus (§G5 (copilot-instructions.md)): feste Sortierung + Seed;
kein time.time in der Entscheidungslogik. Abbruch (Ctrl+C/leere Eingabe)
kalibriert trotzdem mit den bisherigen Verdikten (best effort, §0c-Geist).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTEFAKT = ROOT / "models" / "ohr" / "ohr_prior.json"


def _audio_dateien(verzeichnis: Path) -> list[Path]:
    return sorted(
        [p for p in verzeichnis.rglob("*") if p.suffix.lower() in (".wav", ".flac") and p.is_file()],
        key=lambda p: str(p),
    )


def erkenne_paare(verzeichnis: Path) -> list[dict[str, str]]:
    """Erkennt A/B-Paare aus den unterstützten Layouts — deterministisch sortiert."""
    verzeichnis = Path(verzeichnis)
    if not verzeichnis.is_dir():
        raise FileNotFoundError(f"Verzeichnis fehlt: {verzeichnis}")
    a_dir = verzeichnis / "A"
    b_dir = verzeichnis / "B"
    if a_dir.is_dir() and b_dir.is_dir():
        a_files = {p.name: p for p in _audio_dateien(a_dir)}
        b_files = {p.name: p for p in _audio_dateien(b_dir)}
        gemeinsame = sorted(set(a_files) & set(b_files))
        return [{"id": n.rsplit(".", 1)[0], "a": str(a_files[n]), "b": str(b_files[n])} for n in gemeinsame]

    # Restauriert-Layout (echte Aurik-Paare, 2026-11-10):
    # <stamm>_restauriert*.wav ↔ <stamm>.wav bzw. <stamm>_ORIGINAL.wav
    audio = _audio_dateien(verzeichnis)
    restauriert: dict[str, Path] = {}
    for p in audio:
        if "_restauriert" in p.stem:
            basis = p.stem.split("_restauriert")[0]
            restauriert[basis] = p
    paare_restauriert: list[dict[str, str]] = []
    if restauriert:
        by_name = {p.stem: p for p in audio}
        for basis in sorted(restauriert):
            original = by_name.get(basis) or by_name.get(basis + "_ORIGINAL") or by_name.get(basis + "_original")
            if original:
                paare_restauriert.append({"id": basis, "a": str(original), "b": str(restauriert[basis])})

    # ORIGINAL-Varianten-Layout (2026-11-10): <stamm>_ORIGINAL ↔ <stamm>_<variante>
    # (z. B. Roxy Music ML-ADAPT-v2/v3 — die stärksten Restaurierungs-Vergleiche)
    paare_original: list[dict[str, str]] = []
    original_staemme: dict[str, Path] = {}
    for p in audio:
        if p.stem.endswith("_ORIGINAL"):
            original_staemme[p.stem.rsplit("_ORIGINAL", 1)[0]] = p
    for stamm in sorted(original_staemme):
        orig = original_staemme[stamm]
        kandidaten = [p for p in audio if p.stem.startswith(stamm + "_") and p != orig and "_restauriert" not in p.stem]
        for var in sorted(kandidaten, key=lambda p: p.stem):
            paare_original.append({"id": var.stem, "a": str(orig), "b": str(var)})

    if paare_restauriert or paare_original:
        return sorted(paare_restauriert + paare_original, key=lambda p: p["id"])

    # Suffix-Layout: <stamm>.a.wav / <stamm>.b.wav
    a_by_stamm: dict[str, Path] = {}
    b_by_stamm: dict[str, Path] = {}
    for p in audio:
        if p.name.endswith(".a" + p.suffix.lower()) or p.name.endswith("_a" + p.suffix.lower()):
            stamm = (
                p.name[: -len(".a" + p.suffix.lower())]
                if p.name.endswith(".a" + p.suffix.lower())
                else p.name[: -len("_a" + p.suffix.lower())]
            )
            a_by_stamm[stamm] = p
        elif p.name.endswith(".b" + p.suffix.lower()) or p.name.endswith("_b" + p.suffix.lower()):
            stamm = (
                p.name[: -len(".b" + p.suffix.lower())]
                if p.name.endswith(".b" + p.suffix.lower())
                else p.name[: -len("_b" + p.suffix.lower())]
            )
            b_by_stamm[stamm] = p
    if a_by_stamm and b_by_stamm:
        gemeinsame = sorted(set(a_by_stamm) & set(b_by_stamm))
        return [{"id": n, "a": str(a_by_stamm[n]), "b": str(b_by_stamm[n])} for n in gemeinsame]

    # Unterordner-Layout: <paar>/a.wav + <paar>/b.wav
    paare: list[dict[str, str]] = []
    for sub in sorted(verzeichnis.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue
        kandidaten = {p.stem.lower(): p for p in _audio_dateien(sub)}
        if "a" in kandidaten and "b" in kandidaten:
            paare.append({"id": sub.name, "a": str(kandidaten["a"]), "b": str(kandidaten["b"])})
    return paare


def _lade_48k(pfad: Path):
    """Lädt eine Audiodatei als Mono-float32 @ 48 kHz (scipy-Resampling)."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(pfad), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    x = np.asarray(data, dtype=np.float64)
    if sr != 48000:
        import math

        from scipy.signal import resample_poly

        gcd = math.gcd(sr, 48000)
        x = resample_poly(x, 48000 // gcd, sr // gcd)
    return x.astype(np.float32)


def defekt_variante(audio, variante: str, rng) -> np.ndarray:
    """Kontrollierte, deterministische Defekt-Variante (Stärke bewusst hörbar).

    Die Bibliothek deckt ALLE 7 Ohr-Feature-Dimensionen ab
    (ohr_prior.FEATURE_NAMEN): defektlast (rausch, brummen, kammfilter),
    roughness (kammfilter, wowflutter, verzerrung), sharpness (bandlimit,
    saettigung, verzerrung), loudness (lautheit, kompression), fluctuation
    (tremolo, wowflutter, kompression), sii (bandlimit), musik_mos (alle).
    """
    x = np.asarray(audio, dtype=np.float64)
    if variante == "rausch":
        ziel_snr_db = 24.0
        leistung = float(np.mean(x * x)) + 1e-12
        rausch_leistung = leistung / (10 ** (ziel_snr_db / 10.0))
        n = rng.standard_normal(x.size)
        return (x + np.sqrt(rausch_leistung) * n).astype(np.float32)
    if variante == "bandlimit":
        from scipy.signal import firwin, lfilter

        taps = firwin(65, 8000.0, fs=48000.0)
        y = lfilter(taps, 1.0, x)
        return y.astype(np.float32)
    if variante == "saettigung":
        return (np.tanh(3.0 * x) / np.tanh(3.0)).astype(np.float32)
    if variante == "kammfilter":
        # Phasen-Interferenz: Delay 5 ms, Feedback 0.7, Mix 50 % — typisch
        # für Mehrweg-Übertragungen (Kamm-Metrik des Sweet-Spots).
        delay = int(0.005 * 48000)
        out = x.copy()
        out[delay:] += 0.7 * x[:-delay]
        out = 0.5 * x + 0.5 * out
        out /= float(np.abs(out).max()) + 1e-9
        return out.astype(np.float32)
    if variante == "wowflutter":
        # Periodische Geschwindigkeitsschwankung (4 Hz, ±1 %) — Vinyl/Tape.
        from scipy.interpolate import interp1d

        n = x.size
        t = np.arange(n, dtype=np.float64) / 48000.0
        warp = t + 0.01 * np.sin(2 * np.pi * 4.0 * t) / (2 * np.pi * 4.0)
        warp = np.clip(warp, warp[0], warp[-1])
        f = interp1d(t, x, kind="linear", fill_value="extrapolate")
        return f(warp).astype(np.float32)
    if variante == "kompression":
        # Harte Dynamik-Kompression: Envelope-Follower, Threshold −20 dBFS,
        # Ratio 4:1, deterministisch.
        from scipy.signal import lfilter

        env = np.abs(x)
        b = [1.0]
        a = [1.0, -0.999]
        smooth = lfilter(b, a, env)
        threshold = 10 ** (-20.0 / 20)
        gain = np.ones_like(x)
        over = smooth > threshold
        gain[over] = (threshold + (smooth[over] - threshold) / 4.0) / smooth[over]
        return (x * gain).astype(np.float32)
    if variante == "brummen":
        # Netzbrumm 50 Hz + Harmonische bei −30 dB (tonale Defektlast).
        t = np.arange(x.size, dtype=np.float64) / 48000.0
        brumm = 0.0
        for harmonische, amp in ((1, 0.0316), (2, 0.0158), (3, 0.0079)):
            brumm += amp * np.sin(2 * np.pi * 50.0 * harmonische * t + float(rng.uniform(0, 2 * np.pi)))
        return (x + brumm).astype(np.float32)
    if variante == "tremolo":
        # Amplitudenmodulation 5 Hz, 30 % Tiefe — Fluktuationsstärke direkt.
        t = np.arange(x.size, dtype=np.float64) / 48000.0
        mod = 1.0 - 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 5.0 * t))
        return (x * mod).astype(np.float32)
    if variante == "verzerrung":
        # Kubische Nichtlinearität — Obertöne, Rauigkeit.
        y = x + 0.3 * x**3
        y /= float(np.abs(y).max()) + 1e-9
        return y.astype(np.float32)
    if variante == "lautheit":
        # Pegel −6 dB — kalibriert die Loudness-Dimension direkt.
        return (x * 0.5).astype(np.float32)
    raise ValueError(f"Unbekannte Variante: {variante}")


def _mittel_ausschnitt(audio, sekunden: float):
    """Deterministischer Mittel-Ausschnitt (§G5 (copilot-instructions.md)) —
    wie der StrategieDenker ihn für HPE/Goosebumps nutzt: 20 s aus der
    Song-Mitte genügen für ein Kalibrierungs-Urteil.
    """
    import numpy as np

    arr = np.asarray(audio)
    n = arr.shape[-1]
    want = int(round(sekunden * 48000))
    if n <= want or sekunden <= 0:
        return arr
    start = (n - want) // 2
    return arr[..., start : start + want]


def erzeuge_paare(
    verzeichnis: Path,
    ausgang: Path,
    varianten: list[str],
    seed: int,
    ausschnitt_s: float = 10.0,
) -> list[dict[str, str]]:
    """Erzeugt A/B-Paare aus einem Musik-Ordner: A = Original, B = Defekt-Variante.
    Deterministisch (Seed); Ausgabe als A/-B/-Ordnerlayout unter `ausgang`.
    `ausschnitt_s` > 0: nur der Mittel-Ausschnitt wird verglichen (Standard 10 s).
    """
    import numpy as np
    import soundfile as sf

    verzeichnis = Path(verzeichnis)
    ausgang = Path(ausgang)
    if not verzeichnis.is_dir():
        raise FileNotFoundError(f"Musik-Verzeichnis fehlt: {verzeichnis}")
    rng = np.random.default_rng(seed)
    (ausgang / "A").mkdir(parents=True, exist_ok=True)
    (ausgang / "B").mkdir(parents=True, exist_ok=True)
    paare: list[dict[str, str]] = []
    dateien = _audio_dateien(verzeichnis)
    for idx, pfad in enumerate(dateien, start=1):
        print(f"  ({idx}/{len(dateien)}) Erzeuge: {pfad.name} …", flush=True)
        original = _lade_48k(pfad)
        original = _mittel_ausschnitt(original, ausschnitt_s)
        for variante in varianten:
            b = defekt_variante(original, variante, rng)
            stamm = f"{pfad.stem}__{variante}"
            a_path = ausgang / "A" / f"{stamm}.wav"
            b_path = ausgang / "B" / f"{stamm}.wav"
            sf.write(str(a_path), original, 48000)
            sf.write(str(b_path), b, 48000)
            paare.append({"id": stamm, "a": str(a_path), "b": str(b_path)})
    return paare


def finde_player() -> str | None:
    """Erster verfügbarer CLI-Audio-Player (ffplay > aplay > paplay)."""
    import shutil

    for kandidat in ("ffplay", "aplay", "paplay"):
        if shutil.which(kandidat):
            return kandidat
    return None


def spiele_ab(pfad, player: str) -> bool:
    """Spielt eine WAV-Datei über den CLI-Player — wirft nie, liefert Erfolg."""
    if player not in ("ffplay", "aplay", "paplay"):
        print(f"  (Unbekannter Player: {player})")
        return False
    if not Path(pfad).is_file():
        print(f"  (Abspielen übersprungen — Datei fehlt: {pfad})")
        return False
    try:
        if player == "paplay":
            subprocess.run(["paplay", str(pfad)], check=True, capture_output=True)
        elif player == "aplay":
            subprocess.run(["aplay", "-q", str(pfad)], check=True, capture_output=True)
        else:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(pfad)],
                check=True,
                capture_output=True,
            )
        return True
    except Exception as exc:
        print(f"  (Abspielen fehlgeschlagen: {type(exc).__name__} — Datei: {pfad})")
        return False


def frage_verdikte(
    trials: list[dict],
    eingaben,
    *,
    player: str | None = None,
    basis: Path | None = None,
    spieler_fn=None,
) -> list[dict[str, str]]:
    """Interaktive A/B-Abfrage — `eingaben` ist ein Iterable von Antworten
    (in Produktion: stdin; in Tests: Liste).

    Mit `player`: Trials werden automatisch abgespielt (links, dann rechts);
    Enter = beide nochmal hören; q = Abbruch. Ohne player: Leere Eingabe = Abbruch
    (bisheriges Verhalten).
    """
    _spiel = spieler_fn or (lambda p: spiele_ab(p, player) if player else True)
    verdikte: list[dict[str, str]] = []
    for trial in trials:
        tid = trial["trial_id"]
        links = trial["stimuli"]["left"]
        rechts = trial["stimuli"]["right"]
        links_pfad = (basis / links) if basis else Path(links)
        rechts_pfad = (basis / rechts) if basis else Path(rechts)
        print(f"\n── Trial {tid} ──────────────────────────────")
        print(f"  links : {links}")
        print(f"  rechts: {rechts}")
        while True:
            if player:
                print("  ▶ links …")
                _spiel(links_pfad)
                print("  ▶ rechts …")
                _spiel(rechts_pfad)
            antwort_raw = next(eingaben, None)
            if antwort_raw is None:
                # EOF (gepipete Eingabe beendet) — interaktiv tritt das nicht auf.
                print("  (Eingabe beendet — bisherige Verdikte werden verwendet)")
                return verdikte
            antwort = str(antwort_raw).strip().lower()
            if not antwort:
                if player:
                    print("  (Enter = nochmal hören …)")
                    continue
                print("  (Abbruch — bisherige Verdikte werden verwendet)")
                return verdikte
            if antwort == "q" and player:
                print("  (Abbruch — bisherige Verdikte werden verwendet)")
                return verdikte
            if antwort in ("l", "r"):
                verdikte.append({"trial_id": tid, "preferred": "left" if antwort == "l" else "right"})
                break
            if antwort == "=":
                verdikte.append({"trial_id": tid, "preferred": "tie"})
                break
            print("  Bitte l (links), r (rechts) oder = (gleich) eingeben.")
    return verdikte


def run(cmd: list[str]) -> None:
    subprocess.run([sys.executable, "-B", *cmd], cwd=ROOT, check=True, capture_output=True)


def baue_session(verdikte: list[dict[str, str]], key: dict, sr_hint: int = 48000) -> dict:
    """Baut die Ohr-Urteils-Session (aurik.ohr_urteile/1) direkt im Panel —
    unabhängig vom ab_harness-Subkommando (dessen Datei teilt sich das Repo
    mit der parallelen Sitzung). Entblindet über den Schlüssel, übernimmt die
    ORIGINAL-Kandidatenpfade; Gleichstände fliegen raus (dokumentiert).
    """
    mapping: dict = key["mapping"]
    pairs: list[dict[str, str]] = []
    ties = 0
    for v in verdikte:
        tid = str(v["trial_id"])
        if tid not in mapping:
            raise KeyError(f"Unbekannte trial_id in Verdikten: {tid}")
        pref = str(v["preferred"]).lower()
        if pref not in ("left", "right", "tie"):
            raise ValueError(f"preferred muss left|right|tie sein: {v!r}")
        if pref == "tie":
            ties += 1
            continue
        winner = str(mapping[tid][pref])  # "A" | "B"
        paths = mapping[tid].get("paths")
        if not paths:
            raise KeyError(f"Trial {tid} ohne Originalpfade — Key zu alt, Panel neu starten")
        pairs.append({"a": str(paths["A"]), "b": str(paths["B"]), "choice": winner})
    return {
        "schema": "aurik.ohr_urteile/1",
        "sr_hint": sr_hint,
        "pairs": pairs,
        "gleichstaende_ausgeschlossen": ties,
        "quelle": "scripts/ohr_panel.py",
    }


def erstelle_trials(
    paare: list[dict[str, str]],
    work: Path,
    seed: int,
    wiederholen: int = 1,
    ausschnitt_s: float = 10.0,
) -> tuple[dict, dict]:
    """Verblindetes Versuchsblatt + Schlüssel DIREKT im Panel (autark — die
    ab_harness-Datei teilt sich das Repo mit der parallelen Sitzung).
    Deterministisch (Seed, §G5 (copilot-instructions.md)); der Schlüssel trägt
    die Originalpfade je Trial für die spätere Entblindung.

    JEDE Stimulus-Datei wird auf den deterministischen Mittel-Ausschnitt
    (ausschnitt_s) gekürzt — auch ERKANNTE Paare (restauriert/A/B-Ordner),
    damit nie ganze Songs abgespielt werden.
    """
    import random

    import soundfile as sf

    rng = random.Random(seed)
    stimuli_dir = work / "stimuli"
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict] = []
    key: dict[str, dict] = {}
    for item in paare:
        item_id = str(item["id"])
        for rep in range(wiederholen):
            trial_id = f"{item_id}__r{rep}"
            left_is_a = rng.random() < 0.5
            mapping = {"left": "a" if left_is_a else "b", "right": "b" if left_is_a else "a"}
            slots: dict[str, str] = {}
            for slot in ("left", "right"):
                src = Path(item[mapping[slot]])
                dst = stimuli_dir / f"{trial_id}_{slot}.wav"
                # Mittel-Ausschnitt statt Ganztrack (§G5 (copilot-instructions.md))
                audio = _mittel_ausschnitt(_lade_48k(src), ausschnitt_s)
                sf.write(str(dst), audio, 48000)
                slots[slot] = str(dst.relative_to(work))
            trials.append(
                {
                    "trial_id": trial_id,
                    "item_id": item_id,
                    "repeat": rep,
                    "stimuli": slots,
                }
            )
            key[trial_id] = {
                "left": mapping["left"].upper(),
                "right": mapping["right"].upper(),
                "paths": {"A": str(item["a"]), "B": str(item["b"])},
            }
    return {"schema": "aurik.ab_trials/1", "seed": seed, "trials": trials}, {
        "schema": "aurik.ab_key/1",
        "mapping": key,
    }


def erfrage_verzeichnis(eingaben) -> Path:
    """Fragt interaktiv nach dem Musik-/Paar-Ordner; Enter nimmt den ersten
    vorhandenen Vorschlag (~/Musik, ~/Music, ~/Schreibtisch/Musik).
    """
    vorschlaege = [
        p for p in (Path.home() / "Musik", Path.home() / "Music", Path.home() / "Schreibtisch" / "Musik") if p.is_dir()
    ]
    vorschlag = vorschlaege[0] if vorschlaege else None
    while True:
        if vorschlag:
            print(f"Ordner [Enter = {vorschlag}]: ", end="", flush=True)
        else:
            print("Ordner: ", end="", flush=True)
        antwort = next(eingaben, None)
        if antwort is None:
            if vorschlag:
                return vorschlag
            raise KeyboardInterrupt("kein Ordner angegeben")
        antwort = str(antwort).strip()
        if not antwort and vorschlag:
            return vorschlag
        kandidat = Path(antwort).expanduser()
        if kandidat.is_dir():
            return kandidat
        print(f"  '{antwort}' ist kein lesbares Verzeichnis — bitte nochmal.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ohr-Panel — Ein-Kommando-Kalibrierung (§7, M6)")
    ap.add_argument(
        "--verzeichnis",
        default=None,
        help="Musik-Ordner (mit --erzeuge-paare) oder Ordner mit fertigen A/B-Paaren (sonst interaktiv erfragt)",
    )
    ap.add_argument(
        "--erzeuge-paare", action="store_true", help="Paare selbst erzeugen: Original vs. Defekt-Variante je Musikstück"
    )
    ap.add_argument(
        "--varianten",
        default="rausch,bandlimit,saettigung",
        help="Defekt-Varianten (Komma-getrennt: rausch, bandlimit, saettigung)",
    )
    ap.add_argument(
        "--ausschnitt-sekunden",
        type=float,
        default=10.0,
        help="Nur den Mittel-Ausschnitt dieser Länge vergleichen (0 = ganzer Song)",
    )
    ap.add_argument(
        "--paar-ausgang", default=None, help="Ausgabe-Ordner der erzeugten Paare (Default: <arbeits-verzeichnis>/paare)"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wiederholen", type=int, default=1, help="Wiederholungen je Paar (verblindet)")
    ap.add_argument("--pseudo-count", type=float, default=10.0)
    ap.add_argument("--arbeits-verzeichnis", default=".ohr_panel", help="Arbeits-Ordner (Blatt, Key, Session)")
    ap.add_argument(
        "--kein-player", action="store_true", help="Automatisches Abspielen deaktivieren (Pfade nur anzeigen)"
    )
    args = ap.parse_args(argv)

    # Absolut auflösen: das Panel läuft von beliebigen Verzeichnissen aus, die
    # Subprozesse (ab_harness/ohr_kalibrieren) aber mit cwd=ROOT — relative
    # Pfade müssten sonst doppelt interpretiert werden.
    arbeits_dir = Path(args.arbeits_verzeichnis).resolve()
    ausgang_default = Path(args.paar_ausgang).resolve() if args.paar_ausgang else arbeits_dir / "paare"

    verzeichnis = Path(args.verzeichnis).expanduser() if args.verzeichnis else None
    if verzeichnis is None or not verzeichnis.is_dir():
        if verzeichnis is not None:
            print(f"'{verzeichnis}' ist kein lesbares Verzeichnis.")
        verzeichnis = erfrage_verzeichnis(eingaben=iter(sys.stdin.readline, ""))

    if args.erzeuge_paare:
        varianten = [v.strip().lower() for v in args.varianten.split(",") if v.strip()]
        paare = erzeuge_paare(verzeichnis, ausgang_default, varianten, args.seed, ausschnitt_s=args.ausschnitt_sekunden)
        if not paare:
            print(f"Keine Audiodateien (.wav/.flac) in {verzeichnis} gefunden.")
            return 2
        print(f"Paare erzeugt: {len(paare)} ({', '.join(varianten)} je Stück) → {ausgang_default}")
    else:
        paare = erkenne_paare(verzeichnis)
        if paare:
            print(f"Gefunden: {len(paare)} Paar(e) × {args.wiederholen} Wiederholung(en)")
        else:
            # Komfort: keine fertigen Paare → automatisch aus der Musik erzeugen.
            varianten = [v.strip().lower() for v in args.varianten.split(",") if v.strip()]
            try:
                paare = erzeuge_paare(
                    verzeichnis, ausgang_default, varianten, args.seed, ausschnitt_s=args.ausschnitt_sekunden
                )
            except FileNotFoundError:
                paare = []
            if paare:
                print(
                    f"Keine fertigen A/B-Paare — {len(paare)} Paar(e) automatisch aus der Musik erzeugt "
                    f"({', '.join(varianten)} je Stück) → {ausgang_default}"
                )
            else:
                print("Keine A/B-Paare und keine Audiodateien (.wav/.flac) gefunden.")
                print("Layouts für fertige Paare: A/-B/-Ordner, <name>.a/.b.wav, oder <paar>/{a,b}.wav")
                return 2

    work = arbeits_dir
    items = work / "items.json"
    items.parent.mkdir(parents=True, exist_ok=True)
    items.write_text(json.dumps(paare, indent=2, ensure_ascii=False), encoding="utf-8")

    # Verblindung DIREKT im Panel (autark, §G5 (copilot-instructions.md)).
    sheet_doc, key_doc = erstelle_trials(
        paare, work, args.seed, wiederholen=args.wiederholen, ausschnitt_s=args.ausschnitt_sekunden
    )
    (work / "trials.json").write_text(json.dumps(sheet_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    (work / "key.json").write_text(json.dumps(key_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    sheet = sheet_doc
    player = None if args.kein_player else finde_player()
    if player:
        print(f"\nAudio-Player: {player} — Trials werden automatisch abgespielt (links → rechts).")
        print("Antwort: l / r / = — Enter = nochmal hören, q = Abbruch.")
    else:
        print("\nKein CLI-Audio-Player gefunden (paplay/aplay/ffplay) — bitte die Dateien im eigenen Player öffnen.")
    verdikte = frage_verdikte(
        sheet["trials"],
        eingaben=iter(sys.stdin.readline, ""),
        player=player,
        basis=work,
    )
    if not verdikte:
        print("Keine Verdikte — kein Artefakt geschrieben.")
        return 1

    verdikte_path = work / "verdikte.json"
    verdikte_path.write_text(json.dumps(verdikte, indent=2, ensure_ascii=False), encoding="utf-8")
    session = work / "ohr_urteile.json"
    # Session-Baunng direkt im Panel (robust gegen parallele Repo-Schreibvorgänge).
    key_doc = json.loads((work / "key.json").read_text(encoding="utf-8"))
    session_doc = baue_session(verdikte, key_doc, sr_hint=48000)
    session.write_text(json.dumps(session_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Ohr-Urteils-Session: {len(session_doc['pairs'])} Paare "
        f"({session_doc['gleichstaende_ausgeschlossen']} Gleichstände ausgeschlossen) → {session}"
    )
    if not session_doc["pairs"]:
        print("Keine entscheidenden Urteile (nur Gleichstände) — das bestehende Ohr-Artefakt bleibt unverändert.")
        return 1
    run(
        [
            "scripts/ohr_kalibrieren.py",
            "--urteile",
            str(session),
            "--out",
            str(ARTEFAKT),
            "--seed",
            str(args.seed),
            "--pseudo-count",
            str(args.pseudo_count),
        ]
    )

    from backend.core.dsp.ohr_brief import erstelle_ohr_brief

    brief = erstelle_ohr_brief(artefakt=ARTEFAKT)
    stufen = {
        0: "neutral (kein Artefakt)",
        1: "Priorisierung aktiv",
        2: "Zonen-Kalibrierung (±5 %) aktiv",
        3: "verstärkte Kalibrierung (±8 %) aktiv",
    }
    print("\n── Ohr-Status ──────────────────────────────")
    print(f"  Urteile : {brief.ohr_prior.n_labels if brief.ohr_prior else 0}")
    print(f"  getragen: {brief.getragen:.2f}")
    print(f"  Stufe   : {brief.ohr_stufe()} — {stufen[brief.ohr_stufe()]}")
    print(f"  Artefakt: {ARTEFAKT}")
    print("  Ab jetzt personalisiert Aurik die Restaurierung nach diesem Ohr.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
