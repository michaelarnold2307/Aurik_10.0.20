#!/usr/bin/env python3
"""hoerpanel_player — der Hör-Player für die Golden-Ear-Schwellenmessung.

Der einzige Mensch-Schritt der Wohlklang-Roadmap: Hörproben bewerten, ohne
Dateien auszufüllen. Der Player lädt eine Schwellen-Studie (gebaut von
``scripts/mushra_harness.py thresholds-build``), zeigt je Trial zwei
Intervalle A/B (2AFC, doppelblind) und zeichnet die Antworten als
``answers.csv`` auf — direkt weiterverwertbar von
``python scripts/mushra_harness.py thresholds-fit``.

Doppelblind (wie die MUSHRA-Sessions): die Lösung (``trial_key.json``)
verlässt NIEMALS den Server. Reihenfolge je Hörer deterministisch
(permutiert, Seed aus meta.json). Abbruch-/Wiederaufnahme-fähig: bereits
beantwortete Trials werden übersprungen, die CSV wird atomar geschrieben.

Abgrenzung (Vorhandensein-Prüfung 2026-10-03): ``Aurik10/ui/audio_player.py``
ist der Qt-Stream-Player der App (Quellwechsel Original↔Restauriert) — kein
Hörtest-Player. Ein Browser-Player war bislang nicht vorhanden und ist für
Panel-Teilnehmer ohne App-Installation die professionellere Form.

Lokal starten:
  python scripts/hoerpanel_player.py            # nimmt die neueste Studie
  python scripts/hoerpanel_player.py --study output_audio/mushra/thresholds_… --port 8765

Nur stdlib (§Runtime-Sparsamkeit), deterministisch (§G5 (GEBOTE.md)),
fail-closed bei fehlender Studie (§V6 (VERBOTEN.md)).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "output_audio" / "mushra"


# --------------------------------------------------------------------------
# Studien-Modell
# --------------------------------------------------------------------------
@dataclass
class Study:
    """Eine Schwellen-Studie: Trials + Lösung (serverseitig, doppelblind)."""

    study_dir: Path
    meta: dict
    key: dict[str, dict]  # trial_id → {defective_interval, defect_class, …}
    trials_dir: Path


def load_study(study_dir: Path) -> Study:
    """Lädt Studie (meta.json + trial_key.json + trials/); fail-closed (§V6 (VERBOTEN.md))."""
    study_dir = Path(study_dir)
    meta_path = study_dir / "meta.json"
    key_path = study_dir / "trial_key.json"
    trials_dir = study_dir / "trials"
    if not meta_path.is_file() or not key_path.is_file() or not trials_dir.is_dir():
        raise FileNotFoundError(f"Keine vollständige Schwellen-Studie in {study_dir} (meta/trial_key/trials)")
    key = {row["trial_id"]: row for row in json.loads(key_path.read_text(encoding="utf-8"))}
    return Study(
        study_dir=study_dir, meta=json.loads(meta_path.read_text(encoding="utf-8")), key=key, trials_dir=trials_dir
    )


def latest_study(root: Path = OUT_ROOT) -> Path | None:
    """Neueste ``thresholds_*``-Studie (None wenn keine existiert)."""
    candidates = sorted(p for p in Path(root).glob("thresholds_*") if (p / "trial_key.json").is_file())
    return candidates[-1] if candidates else None


def listener_order(study: Study, listener: str) -> list[str]:
    """Deterministische Trial-Reihenfolge je Hörer (Permutation, nie die Lösung)."""
    seed = (int(study.meta.get("seed", 0)) * 1_000_003 + sum(ord(c) for c in listener)) % (2**32)
    ids = sorted(study.key)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


def session_payload(study: Study, listener: str, store: AnswerStore) -> dict:
    """Alles, was der Client sehen darf — ausdrücklich OHNE Ground Truth."""
    order = listener_order(study, listener)
    answered = store.answered_for(listener)
    remaining = [t for t in order if t not in answered]
    return {
        "listener": listener,
        "order": remaining,
        "total": len(order),
        "answered": len(answered),
        "build": str(study.meta.get("build", "")),
        # Aufgabenart (Wohlklang-Vertrag, Beleg 4): "defect" = Defekt-Schwelle,
        # "preference" = blindes A/B „welches klingt besser?“. Die Lösung
        # (defective_interval) bleibt in trial_key.json und geht NIE zum Client.
        "task": str(study.meta.get("task", "defect")),
        "question": str(study.meta.get("question", "")),
    }


# --------------------------------------------------------------------------
# Antworten (CSV-kompatibel zu mushra_harness.thresholds-fit)
# --------------------------------------------------------------------------
CSV_HEADER = ["listener", "trial_id", "answer_interval"]


@dataclass
class AnswerStore:
    """Antworten-Speicher: atomare CSV-Schreibweise, wiederaufnahmefähig."""

    csv_path: Path
    rows: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, study_dir: Path) -> AnswerStore:
        store = cls(csv_path=Path(study_dir) / "answers.csv")
        if store.csv_path.is_file():
            with open(store.csv_path, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("listener") and row.get("trial_id") and row.get("answer_interval") in ("A", "B"):
                        store.rows.append({k: row[k] for k in CSV_HEADER})
        return store

    def answered_for(self, listener: str) -> set[str]:
        return {r["trial_id"] for r in self.rows if r["listener"] == listener}

    def record(self, listener: str, trial_id: str, answer_interval: str) -> bool:
        """Zeichnet eine Antwort auf (False bei Duplikat/Unbekanntem Trial/„gefälschter“ Antwort)."""
        answer = str(answer_interval).strip().upper()
        if answer not in ("A", "B") or trial_id in self.answered_for(listener):
            return False
        self.rows.append({"listener": listener, "trial_id": trial_id, "answer_interval": answer})
        self._save()
        return True

    def _save(self) -> None:
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.csv_path)


# --------------------------------------------------------------------------
# HTTP-Server (stdlib)
# --------------------------------------------------------------------------
def make_handler(study: Study, store: AnswerStore):
    """Handler-Fabrik: doppelblind — trial_key.json verlässt den Server nie."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # leise, deterministisch
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE.encode("utf-8"))
                return
            if url.path == "/api/session":
                qs = parse_qs(url.query)
                listener = (qs.get("listener") or ["L01"])[0][:32]
                self._json(200, session_payload(study, listener, store))
                return
            if url.path.startswith("/trials/"):
                name = Path(unquote(url.path[len("/trials/") :])).name  # nur Dateiname, kein Pfadaufstieg
                target = (study.trials_dir / name).resolve()
                if target.parent != study.trials_dir.resolve() or not target.is_file():
                    self._send(404, b"nicht gefunden")
                    return
                self._send(200, target.read_bytes(), "audio/wav")
                return
            self._send(404, b"nicht gefunden")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/answer":
                self._send(404, b"nicht gefunden")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, OSError):
                self._json(400, {"ok": False, "error": "ungueltige Anfrage"})
                return
            listener = str(payload.get("listener", ""))[:32]
            trial_id = str(payload.get("trial_id", ""))
            answer = str(payload.get("answer_interval", ""))
            if trial_id not in study.key or not listener:
                self._json(400, {"ok": False, "error": "unbekanntes Trial oder Hoerer"})
                return
            ok = store.record(listener, trial_id, answer)
            self._json(200 if ok else 409, {"ok": ok, "answered": len(store.answered_for(listener))})

    return Handler


def serve(study_dir: Path, port: int = 8765) -> ThreadingHTTPServer:
    """Startet den Player-Server; gibt den Server zurück (für Tests auch port=0)."""
    study = load_study(study_dir)
    store = AnswerStore.load(study.study_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(study, store))
    return server


def main() -> int:
    ap = argparse.ArgumentParser(description="Hör-Player für die Golden-Ear-Schwellenmessung (2AFC, doppelblind)")
    ap.add_argument("--study", default=None, help="Studien-Ordner (thresholds_*); Default: neueste")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    study_dir = Path(args.study) if args.study else latest_study()
    if study_dir is None:
        print(
            "Keine Schwellen-Studie gefunden — zuerst:",
            "python scripts/mushra_harness.py thresholds-build",
            file=sys.stderr,
        )
        return 2
    try:
        server = serve(study_dir, args.port)
    except FileNotFoundError as exc:
        print(f"Studie nicht ladbar: {exc}", file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    print("=== Aurik Hör-Panel (Golden-Ear Schwellenmessung) ===")
    print(f"Studie: {study_dir}")
    display_host = host.decode() if isinstance(host, bytes) else host
    print(f"Player:  http://{display_host}:{port}/")
    print("Antworten landen automatisch in answers.csv — Strg+C beendet.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBeendet — Antworten gespeichert.")
    finally:
        server.server_close()
    return 0


# --------------------------------------------------------------------------
# UI (deutsch, barrierearm, große Ziele, keine Animation)
# --------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aurik Hör-Panel</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background:#14161a; color:#eef1f5;
         margin:0; display:flex; justify-content:center; }
  .wrap { width:min(760px, 94vw); padding:28px 16px 48px; }
  h1 { font-size:22px; letter-spacing:.2px; margin:0 0 6px; }
  .sub { color:#9aa4b2; font-size:14px; margin-bottom:22px; }
  .card { background:#1d2027; border:1px solid #2a2f38; border-radius:12px; padding:22px; margin-bottom:16px; }
  button { font-size:17px; padding:14px 22px; border-radius:10px; border:1px solid #3a4150;
           background:#262b34; color:#eef1f5; cursor:pointer; min-width:120px; }
  button:hover { background:#303846; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.choose { background:#2e4d8f; border-color:#3c63b8; font-weight:600; }
  button.choose:hover { background:#38599f; }
  input[type=text] { font-size:17px; padding:12px; border-radius:8px; border:1px solid #3a4150;
                     background:#14161a; color:#eef1f5; width:200px; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin:10px 0; }
  .prog { height:8px; background:#2a2f38; border-radius:4px; overflow:hidden; margin:14px 0 20px; }
  .prog i { display:block; height:100%; background:#4c8bf5; }
  .hint { color:#9aa4b2; font-size:13px; margin-top:10px; }
  .warn { color:#e8c268; font-size:13px; }
  .ok { color:#6fce8f; }
  label { font-size:15px; }
  .iv { flex:1; text-align:center; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🎧 Aurik Hör-Panel</h1>
  <div class="sub">Golden-Ear Schwellenmessung — bitte am Kopfhörer, in ruhiger Umgebung. Lautstärke einmal einstellen und dann nicht mehr verändern.</div>

  <div class="card" id="start">
    <div class="row">
      <label>Hörer-Kürzel:</label>
      <input type="text" id="listener" value="L01" maxlength="32">
    </div>
    <div class="row">
      <label><input type="checkbox" id="phones"> Ich trage Kopfhörer und bin in ruhiger Umgebung</label>
    </div>
    <button class="choose" id="btnStart" disabled>Start</button>
    <div class="hint" id="startHint">Sie hören je Runde zwei Ausschnitte (A und B). In genau einem ist ein Defekt versteckt — klicken Sie, in welchem Sie ihn hören. Es gibt kein Richtig oder Falsch im Tonfall, nur Ihr Gehör.</div>
    <div class="hint" id="startQuestion" style="display:none"></div>
  </div>

  <div class="card" id="trial" style="display:none">
    <div class="sub" id="progressText">Trial 1 von 1</div>
    <div class="prog"><i id="progBar" style="width:0%"></i></div>
    <div class="row">
      <div class="iv"><button id="playA">▶ A hören</button></div>
      <div class="iv"><button id="playB">▶ B hören</button></div>
    </div>
    <div class="row" style="margin-top:22px">
      <div class="iv"><button class="choose" id="pickA">Defekt ist in A</button></div>
      <div class="iv"><button class="choose" id="pickB">Defekt ist in B</button></div>
    </div>
    <div class="hint" id="trialHint">Tastatur: <b>A</b>/<b>B</b> = abspielen, <b>1</b>/<b>2</b> = wählen. Beide Ausschnitte dürfen beliebig oft wiederholt werden.</div>
    <div class="warn" id="listenHint" style="display:none">Bitte hören Sie sich beide Ausschnitte an, bevor Sie wählen.</div>
  </div>

  <div class="card" id="done" style="display:none">
    <div class="ok"><b>Vielen Dank!</b></div>
    <div class="hint" id="doneText"></div>
  </div>
</div>
<script>
(function () {
  let listener = "L01", order = [], idx = 0, played = {}, total = 0;
  const $ = (id) => document.getElementById(id);
  const audio = new Audio();
  let current = null;

  function play(iv) {
    current = iv;
    audio.src = "/trials/" + encodeURIComponent(order[idx] + "__" + iv + ".wav");
    audio.play();
    played[order[idx] + iv] = true;
    $("listenHint").style.display = "none";
  }
  function bothPlayed() {
    return played[order[idx] + "A"] && played[order[idx] + "B"];
  }
  function render() {
    $("progressText").textContent = "Trial " + (idx + 1) + " von " + order.length;
    $("progBar").style.width = (100 * idx / Math.max(order.length, 1)) + "%";
  }
  function next() {
    idx += 1;
    if (idx >= order.length) {
      $("trial").style.display = "none";
      $("done").style.display = "block";
      $("doneText").textContent = "Ihre Antworten wurden automatisch gespeichert (answers.csv, " +
        total + " Trials). Sie können das Fenster schließen.";
      return;
    }
    render();
  }
  function pick(iv) {
    if (!bothPlayed()) { $("listenHint").style.display = "block"; return; }
    fetch("/api/answer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ listener: listener, trial_id: order[idx], answer_interval: iv })
    }).then(function () { total += 1; next(); });
  }

  $("phones").addEventListener("change", function (e) { $("btnStart").disabled = !e.target.checked; });
  $("btnStart").addEventListener("click", function () {
    listener = ($("listener").value || "L01").trim().slice(0, 32);
    fetch("/api/session?listener=" + encodeURIComponent(listener))
      .then(function (r) { return r.json(); })
      .then(function (s) {
        order = s.order; total = s.answered || 0;
        if (s.task === "preference") {
          $("pickA").textContent = "A klingt besser";
          $("pickB").textContent = "B klingt besser";
          $("startHint").textContent = "Sie hören je Runde zwei Ausschnitte (A und B) desselben Musikabschnitts. " +
            "Klicken Sie den, der Ihnen natürlicher bzw. angenehmer erscheint. Verlassen Sie sich nur auf Ihr Gehör.";
          if (s.question) {
            $("startQuestion").textContent = "Ihre Aufgabe: " + s.question;
            $("startQuestion").style.display = "block";
          }
        } else if (s.question) {
          $("startQuestion").textContent = "Ihre Aufgabe: " + s.question;
          $("startQuestion").style.display = "block";
        }
        if (!order.length) {
          $("start").style.display = "none";
          $("done").style.display = "block";
          $("doneText").textContent = "Für " + listener + " sind bereits alle Trials beantwortet. Vielen Dank!";
          return;
        }
        $("start").style.display = "none";
        $("trial").style.display = "block";
        render();
      });
  });
  $("playA").addEventListener("click", function () { play("A"); });
  $("playB").addEventListener("click", function () { play("B"); });
  $("pickA").addEventListener("click", function () { pick("A"); });
  $("pickB").addEventListener("click", function () { pick("B"); });
  document.addEventListener("keydown", function (e) {
    if ($("trial").style.display === "none") return;
    const k = e.key.toLowerCase();
    if (k === "a") play("A");
    else if (k === "b") play("B");
    else if (k === "1") pick("A");
    else if (k === "2") pick("B");
  });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
