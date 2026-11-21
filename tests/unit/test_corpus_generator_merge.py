"""§WP-7 (2026-10-09): Corpus-Generator — Merge-Manifeste + Slug-Namensschema.

Befund vor dem Fix: `write_manifests` rief `build_manifest(material, files)`
OHNE Source-Metadaten auf und SCHRIEB das Manifest komplett neu — ein Akquise-
Lauf hätte die bestehenden Einträge (synthetisch + bereits verifizierte Reals)
aus dem Manifest gelöscht und die neuen Einträge ohne Lizenz/Attribution/URL
abgelegt. Fix (diese Datei pinnt ihn):

  - MERGE: Bestands-Einträge bleiben erhalten (Dedupe per `file`), neue werden
    angehängt; `description`/`corpus_version` des Bestands bleiben.
  - Nicht lesbares Bestands-Manifest ⇒ Schreiben wird übersprungen (kein stiller
    Datenverlust, §V6 copilot-instructions.md).
  - Source-Metadaten (license/source_url/source_attribution/era_year/genre)
    fließen in die neuen Einträge.
  - Kanonisches Dateinamen-Schema `<material>_<era>_<slug>.<ext>` (`_slugify`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from scripts import generate_corpus_from_public_domain as gen


class _FakeHTTPResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None, chunks: tuple[bytes, ...] = ()):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"unexpected HTTP status {self.status_code}")

    def iter_content(self, chunk_size: int):
        del chunk_size
        return iter(self._chunks)


class _FakeHTTPSession:
    def __init__(self, responses: list[_FakeHTTPResponse]):
        self.responses = responses
        self.calls: list[str] = []
        self.trust_env = True
        self.proxies = {"http": "http://proxy.test"}
        self.adapters: dict[str, object] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get(self, url: str, **_kwargs):
        self.calls.append(url)
        return self.responses.pop(0)

    def mount(self, prefix: str, adapter) -> None:
        self.adapters[prefix] = adapter


def _wav(path: Path, dur_s: float = 0.5) -> None:
    sr = 48000
    sf.write(path, np.zeros(int(sr * dur_s), dtype=np.float32), sr)


def test_slugify_kanonisches_schema() -> None:
    assert gen._slugify("Enrico%20Caruso%2C%20Rigoletto%3A%20La%20donna") == "enrico_caruso_rigoletto_la_donna"
    assert gen._slugify("78_dirty-hands-dirty-face") == "78_dirty_hands_dirty_face"
    assert gen._slugify("---") == "import"


def test_write_manifests_merged_und_metadaten(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gen, "CORPUS_ROOT", tmp_path)
    mat = tmp_path / "vinyl"
    (mat / "damaged").mkdir(parents=True)
    (mat / "manifest.yaml").write_text(
        yaml.dump(
            {
                "corpus_version": "1.0.0",
                "material": "vinyl",
                "description": "BESTAND",
                "entries": [
                    {
                        "file": "damaged/alt.wav",
                        "duration_s": 0.5,
                        "sample_rate": 48000,
                        "material": "vinyl",
                        "era_year": 1965,
                        "genre": "rock",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    newf = mat / "damaged" / "vinyl_1920_neu.wav"
    _wav(newf)
    written = gen.write_manifests(
        {
            "vinyl": [
                (
                    newf,
                    {
                        "url": "https://archive.org/x/neu.mp3",
                        "_dest_name": "vinyl_1920_neu.wav",
                        "era_year": 1920,
                        "license": "Public Domain",
                        "attribution": "Test (IA)",
                    },
                )
            ]
        }
    )
    assert written
    data = yaml.safe_load((mat / "manifest.yaml").read_text(encoding="utf-8"))
    files = [e["file"] for e in data["entries"]]
    assert "damaged/alt.wav" in files, "Bestands-Eintrag wurde gelöscht (§WP-7-Merge verletzt)"
    assert "damaged/vinyl_1920_neu.wav" in files
    assert data["description"] == "BESTAND"
    neu = next(e for e in data["entries"] if e["file"] == "damaged/vinyl_1920_neu.wav")
    assert neu["license"] == "Public Domain"
    assert neu["source_url"].startswith("https://archive.org/")
    assert neu["era_year"] == 1920


def test_write_manifests_ueberspringt_unlesbaren_bestand(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gen, "CORPUS_ROOT", tmp_path)
    mat = tmp_path / "vinyl"
    (mat / "damaged").mkdir(parents=True)
    manifest = mat / "manifest.yaml"
    manifest.write_bytes(b"\x00\x01 kaputt \xff")
    newf = mat / "damaged" / "vinyl_1920_neu.wav"
    _wav(newf)
    written = gen.write_manifests({"vinyl": [(newf, {"_dest_name": "vinyl_1920_neu.wav", "url": "u"})]})
    assert written == [], "Bei unlesbarem Bestand darf NICHTS geschrieben werden (§V6 copilot-instructions.md)"
    assert manifest.read_bytes() == b"\x00\x01 kaputt \xff", "Bestands-Manifest wurde überschrieben"


def test_dedupe_gleiche_datei_wird_nicht_doppelt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gen, "CORPUS_ROOT", tmp_path)
    mat = tmp_path / "vinyl"
    (mat / "damaged").mkdir(parents=True)
    newf = mat / "damaged" / "vinyl_1920_neu.wav"
    _wav(newf)
    pair = (newf, {"_dest_name": "vinyl_1920_neu.wav", "url": "u"})
    gen.write_manifests({"vinyl": [pair]})
    gen.write_manifests({"vinyl": [pair]})
    data = yaml.safe_load((mat / "manifest.yaml").read_text(encoding="utf-8"))
    assert len([e for e in data["entries"] if e["file"] == "damaged/vinyl_1920_neu.wav"]) == 1


def test_license_gate_pd_cc0_allowlist() -> None:
    """§WP-7-Lizenzvertrag (corpus/README.md): nur PD oder CC0."""
    assert gen._license_allowed("Public Domain")
    assert gen._license_allowed("Public Domain Mark 1.0")
    assert gen._license_allowed("CC0")
    assert gen._license_allowed("cc0 1.0")
    assert not gen._license_allowed("CC BY-NC 4.0")
    assert not gen._license_allowed("CC BY-SA 3.0")
    assert not gen._license_allowed("")
    assert not gen._license_allowed(None)


def test_user_url_download_streams_to_file_after_public_address_check(tmp_path: Path, monkeypatch) -> None:
    import requests
    import urllib3.util.connection

    session = _FakeHTTPSession([_FakeHTTPResponse(200, chunks=(b"audio", b"data"))])
    monkeypatch.setattr(requests, "Session", lambda: session)
    connected_addresses = []

    def _fake_create_connection(address, *_args, **_kwargs):
        connected_addresses.append(address)
        return object()

    monkeypatch.setattr(urllib3.util.connection, "create_connection", _fake_create_connection)
    monkeypatch.setattr(
        gen.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(gen.socket.AF_INET, gen.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
    )
    dest = tmp_path / "download.wav"

    assert gen.download_user_url_file("https://example.test/download.wav", dest)
    assert dest.read_bytes() == b"audiodata"
    assert session.calls == ["https://example.test/download.wav"]
    assert session.trust_env is False
    assert session.proxies == {}
    from typing import Any

    adapter: Any = session.adapters["https://"]
    assert adapter.pinned_addresses == ("93.184.216.34",)

    request = requests.Request("GET", "https://example.test/download.wav").prepare()
    pool = adapter.get_connection_with_tls_context(request, verify=True)
    connection = pool.ConnectionCls(host="example.test", port=443, timeout=1)
    assert connection.host == "example.test"
    connection._new_conn()
    assert connected_addresses == [("93.184.216.34", 443)]


def test_user_url_download_rejects_private_redirect(tmp_path: Path, monkeypatch) -> None:
    import requests

    session = _FakeHTTPSession([_FakeHTTPResponse(302, {"Location": "http://127.0.0.1/private"})])
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(
        gen.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(gen.socket.AF_INET, gen.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
    )

    assert not gen.download_user_url_file("http://example.test/start", tmp_path / "download.wav")
    assert session.calls == ["http://example.test/start"]
    assert not (tmp_path / "download.wav").exists()


def test_generate_ueberspringt_nicht_pd_lizenz(tmp_path: Path, monkeypatch, caplog) -> None:
    """CC/unklare Lizenzquellen werden sichtbar übersprungen (§V6 copilot-instructions.md)."""
    monkeypatch.setattr(gen, "CORPUS_ROOT", tmp_path)
    (tmp_path / "vinyl" / "damaged").mkdir(parents=True)
    fake = {
        "vinyl": [
            {"url": "https://example.org/a.mp3", "era_year": 1960, "license": "CC BY-NC 4.0"},
            {"url": "https://example.org/b.mp3", "era_year": 1960, "license": "Public Domain", "slug": "b"},
            {"url": "https://example.org/c.mp3", "era_year": 1960, "license": "CC0"},
        ]
    }
    monkeypatch.setattr(gen, "PUBLIC_DOMAIN_SOURCES", fake)
    with caplog.at_level(logging.WARNING):
        result = gen.generate_from_sources(material="vinyl", count=5, dry_run=True)
    names = [p.name for p, _m in result["vinyl"]]
    assert "vinyl_1960_b.mp3" in names
    assert "vinyl_1960_c.mp3" in names
    assert all("vinyl_1960_a" not in n for n in names), "CC-BY-NC-Quelle wurde NICHT übersprungen"
    assert any("nicht PD/CC0" in r.message for r in caplog.records), (
        "Überspringen ohne Warnung (§V6 copilot-instructions.md verletzt)"
    )


def test_check_coverage_zaehlt_realtraeger(tmp_path: Path, monkeypatch) -> None:
    """Real-Träger = Eintrag mit source_url; synthetische zählen nicht (D-K1-2)."""
    monkeypatch.setattr(gen, "CORPUS_ROOT", tmp_path)
    mat = tmp_path / "vinyl"
    mat.mkdir()
    (mat / "manifest.yaml").write_text(
        yaml.dump(
            {
                "corpus_version": "1.0.0",
                "material": "vinyl",
                "entries": [
                    {"file": "a.wav", "source_url": "https://x/a"},
                    {"file": "b.wav"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cov = gen.check_coverage()
    assert cov["per_material"]["vinyl"] == 2
    assert cov["per_material_real"]["vinyl"] == 1
    assert cov["wp7_real_gap"]["vinyl"] == gen.WP7_REAL_TARGET_PER_MATERIAL - 1
    assert cov["wp7_real_met"] is False
