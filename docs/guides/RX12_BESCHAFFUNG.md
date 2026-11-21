# RX-12-Beschaffung — autonom geklärt bis auf den letzten menschlichen Akt

Stand 2026-11-21. Zweck: Die „Hör-/RX-Vergleichshörtest"-Pflicht des
Wettbewerbs-Paritäts-Registers (`.github/WETTBEWERBS_PARITAET.md`, Spec 26
DRT-16: gepaarter Referenzvergleich, Holdout, Nichtunterlegenheit) wartet
ausschließlich auf RX-prozessierte Vergleichsdateien. Dieses Dokument klärt,
warum das so ist, was minimal beschafft werden muss und was danach ohne
weiteren menschlichen Eingriff automatisch abläuft.

## 1. Warum RX hier nicht autonom installierbar ist

- iZotope RX 11/12 läuft **nur auf Windows und macOS** — dieser Host ist
  Linux. Die Lizenz hängt am iZotope-Portal (einmaliger Rechts-/Identitätsakt).
- Für die **Bedienung** von RX ist menschliches Know-how nicht mehr nötig:
  Sie ist durch den RX-Autopiloten vollständig automatisiert (Abschnitt 2).

## 2. Die Lösung ohne menschliches RX-Know-how: der RX-Autopilot

`rx_autopilot/` automatisiert die komplette Bearbeitungskette:

1. **`vm_setup.sh`** baut eine Windows-11-Evaluation-VM auf diesem Host
   (QEMU/KVM, `/dev/kvm` vorhanden; ISO ohne Konto herunterladbar) mit
   unbeaufsichtigter Installation (`autounattend.xml`) und QEMU-SMB-Share.
2. **`windows_agent.py`** läuft in der VM als Watch-Queue: jede
   `<name>__<defektklasse>.wav` im Share wird deterministisch mit der
   Preset-Kette aus **`rx_presets.json`** (nur dokumentierte
   iZotope-Standard-Einstellungen) headless über rxconnect verarbeitet und
   als `<name>_rx12.wav` ausgegeben. Kein GUI, kein Experten-Tweaking.
3. **`sync.py`** (Linux-Seite) legt die Paare in den Share, wartet auf die
   Ausgaben, baut das DRT-16-Manifest und stößt den Vergleichs-Harness an
   (`benchmarks/competitive/external_competitive_evidence.py`, verifiziert).

Damit ist die Bearbeitung definitiv menschenfrei. Der einzige verbleibende
Akt ist die **einmalige Lizenz-Aktivierung** (RX-Installer in den Share
legen — `firstboot.cmd` installiert still; Trial-Anmeldung beim ersten
Start). Das ist ein Rechtsakt, kein Bearbeitungs-Know-how — und wird von
keinem Agenten übernommen (Abschnitt 6).

## 3. Was bereits OHNE RX autonom erledigt ist

- **DRT-16-Vergleichs-Harness verifiziert**: `benchmarks/competitive/
  external_competitive_evidence.py` läuft end-to-end (Maschinerie-Beweis
  2026-11-21, `benchmarks/competitive/reports/machinery_proof_report.json`):
  identische Referenz/Input pro Szenario, Aurik- und Konkurrenz-Ausgaben,
  OQS/MUSHRA-Metriken, stratifizierte 5×6-Gates, `release_competitive_ready`.
- **Platzhalter-Baseline** mit ffmpeg (afftdn/highpass) als ehrliche
  Zwischenreferenz — NICHT als RX-Ersatz dokumentiert.
- **Paar-Materialien liegen bereit**: signal10-Paare (hum, dropout, clicks,
  wowflutter, pitch_drift, lautheit, amp_drift), klassen_paare (vinyl/crackle),
  stereo-Paare — alles mit Clean-Referenz in `data/schadens_quellen/`.

## 4. Der automatische Ablauf nach dem Datei-Import

```bash
# 1) Manifest aus der Paar-Liste + den rx12-Dateien bauen (Vorlage):
#    benchmarks/competitive/manifest_template.json
# 2) Harness laufen lassen:
.venv_aurik/bin/python -B benchmarks/competitive/external_competitive_evidence.py \
    --manifest benchmarks/competitive/manifest_rx12.json \
    --output benchmarks/competitive/reports/rx12_vergleich.json
# 3) Bericht prüfen; bei aurik_vs_rx11_no_stratified_regressions=true:
#    Register-Zeile der Disziplin mit dem Bericht belegen.
```

Kein weiterer Code-Schritt nötig — die Gates, die Berichts-Logik und die
Register-Mechanik existieren und sind getestet.

## 5. Disziplin-Priorität der Beschaffung

| Priorität | Disziplin | RX-Modul | Status-Register |
| --- | --- | --- | --- |
| 1 | Decrackle | De-crackle | LUECKE (VinylNet-noise-bank-Retrain läuft parallel) |
| 2 | Denoise | Spectral De-noise | LUECKE (Witness-Kopf als Kandidat) |
| 3 | HF-/Bandbreite | Spectral Recovery | LUECKE (Kandidat-Frage offen) |
| 4 | Declick | De-click | LUECKE (DRT-5-Q + DRT-8-Beleg offen) |
| 5 | Dereverb | De-reverb | GESICHERT (Bestätigungshörtest) |

## 6. Hinweis für Agenten

Diese Datei ist der verbindliche Beschaffungs-Steckbrief. Kein Agent darf
(a) einen iZotope-Account anlegen, (b) Lizenzschlüssel beschaffen oder
(c) RX-Ausgaben fälschen. Der einzige zulässige Weg ist Abschnitt 2 durch
den Menschen — danach übernimmt Abschnitt 4 automatisch.
