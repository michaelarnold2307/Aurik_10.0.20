# Aurik 10 - CLI

Kommandozeilenschnittstelle für headless Restaurierungsläufe und Batch-Exports.

- Einstiegspunkt: `cli/aurik_cli.py` (Aufruf: `python3 -B cli/aurik_cli.py --help`)
- Restaurierungsmodus: `--mode Restoration` (Standard) oder `--mode "Studio 2026"`
- Export: `--bit-depth 16|24|32`, `--output-sr 44100|48000`, ABX via `--abx`
- Der Export folgt dem §0c-Vertrag: bei Gate-Fail wird das bestmögliche Ergebnis
  mit Status „degraded“ exportiert — kein Hardstop ohne Ausgabedatei.
