- Umsetzung der Stabilitaets-Empfehlung vom 2026-05-23.
- `.vscode/tasks.json` angepasst:
  - `pytest: Vollsuite STABIL (seriell, low-overhead)` nutzt jetzt `run_tests_chunked_safe.sh` mit
    `AURIK_BATCH_FILES=8`, `AURIK_OOM_RETRY_MEM_GB=20`, `AURIK_OOM_RETRY_SWAP_MB=6144`.
  - `pytest: Chunk C (restliche tests/, ohne unit+goals+integration)` nutzt identische Chunk/OOM-Parameter.
- `run_tests_chunked_safe.sh` Default-Werte angepasst:
  - `AURIK_BATCH_FILES` default von 40 -> 8.
  - `AURIK_OOM_RETRY_MEM_GB` default von 16 -> 20.
  - `AURIK_OOM_RETRY_SWAP_MB` default von 4096 -> 6144.
- Schnellverifikation mit `AURIK_CHUNK_LIMIT=1` auf Vollsuite-Pfad: erfolgreich, Chunk 1/80 gruen.
