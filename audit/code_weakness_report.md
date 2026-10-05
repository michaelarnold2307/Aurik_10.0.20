# Code-Schwachstellen-Report (Watchdog)

- Erzeugt: 2026-10-05T05:53:45.459039
- Geprüfte Dateien: 1691 (Dauer: 103.352s)
- Befunde gesamt: **6**
  - critical: 0
  - high: 0
  - medium: 0
  - low: 6
- Pro Regel: wallclock_ttl_housekeeping=6
- Unterdrückte Befunde (unter Schwelle/Kappung, bewusst sichtbar): determinism_time_usage=318
  (Schwellen: AST-Cap 3/Datei, print ≥ 3 im echten Code, Top-N 10, max_findings. determinism_time_usage listet NUR Wall-Clock in Entscheidungslogik (§G5 (GEBOTE.md)); Messungen und Zeitstempel sind kein Verstoß — persistierte Zeitstempel MÜSSEN Wall-Clock bleiben, time.monotonic() ist prozesslokal und über Prozess-/Session-Grenzen bedeutungslos. Unterdrückt heißt nicht: nicht vorhanden.)

## LOW (6)

- `backend/core/artist_fingerprint.py:199` — **wallclock_ttl_housekeeping** (§G5 (copilot-instructions.md) — Abgrenzung TTL-Haushalt)
  - Wall-Clock in TTL-/Datei-Lebenszyklus-Entscheidung (kein Audio-Determinismus-Verstoß)
  - Evidenz: `if time.time() - fp.last_updated > MAX_FINGERPRINT_AGE_DAYS * 86400:`
  - Empfehlung: Bewusst: Jede TTL-Politik (Prüfpunkt-/Cache-Ablauf) braucht Wall-Clock — der Wert wird persistiert und muss über Prozess-/Boot-Grenzen vergleichbar bleiben. Kein §G5-Verstoß, solange die Entscheidung das Restaurierungs-Audio nicht beeinflusst. Sichtbar ausgewiesen, damit die Prüfung vollständig bleibt.
- `backend/core/artist_fingerprint.py:233` — **wallclock_ttl_housekeeping** (§G5 (copilot-instructions.md) — Abgrenzung TTL-Haushalt)
  - Wall-Clock in TTL-/Datei-Lebenszyklus-Entscheidung (kein Audio-Determinismus-Verstoß)
  - Evidenz: `if time.time() - fp.last_updated > MAX_FINGERPRINT_AGE_DAYS * 86400:`
  - Empfehlung: Bewusst: Jede TTL-Politik (Prüfpunkt-/Cache-Ablauf) braucht Wall-Clock — der Wert wird persistiert und muss über Prozess-/Boot-Grenzen vergleichbar bleiben. Kein §G5-Verstoß, solange die Entscheidung das Restaurierungs-Audio nicht beeinflusst. Sichtbar ausgewiesen, damit die Prüfung vollständig bleibt.
- `backend/core/artist_fingerprint.py:318` — **wallclock_ttl_housekeeping** (§G5 (copilot-instructions.md) — Abgrenzung TTL-Haushalt)
  - Wall-Clock in TTL-/Datei-Lebenszyklus-Entscheidung (kein Audio-Determinismus-Verstoß)
  - Evidenz: `cutoff = time.time() - max_age_days * 86400`
  - Empfehlung: Bewusst: Jede TTL-Politik (Prüfpunkt-/Cache-Ablauf) braucht Wall-Clock — der Wert wird persistiert und muss über Prozess-/Boot-Grenzen vergleichbar bleiben. Kein §G5-Verstoß, solange die Entscheidung das Restaurierungs-Audio nicht beeinflusst. Sichtbar ausgewiesen, damit die Prüfung vollständig bleibt.
- `backend/core/crash_recovery_guard.py:198` — **wallclock_ttl_housekeeping** (§G5 (copilot-instructions.md) — Abgrenzung TTL-Haushalt)
  - Wall-Clock in TTL-/Datei-Lebenszyklus-Entscheidung (kein Audio-Determinismus-Verstoß)
  - Evidenz: `now = time.time()`
  - Empfehlung: Bewusst: Jede TTL-Politik (Prüfpunkt-/Cache-Ablauf) braucht Wall-Clock — der Wert wird persistiert und muss über Prozess-/Boot-Grenzen vergleichbar bleiben. Kein §G5-Verstoß, solange die Entscheidung das Restaurierungs-Audio nicht beeinflusst. Sichtbar ausgewiesen, damit die Prüfung vollständig bleibt.
- `backend/core/recovery_checkpoint.py:261` — **wallclock_ttl_housekeeping** (§G5 (copilot-instructions.md) — Abgrenzung TTL-Haushalt)
  - Wall-Clock in TTL-/Datei-Lebenszyklus-Entscheidung (kein Audio-Determinismus-Verstoß)
  - Evidenz: `now = time.time()`
  - Empfehlung: Bewusst: Jede TTL-Politik (Prüfpunkt-/Cache-Ablauf) braucht Wall-Clock — der Wert wird persistiert und muss über Prozess-/Boot-Grenzen vergleichbar bleiben. Kein §G5-Verstoß, solange die Entscheidung das Restaurierungs-Audio nicht beeinflusst. Sichtbar ausgewiesen, damit die Prüfung vollständig bleibt.
- `backend/core/recovery_checkpoint.py:420` — **wallclock_ttl_housekeeping** (§G5 (copilot-instructions.md) — Abgrenzung TTL-Haushalt)
  - Wall-Clock in TTL-/Datei-Lebenszyklus-Entscheidung (kein Audio-Determinismus-Verstoß)
  - Evidenz: `now = time.time()`
  - Empfehlung: Bewusst: Jede TTL-Politik (Prüfpunkt-/Cache-Ablauf) braucht Wall-Clock — der Wert wird persistiert und muss über Prozess-/Boot-Grenzen vergleichbar bleiben. Kein §G5-Verstoß, solange die Entscheidung das Restaurierungs-Audio nicht beeinflusst. Sichtbar ausgewiesen, damit die Prüfung vollständig bleibt.
