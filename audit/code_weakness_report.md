# Code-Schwachstellen-Report (Watchdog)

- Erzeugt: 2026-11-14T23:37:14.004923
- Geprüfte Dateien: 1791 (Dauer: 136.213s)
- Befunde gesamt: **8**
  - critical: 0
  - high: 8
  - medium: 0
  - low: 0
- Pro Regel: dither_missing_int_conversion=2, silent_fallback_no_log=6
- Unterdrückte Befunde (unter Schwelle/Kappung, bewusst sichtbar): determinism_time_usage=317, truncated_max_findings=10
  (Schwellen: AST-Cap 3/Datei, print ≥ 3 im echten Code, Top-N 10, max_findings. determinism_time_usage listet NUR Wall-Clock in Entscheidungslogik (§G5 (GEBOTE.md)); Messungen und Zeitstempel sind kein Verstoß — persistierte Zeitstempel MÜSSEN Wall-Clock bleiben, time.monotonic() ist prozesslokal und über Prozess-/Session-Grenzen bedeutungslos. Unterdrückt heißt nicht: nicht vorhanden.)

## HIGH (8)

- `backend/core/dsp/audibility_budget.py:118` — **dither_missing_int_conversion** (§V5 (copilot-instructions.md))
  - Integer-Konversion ohne Dither (bit_depth < 32)
  - Evidenz: `clipped: np.ndarray = np.clip(idx, 0, e.size - 2).astype(np.int32)`
  - Empfehlung: POW-r Type 3 (primär) oder TPDF (Fallback) vor der Konversion anwenden; kein nacktes astype(np.int16). (§V5 (copilot-instructions.md) Truncation-ohne-Dither-Verbot)
- `backend/core/dsp/musik_declip_kopf.py:148` — **dither_missing_int_conversion** (§V5 (copilot-instructions.md))
  - Integer-Konversion ohne Dither (bit_depth < 32)
  - Evidenz: `kanten = np.diff(maske.astype(np.int8))`
  - Empfehlung: POW-r Type 3 (primär) oder TPDF (Fallback) vor der Konversion anwenden; kein nacktes astype(np.int16). (§V5 (copilot-instructions.md) Truncation-ohne-Dither-Verbot)
- `backend/core/defect_scanner.py:693` — **silent_fallback_no_log** (§V6 (copilot-instructions.md))
  - Fallback/Return in except-Block ohne Logging
  - Evidenz: `except Exception:  # §V6 (copilot-instructions.md)`
  - Empfehlung: logger.warning() + Begründung ergänzen, damit ML→DSP-Fallbacks nie stumm bleiben. (§V6 (copilot-instructions.md) Silent-Failure-Verbot)
- `backend/core/defect_scanner.py:720` — **silent_fallback_no_log** (§V6 (copilot-instructions.md))
  - Fallback/Return in except-Block ohne Logging
  - Evidenz: `except Exception:  # §V6 (copilot-instructions.md)`
  - Empfehlung: logger.warning() + Begründung ergänzen, damit ML→DSP-Fallbacks nie stumm bleiben. (§V6 (copilot-instructions.md) Silent-Failure-Verbot)
- `backend/core/dsp/deep_loudness_net.py:82` — **silent_fallback_no_log** (§V6 (copilot-instructions.md))
  - Fallback/Return in except-Block ohne Logging
  - Evidenz: `except Exception:  # §V6 (copilot-instructions.md)`
  - Empfehlung: logger.warning() + Begründung ergänzen, damit ML→DSP-Fallbacks nie stumm bleiben. (§V6 (copilot-instructions.md) Silent-Failure-Verbot)
- `backend/core/dsp/deep_loudness_net.py:100` — **silent_fallback_no_log** (§V6 (copilot-instructions.md))
  - Fallback/Return in except-Block ohne Logging
  - Evidenz: `except Exception:  # §V6 (copilot-instructions.md)`
  - Empfehlung: logger.warning() + Begründung ergänzen, damit ML→DSP-Fallbacks nie stumm bleiben. (§V6 (copilot-instructions.md) Silent-Failure-Verbot)
- `backend/core/dsp/music_mos_oracle.py:79` — **silent_fallback_no_log** (§V6 (copilot-instructions.md))
  - Fallback/Return in except-Block ohne Logging
  - Evidenz: `except (TypeError, ValueError):`
  - Empfehlung: logger.warning() + Begründung ergänzen, damit ML→DSP-Fallbacks nie stumm bleiben. (§V6 (copilot-instructions.md) Silent-Failure-Verbot)
- `backend/core/dsp/music_vad_oracle.py:110` — **silent_fallback_no_log** (§V6 (copilot-instructions.md))
  - Fallback/Return in except-Block ohne Logging
  - Evidenz: `except (TypeError, ValueError):`
  - Empfehlung: logger.warning() + Begründung ergänzen, damit ML→DSP-Fallbacks nie stumm bleiben. (§V6 (copilot-instructions.md) Silent-Failure-Verbot)
