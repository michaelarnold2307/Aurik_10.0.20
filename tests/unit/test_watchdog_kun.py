"""v10 Integrationstest fur Watchdog + Aurik-Agent-Optimierungen."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

FAILED = 0
PASSED = 0


def check(name, condition):
    global FAILED, PASSED
    if not condition:
        print(f"  FAIL: {name}")
        FAILED += 1
    else:
        print(f"  OK: {name}")
        PASSED += 1


# ── Test 1: Imports ──
print("Test: Module imports")
from backend.core.critical_listening_points import CLP_ZONES, VOCAL_ZONES, CLPResult, analyze_critical_zones
from backend.core.dynamics_preserver import DynamicsPreserver, capture_dynamics_profile, restore_dynamics
from backend.core.watchdog_monitor import SignalIntegrity, WatchdogMonitor, WatchdogReport, get_watchdog
from backend.core.whisper_detail_preserver import WhisperPreservationResult, analyze_whisper_detail

check("CLP_ZONES count", len(CLP_ZONES) == 6)
check("CLP_ZONES[0].name = Praesenz", CLP_ZONES[0].name == "Präsenz")
check("VOCAL_ZONES has formants", "formant_f1" in VOCAL_ZONES and "sibilance" in VOCAL_ZONES)

# ── Test 2: Singleton ──
print("Test: Watchdog Singleton")
w1 = get_watchdog()
w2 = get_watchdog()
check("singleton identity", w1 is w2)

# ── Test 3: Pre-Flight ──
print("Test: Pre-Flight Check")
wd = WatchdogMonitor()
audio_clean = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32) * 0.1
ok, issues = wd.pre_flight_check(audio_clean, 48000)
check("clean audio passes pre-flight", ok)

audio_nan = np.full(1000, np.nan, dtype=np.float32)
ok, issues = wd.pre_flight_check(audio_nan, 48000)
check("NaN audio fails pre-flight", not ok)

audio_dc = np.ones(1000, dtype=np.float32) * 0.5
ok, issues = wd.pre_flight_check(audio_dc, 48000)
check("DC audio fails pre-flight", not ok)

# ── Test 4: Phase Tracking ──
print("Test: Phase Tracking")
wd2 = WatchdogMonitor()
wd2.on_phase_start("test")
wd2.on_phase_end("test", audio_clean, 48000)
report = wd2.post_flight_validity(audio_clean, 48000)
check("report type is WatchdogReport", isinstance(report, WatchdogReport))
check("phases monitored", len(report.phase_watches) > 0)
check("all_checks_passed", report.all_checks_passed)

# ── Test 5: Cumulative Strength ──
print("Test: Cumulative Strength Guard")
wd3 = WatchdogMonitor()
wd3.record_cumulative_effect(0.6)
wd3.record_cumulative_effect(0.5)
wd3.record_cumulative_effect(0.3)  # total 1.4 > CUMULATIVE_STRENGTH_WARN
report3 = wd3.post_flight_validity(audio_clean, 48000)
check("cumulative tracked", report3.cumulative_strength > 1.0)

# ── Test 6: Silent Exception Tracking ──
print("Test: Silent Exception Tracking")
wd4 = WatchdogMonitor()
for _ in range(6):
    wd4.record_silent_exception("test_context")
report4 = wd4.post_flight_validity(audio_clean, 48000)
check("silent excepts tracked", report4.silent_except_count >= 6)

# ── Test 7: Critical Listening Points ──
print("Test: CLP Analysis")
audio_rand = np.random.randn(48000).astype(np.float32) * 0.1
clp = analyze_critical_zones(audio_rand, 48000)
check("mask produced", clp.critical_mask is not None)
check("zone_scores count", len(clp.zone_scores) == len(CLP_ZONES))
check("vocal_presence range", 0.0 <= clp.vocal_presence <= 1.0)
check("whisper_energy >= 0", clp.whisper_energy >= 0.0)

# ── Test 8: Dynamics ──
print("Test: Dynamics Preservation")
t = np.arange(48000) / 48000
audio_sine = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
profile = capture_dynamics_profile(audio_sine, 48000)
check("crest_factor > 0", profile.crest_factor_db > 0)
check("peak_to_rms > 0", profile.crest_factor_db > 0)
check("dynamic_range_db valid", profile.dynamic_range_db >= 0)

overprocessed = audio_sine * 0.3
restored = restore_dynamics(overprocessed, 48000, profile)
check("restored shape matches", restored.shape == audio_sine.shape)

# ── Test 9: Whisper ──
print("Test: Whisper Detail")
audio_quiet = np.random.randn(48000).astype(np.float32) * 0.001
w_result = analyze_whisper_detail(audio_quiet, 48000)
check("whisper_ratio valid", 0.0 <= w_result.whisper_ratio <= 1.0)
check("segments is list", isinstance(w_result.segments, list))

# ── Summary ──
if __name__ == "__main__":
    print(f"\n{'=' * 50}")
    print(f"Result: {PASSED} passed, {FAILED} failed")
    sys.exit(0 if FAILED == 0 else 1)
