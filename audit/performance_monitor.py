"""
Aurik Performance-Check und Ressourcenmonitoring
- Misst regelmäßig Latenz, Durchsatz, CPU/GPU-Auslastung aller Kernmodule und Plugins
- Dokumentiert Performance-Einbrüche und Ressourcenengpässe im Audit-Log
- Generiert einen Performance-Report für das Entwicklerteam
"""

import json
import time
from pathlib import Path

import psutil


def measure_performance(module_name, func, *args, **kwargs):
    start_time = time.time()
    cpu_start = psutil.cpu_percent(interval=None)
    mem_start = psutil.virtual_memory().percent
    result = func(*args, **kwargs)
    cpu_end = psutil.cpu_percent(interval=None)
    mem_end = psutil.virtual_memory().percent
    end_time = time.time()
    perf_data = {
        "module": module_name,
        "latency": end_time - start_time,
        "cpu_usage": cpu_end - cpu_start,
        "mem_usage": mem_end - mem_start,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return result, perf_data


def log_performance(perf_data, log_path="audit/performance_log.json"):
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        with open(log_file) as f:
            log_data = json.load(f)
    else:
        log_data = []
    log_data.append(perf_data)
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)


def main():
    # Beispiel: Performance-Messung für ein Dummy-Modul
    def dummy_module():
        time.sleep(0.5)
        return "done"

    result, perf_data = measure_performance("dummy_module", dummy_module)
    log_performance(perf_data)
    print(f"Performance-Report: {perf_data}")


if __name__ == "__main__":
    main()
