"""Fail-closed Snyk scans for Aurik's runtime and development dependencies."""

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

SUPPORTED_MANIFESTS = (
    "requirements/requirements_aurik.txt",
    "requirements/requirements_dev.txt",
)


def run(cmd: Sequence[str], cwd: str | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(
            list(cmd),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.returncode, result.stdout
    except OSError as exc:
        return 127, str(exc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan Aurik Python dependency manifests with Snyk.")
    parser.add_argument("--manifest", action="append", choices=SUPPORTED_MANIFESTS)
    args = parser.parse_args(argv)

    cwd = str(Path.cwd())
    manifests = args.manifest or SUPPORTED_MANIFESTS
    report_dir = os.environ.get("SNYK_REPORT_DIR")
    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)

    findings: list[str] = []
    scan_errors: list[str] = []
    for manifest in manifests:
        command = [
            "snyk",
            "test",
            f"--file={manifest}",
            "--package-manager=pip",
            f"--command={sys.executable}",
            "--severity-threshold=high",
        ]
        if report_dir:
            report_path = Path(report_dir) / f"{Path(manifest).stem}.json"
            command.append(f"--json-file-output={report_path}")

        print(f"Running Snyk scan for {manifest}...")
        scan_code, scan_output = run(command, cwd=cwd)
        if scan_output:
            sys.stdout.write(scan_output)
            if not scan_output.endswith("\n"):
                print()

        if scan_code == 1:
            findings.append(manifest)
        elif scan_code != 0:
            scan_errors.append(f"{manifest} (exit code {scan_code})")

    if findings:
        print(f"High-severity Snyk findings in: {', '.join(findings)}; commit blocked.")
    if scan_errors:
        print(f"Snyk scans failed for: {', '.join(scan_errors)}; commit blocked.")
    if findings or scan_errors:
        return 1

    print("All Snyk dependency scans passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
