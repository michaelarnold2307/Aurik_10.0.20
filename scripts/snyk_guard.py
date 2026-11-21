"""Fail-closed Snyk pre-commit guard.

Exit code 0 from ``snyk test`` means the scan completed with no findings;
exit code 1 means findings were found. Operational scan failures block the
commit and never trigger an automatic fix.
"""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


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


def main() -> int:
    cwd = str(Path.cwd())
    test_command = ["snyk", "test", "--severity-threshold=high"]

    print("Running Snyk security scan...")
    scan_code, scan_output = run(test_command, cwd=cwd)
    print(scan_output)
    if scan_code == 0:
        print("No high-severity vulnerabilities detected.")
        return 0
    if scan_code != 1:
        print(f"Snyk scan failed with exit code {scan_code}; commit blocked.")
        return 1

    print("Snyk reported vulnerabilities; attempting automatic fix...")
    fix_code, fix_output = run(["snyk", "fix"], cwd=cwd)
    print(fix_output)
    if fix_code != 0:
        print(f"Snyk fix failed with exit code {fix_code}; commit blocked.")
        return 1

    retest_code, retest_output = run(test_command, cwd=cwd)
    print(retest_output)
    if retest_code == 0:
        print("No high-severity vulnerabilities remain.")
        return 0
    if retest_code == 1:
        print("High-severity vulnerabilities remain after the fix; commit blocked.")
    else:
        print(f"Snyk retest failed with exit code {retest_code}; commit blocked.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
