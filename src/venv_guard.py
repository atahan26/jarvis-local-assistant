"""Hard fail if we are not running inside the project's own .venv.

Spec non-negotiable §1: everything runs in .venv, never system Python, never
--user. This is imported first by every entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VENV = PROJECT_ROOT / ".venv"


def assert_venv(strict_path: bool = True) -> None:
    if sys.prefix == sys.base_prefix:
        raise SystemExit(
            "REFUSING TO RUN: not inside a virtual environment.\n"
            f"  sys.prefix      = {sys.prefix}\n"
            f"  sys.base_prefix = {sys.base_prefix}\n"
            f"Activate it first:  {EXPECTED_VENV / 'Scripts' / 'Activate.ps1'}"
        )

    if strict_path:
        actual = Path(sys.prefix).resolve()
        if actual != EXPECTED_VENV.resolve():
            raise SystemExit(
                "REFUSING TO RUN: inside a virtual environment, but not this "
                "project's.\n"
                f"  active   = {actual}\n"
                f"  expected = {EXPECTED_VENV.resolve()}"
            )

    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            "REFUSING TO RUN: spec §4 requires Python 3.11 for wheel "
            f"availability; this interpreter is {sys.version.split()[0]}."
        )


if __name__ == "__main__":
    assert_venv()
    print(f"venv OK: {sys.prefix}")
    print(f"python : {sys.version.split()[0]}")
