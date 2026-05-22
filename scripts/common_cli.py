from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, args: list[str], cwd: Path | None = None) -> int:
    print(f"[RUN ] {name}")
    result = subprocess.run(args, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"[FAIL] {name} (exit={result.returncode})")
        return result.returncode
    print(f"[PASS] {name}")
    return 0


def print_command_list(title: str, commands: dict[str, str]) -> None:
    print(title)
    for name, description in sorted(commands.items()):
        print(f"  - {name:<20} {description}")


def python_command(script_path: Path, *args: str) -> list[str]:
    return [sys.executable, str(script_path), *args]
