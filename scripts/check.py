import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name, args):
    print(f"[RUN ] {name}")
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        print(f"[FAIL] {name} (exit={result.returncode})")
        return result.returncode
    print(f"[PASS] {name}")
    return 0


def main():
    steps = [
        ("regression_smoke", [sys.executable, str(ROOT / "scripts" / "regression_smoke.py")]),
    ]

    for name, args in steps:
        code = run_step(name, args)
        if code != 0:
            return code

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
