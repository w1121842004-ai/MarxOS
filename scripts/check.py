import argparse
from pathlib import Path

from common_cli import python_command, run_step


ROOT = Path(__file__).resolve().parents[1]


QUICK_STEPS = [
    ("validate_index_manifest", python_command(ROOT / "scripts" / "validate_index_manifest.py")),
    ("validate_maps", python_command(ROOT / "scripts" / "validate_maps.py")),
    ("regression_smoke", python_command(ROOT / "scripts" / "regression_smoke.py")),
    ("tests_p0", python_command(ROOT / "scripts" / "test.py", "p0")),
]

FULL_EXTRA_STEPS = [
    ("evaluate_retrieval", python_command(ROOT / "scripts" / "evaluate_retrieval.py")),
    ("evaluate_citation_pages", python_command(ROOT / "scripts" / "evaluate_citation_pages.py")),
    ("evaluate_eval_dataset", python_command(ROOT / "scripts" / "evaluate_eval_dataset.py")),
]


def main():
    parser = argparse.ArgumentParser(description="Run MarxOS regression and evaluation checks.")
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick: fast local regressions; full: includes retrieval/dataset evaluations",
    )
    args = parser.parse_args()

    steps = list(QUICK_STEPS)
    if args.mode == "full":
        steps.extend(FULL_EXTRA_STEPS)

    for name, cmd in steps:
        code = run_step(name, cmd, cwd=ROOT)
        if code != 0:
            return code

    print(f"All {args.mode} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
