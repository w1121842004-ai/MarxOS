import argparse
import sys
from pathlib import Path

from common_cli import print_command_list, run_step


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"


SUITES = {
    "p0": [
        "test_runtime_health.py",
        "test_launch_contract.py",
        "test_unit_isolation.py",
        "test_run_query_regressions.py",
        "test_web_api.py",
        "test_web_smoke_runner.py",
    ],
    "app": [
        "test_unit_isolation.py",
        "test_run_query_regressions.py",
        "test_app_local_paths.py",
    ],
    "web": [
        "test_web_api.py",
        "test_web_smoke_runner.py",
    ],
    "rag": [
        "test_core_bibliography.py",
        "test_page_metadata_inference.py",
        "test_paragraph_cache.py",
        "test_document_contract.py",
        "test_document_audit_cli.py",
    ],
}
SUITES["all"] = SUITES["app"] + SUITES["web"] + SUITES["rag"]
SUITE_DESCRIPTIONS = {
    "p0": "Stable baseline tests: runtime health, launch contract, offline isolation, Web contract",
    "app": "CLI/query orchestration, local answers, quote and citation regressions",
    "web": "Web API payload, evidence, and metrics behavior",
    "rag": "Bibliography, page metadata inference, and paragraph cache logic",
    "all": "Run every grouped unittest suite",
}


def unittest_discover_args(pattern: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(TESTS_DIR),
        "-p",
        pattern,
    ]


def run_suite(name: str) -> int:
    for filename in SUITES[name]:
        code = run_step(
            f"tests:{name}:{filename}",
            unittest_discover_args(filename),
            cwd=ROOT,
            quiet_success=True,
        )
        if code != 0:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run grouped MarxOS unittest suites.")
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        help="Which grouped unittest suite to run. Use 'list' to show suites.",
    )
    args = parser.parse_args()
    if args.suite == "list":
        print_command_list("Available test suites:", SUITE_DESCRIPTIONS)
        return 0
    if args.suite not in SUITES:
        parser.error(f"unknown suite: {args.suite}")
    return run_suite(args.suite)


if __name__ == "__main__":
    raise SystemExit(main())
