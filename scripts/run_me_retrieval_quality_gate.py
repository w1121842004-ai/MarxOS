from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_SCRIPT = ROOT_DIR / "scripts" / "audit_me_article_locator_retrieval.py"
DEFAULT_REPORT = ROOT_DIR / "logs" / "me_retrieval_quality_gate_latest.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def run_article_audit(args) -> dict:
    report_path = ROOT_DIR / "logs" / "me_article_locator_gate_report.json"
    command = [
        sys.executable,
        str(DEFAULT_AUDIT_SCRIPT),
        "--sample-size",
        str(args.sample_size),
        "--report",
        str(report_path),
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    run_command(command)
    return load_json(report_path)


def run_letter_audit(args) -> dict:
    report_path = ROOT_DIR / "logs" / "me_letter_locator_gate_report.json"
    command = [
        sys.executable,
        str(DEFAULT_AUDIT_SCRIPT),
        "--letters-only",
        "--sample-size",
        str(args.sample_size),
        "--report",
        str(report_path),
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    run_command(command)
    return load_json(report_path)


def evaluate_gate(article_report: dict, letter_report: dict, args) -> dict:
    article_summary = article_report["summary"]
    letter_summary = letter_report["summary"]
    letter_total = int(letter_summary.get("letter_cases") or letter_summary.get("total") or 0)
    letter_title_hits = int(letter_summary.get("letter_title_hit_at_k") or 0)

    checks = [
        {
            "name": "article_pass_rate",
            "actual": article_summary.get("pass_rate"),
            "expected": f">= {args.article_pass_rate}",
            "ok": float(article_summary.get("pass_rate") or 0) >= args.article_pass_rate,
        },
        {
            "name": "letter_title_hit_at_k",
            "actual": f"{letter_title_hits}/{letter_total}",
            "expected": "all letter cases",
            "ok": letter_total > 0 and letter_title_hits == letter_total,
        },
        {
            "name": "letter_fail_count",
            "actual": letter_summary.get("fail"),
            "expected": "0",
            "ok": int(letter_summary.get("fail") or 0) == 0,
        },
    ]
    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "article_pass_rate_threshold": args.article_pass_rate,
        },
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "article_summary": article_summary,
        "letter_summary": letter_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ME retrieval quality gate.")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--article-pass-rate", type=float, default=0.95)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    article_report = run_article_audit(args)
    letter_report = run_letter_audit(args)
    gate = evaluate_gate(article_report, letter_report, args)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nGate summary:")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if gate["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
