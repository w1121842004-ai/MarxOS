from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"


def run_json_report(command: list[str], report: Path) -> tuple[int, dict]:
    completed = subprocess.run(command, cwd=ROOT, text=True)
    data = {}
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
    return completed.returncode, data


def summary(data: dict) -> dict:
    return data.get("summary") or data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--ambiguous-sample-size", type=int, default=120)
    parser.add_argument("--answer-sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--report", type=Path, default=ROOT / "logs" / "me_enterprise_retrieval_gate_latest.json")
    args = parser.parse_args()

    reports = {
        "retrieval": ROOT / "logs" / "me_retrieval_quality_gate_enterprise.json",
        "ambiguous": ROOT / "logs" / "me_ambiguous_locator_gate_report.json",
        "answer": ROOT / "logs" / "me_answer_contract_gate_report.json",
    }
    commands = {
        "retrieval": [
            str(PYTHON), "-u", "scripts/run_me_retrieval_quality_gate.py",
            "--sample-size", str(args.sample_size),
            "--seed", str(args.seed),
            "--report", str(reports["retrieval"]),
        ],
        "ambiguous": [
            str(PYTHON), "-u", "scripts/audit_me_ambiguous_locator_retrieval.py",
            "--sample-size", str(args.ambiguous_sample_size),
            "--seed", str(args.seed),
            "--report", str(reports["ambiguous"]),
        ],
        "answer": [
            str(PYTHON), "-u", "scripts/audit_me_answer_contracts.py",
            "--sample-size", str(args.answer_sample_size),
            "--seed", str(args.seed),
            "--report", str(reports["answer"]),
        ],
    }

    results = {}
    ok = True
    for name, command in commands.items():
        code, data = run_json_report(command, reports[name])
        results[name] = {
            "exit_code": code,
            "report": str(reports[name]),
            "summary": summary(data),
        }
        ok = ok and code == 0

    report = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
        },
        "ok": ok,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
