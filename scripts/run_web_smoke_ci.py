"""Run the versioned Web response-contract smoke suite without external services."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import web_app


DEFAULT_DATASET = ROOT / "tests" / "fixtures" / "web_smoke_v1.json"
REQUIRED_RESPONSE_FIELDS = {
    "intent", "mode", "answer", "evidence", "citation_audit", "topic",
    "crag", "timing", "elapsed_ms", "memory_turns",
}
VALID_MODES = {"auto", "fast", "standard", "deep", "precise"}
EXPECTED_RESPONSE_MODE = {
    "auto": "fast",
    "fast": "fast",
    "standard": "standard",
    "deep": "deep",
    "precise": "fast",
}


def load_dataset(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 1:
        raise ValueError("web smoke dataset schema_version must be 1")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("web smoke dataset must contain a non-empty cases list")
    labels = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        label = case.get("label")
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError(f"case {index} has a missing or duplicate label")
        labels.add(label)
        if not isinstance(case.get("query"), str):
            raise ValueError(f"case {label} query must be a string")
        if case.get("mode", "auto") not in VALID_MODES:
            raise ValueError(f"case {label} has an invalid mode")
    return dataset


def _fake_run_query(_query: str, **_kwargs) -> str:
    return "离线 Web smoke 固定回答。"


def _validate_result(status: int, payload: dict, requested_mode: str) -> list[str]:
    errors = []
    if status != 200:
        errors.append(f"status={status}")
    missing = sorted(REQUIRED_RESPONSE_FIELDS - payload.keys())
    if missing:
        errors.append("missing_fields=" + ",".join(missing))
    if status == 200 and not isinstance(payload.get("answer"), str):
        errors.append("answer must be a string")
    if status == 200 and not isinstance(payload.get("evidence"), list):
        errors.append("evidence must be a list")
    expected_mode = EXPECTED_RESPONSE_MODE[requested_mode]
    if status == 200 and payload.get("mode") != expected_mode:
        errors.append(f"mode={payload.get('mode')!r}, expected={expected_mode!r}")
    return errors


def run_dataset(dataset: dict, report_path: Path | None = None) -> dict:
    handler = web_app.MarxOSHandler.__new__(web_app.MarxOSHandler)
    results = []
    with (
        patch.object(web_app.app, "run_query", side_effect=_fake_run_query),
        patch.object(web_app.app, "classify_query", return_value="rag_answer"),
        patch.object(web_app.MarxOSHandler, "_append_metrics_log", return_value=None),
        patch.object(web_app.app, "LAST_EVIDENCE", []),
        patch.object(web_app.app, "LAST_CITATION_AUDIT", {}),
        patch.object(web_app.app, "LAST_TOPIC_INFO", {}),
        patch.object(web_app.app, "LAST_CRAG_REPORT", {}),
        patch.object(web_app.app, "LAST_TIMING", {}),
    ):
        with contextlib.redirect_stderr(io.StringIO()):
            for case in dataset["cases"]:
                requested_mode = case.get("mode", "auto")
                status, payload = handler._run_ask_payload(
                    {"query": case["query"], "mode": requested_mode, "history": []}
                )
                errors = _validate_result(status, payload, requested_mode)
                results.append(
                    {
                        "label": case["label"],
                        "status": status,
                        "response_mode": payload.get("mode"),
                        "intent": payload.get("intent"),
                        "ok": not errors,
                        "errors": errors,
                    }
                )

    failed = sum(not item["ok"] for item in results)
    report = {
        "schema_version": dataset["schema_version"],
        "summary": {"total": len(results), "passed": len(results) - failed, "failed": failed},
        "results": results,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_dataset(load_dataset(args.dataset), report_path=args.report)
    print(json.dumps(report["summary"], ensure_ascii=False))
    for result in report["results"]:
        if not result["ok"]:
            print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
