from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from marxos.data.document_contract import audit_document_records


def _read_jsonl(path_value: str) -> tuple[list[dict], str]:
    if path_value == "-":
        content = sys.stdin.read()
    else:
        content = Path(path_value).read_text(encoding="utf-8")
    records = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid_jsonl at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid_jsonl at line {line_number}: record must be an object")
        records.append(value)
    return records, hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MarxOS document metadata without loading models or indexes.")
    parser.add_argument("--input", required=True, help="JSONL record cache, or - for stdin")
    parser.add_argument("--report", help="Optional JSON report path")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print only the compact production summary (the --report file remains complete)",
    )
    args = parser.parse_args()
    try:
        records, checksum = _read_jsonl(args.input)
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema_version": "document-audit/v1", "error": str(exc)}, ensure_ascii=False))
        return 2

    report = audit_document_records(records)
    report = {
        **report,
        "input": {"kind": "jsonl", "path": args.input, "sha256": checksum},
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    if args.summary_only:
        sys.stdout.write(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(payload)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
