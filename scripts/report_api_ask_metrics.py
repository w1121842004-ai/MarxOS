import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


DEFAULT_METRICS_PATH = Path("logs/api_ask_metrics.jsonl")


def load_rows(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("event") == "api_ask":
            rows.append(row)
    return rows


def row_day(row):
    ts = row.get("ts")
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "unknown"


def pct(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def summarize(rows):
    total = len(rows)
    evidence_positive = sum(1 for r in rows if int(r.get("evidence_count", 0) or 0) > 0)
    fallback_used = sum(1 for r in rows if bool(r.get("fallback_used")))
    with_citations = sum(1 for r in rows if int(r.get("citation_lines_count", 0) or 0) > 0)
    matched_positive = sum(1 for r in rows if int(r.get("matched_count", 0) or 0) > 0)
    avg_elapsed = round(sum(int(r.get("elapsed_ms", 0) or 0) for r in rows) / total, 2) if total else 0.0
    intent_counter = Counter((r.get("intent") or "-") for r in rows)
    return {
        "total_requests": total,
        "button_visible_rate": round(pct(evidence_positive, total), 2),
        "fallback_rate": round(pct(fallback_used, total), 2),
        "citation_answer_rate": round(pct(with_citations, total), 2),
        "matched_rate": round(pct(matched_positive, total), 2),
        "avg_elapsed_ms": avg_elapsed,
        "intent_top": intent_counter.most_common(5),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate api_ask metrics and print daily report.")
    parser.add_argument("--input", default=str(DEFAULT_METRICS_PATH), help="metrics jsonl path")
    parser.add_argument("--day", default="", help="YYYY-MM-DD; empty means all days")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    if args.day:
        rows = [r for r in rows if row_day(r) == args.day]
    report = summarize(rows)
    report["day"] = args.day or "all"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
