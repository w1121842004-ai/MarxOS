#!/usr/bin/env python3
"""失败案例汇总：按种类/意图聚合 logs/query_misses.jsonl，列出待回灌样本。

用法：
  python scripts/report_query_misses.py            # 全量汇总
  python scripts/report_query_misses.py --kind quote_unconfirmed
  python scripts/report_query_misses.py --limit 30  # 最近 N 条明细
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marxos.web.miss_log import miss_log_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="", help="只看某一失败种类")
    parser.add_argument("--limit", type=int, default=30, help="明细条数（0=只看汇总）")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    log_path = args.log or miss_log_path()
    if not log_path.exists():
        print(f"暂无失败案例（{log_path} 不存在）")
        return 0

    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if args.kind:
        records = [r for r in records if r.get("kind") == args.kind]

    kind_counts = Counter(r.get("kind") for r in records)
    intent_counts = Counter(r.get("intent") for r in records)
    print(f"总失败案例：{len(records)}")
    print("\n按失败种类：")
    for kind, count in kind_counts.most_common():
        print(f"  {kind}: {count}")
    print("\n按意图：")
    for intent, count in intent_counts.most_common():
        print(f"  {intent or '(未记录)'}: {count}")

    if args.limit > 0 and records:
        print(f"\n最近 {min(args.limit, len(records))} 条明细：")
        for record in records[-args.limit:]:
            print(
                f"  [{record.get('kind')}] {record.get('query', '')[:60]} "
                f"| intent={record.get('intent')} mode={record.get('mode')} "
                f"crag={record.get('crag_score')} detail={record.get('detail', '')[:60]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
