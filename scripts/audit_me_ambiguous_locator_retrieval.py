from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app


DEFAULT_LOCATORS = ROOT_DIR / "rag" / "me_article_locators.json"
DEFAULT_REPORT = ROOT_DIR / "logs" / "me_ambiguous_locator_audit_latest.json"


def norm(text) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", str(text or "")).lower()


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_article(locator: dict) -> bool:
    return (
        locator.get("locator_type", "article") == "article"
        and locator.get("active") is not False
        and re.fullmatch(r"me\d{2}[abc]?\.pdf", str(locator.get("source") or "").lower())
        and locator.get("title")
        and as_int(locator.get("start_page")) is not None
    )


def load_ambiguous_groups(path: Path, min_locations: int) -> list[dict]:
    locators = json.loads(path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for locator in locators:
        if not isinstance(locator, dict) or not is_article(locator):
            continue
        groups[norm(locator.get("title"))].append(locator)

    ambiguous = []
    for title_key, items in groups.items():
        locations = {
            (
                item.get("source"),
                as_int(item.get("start_page")),
                as_int(item.get("end_page")),
            )
            for item in items
        }
        if len(locations) < min_locations:
            continue
        title = min((item.get("title") or "" for item in items), key=len)
        ambiguous.append({"title": title, "title_key": title_key, "items": items, "location_count": len(locations)})
    filtered = []
    for group in ambiguous:
        constraints = app.constraints_from_query(
            f"《{group['title']}》在马恩全集中对应哪些可能篇目？请列出候选卷册和页码范围。"
        )
        if constraints.get("ambiguous_locator") and len(constraints.get("entries") or []) >= 2:
            filtered.append(group)
    return sorted(filtered, key=lambda item: (-item["location_count"], len(item["title"])))


def build_case(group: dict, index: int, rng: random.Random) -> dict:
    title = group["title"]
    template = rng.choice([
        "《{title}》在马恩全集中对应哪些可能篇目？请列出候选卷册和页码范围。",
        "请检索《{title}》，如果有多个同名篇目请不要只给一个结论。",
        "《{title}》出自哪里？请先判断是否存在同名歧义。",
    ])
    expected = []
    seen = set()
    for item in group["items"]:
        key = (item.get("source"), as_int(item.get("start_page")), as_int(item.get("end_page")))
        if key in seen:
            continue
        seen.add(key)
        expected.append(
            {
                "source": item.get("source"),
                "start_page": as_int(item.get("start_page")),
                "end_page": as_int(item.get("end_page")),
                "title": item.get("title"),
            }
        )
    return {
        "id": f"me_ambiguous_locator_{index:04d}",
        "query": template.format(title=title),
        "title": title,
        "expected_locations": expected,
    }


def location_hit(doc, expected) -> bool:
    metadata = app.normalize_metadata(doc.metadata)
    if metadata.get("source") != expected.get("source"):
        return False
    page = as_int(metadata.get("citation_page") or metadata.get("printed_page") or metadata.get("page"))
    if page is None:
        return False
    return expected["start_page"] <= page <= expected["end_page"]


def constraint_location_hit(entry, expected) -> bool:
    if entry.get("source") != expected.get("source"):
        return False
    start = as_int(entry.get("start_page"))
    end = as_int(entry.get("end_page"))
    return start == expected.get("start_page") and end == expected.get("end_page")


def evaluate_case(case: dict, docs: list, constraints: dict) -> dict:
    hit_locations = []
    for expected in case["expected_locations"]:
        if any(location_hit(doc, expected) for doc in docs):
            hit_locations.append(expected)
    constraint_hits = []
    for expected in case["expected_locations"]:
        if any(constraint_location_hit(entry, expected) for entry in constraints.get("entries") or []):
            constraint_hits.append(expected)
    unique_sources_pages = {
        (
            (doc.metadata or {}).get("source"),
            as_int((doc.metadata or {}).get("citation_page") or (doc.metadata or {}).get("page")),
        )
        for doc in docs
    }
    issues = []
    best_hit_count = max(len(hit_locations), len(constraint_hits))
    if best_hit_count < min(2, len(case["expected_locations"])):
        issues.append("missed_multiple_ambiguous_candidates")
    if (
        len(unique_sources_pages) < 2
        and len(constraint_hits) < 2
        and len(case["expected_locations"]) > 1
    ):
        issues.append("collapsed_to_single_candidate")
    return {
        "id": case["id"],
        "query": case["query"],
        "title": case["title"],
        "expected_count": len(case["expected_locations"]),
        "hit_count": len(hit_locations),
        "constraint_hit_count": len(constraint_hits),
        "ambiguous_locator": bool(constraints.get("ambiguous_locator")),
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "top_docs": [
            {
                "rank": rank,
                "source": (doc.metadata or {}).get("source"),
                "citation_page": (doc.metadata or {}).get("citation_page"),
                "page": (doc.metadata or {}).get("page"),
                "article": (doc.metadata or {}).get("article") or (doc.metadata or {}).get("section"),
                "match_type": (doc.metadata or {}).get("match_type"),
            }
            for rank, doc in enumerate(docs[:8], start=1)
        ],
        "constraint_entries": [
            {
                "source": entry.get("source"),
                "start_page": entry.get("start_page"),
                "end_page": entry.get("end_page"),
                "article": entry.get("article") or entry.get("classic_title"),
            }
            for entry in (constraints.get("entries") or [])[:12]
        ],
    }


def summarize(results):
    total = len(results)
    passed = sum(1 for item in results if item["status"] == "pass")
    return {
        "total": total,
        "pass": passed,
        "fail": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "issues": dict(Counter(issue for item in results for issue in item["issues"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locators", type=Path, default=DEFAULT_LOCATORS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-locations", type=int, default=2)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    groups = load_ambiguous_groups(args.locators, args.min_locations)
    if len(groups) < args.sample_size:
        sample = groups
    else:
        sample = rng.sample(groups, args.sample_size)
    cases = [build_case(group, index, rng) for index, group in enumerate(sample, start=1)]
    db = app.load_vectorstore()
    results = []
    for index, case in enumerate(cases, start=1):
        docs = app.retrieve_documents(case["query"], db, k=args.top_k)
        constraints = app.constraints_from_query(case["query"])
        result = evaluate_case(case, docs, constraints)
        results.append(result)
        print(
            f"[{result['status'].upper():4}] {index:03d}/{len(cases):03d} "
            f"hits={result['hit_count']}/{result['expected_count']} "
            f"constraint_hits={result['constraint_hit_count']} {case['title']}"
        )
        if result["issues"]:
            print(f"       issues={','.join(result['issues'])}")

    report = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_size": len(cases),
            "seed": args.seed,
            "ambiguous_group_count": len(groups),
        },
        "summary": summarize(results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    raise SystemExit(0 if report["summary"]["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
