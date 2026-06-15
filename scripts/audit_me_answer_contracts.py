from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MARXOS_HYBRID_RETRIEVAL", "0")

import app


DEFAULT_LOCATORS = ROOT_DIR / "rag" / "me_article_locators.json"
DEFAULT_REPORT = ROOT_DIR / "logs" / "me_answer_contract_audit_latest.json"


def normalize_text(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", str(value or "")).lower()


def load_locators(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if isinstance(item, dict) and item.get("active") is not False]


def sample_cases(locators: list[dict], sample_size: int, seed: int | None) -> list[dict]:
    rng = random.Random(seed if seed is not None else random.SystemRandom().randint(1, 2**31 - 1))
    articles = [
        item for item in locators
        if item.get("locator_type") == "article" and item.get("primary") is not False
    ]
    letters = [
        item for item in locators
        if item.get("locator_type") == "letter" or item.get("is_letter")
    ]
    half = max(1, sample_size // 2)
    article_sample = rng.sample(articles, min(half, len(articles)))
    letter_sample = rng.sample(letters, min(sample_size - len(article_sample), len(letters)))
    cases = []
    for locator in article_sample:
        title = locator["title"]
        cases.append(
            {
                "kind": "article",
                "title": title,
                "source": locator.get("source"),
                "query": f"《{title}》主要讨论什么？请结合马恩全集原文说明。",
            }
        )
    for locator in letter_sample:
        title = locator["title"]
        cases.append(
            {
                "kind": "letter",
                "title": title,
                "source": locator.get("source"),
                "query": f"《{title}》这封信主要谈了什么？",
            }
        )
    rng.shuffle(cases)
    return cases


def evaluate_case(case: dict, db, top_k: int) -> dict:
    docs = app.retrieve_documents(case["query"], db, k=top_k, allow_exact_quote=False)
    top = docs[0] if docs else None
    metadata = app.normalize_metadata(top.metadata) if top else {}
    citation = app.format_citation(metadata, include_article=True) if top else ""
    context = app.build_context(docs[: min(top_k, 5)], "rag_answer") if docs else ""

    issues = []
    warnings = []
    if not docs:
        issues.append("no_docs")

    if case["kind"] == "letter":
        if not metadata.get("no_page_citation"):
            issues.append("letter_missing_no_page_citation_flag")
        if "页" in citation or "页" in context.split("原文：", 1)[0]:
            issues.append("letter_context_exposes_page_citation")
        title_norm = normalize_text(case["title"])
        citation_norm = normalize_text(citation)
        if title_norm and title_norm not in citation_norm:
            issues.append("letter_title_missing_from_citation")
        if metadata.get("source") != case.get("source"):
            warnings.append("letter_source_not_top1")
    else:
        if metadata.get("no_page_citation"):
            issues.append("article_marked_no_page_citation")
        if "页" not in citation:
            issues.append("article_citation_missing_page")
        if metadata.get("source") != case.get("source"):
            warnings.append("article_source_not_top1")

    return {
        "query": case["query"],
        "kind": case["kind"],
        "expected_source": case.get("source"),
        "top_source": metadata.get("source"),
        "top_article": metadata.get("article") or metadata.get("section"),
        "citation": citation,
        "no_page_citation": bool(metadata.get("no_page_citation")),
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "warnings": warnings,
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item["status"] == "pass")
    return {
        "total": total,
        "pass": passed,
        "fail": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "by_kind": {
            kind: {
                "total": count,
                "pass": sum(1 for item in results if item["kind"] == kind and item["status"] == "pass"),
            }
            for kind, count in Counter(item["kind"] for item in results).items()
        },
        "issues": dict(Counter(issue for item in results for issue in item["issues"])),
        "warnings": dict(Counter(warning for item in results for warning in item.get("warnings", []))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit answer-side citation contracts for ME retrieval.")
    parser.add_argument("--locators", type=Path, default=DEFAULT_LOCATORS)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    locators = load_locators(args.locators)
    cases = sample_cases(locators, args.sample_size, args.seed)
    db = app.load_vectorstore()
    results = []
    for index, case in enumerate(cases, start=1):
        result = evaluate_case(case, db, args.top_k)
        results.append(result)
        print(
            f"[{result['status'].upper():4}] {index:03d}/{len(cases):03d} "
            f"{case['kind']} top={result['top_source']} {case['query'][:70]}",
            flush=True,
        )
        if result["issues"]:
            print(f"       issues={','.join(result['issues'])}", flush=True)

    report = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "top_k": args.top_k,
        },
        "summary": summarize(results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    return 0 if report["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
