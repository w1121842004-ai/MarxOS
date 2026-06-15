from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["MARXOS_HYBRID_RETRIEVAL"] = "0"

import app


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DATASET_PATH = ROOT_DIR / "eval_dataset_me_200.json"
REPORT_PATH = ROOT_DIR / "logs" / "enterprise_eval_dataset_results.json"


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", str(value or "")).lower()


def expected_sources(case: dict) -> set[str]:
    sources = {str(item).strip() for item in case.get("source_scope") or [] if str(item).strip()}
    for citation in case.get("expected_citations") or []:
        source = str((citation or {}).get("source") or "").strip()
        if source:
            sources.add(source)
    return sources


def expected_pages(case: dict) -> list[tuple[str, int | None]]:
    pages = []
    for citation in case.get("expected_citations") or []:
        source = str((citation or {}).get("source") or "").strip()
        page = as_int((citation or {}).get("citation_page"))
        if source:
            pages.append((source, page))
    return pages


def summarize_doc(doc, rank: int) -> dict:
    metadata = app.normalize_metadata(doc.metadata)
    return {
        "rank": rank,
        "source": metadata.get("source"),
        "book": metadata.get("book"),
        "article": metadata.get("article") or metadata.get("section"),
        "citation_page": metadata.get("citation_page"),
        "printed_page": metadata.get("printed_page"),
        "pdf_page": metadata.get("pdf_page"),
        "match_type": metadata.get("match_type"),
        "hybrid_source": metadata.get("hybrid_source"),
        "snippet": " ".join(str(doc.page_content or "").split())[:220],
    }


def is_me_volume_source(source: object) -> bool:
    return bool(re.fullmatch(r"me\d{2}[abc]?\.pdf", str(source or "").lower()))


def filter_docs_by_corpus(docs: list, corpus: str) -> list:
    if corpus == "all":
        return docs
    if corpus == "me":
        return [doc for doc in docs if is_me_volume_source((doc.metadata or {}).get("source"))]
    allowed = {item.strip() for item in corpus.split(",") if item.strip()}
    if allowed:
        return [doc for doc in docs if (doc.metadata or {}).get("source") in allowed]
    return docs


def source_hit_rank(docs: list, sources: set[str]) -> int | None:
    if not sources:
        return None
    for rank, doc in enumerate(docs, start=1):
        if (doc.metadata or {}).get("source") in sources:
            return rank
    return None


def page_hit_rank(docs: list, expected: list[tuple[str, int | None]], tolerance: int) -> int | None:
    expected = [(source, page) for source, page in expected if page is not None]
    if not expected:
        return None
    for rank, doc in enumerate(docs, start=1):
        metadata = app.normalize_metadata(doc.metadata)
        source = metadata.get("source")
        candidates = [
            as_int(metadata.get("citation_page")),
            as_int(metadata.get("printed_page")),
            as_int(metadata.get("pdf_page")),
        ]
        for expected_source, expected_page in expected:
            if source != expected_source:
                continue
            if any(page is not None and abs(page - expected_page) <= tolerance for page in candidates):
                return rank
    return None


def quote_hit(docs: list, case: dict) -> bool | None:
    quotes = [
        normalize_text((citation or {}).get("quote"))
        for citation in case.get("expected_citations") or []
        if normalize_text((citation or {}).get("quote"))
    ]
    if not quotes:
        return None
    combined = normalize_text("\n".join(str(doc.page_content or "") for doc in docs))
    return any(quote and quote in combined for quote in quotes)


def evaluate_case(case: dict, docs: list, page_tolerance: int) -> dict:
    evaluation_mode = case.get("evaluation_mode") or "strict_citation"
    sources = expected_sources(case)
    pages = expected_pages(case)
    src_rank = source_hit_rank(docs, sources)
    pg_rank = page_hit_rank(docs, pages, page_tolerance)
    q_hit = quote_hit(docs, case)

    issues = []
    warnings = []
    if evaluation_mode in {"strict_citation", "source_required"} and sources and src_rank is None:
        issues.append("missing_expected_source")
    if evaluation_mode == "source_preferred" and sources and src_rank is None:
        warnings.append("missing_preferred_source")
    if evaluation_mode == "strict_citation" and any(page is not None for _source, page in pages) and pg_rank is None:
        issues.append("missing_expected_page")
    if evaluation_mode == "strict_citation" and q_hit is False:
        issues.append("missing_expected_quote")

    status = "pass" if not issues else "fail"
    return {
        "id": case.get("id"),
        "query": case.get("query"),
        "question_type": case.get("question_type"),
        "evaluation_mode": evaluation_mode,
        "difficulty": case.get("difficulty"),
        "discipline": case.get("discipline"),
        "expected_sources": sorted(sources),
        "expected_pages": [{"source": source, "page": page} for source, page in pages],
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "source_hit_rank": src_rank,
        "page_hit_rank": pg_rank,
        "quote_hit": q_hit,
        "top_docs": [summarize_doc(doc, rank) for rank, doc in enumerate(docs, start=1)],
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for result in results if result["status"] == "pass")
    failed = total - passed
    by_type = {}
    for question_type, group_total in Counter(result.get("question_type") for result in results).items():
        group_pass = sum(
            1
            for result in results
            if result.get("question_type") == question_type and result["status"] == "pass"
        )
        by_type[question_type] = {
            "total": group_total,
            "pass": group_pass,
            "pass_rate": round(group_pass / group_total, 4) if group_total else 0,
        }
    return {
        "total": total,
        "pass": passed,
        "fail": failed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "by_type": by_type,
        "by_mode": {
            mode: {
                "total": group_total,
                "pass": group_pass,
                "pass_rate": round(group_pass / group_total, 4) if group_total else 0,
            }
            for mode, group_total in Counter(result.get("evaluation_mode") for result in results).items()
            for group_pass in [
                sum(
                    1
                    for result in results
                    if result.get("evaluation_mode") == mode and result["status"] == "pass"
                )
            ]
        },
        "issues": dict(Counter(issue for result in results for issue in result["issues"])),
        "warnings": dict(Counter(warning for result in results for warning in result.get("warnings", []))),
        "source_hit_at_1": sum(1 for result in results if result.get("source_hit_rank") == 1),
        "source_hit_at_k": sum(1 for result in results if result.get("source_hit_rank") is not None),
        "page_hit_at_k": sum(1 for result in results if result.get("page_hit_rank") is not None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the enterprise Marx/Engels eval dataset.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="comma-separated case ids to evaluate")
    parser.add_argument("--page-tolerance", type=int, default=2)
    parser.add_argument(
        "--corpus",
        default="me",
        help="Filter retrieved docs before scoring: me, all, or comma-separated source filenames.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.ids:
        wanted_ids = {case_id.strip() for case_id in args.ids.split(",") if case_id.strip()}
        dataset = [case for case in dataset if case.get("id") in wanted_ids]
    if args.limit:
        dataset = dataset[: args.limit]

    db = app.load_vectorstore()
    results = []
    for index, case in enumerate(dataset, start=1):
        query = case["query"]
        docs = app.retrieve_documents(query, db, k=max(args.top_k * 4, args.top_k), allow_exact_quote=False)
        docs = filter_docs_by_corpus(docs, args.corpus)[: args.top_k]
        result = evaluate_case(case, docs, page_tolerance=args.page_tolerance)
        results.append(result)
        top1 = result["top_docs"][0] if result["top_docs"] else {}
        print(
            f"[{result['status'].upper():4}] {index:03d}/{len(dataset):03d} "
            f"{case.get('id')} src_rank={result['source_hit_rank']} page_rank={result['page_hit_rank']} "
            f"top1={top1.get('source')}:{top1.get('citation_page')} {query[:80]}",
            flush=True,
        )
        if result["issues"]:
            print(f"       issues={','.join(result['issues'])}", flush=True)

    summary = summarize(results)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
