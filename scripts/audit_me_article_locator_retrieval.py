from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MARXOS_HYBRID_RETRIEVAL", "0")

import app


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_LOCATORS = ROOT_DIR / "rag" / "me_article_locators.json"
DEFAULT_DATASET = ROOT_DIR / "logs" / "me_article_locator_audit_dataset_latest.json"
DEFAULT_REPORT = ROOT_DIR / "logs" / "me_article_locator_audit_report_latest.json"

QUERY_TEMPLATES = [
    ("title_summary", "《{title}》主要讨论什么？请结合马恩全集原文说明。"),
    ("title_location", "请定位《{title}》在马恩全集中的卷册和页码。"),
    ("source_context", "《{title}》出自哪里？请给出全集卷册、页码并概括上下文。"),
    ("argument_analysis", "马克思或恩格斯在《{title}》中如何展开论述？请依据原文分析。"),
    ("citation_need", "我需要引用《{title}》，请先检索全集原文并说明应查哪一卷哪几页。"),
]
LETTER_QUERY_TEMPLATES = [
    ("letter_summary", "《{title}》这封信主要谈了什么？"),
    ("letter_context", "请检索《{title}》这封信，并说明这是哪一封信。"),
    ("letter_analysis", "马克思或恩格斯在《{title}》这封信中表达了什么看法？"),
]
NON_BODY_QUERY_TEMPLATES = [
    ("non_body_location", "请检索《{title}》这个资料项，并说明它属于哪类材料。"),
    ("non_body_context", "《{title}》在马恩全集中对应什么资料项？"),
]
LETTER_DATE_RE = re.compile(r"[（(][^）)]*(?:\d{1,2}月|\d{4}年|约|初|末|左右)")
NON_BODY_TITLE_MARKERS = [
    "索引",
    "目录",
    "年表",
    "名单",
    "书目",
    "插图",
    "图版",
    "地图",
    "照片",
    "画像",
    "示意图",
    "平面图",
    "草图",
    "手稿的一页",
    "封面",
    "扉页",
    "第一页",
    "出版说明",
    "编者注",
    "译者注",
    "注释",
    "附录",
    "请柬",
    "邀请信",
    "申请书",
    "证书",
]


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", str(value or "")).lower()


def is_me_source(source: object) -> bool:
    return bool(re.fullmatch(r"me\d{2}[abc]?\.pdf", str(source or "").lower()))


def clean_title(title: object) -> str:
    return str(title or "").strip().strip("《》[]【】()（）\"'")


def is_letter_title(title: object) -> bool:
    title = str(title or "")
    if not title:
        return False
    return "致" in title


def locator_type(locator: dict, title: str) -> str:
    explicit = str(locator.get("locator_type") or "").strip()
    if explicit:
        return explicit
    if locator.get("is_letter") or is_letter_title(title):
        return "letter"
    if any(marker in title for marker in NON_BODY_TITLE_MARKERS):
        return "non_body"
    return "article"


def title_quality(title: str) -> int:
    normalized = normalize_text(title)
    chinese_chars = sum(1 for char in title if "\u4e00" <= char <= "\u9fff")
    punctuation = sum(1 for char in title if char in ".。!！?？,，;；:：…·•-—_[]()（）'\"")
    score = 100
    if len(normalized) < 5:
        score -= 50
    if len(normalized) > 60:
        score -= 20
    if punctuation / max(len(title), 1) > 0.2:
        score -= 20
    if chinese_chars == 0:
        score -= 80
    return score


def load_locators(path: Path) -> list[dict]:
    locators = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(locators, list):
        raise ValueError(f"Locator file must contain a list: {path}")
    return [
        item
        for item in locators
        if isinstance(item, dict)
        and item.get("active") is not False
        and is_me_source(item.get("source"))
        and clean_title(item.get("title"))
        and as_int(item.get("start_page")) is not None
        and as_int(item.get("end_page")) is not None
    ]


def sample_locators(
    locators: list[dict],
    rng: random.Random,
    sample_size: int,
    include_derivative: bool,
    require_pdf_start: bool,
    min_title_quality: int,
    include_letters: bool,
    letters_only: bool,
    include_non_body: bool,
    non_body_only: bool,
    include_ambiguous_titles: bool,
) -> list[dict]:
    candidates = []
    seen = set()
    title_locations = defaultdict(set)
    for locator in locators:
        title = clean_title(locator.get("title"))
        loc_type = locator_type(locator, title)
        if loc_type != "article":
            continue
        title_key = normalize_text(title)
        if not title_key:
            continue
        title_locations[title_key].add(
            (
                locator.get("source"),
                as_int(locator.get("start_page")),
                as_int(locator.get("end_page")),
            )
        )
    ambiguous_titles = {
        title_key for title_key, locations in title_locations.items()
        if len(locations) > 1
    }

    for locator in locators:
        title = clean_title(locator.get("title"))
        loc_type = locator_type(locator, title)
        is_letter = loc_type == "letter"
        is_non_body = loc_type == "non_body"
        if letters_only and not is_letter:
            continue
        if non_body_only and not is_non_body:
            continue
        if not letters_only and not non_body_only and is_non_body and not include_non_body:
            continue
        if is_letter and not (include_letters or letters_only):
            continue
        if not include_derivative and locator.get("primary") is False:
            continue
        if require_pdf_start and as_int(locator.get("pdf_start_page")) is None:
            continue
        if title_quality(title) < min_title_quality:
            continue
        if (
            not include_ambiguous_titles
            and loc_type == "article"
            and normalize_text(title) in ambiguous_titles
        ):
            continue
        key = (
            locator.get("source"),
            normalize_text(title),
            as_int(locator.get("start_page")),
            as_int(locator.get("end_page")),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(locator)

    if len(candidates) < sample_size:
        raise ValueError(
            f"Only {len(candidates)} locators match filters, cannot sample {sample_size}."
        )
    sample_locators.last_ambiguous_title_count = len(ambiguous_titles)
    sample_locators.last_candidate_count = len(candidates)
    return rng.sample(candidates, sample_size)


def build_case(locator: dict, index: int, rng: random.Random) -> dict:
    title = clean_title(locator.get("title"))
    loc_type = locator_type(locator, title)
    is_letter = loc_type == "letter"
    is_non_body = loc_type == "non_body"
    if is_letter:
        question_type, template = rng.choice(LETTER_QUERY_TEMPLATES)
    elif is_non_body:
        question_type, template = rng.choice(NON_BODY_QUERY_TEMPLATES)
    else:
        question_type, template = rng.choice(QUERY_TEMPLATES)
    source = locator.get("source")
    start_page = as_int(locator.get("start_page"))
    end_page = as_int(locator.get("end_page"))
    return {
        "id": f"me_article_audit_{index:04d}",
        "query": template.format(title=title),
        "question_type": question_type,
        "evaluation_mode": (
            "letter_locator" if is_letter
            else "non_body_locator" if is_non_body
            else "strict_article_locator"
        ),
        "source_scope": [source],
        "expected_article": title,
        "expected_source": source,
        "expected_page_range": {
            "source": source,
            "start_page": start_page,
            "end_page": end_page,
            "pdf_start_page": as_int(locator.get("pdf_start_page")),
            "pdf_end_page": as_int(locator.get("pdf_end_page")),
        },
        "locator": {
            "title": title,
            "book": locator.get("book"),
            "source": source,
            "start_page": start_page,
            "end_page": end_page,
            "level": locator.get("level"),
            "parent": locator.get("parent"),
            "primary": locator.get("primary"),
            "is_letter": is_letter,
            "locator_type": loc_type,
        },
        "is_letter": is_letter,
        "is_non_body": is_non_body,
        "locator_type": loc_type,
    }


def build_dataset(locators: list[dict], args) -> dict:
    if args.sample_size:
        sample_size = args.sample_size
    else:
        sample_size = random.SystemRandom().randint(args.min_sample, args.max_sample)
    if sample_size < 1:
        raise ValueError("--sample-size must be positive")
    if sample_size < 300 or sample_size > 500:
        raise ValueError("Independent article audit sample size must be between 300 and 500.")

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(1, 2**31 - 1)
    rng = random.Random(seed)
    sampled = sample_locators(
        locators,
        rng,
        sample_size,
        include_derivative=args.include_derivative,
        require_pdf_start=args.require_pdf_start,
        min_title_quality=args.min_title_quality,
        include_letters=args.include_letters,
        letters_only=args.letters_only,
        include_non_body=args.include_non_body,
        non_body_only=args.non_body_only,
        include_ambiguous_titles=args.include_ambiguous_titles,
    )
    cases = [build_case(locator, index, rng) for index, locator in enumerate(sampled, start=1)]
    return {
        "meta": {
            "kind": "me_article_locator_independent_audit",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "sample_size": sample_size,
            "locator_path": str(args.locators),
            "include_derivative": args.include_derivative,
            "require_pdf_start": args.require_pdf_start,
            "min_title_quality": args.min_title_quality,
            "include_letters": args.include_letters,
            "letters_only": args.letters_only,
            "include_non_body": args.include_non_body,
            "non_body_only": args.non_body_only,
            "include_ambiguous_titles": args.include_ambiguous_titles,
            "ambiguous_title_count": getattr(sample_locators, "last_ambiguous_title_count", 0),
            "candidate_count": getattr(sample_locators, "last_candidate_count", len(sampled)),
        },
        "cases": cases,
    }


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
        "entry_type": metadata.get("entry_type"),
        "snippet": " ".join(str(doc.page_content or "").split())[:220],
    }


def page_in_range(value, start, end, tolerance: int) -> bool:
    page = as_int(value)
    start = as_int(start)
    end = as_int(end)
    if page is None or start is None or end is None:
        return False
    return start - tolerance <= page <= end + tolerance


def page_range_hit(doc, expected: dict, tolerance: int) -> bool:
    metadata = app.normalize_metadata(doc.metadata)
    source = metadata.get("source")
    if source != expected.get("source"):
        return False

    if page_in_range(metadata.get("citation_page"), expected.get("start_page"), expected.get("end_page"), tolerance):
        return True
    if page_in_range(metadata.get("printed_page"), expected.get("start_page"), expected.get("end_page"), tolerance):
        return True

    pdf_start = expected.get("pdf_start_page")
    pdf_end = expected.get("pdf_end_page")
    if pdf_start is not None and pdf_end is not None:
        return page_in_range(metadata.get("pdf_page"), pdf_start, pdf_end, tolerance)
    if pdf_start is not None:
        return page_in_range(metadata.get("pdf_page"), pdf_start, pdf_start, tolerance)
    return False


def evaluate_case(case: dict, docs: list, page_tolerance: int) -> dict:
    expected = case["expected_page_range"]
    expected_source = expected.get("source")
    top_docs = [summarize_doc(doc, rank) for rank, doc in enumerate(docs, start=1)]

    source_hit_rank = None
    page_hit_rank = None
    letter_title_hit_rank = None
    expected_title_norm = normalize_text(case.get("expected_article"))
    for rank, doc in enumerate(docs, start=1):
        metadata = app.normalize_metadata(doc.metadata)
        if source_hit_rank is None and metadata.get("source") == expected_source:
            source_hit_rank = rank
        doc_title_norm = normalize_text(
            metadata.get("letter_title")
            or metadata.get("locator_title")
            or metadata.get("article")
            or metadata.get("section")
        )
        if (
            letter_title_hit_rank is None
            and expected_title_norm
            and doc_title_norm
            and (expected_title_norm in doc_title_norm or doc_title_norm in expected_title_norm)
        ):
            letter_title_hit_rank = rank
        if page_hit_rank is None and page_range_hit(doc, expected, page_tolerance):
            page_hit_rank = rank

    constraints = app.constraints_from_query(case["query"])
    constraint_sources = sorted(constraints.get("sources") or [])
    constraint_page_ranges = {
        source: ranges for source, ranges in (constraints.get("page_ranges") or {}).items()
    }
    constraint_source_hit = expected_source in constraint_sources
    is_letter = bool(case.get("is_letter"))
    is_non_body = bool(case.get("is_non_body"))
    constraint_range_hit = is_letter
    for start, end in constraint_page_ranges.get(expected_source, []):
        if (
            page_in_range(expected.get("start_page"), start, end, page_tolerance)
            or page_in_range(start, expected.get("start_page"), expected.get("end_page"), page_tolerance)
        ):
            constraint_range_hit = True

    issues = []
    warnings = []
    if not constraint_source_hit:
        issues.append("constraint_missed_source")
    if not is_letter and not is_non_body and not constraint_range_hit:
        issues.append("constraint_missed_page_range")
    if is_letter and source_hit_rank is None:
        warnings.append("retrieval_missed_letter_source")
    elif source_hit_rank is None:
        issues.append("retrieval_missed_source")
    if is_letter and letter_title_hit_rank is None:
        issues.append("retrieval_missed_letter_title")
    if is_non_body and page_hit_rank is None:
        warnings.append("retrieval_missed_non_body_page_range")
    elif not is_letter and page_hit_rank is None:
        issues.append("retrieval_missed_page_range")

    return {
        "id": case["id"],
        "query": case["query"],
        "question_type": case["question_type"],
        "expected_source": expected_source,
        "expected_article": case.get("expected_article"),
        "expected_page_range": expected,
        "is_letter": is_letter,
        "is_non_body": is_non_body,
        "locator_type": case.get("locator_type"),
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "warnings": warnings,
        "constraint_source_hit": constraint_source_hit,
        "constraint_range_hit": constraint_range_hit,
        "constraint_sources": constraint_sources,
        "constraint_page_ranges": constraint_page_ranges,
        "constraint_flags": {
            "article_locator": bool(constraints.get("article_locator")),
            "high_precision_locator": bool(constraints.get("high_precision_locator")),
            "explicit_volume": bool(constraints.get("explicit_volume")),
            "strict_title": bool(constraints.get("strict_title")),
        },
        "source_hit_rank": source_hit_rank,
        "page_hit_rank": page_hit_rank,
        "letter_title_hit_rank": letter_title_hit_rank,
        "top_docs": top_docs,
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item["status"] == "pass")
    issue_counts = Counter(issue for item in results for issue in item["issues"])
    warning_counts = Counter(warning for item in results for warning in item.get("warnings", []))
    by_type = {}
    for question_type, count in Counter(item["question_type"] for item in results).items():
        type_passed = sum(
            1 for item in results if item["question_type"] == question_type and item["status"] == "pass"
        )
        by_type[question_type] = {
            "total": count,
            "pass": type_passed,
            "pass_rate": round(type_passed / count, 4) if count else 0,
        }
    return {
        "total": total,
        "pass": passed,
        "fail": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "by_type": by_type,
        "issues": dict(issue_counts),
        "warnings": dict(warning_counts),
        "constraint_source_hit": sum(1 for item in results if item["constraint_source_hit"]),
        "constraint_range_hit": sum(1 for item in results if item["constraint_range_hit"]),
        "source_hit_at_1": sum(1 for item in results if item.get("source_hit_rank") == 1),
        "source_hit_at_k": sum(1 for item in results if item.get("source_hit_rank") is not None),
        "page_range_hit_at_1": sum(1 for item in results if item.get("page_hit_rank") == 1),
        "page_range_hit_at_k": sum(1 for item in results if item.get("page_hit_rank") is not None),
        "letter_cases": sum(1 for item in results if item.get("is_letter")),
        "non_body_cases": sum(1 for item in results if item.get("is_non_body")),
        "letter_title_hit_at_k": sum(
            1 for item in results
            if item.get("is_letter") and item.get("letter_title_hit_rank") is not None
        ),
    }


def run_audit(dataset: dict, args) -> dict:
    db = app.load_vectorstore()
    results = []
    cases = dataset["cases"]
    for index, case in enumerate(cases, start=1):
        docs = app.retrieve_documents(
            case["query"],
            db,
            k=max(args.top_k * 4, args.top_k),
            allow_exact_quote=False,
        )[: args.top_k]
        result = evaluate_case(case, docs, page_tolerance=args.page_tolerance)
        results.append(result)
        top1 = result["top_docs"][0] if result["top_docs"] else {}
        print(
            f"[{result['status'].upper():4}] {index:03d}/{len(cases):03d} "
            f"{case['id']} src_rank={result['source_hit_rank']} "
            f"page_rank={result['page_hit_rank']} top1={top1.get('source')}:{top1.get('citation_page')} "
            f"{case['query'][:80]}",
            flush=True,
        )
        if result["issues"]:
            print(f"       issues={','.join(result['issues'])}", flush=True)

    return {
        "meta": {
            **dataset["meta"],
            "top_k": args.top_k,
            "page_tolerance": args.page_tolerance,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": summarize(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Random independent audit for ME article locator retrieval."
    )
    parser.add_argument("--locators", type=Path, default=DEFAULT_LOCATORS)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--min-sample", type=int, default=300)
    parser.add_argument("--max-sample", type=int, default=500)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--page-tolerance", type=int, default=2)
    parser.add_argument("--include-derivative", action="store_true")
    parser.add_argument("--include-letters", action="store_true")
    parser.add_argument("--letters-only", action="store_true")
    parser.add_argument("--include-non-body", action="store_true")
    parser.add_argument("--non-body-only", action="store_true")
    parser.add_argument("--include-ambiguous-titles", action="store_true")
    parser.add_argument("--require-pdf-start", action="store_true")
    parser.add_argument("--min-title-quality", type=int, default=40)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    if args.min_sample < 300 or args.max_sample > 500 or args.min_sample > args.max_sample:
        raise ValueError("--min-sample/--max-sample must stay within 300..500.")

    locators = load_locators(args.locators)
    dataset = build_dataset(locators, args)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": str(args.dataset_output),
                "sample_size": dataset["meta"]["sample_size"],
                "seed": dataset["meta"]["seed"],
                "generate_only": args.generate_only,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if args.generate_only:
        return 0

    report = run_audit(dataset, args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nSummary:")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
