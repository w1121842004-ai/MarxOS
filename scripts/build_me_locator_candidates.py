from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval_dataset_me_200.json"
DEFAULT_OCR_CACHE = ROOT / "data" / "ocr_cache"
DEFAULT_OUTPUT = ROOT / "rag" / "me_high_precision_locators.json"

FULLWIDTH_DIGIT_MAP = str.maketrans("０１２３４５６７８９", "0123456789")

STOPWORDS = {
    "如何",
    "什么",
    "为什么",
    "说明",
    "分析",
    "理解",
    "指出",
    "认为",
    "马克思",
    "恩格斯",
    "原文",
    "出处",
    "关系",
    "意义",
    "作用",
    "这种",
    "这个",
    "问题",
    "比较",
    "综述",
    "请以",
    "依据",
}


def normalize_text(value: object) -> str:
    text = str(value or "").translate(FULLWIDTH_DIGIT_MAP)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text).lower()


def compact_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").translate(FULLWIDTH_DIGIT_MAP)).strip()


def source_stem(source: str) -> str:
    return source.replace(".pdf", "")


def read_ocr_page(path: Path) -> str:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        return compact_text(payload.get("cleaned_text") or payload.get("raw_text") or "")
    try:
        return compact_text(path.read_text(encoding="utf-8"))
    except OSError:
        return ""


def page_num_from_path(path: Path) -> int | None:
    match = re.search(r"page_(\d+)\.(?:txt|json)$", path.name)
    return int(match.group(1)) if match else None


def find_pdf_page(source: str, citation_page: int | None, quote: str, ocr_cache: Path) -> int | None:
    source_dir = ocr_cache / source_stem(source)
    if not source_dir.exists():
        return None

    normalized_quote = normalize_text(quote)
    direct_candidates = []
    if citation_page is not None:
        direct_candidates.extend(
            [
                source_dir / f"page_{citation_page}.json",
                source_dir / f"page_{citation_page}.txt",
            ]
        )
    for path in direct_candidates:
        if path.exists() and normalized_quote and normalized_quote in normalize_text(read_ocr_page(path)):
            return page_num_from_path(path)

    if not normalized_quote:
        return citation_page

    paths = sorted(
        source_dir.glob("page_*.*"),
        key=lambda item: page_num_from_path(item) or 0,
    )
    for path in paths:
        if path.suffix.lower() not in {".json", ".txt"}:
            continue
        if normalized_quote in normalize_text(read_ocr_page(path)):
            return page_num_from_path(path)

    # If OCR has line breaks or minor cleanup differences, a shorter prefix is often enough.
    quote_prefix = normalized_quote[: max(18, min(36, len(normalized_quote)))]
    for path in paths:
        if path.suffix.lower() not in {".json", ".txt"}:
            continue
        if quote_prefix and quote_prefix in normalize_text(read_ocr_page(path)):
            return page_num_from_path(path)
    return citation_page


def quoted_terms(query: str) -> list[str]:
    terms = []
    for pattern in [r"《([^》]{2,})》", r"‘([^’]{2,})’", r"“([^”]{2,})”", r'"([^"]{2,})"']:
        terms.extend(match.group(1).strip() for match in re.finditer(pattern, query))
    return terms


def date_terms(query: str) -> list[str]:
    return re.findall(r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?", query)


def candidate_terms(query: str, tags: list[str], expected_work: str, quote: str) -> list[str]:
    query = compact_text(query)
    terms: list[str] = []
    terms.extend(date_terms(query))
    terms.extend(quoted_terms(query))
    for tag in tags or []:
        tag = compact_text(tag)
        if len(normalize_text(tag)) >= 2 and normalize_text(tag) in normalize_text(query):
            terms.append(tag)

    # Extract medium-length Chinese noun-ish spans split by common function words.
    for piece in re.split(r"[，。？！、；：\s]|如何|为何|为什么|什么|怎样|怎么|请|在|中|的|与|和|及|对|为|把|从|看|论述|分析|理解|说明|指出|认为|比较|综述", query):
        piece = piece.strip("《》“”‘’\"'")
        if 2 <= len(piece) <= 18 and piece not in STOPWORDS:
            terms.append(piece)

    if expected_work and normalize_text(expected_work) in normalize_text(query):
        terms.append(str(expected_work).strip("《》"))

    deduped = []
    seen = set()
    for term in terms:
        normalized = normalize_text(term)
        if len(normalized) < 2 or normalized in seen or term in STOPWORDS:
            continue
        seen.add(normalized)
        deduped.append(term)
    return deduped[:4]


def build_locators(dataset: list[dict], ocr_cache: Path) -> list[dict]:
    locators = []
    seen = set()
    for item in dataset:
        query = item.get("query") or ""
        expected_citations = item.get("expected_citations") or []
        if not expected_citations:
            continue
        citation = expected_citations[0] or {}
        source = str(citation.get("source") or "").strip()
        if not re.fullmatch(r"me\d{2}[abc]?\.pdf", source.lower()):
            continue
        citation_page = citation.get("citation_page")
        try:
            citation_page = int(citation_page)
        except (TypeError, ValueError):
            citation_page = None
        if citation_page is None:
            continue
        quote = str(citation.get("quote") or "")
        tokens = candidate_terms(
            query,
            item.get("tags") or [],
            item.get("expected_work") or citation.get("article") or "",
            quote,
        )
        if len(tokens) < 2:
            continue
        pdf_page = find_pdf_page(source, citation_page, quote, ocr_cache)
        key = (source, citation_page, normalize_text(query))
        if key in seen:
            continue
        seen.add(key)
        locator = {
            "id": item.get("id"),
            "active": True,
            "tokens_all": tokens,
            "title": item.get("expected_work") or citation.get("article") or source,
            "source": source,
            "page": citation_page,
            "article": citation.get("article") or item.get("expected_work") or "",
            "quote": quote,
            "query": query,
        }
        if pdf_page is not None and pdf_page != citation_page:
            locator["pdf_page"] = pdf_page
        locators.append(locator)
    locators.sort(key=lambda item: (item["source"], item["page"], item["id"] or ""))
    return locators


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ME high precision locator candidates from eval citations.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ocr-cache", type=Path, default=DEFAULT_OCR_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inactive", action="store_true", help="write candidates with active=false")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    locators = build_locators(dataset, args.ocr_cache)
    if args.inactive:
        for locator in locators:
            locator["active"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(locators, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    by_source = {}
    with_pdf = 0
    for locator in locators:
        by_source[locator["source"]] = by_source.get(locator["source"], 0) + 1
        if locator.get("pdf_page") is not None:
            with_pdf += 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": len(locators),
                "sources": len(by_source),
                "with_pdf_page_override": with_pdf,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
