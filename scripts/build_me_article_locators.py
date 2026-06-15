from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLE_MAP = ROOT / "rag" / "article_map.json"
DEFAULT_OCR_CACHE = ROOT / "data" / "ocr_cache"
DEFAULT_OUTPUT = ROOT / "rag" / "me_article_locators.json"
DEFAULT_LETTER_OUTPUT = ROOT / "rag" / "me_letter_locators.json"
DEFAULT_NON_BODY_OUTPUT = ROOT / "rag" / "me_non_body_locators.json"
LETTER_DATE_RE = re.compile(r"[（(][^）)]*(?:\d{1,2}月|\d{4}年|约|初|末|左右)")

FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

BAD_TITLE_MARKERS = [
    "本PDF文件",
    "S22PDF",
    "pdf@",
    "目录",
    "插图",
    "人名索引",
    "文献索引",
    "名目索引",
    "期刊索引",
]
DERIVATIVE_TITLE_MARKERS = [
    "封面",
    "扉页",
    "第一页",
    "手稿的一页",
    "中译文",
    "序言",
    "导言",
    "跋",
    "说明",
    "附录",
]
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


def normalize_for_match(text: object) -> str:
    text = str(text or "").translate(FULLWIDTH_DIGITS)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text).lower()


def clean_title(title: object) -> str:
    title = str(title or "").translate(FULLWIDTH_DIGITS)
    title = re.split(r"[.·•…]{3,}", title, maxsplit=1)[0]
    title = re.sub(r"^[\d\s、.．·•-]+", "", title)
    title = re.sub(r"\s+", "", title)
    title = title.strip(" ，。；：:、（）()[]【】《》“”\"'")
    return title[:120]


def is_me_source(source: str) -> bool:
    return bool(re.fullmatch(r"me\d{2}[abc]?\.pdf", str(source or "").lower()))


def is_valid_title(title: str) -> bool:
    normalized = normalize_for_match(title)
    if len(normalized) < 4:
        return False
    if normalized.isdigit():
        return False
    if any(marker in title for marker in BAD_TITLE_MARKERS):
        return False
    if re.fullmatch(r"[第上下中一二三四五六七八九十0-9卷册部分篇章节目]+", title):
        return False
    return True


def is_primary_title(title: str) -> bool:
    return not any(marker in title for marker in DERIVATIVE_TITLE_MARKERS)


def is_letter_title(title: str) -> bool:
    title = str(title or "")
    if not title:
        return False
    return "致" in title


def locator_type_for_title(title: str) -> str:
    if is_letter_title(title):
        return "letter"
    if any(marker in title for marker in NON_BODY_TITLE_MARKERS):
        return "non_body"
    return "article"


def page_num_from_path(path: Path) -> int | None:
    match = re.search(r"page_(\d+)\.(?:txt|json)$", path.name)
    return int(match.group(1)) if match else None


def read_page_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return str(payload.get("cleaned_text") or payload.get("raw_text") or "")
        return path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return ""


def infer_printed_page(text: str, pdf_page: int | None) -> int | None:
    text = str(text or "").translate(FULLWIDTH_DIGITS)
    text = text.split("本PDF文件")[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    edge_lines = lines[:4] + lines[-4:]

    candidates = []
    for line in edge_lines:
        compact = re.sub(r"\s+", "", line)
        match = re.fullmatch(r"[-—–_/\\.]*([0-9]{1,4})[-—–_/\\.]*", compact)
        if not match:
            continue
        printed = int(match.group(1))
        if printed <= 0 or printed > 1200 or 1700 <= printed <= 2099:
            continue
        if pdf_page is not None and not (-5 <= pdf_page - printed <= 160):
            continue
        candidates.append(printed)

    if candidates:
        return candidates[0]

    tail = text[max(0, len(text) - 120):]
    for match in reversed(list(re.finditer(r"([0-9]{3,4})(?=\D{0,40}$)", tail))):
        printed = int(match.group(1)[::-1].lstrip("0") or "0")
        if printed <= 0 or printed > 1200 or 1700 <= printed <= 2099:
            continue
        if pdf_page is not None and not (-5 <= pdf_page - printed <= 160):
            continue
        return printed
    return None


def build_printed_to_pdf_map(source: str, ocr_cache: Path) -> dict[int, int]:
    source_dir = ocr_cache / source.replace(".pdf", "")
    if not source_dir.exists():
        return {}

    mapping = {}
    for path in sorted(source_dir.glob("page_*.*"), key=lambda item: page_num_from_path(item) or 0):
        if path.suffix.lower() not in {".json", ".txt"}:
            continue
        pdf_page = page_num_from_path(path)
        printed = infer_printed_page(read_page_text(path), pdf_page)
        if printed is not None and printed not in mapping and pdf_page is not None:
            mapping[printed] = pdf_page
    return mapping


def nearest_pdf_page(mapping: dict[int, int], printed_page: int | None) -> int | None:
    if printed_page is None or not mapping:
        return None
    if printed_page in mapping:
        return mapping[printed_page]
    candidates = [
        (abs(printed - printed_page), printed, pdf)
        for printed, pdf in mapping.items()
        if abs(printed - printed_page) <= 3
    ]
    if not candidates:
        return None
    candidates.sort()
    distance, printed, pdf = candidates[0]
    return pdf + (printed_page - printed)


def build_locators(article_map: dict, ocr_cache: Path) -> list[dict]:
    locators = []
    seen = set()
    pdf_maps: dict[str, dict[int, int]] = {}

    for source, source_map in sorted(article_map.items()):
        if not is_me_source(source):
            continue
        pdf_maps[source] = build_printed_to_pdf_map(source, ocr_cache)
        book = source_map.get("book") or source

        for entry in source_map.get("entries") or []:
            title = clean_title(entry.get("title"))
            if not is_valid_title(title):
                continue
            start_page = entry.get("start_printed_page")
            end_page = entry.get("end_printed_page")
            try:
                start_page = int(start_page)
                end_page = int(end_page)
            except (TypeError, ValueError):
                continue
            if start_page <= 0 or end_page < start_page or end_page > 1200:
                continue
            key = (source, normalize_for_match(title), start_page, end_page)
            if key in seen:
                continue
            seen.add(key)

            mapping = pdf_maps[source]
            locator = {
                "active": True,
                "title": title,
                "source": source,
                "book": book,
                "start_page": start_page,
                "end_page": end_page,
                "level": entry.get("level"),
                "parent": clean_title(entry.get("parent")),
                "primary": is_primary_title(title),
                "aliases": [title],
            }
            locator_type = locator_type_for_title(title)
            locator["locator_type"] = locator_type
            if locator_type == "letter":
                locator["is_letter"] = True
                locator["no_page_citation"] = True
                locator["citation_mode"] = "letter_title"
            elif locator_type == "non_body":
                locator["non_body"] = True
            pdf_start = nearest_pdf_page(mapping, start_page)
            pdf_end = nearest_pdf_page(mapping, end_page)
            if pdf_start is not None:
                locator["pdf_start_page"] = pdf_start
            if pdf_end is not None:
                locator["pdf_end_page"] = pdf_end
            locators.append(locator)

    locators.sort(
        key=lambda item: (
            item["source"],
            item["start_page"],
            item.get("level") or 99,
            len(normalize_for_match(item["title"])),
        )
    )
    return locators


def main() -> int:
    parser = argparse.ArgumentParser(description="Build article-level ME locators from article_map and OCR page numbers.")
    parser.add_argument("--article-map", type=Path, default=DEFAULT_ARTICLE_MAP)
    parser.add_argument("--ocr-cache", type=Path, default=DEFAULT_OCR_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--letter-output", type=Path, default=DEFAULT_LETTER_OUTPUT)
    parser.add_argument("--non-body-output", type=Path, default=DEFAULT_NON_BODY_OUTPUT)
    args = parser.parse_args()

    article_map = json.loads(args.article_map.read_text(encoding="utf-8"))
    locators = build_locators(article_map, args.ocr_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(locators, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    letter_locators = [item for item in locators if item.get("is_letter")]
    args.letter_output.parent.mkdir(parents=True, exist_ok=True)
    args.letter_output.write_text(
        json.dumps(letter_locators, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    non_body_locators = [item for item in locators if item.get("locator_type") == "non_body"]
    args.non_body_output.parent.mkdir(parents=True, exist_ok=True)
    args.non_body_output.write_text(
        json.dumps(non_body_locators, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": len(locators),
                "sources": len({item["source"] for item in locators}),
                "with_pdf_start": sum(1 for item in locators if item.get("pdf_start_page") is not None),
                "letter_output": str(args.letter_output),
                "letters": len(letter_locators),
                "non_body_output": str(args.non_body_output),
                "non_body": len(non_body_locators),
                "articles": sum(1 for item in locators if item.get("locator_type") == "article"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
