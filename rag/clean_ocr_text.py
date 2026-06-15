import re


HEADER_FOOTER_PATTERNS = [
    r"^[-—–]?\s*\d{1,4}\s*[-—–]?$",
    r"^第\s*\d+\s*页$",
    r"^马克思恩格斯(文集|选集|全集).{0,12}$",
    r"^中文马克思主义文库$",
    r"^www\..+$",
    r"^http[s]?://.+$",
]

PDF_BOILERPLATE_MARKERS = [
    "本PDF文件",
    "S22PDF",
    "pdf@",
    "home.icm.ac.cn",
    "pdfFactory",
    "fineprint",
]

AUTHOR_NAMES = [
    "马克思",
    "恩格斯",
    "马克思恩格斯",
    "卡·马克思",
    "弗·恩格斯",
    "卡.马克思",
    "弗.恩格斯",
]

TITLE_KEYWORDS = [
    "共产党宣言",
    "资本论",
    "哥达纲领批判",
    "黑格尔法哲学批判",
    "反杜林论",
    "自然辩证法",
    "家庭、私有制和国家的起源",
    "社会主义从空想到科学的发展",
    "路易·波拿巴的雾月十八日",
    "法兰西内战",
    "德意志意识形态",
    "1844年经济学哲学手稿",
    "费尔巴哈",
]

LEADER_CHARS_RE = re.compile(r"[.．·•⋯…]{3,}")
PAGE_RANGE_RE = re.compile(r"(?<!\d)\d{1,4}\s*[-—–]\s*\d{1,4}(?!\d)")
LINE_END_PAGE_RE = re.compile(r"(?<!\d)\d{1,4}(?!\d)\s*$")


def normalize_ocr_text(text):
    text = (text or "").replace("\u3000", " ")
    for marker in PDF_BOILERPLATE_MARKERS:
        text = text.split(marker)[0]
    text = text.replace("⋯", "…")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    return text


def normalize_roman_heading(line):
    line = re.sub(r"^111(?=[.．、\s])", "III", line)
    line = re.sub(r"^11(?=[.．、\s])", "II", line)

    return line


def normalized_lines(text):
    lines = []

    for raw_line in normalize_ocr_text(text).splitlines():
        line = normalize_roman_heading(raw_line.strip())
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    return lines


def is_header_footer_line(line):
    compact = re.sub(r"\s+", "", line)

    for marker in PDF_BOILERPLATE_MARKERS:
        if marker in line:
            return True

    return any(re.fullmatch(pattern, compact) or re.fullmatch(pattern, line) for pattern in HEADER_FOOTER_PATTERNS)


def clean_toc_line(line):
    line = re.sub(r"(?<=\d)\s*[一—–]\s*(?=\d)", "-", line)
    line = re.sub(r"(?:[.．·•…]\s*){3,}", " ", line)
    line = LEADER_CHARS_RE.sub(" ", line)
    line = re.sub(r"\s+[，,;；:\"“”']+\s*(?=\d{1,4}(?:\s*[-—–]\s*\d{1,4})?$)", " ", line)
    line = re.sub(r"[,:，：;；]{2,}", " ", line)
    line = re.sub(r"\s+", " ", line)

    return line.strip(" .．·•…，,;；")


def clean_body_line(line):
    line = clean_toc_line(line)
    line = re.sub(r"[�□■]+", "", line)
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def toc_score(lines):
    if not lines:
        return 0, []

    text = "\n".join(lines)
    leader_count = sum(1 for line in lines if LEADER_CHARS_RE.search(line))
    range_count = len(PAGE_RANGE_RE.findall(text))
    ending_page_count = sum(1 for line in lines if LINE_END_PAGE_RE.search(line))
    toc_label = any(line in {"目录", "目次"} or "目录" in line[:8] or "目次" in line[:8] for line in lines)
    reasons = []
    score = 0

    if toc_label:
        score += 3
        reasons.append("toc_label")

    if leader_count >= 3:
        score += 3
        reasons.append("many_dot_leaders")
    elif leader_count >= 1:
        score += 1
        reasons.append("dot_leader")

    if range_count >= 2:
        score += 3
        reasons.append("multiple_page_ranges")
    elif range_count == 1:
        score += 1
        reasons.append("page_range")

    if ending_page_count >= 5:
        score += 2
        reasons.append("many_line_end_pages")

    return score, reasons


def extract_title_candidate(lines):
    candidates = []

    for line in lines[:12]:
        line = clean_body_line(line)

        if not line:
            continue

        if is_header_footer_line(line):
            continue

        if line in {"目录", "目次"}:
            continue

        if line in AUTHOR_NAMES:
            continue

        if len(line) > 80:
            continue

        has_title_marker = "《" in line or "》" in line or any(keyword in line for keyword in TITLE_KEYWORDS)
        looks_like_title = len(line) <= 40 and not re.search(r"[。！？；]", line)

        if has_title_marker or looks_like_title:
            candidates.append(line.strip())

    return candidates[0] if candidates else None


def extract_author_candidate(lines):
    text = "\n".join(lines[:12])

    for author in AUTHOR_NAMES:
        if author in text:
            return author

    return None


def is_title_page(lines, title_candidate, author_candidate):
    text = "\n".join(lines)
    compact_len = len(re.sub(r"\s+", "", text))

    if compact_len > 260 or len(lines) > 16:
        return False

    if author_candidate and title_candidate:
        return True

    if title_candidate and any(keyword in title_candidate for keyword in TITLE_KEYWORDS):
        return True

    if compact_len <= 120 and title_candidate and ("《" in text or "》" in text):
        return True

    return False


def clean_ocr_page(raw_text, source=None, page_num=None, book_title=None):
    lines = normalized_lines(raw_text)
    score, reasons = toc_score(lines)
    title_candidate = extract_title_candidate(lines)
    author_candidate = extract_author_candidate(lines)
    title_page = is_title_page(lines, title_candidate, author_candidate)

    if not lines:
        page_type = "blank"
    elif score >= 4:
        page_type = "toc"
    elif title_page:
        page_type = "title_page"
        reasons.append("short_title_page")
    else:
        page_type = "body"

    if page_type == "toc":
        title_candidate = None
        author_candidate = None

    cleaned_lines = []
    for line in lines:
        if is_header_footer_line(line):
            continue

        cleaned = clean_toc_line(line) if page_type == "toc" else clean_body_line(line)

        if cleaned:
            cleaned_lines.append(cleaned)

    cleaned_text = "\n".join(cleaned_lines)

    return {
        "raw_text": raw_text or "",
        "cleaned_text": cleaned_text,
        "page_type": page_type,
        "is_toc": page_type == "toc",
        "is_title_page": page_type == "title_page",
        "title_candidate": title_candidate,
        "author_candidate": author_candidate,
        "source": source,
        "page_num": page_num,
        "book_title": book_title,
        "reasons": reasons,
    }


def clean_text(text):
    return clean_ocr_page(text)["cleaned_text"]
