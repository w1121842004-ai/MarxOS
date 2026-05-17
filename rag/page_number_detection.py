import re


FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def normalize_digits(text):
    return str(text or "").translate(FULLWIDTH_DIGITS)


def is_valid_printed_page(page):
    if page <= 0 or page > 1200:
        return False

    # Dates in headers and notes are common OCR traps.
    if 1700 <= page <= 2099:
        return False

    return True


def is_plausible_for_pdf_page(printed_page, pdf_page):
    if printed_page is None or pdf_page is None:
        return True

    return -5 <= pdf_page - printed_page <= 140


def line_page_candidates(text, pdf_page=None):
    """Extract page-number candidates from one trusted header/footer line.

    This function assumes the caller already restricted the line to the page
    margin. It still rejects obvious note references such as "第84-85页".
    """
    compact = re.sub(r"\s+", "", normalize_digits(text))
    if not compact:
        return []

    if "页" in compact or "版" in compact:
        return []

    if re.search(r"\d{1,4}\s*[-—－]\s*\d{1,4}", compact):
        return []

    candidates = []
    whole_line = re.fullmatch(r"[/\\_.\-—－ ]*(\d{1,4})[/\\_.\-—－ ]*", compact)
    edge_numbers = []

    if whole_line:
        edge_numbers.append((whole_line.group(1), "whole_line"))
    else:
        start = re.match(r"^[/\\_.\-—－ ]*(\d{1,4})(?!\d)", compact)
        end = re.search(r"(?<!\d)(\d{1,4})[/\\_.\-—－ ]*$", compact)
        if start:
            edge_numbers.append((start.group(1), "line_start"))
        if end:
            edge_numbers.append((end.group(1), "line_end"))

    for raw, reason in edge_numbers:
        page = int(raw)
        if not is_valid_printed_page(page):
            continue
        if not is_plausible_for_pdf_page(page, pdf_page):
            continue
        candidates.append({"printed_page": page, "reason": reason, "line": text})

    return candidates


def margin_page_candidates(header_text="", footer_text="", pdf_page=None):
    candidates = []

    for region, text in [("header", header_text), ("footer", footer_text)]:
        for line in str(text or "").splitlines():
            for candidate in line_page_candidates(line, pdf_page=pdf_page):
                candidate["region"] = region
                candidates.append(candidate)

    seen = set()
    unique = []
    for candidate in candidates:
        key = (candidate["printed_page"], candidate["region"], candidate["line"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return unique
