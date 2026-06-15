import os
import re
import json

import fitz
from paddleocr import PaddleOCR
from pdf2image import convert_from_path, pdfinfo_from_path
try:
    from clean_ocr_text import clean_ocr_page, clean_text
except ModuleNotFoundError:
    from rag.clean_ocr_text import clean_ocr_page, clean_text

try:
    from page_number_detection import margin_page_candidates
except ModuleNotFoundError:
    from rag.page_number_detection import margin_page_candidates


DATA_DIR = "data"
OCR_CACHE_DIR = "data/ocr_cache"
DEFAULT_POPPLER_PATH = r"D:\EdgeDownload\poppler-26.02.0\Library\bin"
POPPLER_PATH = os.getenv("POPPLER_PATH", DEFAULT_POPPLER_PATH)

if POPPLER_PATH and not os.path.exists(POPPLER_PATH):
    POPPLER_PATH = None

# Optional environment variables:
# START_PAGE=100 END_PAGE=105 limits each PDF to a page range.
# PDF_NAME=me49.pdf limits the run to one PDF.
# TARGET_PDFS=mea01.pdf,mes01.pdf limits the run to selected PDFs.
# SKIP_PDFS=capital.pdf,foo.pdf skips selected PDFs.
# OVERWRITE_OCR=1 rebuilds existing cached txt files.
# FORCE_OCR=1 ignores usable PDF text layers and always runs PaddleOCR.
# TEXT_LAYER_MIN_LENGTH=80 controls when extracted PDF text is considered usable.
# USE_GPU=1 enables PaddleOCR GPU mode.
# ME_VOLUMES_ONLY=1 only processes me01-me50 style PDFs.
# SKIP_OCR_FALLBACK=1 skips pages with short/no text instead of running OCR.
# PROGRESS_EVERY=100 prints page-level progress every N pages. Use 1 for every page.
# OCR_MARGIN_RATIO=0.12 controls the top/bottom image bands used as header/footer.
START_PAGE = int(os.getenv("START_PAGE", "1"))
END_PAGE = os.getenv("END_PAGE")
END_PAGE = int(END_PAGE) if END_PAGE else None
PDF_NAME = os.getenv("PDF_NAME")
TARGET_PDFS = {
    name.strip()
    for name in os.getenv("TARGET_PDFS", "").split(",")
    if name.strip()
}
SKIP_PDFS = {
    name.strip()
    for name in os.getenv("SKIP_PDFS", "capital.pdf").split(",")
    if name.strip()
}
OVERWRITE_OCR = os.getenv("OVERWRITE_OCR") == "1"
FORCE_OCR = os.getenv("FORCE_OCR") == "1"
TEXT_LAYER_MIN_LENGTH = int(os.getenv("TEXT_LAYER_MIN_LENGTH", "80"))
USE_GPU = os.getenv("USE_GPU") == "1"
ME_VOLUMES_ONLY = os.getenv("ME_VOLUMES_ONLY") == "1"
SKIP_OCR_FALLBACK = os.getenv("SKIP_OCR_FALLBACK") == "1"
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "100"))
OCR_MARGIN_RATIO = float(os.getenv("OCR_MARGIN_RATIO", "0.12"))

DATA_DIR_ABS = os.path.abspath(DATA_DIR)
OCR_CACHE_DIR_ABS = os.path.abspath(OCR_CACHE_DIR)


def is_me_volume(filename):
    stem = filename.lower().replace(".pdf", "")

    if not re.fullmatch(r"me\d{2}[abc]?", stem):
        return False

    volume = int(stem[2:4])

    return 1 <= volume <= 50


def should_log_page(page_num):
    return PROGRESS_EVERY > 0 and page_num % PROGRESS_EVERY == 0


def get_cache_path(filename, page_num):
    safe_name = filename.replace(".pdf", "")

    return os.path.join(OCR_CACHE_DIR, safe_name, f"page_{page_num}.txt")


def get_cache_json_path(filename, page_num):
    safe_name = filename.replace(".pdf", "")

    return os.path.join(OCR_CACHE_DIR, safe_name, f"page_{page_num}.json")


def save_cached_page(filename, page_num, raw_text, book_title=None, layout_meta=None):
    cache_path = get_cache_path(filename, page_num)
    json_path = get_cache_json_path(filename, page_num)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cleaned_page = clean_ocr_page(
        raw_text,
        source=filename,
        page_num=page_num,
        book_title=book_title,
    )
    if layout_meta:
        cleaned_page.update(layout_meta)

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(cleaned_page["cleaned_text"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_page, f, ensure_ascii=False, indent=2)

    return cleaned_page


def save_cached_text(filename, page_num, text):
    return save_cached_page(filename, page_num, text)


def extract_text_layer(pdf_path, page_num):
    with fitz.open(pdf_path) as pdf:
        page = pdf.load_page(page_num - 1)
        text = page.get_text("text")

    return text


def create_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        use_gpu=USE_GPU,
        show_log=False,
    )


def bbox_center_y(bbox):
    return sum(point[1] for point in bbox) / len(bbox)


def bbox_to_list(bbox):
    return [[float(point[0]), float(point[1])] for point in bbox]


def layout_from_ocr_result(result, image_size, page_num):
    width, height = image_size
    header_limit = height * OCR_MARGIN_RATIO
    footer_limit = height * (1 - OCR_MARGIN_RATIO)
    layout_lines = []
    header_lines = []
    body_lines = []
    footer_lines = []

    if result and result[0]:
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = float(line[1][1]) if len(line[1]) > 1 else None
            center_y = bbox_center_y(bbox)
            if center_y <= header_limit:
                region = "header"
                header_lines.append(text)
            elif center_y >= footer_limit:
                region = "footer"
                footer_lines.append(text)
            else:
                region = "body"
                body_lines.append(text)

            layout_lines.append(
                {
                    "text": text,
                    "bbox": bbox_to_list(bbox),
                    "confidence": confidence,
                    "region": region,
                }
            )

    header_text = "\n".join(header_lines)
    body_text = "\n".join(body_lines)
    footer_text = "\n".join(footer_lines)

    return {
        "image_width": width,
        "image_height": height,
        "ocr_margin_ratio": OCR_MARGIN_RATIO,
        "layout_lines": layout_lines,
        "header_text": header_text,
        "body_text": body_text,
        "footer_text": footer_text,
        "page_number_candidates": margin_page_candidates(
            header_text,
            footer_text,
            pdf_page=page_num,
        ),
    }


def ocr_pdf_page(ocr, pdf_path, filename, page_num):
    if ocr is None:
        ocr = create_ocr()

    cache_path = get_cache_path(filename, page_num)

    pages = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        poppler_path=POPPLER_PATH,
    )

    image_path = f"temp_{filename}_{page_num}.png".replace(".pdf", "")
    pages[0].save(image_path, "PNG")

    try:
        result = ocr.ocr(image_path, cls=True)
        layout_meta = layout_from_ocr_result(result, pages[0].size, page_num)
        all_text = []

        all_text.extend(layout_meta["header_text"].splitlines())
        all_text.extend(layout_meta["body_text"].splitlines())
        all_text.extend(layout_meta["footer_text"].splitlines())

        raw_text = "\n".join(all_text)
        cleaned_page = save_cached_page(filename, page_num, raw_text, layout_meta=layout_meta)
        print(f"OCR完成：{filename} 第{page_num}页，字符数：{len(cleaned_page['cleaned_text'])}")

        return ocr

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


def cache_pdf_page(ocr, pdf_path, filename, page_num):
    cache_path = get_cache_path(filename, page_num)
    json_path = get_cache_json_path(filename, page_num)

    if os.path.exists(cache_path) and not OVERWRITE_OCR:
        if not os.path.exists(json_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                save_cached_page(filename, page_num, f.read())
        if should_log_page(page_num):
            print(f"缓存已存在，跳过：{filename} 第{page_num}页")
        return ocr

    if FORCE_OCR:
        print(f"FORCE_OCR: {filename} page {page_num}")
        return ocr_pdf_page(ocr, pdf_path, filename, page_num)

    extracted_text = extract_text_layer(pdf_path, page_num)
    extracted_cleaned_text = clean_text(extracted_text)

    if len(extracted_cleaned_text) >= TEXT_LAYER_MIN_LENGTH:
        cleaned_page = save_cached_page(filename, page_num, extracted_text)
        if should_log_page(page_num):
            print(f"文本层完成：{filename} 第{page_num}页，字符数：{len(cleaned_page['cleaned_text'])}")
        return ocr

    if SKIP_OCR_FALLBACK:
        if should_log_page(page_num):
            print(f"文本层过短，跳过：{filename} 第{page_num}页，字符数：{len(extracted_cleaned_text)}")
        return ocr

    print(f"文本层过短，转 OCR：{filename} 第{page_num}页，字符数：{len(extracted_cleaned_text)}")
    return ocr_pdf_page(ocr, pdf_path, filename, page_num)


def iter_pdf_files():
    for root, dirs, files in os.walk(DATA_DIR):
        root_abs = os.path.abspath(root)

        if root_abs == OCR_CACHE_DIR_ABS or root_abs.startswith(OCR_CACHE_DIR_ABS + os.sep):
            continue

        for filename in sorted(files):
            if not filename.endswith(".pdf"):
                continue

            if PDF_NAME and filename != PDF_NAME:
                continue

            if TARGET_PDFS and filename not in TARGET_PDFS:
                continue

            if ME_VOLUMES_ONLY and not is_me_volume(filename):
                continue

            if filename in SKIP_PDFS:
                print(f"跳过 PDF：{filename}")
                continue

            yield os.path.join(root, filename), filename


def main():
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)

    if START_PAGE < 1:
        raise ValueError("START_PAGE 必须大于等于 1")

    ocr = None

    for pdf_path, filename in iter_pdf_files():
        print(f"\n===== 正在处理：{filename} =====\n")

        try:
            pdf_info = pdfinfo_from_path(
                pdf_path,
                poppler_path=POPPLER_PATH,
            )
            total_pages = pdf_info["Pages"]
            last_page = END_PAGE if END_PAGE else total_pages
            last_page = min(last_page, total_pages)

            if START_PAGE > total_pages:
                print(f"跳过：{filename} 只有 {total_pages} 页，START_PAGE={START_PAGE}")
                continue

            if START_PAGE > last_page:
                print(f"跳过：{filename} 页码范围无效：{START_PAGE}-{last_page}")
                continue

            print(f"页码范围：{START_PAGE}-{last_page} / 共 {total_pages} 页")

            for page_num in range(START_PAGE, last_page + 1):
                ocr = cache_pdf_page(ocr, pdf_path, filename, page_num)

        except Exception as e:
            print(f"\nOCR失败：{filename}")
            print(e)


if __name__ == "__main__":
    main()
