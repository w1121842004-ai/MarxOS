import os
import pickle
import shutil
from collections import defaultdict

from build_vectorstore_from_cache import (
    ARTICLE_MAPPING,
    BOOK_MAPPING,
    OCR_CACHE_DIR,
    article_from_map,
    infer_page_metadata,
    is_plausible_for_pdf_page,
)


VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore/marx_reader")
INDEX_PKL = os.path.join(VECTORSTORE_DIR, "index.pkl")
BACKUP_PKL = os.path.join(VECTORSTORE_DIR, "index.pkl.bak")


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def cache_text_for(source, pdf_page):
    stem = source.replace(".pdf", "")
    path = os.path.join(OCR_CACHE_DIR, stem, f"page_{pdf_page}.txt")

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def suspicious_printed_pages(page_records, ignored_pages=None):
    ignored_pages = ignored_pages or set()
    suspicious = set()

    for source, rows in page_records.items():
        printed_rows = [
            (pdf_page, printed_page)
            for pdf_page, printed_page, _chapter in rows.values()
            if printed_page is not None and (source, pdf_page) not in ignored_pages
        ]
        printed_rows.sort(key=lambda item: item[0])

        previous_pdf = None
        previous_printed = None
        for pdf_page, printed_page in printed_rows:
            offset = pdf_page - printed_page
            is_suspicious = offset < -5 or offset > 140

            if not is_suspicious and previous_pdf is not None and previous_printed is not None:
                pdf_delta = pdf_page - previous_pdf
                printed_delta = printed_page - previous_printed
                is_suspicious = (
                    pdf_delta > 0
                    and (printed_delta < -3 or printed_delta > pdf_delta + 8)
                )

            if is_suspicious:
                suspicious.add((source, pdf_page))

            previous_pdf = pdf_page
            previous_printed = printed_page

    return suspicious


def all_suspicious_printed_pages(page_records):
    suspicious = set()

    while True:
        new_suspicious = suspicious_printed_pages(page_records, suspicious)
        new_suspicious -= suspicious
        if not new_suspicious:
            return suspicious
        suspicious.update(new_suspicious)


def main():
    if not os.path.exists(INDEX_PKL):
        raise FileNotFoundError(INDEX_PKL)

    if not os.path.exists(BACKUP_PKL):
        shutil.copy2(INDEX_PKL, BACKUP_PKL)

    with open(INDEX_PKL, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    repaired = 0
    missing_cache = 0
    degraded_printed_pages = 0
    page_records = defaultdict(dict)

    for doc in docstore._dict.values():
        metadata = doc.metadata
        source = metadata.get("source")
        pdf_page = as_int(metadata.get("pdf_page") or metadata.get("page"))

        if not source or pdf_page is None:
            continue

        if pdf_page in page_records[source]:
            continue

        page_text = cache_text_for(source, pdf_page)

        if page_text is None:
            missing_cache += 1
            continue

        fallback_article = ARTICLE_MAPPING.get(source, metadata.get("article", "unknown_article"))
        printed_page, chapter = infer_page_metadata(page_text, fallback_article, pdf_page)
        if not is_plausible_for_pdf_page(printed_page, pdf_page):
            printed_page = None
        page_records[source][pdf_page] = (pdf_page, printed_page, chapter)

    suspicious_pages = all_suspicious_printed_pages(page_records)

    for doc in docstore._dict.values():
        metadata = doc.metadata
        source = metadata.get("source")
        pdf_page = as_int(metadata.get("pdf_page") or metadata.get("page"))

        if not source or pdf_page is None:
            continue

        page_record = page_records.get(source, {}).get(pdf_page)
        if page_record is None:
            continue

        _pdf_page, printed_page, chapter = page_record
        if printed_page is not None and (source, pdf_page) in suspicious_pages:
            printed_page = None
            degraded_printed_pages += 1

        page = printed_page if printed_page is not None else pdf_page
        citation_page = printed_page if printed_page is not None else pdf_page
        citation_page_type = "printed_page" if printed_page is not None else "pdf_page"
        section = article_from_map(source, printed_page)
        article = chapter

        metadata["book"] = BOOK_MAPPING.get(source, metadata.get("book", source.replace(".pdf", "")))
        metadata["article"] = article
        metadata["chapter"] = chapter
        metadata["section"] = section
        metadata["pdf_page"] = pdf_page
        metadata["printed_page"] = printed_page
        metadata["citation_page"] = citation_page
        metadata["citation_page_type"] = citation_page_type
        metadata["page"] = page

        repaired += 1

    with open(INDEX_PKL, "wb") as f:
        pickle.dump((docstore, index_to_docstore_id), f)

    print(f"metadata repaired: {repaired} chunks")
    print(f"missing cache pages: {missing_cache}")
    print(f"degraded suspicious printed-page chunks: {degraded_printed_pages}")
    print(f"backup file: {BACKUP_PKL}")


if __name__ == "__main__":
    main()
