import os
import pickle
import shutil

from build_vectorstore_from_cache import (
    ARTICLE_MAPPING,
    BOOK_MAPPING,
    OCR_CACHE_DIR,
    article_from_map,
    infer_page_metadata,
    is_plausible_for_pdf_page,
)


VECTORSTORE_DIR = "vectorstore/marx_knowledge_base"
INDEX_PKL = os.path.join(VECTORSTORE_DIR, "index.pkl")
BACKUP_PKL = os.path.join(VECTORSTORE_DIR, "index.pkl.bak")


def cache_text_for(source, pdf_page):
    stem = source.replace(".pdf", "")
    path = os.path.join(OCR_CACHE_DIR, stem, f"page_{pdf_page}.txt")

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    if not os.path.exists(INDEX_PKL):
        raise FileNotFoundError(INDEX_PKL)

    if not os.path.exists(BACKUP_PKL):
        shutil.copy2(INDEX_PKL, BACKUP_PKL)

    with open(INDEX_PKL, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    repaired = 0
    missing_cache = 0

    for doc in docstore._dict.values():
        metadata = doc.metadata
        source = metadata.get("source")
        pdf_page = metadata.get("pdf_page") or metadata.get("page")

        if not source or pdf_page is None:
            continue

        page_text = cache_text_for(source, pdf_page)

        if page_text is None:
            missing_cache += 1
            continue

        fallback_article = ARTICLE_MAPPING.get(source, metadata.get("article", "未知篇目"))
        printed_page, chapter = infer_page_metadata(page_text, fallback_article, pdf_page)
        if not is_plausible_for_pdf_page(printed_page, pdf_page):
            printed_page = None
        page = printed_page if printed_page is not None else pdf_page
        section = article_from_map(source, printed_page)
        article = chapter

        metadata["book"] = BOOK_MAPPING.get(source, metadata.get("book", source.replace(".pdf", "")))
        metadata["article"] = article
        metadata["chapter"] = chapter
        metadata["section"] = section
        metadata["pdf_page"] = pdf_page
        metadata["printed_page"] = printed_page
        metadata["page"] = page

        repaired += 1

    with open(INDEX_PKL, "wb") as f:
        pickle.dump((docstore, index_to_docstore_id), f)

    print(f"metadata 修复完成：{repaired} chunks")
    print(f"缺失缓存：{missing_cache} chunks")
    print(f"备份文件：{BACKUP_PKL}")


if __name__ == "__main__":
    main()
