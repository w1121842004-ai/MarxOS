"""
⚠️ DEPRECATED: v1 vectorstore build script — use v2 pipeline instead.

This module's ``main()`` builds a FAISS index by chunking OCR pages directly,
producing chunks WITHOUT ``parent_paragraph_id`` metadata.  Those chunks cannot
be expanded to paragraph windows by ``expand_semantic_parent_docs()``.

**Use the v2 pipeline instead:**

.. code-block:: bash

    python scripts/build_paragraph_cache.py       # Step 1
    python scripts/build_semantic_child_vectorstore.py  # Step 2
    python scripts/build_paragraph_vectorstore.py       # Step 3

The utility functions in this module (``document_from_cache``,
``page_num_from_cache_file``, ``BOOK_MAPPING``, ``is_me_volume``,
``infer_page_metadata*``, ``load_cleaned_cache_page``, ``iter_cache_files``)
remain in active use by other modules and are NOT deprecated.
"""

import os
import re
import shutil
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from marxos.embeddings import HuggingFaceEmbeddings, embedding_encode_kwargs
from marxos.config import get_settings
try:
    from clean_ocr_text import clean_ocr_page
except ModuleNotFoundError:
    from rag.clean_ocr_text import clean_ocr_page
try:
    from page_number_detection import margin_page_candidates
except ModuleNotFoundError:
    from rag.page_number_detection import margin_page_candidates


SETTINGS = get_settings()
OCR_CACHE_DIR = SETTINGS.corpus.ocr_cache_dir
VECTORSTORE_DIR = SETTINGS.index.vectorstore_dir
TEMP_VECTORSTORE_DIR = f"{VECTORSTORE_DIR}_tmp"
ARTICLE_MAP_PATH = SETTINGS.corpus.article_map_path
ARTICLE_MAP_EXTRA_PATHS = SETTINGS.corpus.article_map_extra_paths
PARAGRAPH_CACHE_PATH = SETTINGS.corpus.paragraph_cache_path

EMBEDDING_MODEL = SETTINGS.models.embedding_model

CHUNK_SIZE = SETTINGS.retrieval.semantic_child_chunk_size
CHUNK_OVERLAP = SETTINGS.retrieval.semantic_child_chunk_overlap
MIN_TEXT_LENGTH = 20
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
READ_PROGRESS_EVERY = int(os.getenv("READ_PROGRESS_EVERY", "5000"))
SPLIT_PROGRESS_EVERY = int(os.getenv("SPLIT_PROGRESS_EVERY", "5000"))

# Optional environment variables:
# ME_VOLUMES_ONLY=1 only builds from me01-me50 style caches.
# TARGET_PDFS=mea01.pdf,mes01.pdf only builds selected cached PDFs.
# SKIP_PDFS=capital.pdf,foo.pdf skips selected cached PDFs.
ME_VOLUMES_ONLY = os.getenv("ME_VOLUMES_ONLY") == "1"
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


BOOK_MAPPING = {
    "20210302c.pdf": "马克思恩格斯文献资料 20210302c",
    "capital.pdf": "资本论 第一卷",
    "letter01.pdf": "马克思恩格斯书信集 第1卷",
    "letter02.pdf": "马克思恩格斯书信集 第2卷",
    "letter03.pdf": "马克思恩格斯书信集 第3卷",
    "letter04.pdf": "马克思恩格斯书信集 第4卷",
    "me01.pdf": "马克思恩格斯全集 第1卷",
    "me01a.pdf": "马克思恩格斯全集 第1卷A",
    "me01b.pdf": "马克思恩格斯全集 第1卷B",
    "me02.pdf": "马克思恩格斯全集 第2卷",
    "me03.pdf": "马克思恩格斯全集 第3卷",
    "me04.pdf": "马克思恩格斯全集 第4卷",
    "me05.pdf": "马克思恩格斯全集 第5卷",
    "me06.pdf": "马克思恩格斯全集 第6卷",
    "me07.pdf": "马克思恩格斯全集 第7卷",
    "me08.pdf": "马克思恩格斯全集 第8卷",
    "me09.pdf": "马克思恩格斯全集 第9卷",
    "me10.pdf": "马克思恩格斯全集 第10卷",
    "me11.pdf": "马克思恩格斯全集 第11卷",
    "me12.pdf": "马克思恩格斯全集 第12卷",
    "me13.pdf": "马克思恩格斯全集 第13卷",
    "me14a.pdf": "马克思恩格斯全集 第14卷A",
    "me14b.pdf": "马克思恩格斯全集 第14卷B",
    "me15.pdf": "马克思恩格斯全集 第15卷",
    "me16.pdf": "马克思恩格斯全集 第16卷",
    "me17.pdf": "马克思恩格斯全集 第17卷",
    "me18.pdf": "马克思恩格斯全集 第18卷",
    "me19.pdf": "马克思恩格斯全集 第19卷",
    "me20.pdf": "马克思恩格斯全集 第20卷",
    "me21.pdf": "马克思恩格斯全集 第21卷",
    "me22.pdf": "马克思恩格斯全集 第22卷",
    "me23.pdf": "马克思恩格斯全集 第23卷",
    "me24.pdf": "马克思恩格斯全集 第24卷",
    "me25a.pdf": "马克思恩格斯全集 第25卷A",
    "me25b.pdf": "马克思恩格斯全集 第25卷B",
    "me26a.pdf": "马克思恩格斯全集 第26卷A",
    "me26b.pdf": "马克思恩格斯全集 第26卷B",
    "me26c.pdf": "马克思恩格斯全集 第26卷C",
    "me27.pdf": "马克思恩格斯全集 第27卷",
    "me28a.pdf": "马克思恩格斯全集 第28卷A",
    "me28b.pdf": "马克思恩格斯全集 第28卷B",
    "me29.pdf": "马克思恩格斯全集 第29卷",
    "me30a.pdf": "马克思恩格斯全集 第30卷A",
    "me30b.pdf": "马克思恩格斯全集 第30卷B",
    "me31a.pdf": "马克思恩格斯全集 第31卷A",
    "me31b.pdf": "马克思恩格斯全集 第31卷B",
    "me32.pdf": "马克思恩格斯全集 第32卷",
    "me33.pdf": "马克思恩格斯全集 第33卷",
    "me34.pdf": "马克思恩格斯全集 第34卷",
    "me35.pdf": "马克思恩格斯全集 第35卷",
    "me36.pdf": "马克思恩格斯全集 第36卷",
    "me37.pdf": "马克思恩格斯全集 第37卷",
    "me38.pdf": "马克思恩格斯全集 第38卷",
    "me39a.pdf": "马克思恩格斯全集 第39卷A",
    "me39b.pdf": "马克思恩格斯全集 第39卷B",
    "me40.pdf": "马克思恩格斯全集 第40卷",
    "me41.pdf": "马克思恩格斯全集 第41卷",
    "me42.pdf": "马克思恩格斯全集 第42卷",
    "me43.pdf": "马克思恩格斯全集 第43卷",
    "me44.pdf": "马克思恩格斯全集 第44卷",
    "me45.pdf": "马克思恩格斯全集 第45卷",
    "me46a.pdf": "马克思恩格斯全集 第46卷A",
    "me46b.pdf": "马克思恩格斯全集 第46卷B",
    "me47.pdf": "马克思恩格斯全集 第47卷",
    "me48.pdf": "马克思恩格斯全集 第48卷",
    "me49.pdf": "马克思恩格斯全集 第49卷",
    "me50.pdf": "马克思恩格斯全集 第50卷",
    "me1-39-index.pdf": "马克思恩格斯全集 第1-39卷索引",
    "me40-50-index.pdf": "马克思恩格斯全集 第40-50卷索引",
    "mea01.pdf": "马克思恩格斯文集 第1卷",
    "mea02.pdf": "马克思恩格斯文集 第2卷",
    "mea03.pdf": "马克思恩格斯文集 第3卷",
    "mea04.pdf": "马克思恩格斯文集 第4卷",
    "mea05.pdf": "马克思恩格斯文集 第5卷",
    "mea06.pdf": "马克思恩格斯文集 第6卷",
    "mea07.pdf": "马克思恩格斯文集 第7卷",
    "mea08.pdf": "马克思恩格斯文集 第8卷",
    "mea09.pdf": "马克思恩格斯文集 第9卷",
    "mea10.pdf": "马克思恩格斯文集 第10卷",
    "mega1-mega2.pdf": "马克思恩格斯全集 MEGA1-MEGA2 对照资料",
    "meid.pdf": "马克思恩格斯全集人名索引",
    "men1-39-index.pdf": "马克思恩格斯全集 第1-39卷名目索引",
    "mes01.pdf": "马克思恩格斯选集 第1卷",
    "mes02.pdf": "马克思恩格斯选集 第2卷",
    "mes03.pdf": "马克思恩格斯选集 第3卷",
    "mes04.pdf": "马克思恩格斯选集 第4卷",
}

ARTICLE_MAPPING = {
    filename: title for filename, title in BOOK_MAPPING.items()
}

FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def normalize_digits(text):
    return text.translate(FULLWIDTH_DIGITS)


def strip_pdf_boilerplate(text):
    for marker in ["本PDF文件", "S22PDF", "pdf@", "home.icm.ac.cn"]:
        text = text.split(marker)[0]

    return text


def is_valid_printed_page(page):
    if page <= 0 or page > 1200:
        return False

    # Years are frequent in titles and notes; they are not printed page numbers.
    if 1700 <= page <= 2099:
        return False

    return True


def is_plausible_for_pdf_page(printed_page, pdf_page):
    if printed_page is None or pdf_page is None:
        return True

    # OCR cache page numbers include front matter and inserted plates, so a
    # moderate offset is normal. Hundreds of pages of drift usually means a
    # date, note number, or S22PDF artifact was mistaken for a page number.
    return -80 <= pdf_page - printed_page <= 220


def infer_reversed_footer_page(normalized_text, pdf_page):
    if pdf_page is None:
        return None

    tail = normalized_text[max(0, len(normalized_text) - 80):]
    matches = list(re.finditer(r"(\d{3,4})(?=\D{0,30}$)", tail))

    for match in reversed(matches):
        raw = match.group(1)
        page = int(raw[::-1].lstrip("0") or "0")

        if is_valid_printed_page(page) and is_plausible_for_pdf_page(page, pdf_page):
            return page

    return None


def sanitize_article(article, fallback_article):
    if not article:
        return fallback_article

    for marker in ["本PDF文件", "S22PDF", "pdf@", "home.icm.ac.cn"]:
        article = article.split(marker)[0]

    article = article.strip(" ，。；：:、（）()[]【】")

    if not article:
        return fallback_article

    return article[:40]


def infer_page_metadata(text, fallback_article, pdf_page=None):
    normalized_text = normalize_digits(strip_pdf_boilerplate(text))
    footer_page = infer_reversed_footer_page(normalized_text, pdf_page)

    if footer_page is not None:
        return footer_page, fallback_article

    def candidate_from_margin_line(line):
        compact = re.sub(r"\s+", "", line.strip())
        if not compact:
            return None

        match = re.fullmatch(r"[/\\_. -]*(\d{1,4})[/\\_. -]*", compact)
        if match:
            return int(match.group(1)), None

        return None

    lines = [line for line in normalized_text.splitlines() if line.strip()]
    margin_lines = lines[:4] + lines[-4:]
    candidates = []

    for line in margin_lines:
        candidate = candidate_from_margin_line(line)
        if candidate is None:
            continue

        page, article = candidate
        if not is_valid_printed_page(page):
            continue

        if pdf_page is not None and not is_plausible_for_pdf_page(page, pdf_page):
            continue

        candidates.append((page, article))

    if not candidates:
        return None, fallback_article

    printed_page, article = candidates[0]
    article = sanitize_article(article, fallback_article)

    return printed_page, article


def infer_page_metadata_from_layout(cleaned_page, fallback_article, pdf_page=None):
    """Prefer OCR layout metadata when available.

    New OCR cache files keep header/footer text separate. That is safer than
    guessing from the full page text because body notes often contain page-like
    numbers.
    """
    explicit_candidates = cleaned_page.get("page_number_candidates") or []
    candidates = []

    for candidate in explicit_candidates:
        page = candidate.get("printed_page")
        if page is None:
            continue
        try:
            page = int(page)
        except (TypeError, ValueError):
            continue
        if is_valid_printed_page(page) and is_plausible_for_pdf_page(page, pdf_page):
            candidates.append((page, fallback_article, "ocr_layout"))

    if not candidates and (cleaned_page.get("header_text") or cleaned_page.get("footer_text")):
        for candidate in margin_page_candidates(
            cleaned_page.get("header_text", ""),
            cleaned_page.get("footer_text", ""),
            pdf_page=pdf_page,
        ):
            candidates.append((candidate["printed_page"], fallback_article, "ocr_layout"))

    if candidates:
        printed_page, article, source = candidates[0]
        return printed_page, article, source

    printed_page, article = infer_page_metadata(
        cleaned_page.get("cleaned_text") or cleaned_page.get("raw_text") or "",
        fallback_article,
        pdf_page,
    )
    return printed_page, article, "text_margin"


def load_article_map():
    paths = [ARTICLE_MAP_PATH]
    paths.extend(path for path in ARTICLE_MAP_EXTRA_PATHS.split(os.pathsep) if path)

    merged = {}
    for index, path in enumerate(paths):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if index == 0:
            merged.update(data)
        else:
            for source, payload in data.items():
                merged.setdefault(source, payload)
    return merged


ARTICLE_MAP = load_article_map()


def article_from_map(source, printed_page):
    if printed_page is None:
        return None

    source_map = ARTICLE_MAP.get(source)

    if not source_map:
        return None

    hits = []

    for entry in source_map.get("entries", []):
        start = entry.get("start_printed_page")
        end = entry.get("end_printed_page")

        if start is None or end is None:
            continue

        if start <= printed_page <= end:
            hits.append(entry)

    if not hits:
        return None

    hits.sort(key=lambda entry: (entry["end_printed_page"] - entry["start_printed_page"], -len(entry["title"])))

    return hits[0]["title"]


def printed_page_source_is_untrusted(source):
    """Some source families expose manuscript/note page marks as bare OCR numbers."""
    stem = (source or "").lower().replace(".pdf", "")
    return stem.startswith(("mea", "mes"))


def infer_page_from_sequence(
    source,
    pdf_page,
    printed_page,
    page_source,
    page_sequence_context,
    max_gap=3,
    min_run_for_correction=3,
):
    """Fill short page-number gaps from the previous trusted page in the same source."""
    previous = page_sequence_context.get(source)
    trusted_page_source = page_source in {"ocr_layout", "text_margin"}

    if previous:
        previous_pdf_page = previous.get("pdf_page")
        previous_printed_page = previous.get("printed_page")
        previous_run_length = previous.get("run_length", 1)

        if previous_pdf_page is not None and previous_printed_page is not None:
            gap = pdf_page - previous_pdf_page
            expected_page = previous_printed_page + gap

            if 0 < gap <= max_gap and is_valid_printed_page(expected_page):
                if printed_page is None and is_plausible_for_pdf_page(expected_page, pdf_page):
                    printed_page = expected_page
                    page_source = "page_sequence"
                elif (
                    (not trusted_page_source or printed_page_source_is_untrusted(source))
                    and
                    printed_page != expected_page
                    and previous_run_length >= min_run_for_correction
                    and is_plausible_for_pdf_page(expected_page, pdf_page)
                ):
                    printed_page = expected_page
                    page_source = "page_sequence_corrected"

    if printed_page is not None:
        run_length = 1
        if previous:
            previous_pdf_page = previous.get("pdf_page")
            previous_printed_page = previous.get("printed_page")
            previous_run_length = previous.get("run_length", 1)
            if previous_pdf_page is not None and previous_printed_page is not None:
                if pdf_page - previous_pdf_page == printed_page - previous_printed_page:
                    run_length = previous_run_length + 1

        page_sequence_context[source] = {
            "pdf_page": pdf_page,
            "printed_page": printed_page,
            "page_source": page_source,
            "run_length": run_length,
        }

    return printed_page, page_source


def is_me_volume(filename):
    stem = filename.lower().replace(".pdf", "")

    if not re.fullmatch(r"me\d{2}[abc]?", stem):
        return False

    volume = int(stem[2:4])

    return 1 <= volume <= 50


def page_num_from_cache_file(path):
    match = re.search(r"page_(\d+)\.(?:txt|json)$", path)

    if not match:
        return None

    return int(match.group(1))


def cache_json_path(cache_path):
    return re.sub(r"\.txt$", ".json", cache_path)


def load_cleaned_cache_page(cache_path, source, page_num, book):
    json_path = cache_json_path(cache_path)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            page = json.load(f)

        if "cleaned_text" in page and "page_type" in page:
            return page

    with open(cache_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return clean_ocr_page(
        raw_text,
        source=source,
        page_num=page_num,
        book_title=book,
    )


def iter_cache_files():
    cache_files = []

    for root, dirs, files in os.walk(OCR_CACHE_DIR):
        for filename in files:
            if filename.endswith(".txt"):
                cache_files.append(os.path.join(root, filename))

    return sorted(
        cache_files,
        key=lambda path: (
            os.path.basename(os.path.dirname(path)),
            page_num_from_cache_file(path) or 0,
        ),
    )


def document_from_cache(cache_path, title_context, page_sequence_context=None):
    if page_sequence_context is None:
        page_sequence_context = {}

    page_num = page_num_from_cache_file(cache_path)

    if page_num is None:
        return None

    source_stem = os.path.basename(os.path.dirname(cache_path))
    source = f"{source_stem}.pdf"

    if source in SKIP_PDFS:
        return None

    if TARGET_PDFS and source not in TARGET_PDFS:
        return None

    if ME_VOLUMES_ONLY and not is_me_volume(source):
        return None

    book = BOOK_MAPPING.get(source, source_stem)
    cleaned_page = load_cleaned_cache_page(cache_path, source, page_num, book)
    text = (cleaned_page.get("cleaned_text") or "").strip()
    page_type = cleaned_page.get("page_type") or "body"
    title_candidate = cleaned_page.get("title_candidate")
    author_candidate = cleaned_page.get("author_candidate")

    if page_type == "title_page":
        page_sequence_context.pop(source, None)
        if title_candidate:
            title_context[source] = {
                "title": title_candidate,
                "author": author_candidate,
                "pdf_page": page_num,
            }
        return None

    if page_type == "toc":
        return None

    if len(text) < MIN_TEXT_LENGTH:
        return None

    fallback_article = ARTICLE_MAPPING.get(source, "未知篇目")
    printed_page, chapter, page_source = infer_page_metadata_from_layout(cleaned_page, fallback_article, page_num)
    if not is_plausible_for_pdf_page(printed_page, page_num):
        printed_page = None
    printed_page, page_source = infer_page_from_sequence(
        source,
        page_num,
        printed_page,
        page_source,
        page_sequence_context,
    )
    page = printed_page if printed_page is not None else page_num
    citation_page = printed_page if printed_page is not None else page_num
    citation_page_type = "printed_page" if printed_page is not None else "pdf_page"
    section = article_from_map(source, printed_page)
    title_page_info = title_context.get(source) or {}
    title_page_title = title_page_info.get("title")
    article = section or title_page_title or chapter
    chapter = title_page_title or chapter

    return Document(
        page_content=text,
        metadata={
            "book": book,
            "article": article,
            "chapter": chapter,
            "section": section,
            "page": page,
            "printed_page": printed_page,
            "pdf_page": page_num,
            "citation_page": citation_page,
            "citation_page_type": citation_page_type,
            "page_number_source": page_source,
            "page_number_candidates": cleaned_page.get("page_number_candidates") or [],
            "source": source,
            "ocr": True,
            "page_type": page_type,
            "raw_page_available": bool(cleaned_page.get("raw_text")),
            "title_from_title_page": title_page_title,
            "title_page_pdf_page": title_page_info.get("pdf_page"),
            "author_from_title_page": title_page_info.get("author"),
            "cleaning_reasons": ",".join(cleaned_page.get("reasons") or []),
            "text_source": cleaned_page.get("text_source") or "legacy_ocr",
        },
    )


def main():
    import warnings
    warnings.warn(
        "build_vectorstore_from_cache.py (v1) is deprecated. "
        "Use the v2 pipeline instead:\n"
        "  python scripts/build_paragraph_cache.py\n"
        "  python scripts/build_semantic_child_vectorstore.py\n"
        "  python scripts/build_paragraph_vectorstore.py\n"
        "v1 chunks lack parent_paragraph_id and cannot be expanded "
        "to paragraph windows.",
        DeprecationWarning,
        stacklevel=2,
    )
    all_docs = []
    title_context = {}
    page_sequence_context = {}
    cache_files = iter_cache_files()
    total_cache_files = len(cache_files)

    print(f"扫描到缓存文件：{total_cache_files}", flush=True)

    for index, cache_path in enumerate(cache_files, start=1):
        doc = document_from_cache(cache_path, title_context, page_sequence_context)

        if doc is not None:
            all_docs.append(doc)

        if READ_PROGRESS_EVERY > 0 and index % READ_PROGRESS_EVERY == 0:
            percent = index / total_cache_files * 100
            print(
                f"读取缓存进度：{index}/{total_cache_files} files ({percent:.2f}%)，有效页：{len(all_docs)}",
                flush=True,
            )

    if not all_docs:
        raise RuntimeError("没有可用 OCR 缓存，请先运行 rag/ocr_to_cache.py")

    print(f"读取缓存完成：有效页 {len(all_docs)}", flush=True)
    print("开始切分 chunk...", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "。",
            "！",
            "？",
            "\n\n",
            "\n",
            "；",
            "，",
            "",
        ],
    )

    chunks = []
    total_docs = len(all_docs)

    for index, doc in enumerate(all_docs, start=1):
        chunks.extend(splitter.split_documents([doc]))

        if SPLIT_PROGRESS_EVERY > 0 and index % SPLIT_PROGRESS_EVERY == 0:
            percent = index / total_docs * 100
            print(
                f"Chunk 进度：{index}/{total_docs} pages ({percent:.2f}%)，当前 chunks：{len(chunks)}",
                flush=True,
            )

    print(f"读取缓存页数：{len(all_docs)}", flush=True)
    print(f"总 chunk 数量：{len(chunks)}", flush=True)
    print(f"Embedding 批大小：{BATCH_SIZE}", flush=True)
    print("开始加载 embedding 模型...", flush=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs=embedding_encode_kwargs(EMBEDDING_MODEL),
    )

    print("embedding 模型加载完成，开始构建 FAISS...", flush=True)

    if os.path.exists(TEMP_VECTORSTORE_DIR):
        shutil.rmtree(TEMP_VECTORSTORE_DIR)

    vectorstore = None
    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_chunks)
        batch = chunks[start:end]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(
                batch,
                embeddings,
            )
        else:
            vectorstore.add_documents(batch)

        percent = end / total_chunks * 100
        print(f"Embedding 进度：{end}/{total_chunks} chunks ({percent:.2f}%)", flush=True)

    if vectorstore is None:
        raise RuntimeError("没有可用于构建向量库的 chunk")

    vectorstore.save_local(TEMP_VECTORSTORE_DIR)

    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)

    shutil.move(TEMP_VECTORSTORE_DIR, VECTORSTORE_DIR)

    print(f"\n===== MarxOS 知识库构建完成：{VECTORSTORE_DIR} =====", flush=True)


if __name__ == "__main__":
    main()
