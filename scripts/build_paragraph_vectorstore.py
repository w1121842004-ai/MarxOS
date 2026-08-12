from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_community.vectorstores import FAISS
from marxos.embeddings import HuggingFaceEmbeddings, embedding_encode_kwargs
from marxos.config import get_settings

from rag.paragraph_cache import paragraph_record_to_document, read_paragraph_cache  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SETTINGS = get_settings()
EMBEDDING_MODEL = SETTINGS.models.embedding_model
PARAGRAPH_CACHE_PATH = Path(os.getenv("PARAGRAPH_CACHE_PATH", SETTINGS.corpus.paragraph_cache_path))
PARAGRAPH_VECTORSTORE_DIR = Path(SETTINGS.index.paragraph_vectorstore_dir)
TEMP_VECTORSTORE_DIR = Path(f"{PARAGRAPH_VECTORSTORE_DIR}_tmp")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))


def main() -> None:
    if not PARAGRAPH_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"paragraph cache not found: {PARAGRAPH_CACHE_PATH}. "
            "Run scripts/build_paragraph_cache.py first."
        )

    records = read_paragraph_cache(PARAGRAPH_CACHE_PATH)
    documents = [
        paragraph_record_to_document(record)
        for record in records
        if record.get("paragraph_text")
    ]

    if not documents:
        raise RuntimeError("no paragraph documents available")

    print(f"paragraph cache: {PARAGRAPH_CACHE_PATH}", flush=True)
    print(f"paragraph documents: {len(documents)}", flush=True)
    print(f"vectorstore: {PARAGRAPH_VECTORSTORE_DIR}", flush=True)
    print(f"batch size: {BATCH_SIZE}", flush=True)
    print("loading embedding model...", flush=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs=embedding_encode_kwargs(EMBEDDING_MODEL),
    )

    if TEMP_VECTORSTORE_DIR.exists():
        shutil.rmtree(TEMP_VECTORSTORE_DIR)

    vectorstore = None
    total = len(documents)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = documents[start:end]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        percent = end / total * 100
        print(f"embedding progress: {end}/{total} paragraphs ({percent:.2f}%)", flush=True)

    if vectorstore is None:
        raise RuntimeError("failed to build paragraph vectorstore")

    vectorstore.save_local(TEMP_VECTORSTORE_DIR)

    if PARAGRAPH_VECTORSTORE_DIR.exists():
        shutil.rmtree(PARAGRAPH_VECTORSTORE_DIR)

    shutil.move(TEMP_VECTORSTORE_DIR, PARAGRAPH_VECTORSTORE_DIR)
    print(f"paragraph vectorstore built: {PARAGRAPH_VECTORSTORE_DIR}", flush=True)


if __name__ == "__main__":
    main()
