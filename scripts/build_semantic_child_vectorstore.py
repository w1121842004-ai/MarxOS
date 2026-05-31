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
from marxos_embeddings import HuggingFaceEmbeddings

from rag.paragraph_cache import read_paragraph_cache  # noqa: E402
from rag.semantic_retrieval import build_semantic_child_documents  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PARAGRAPH_CACHE_PATH = Path(os.getenv("PARAGRAPH_CACHE_PATH", "data/paragraph_cache_core.jsonl"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "vectorstore/marx_reader_core"))
TEMP_VECTORSTORE_DIR = Path(f"{VECTORSTORE_DIR}_tmp")
CHUNK_SIZE = int(os.getenv("SEMANTIC_CHILD_CHUNK_SIZE", "180"))
CHUNK_OVERLAP = int(os.getenv("SEMANTIC_CHILD_CHUNK_OVERLAP", "40"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))


def main() -> None:
    if not PARAGRAPH_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"paragraph cache not found: {PARAGRAPH_CACHE_PATH}. "
            "Run scripts/build_paragraph_cache.py first."
        )

    records = read_paragraph_cache(PARAGRAPH_CACHE_PATH)
    if not records:
        raise RuntimeError("no paragraph records available")

    documents = build_semantic_child_documents(
        records,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    if not documents:
        raise RuntimeError("no semantic child documents available")

    print(f"paragraph cache: {PARAGRAPH_CACHE_PATH}", flush=True)
    print(f"paragraph records: {len(records)}", flush=True)
    print(f"semantic child chunks: {len(documents)}", flush=True)
    print(f"vectorstore: {VECTORSTORE_DIR}", flush=True)
    print(f"chunk size: {CHUNK_SIZE}", flush=True)
    print(f"chunk overlap: {CHUNK_OVERLAP}", flush=True)
    print(f"batch size: {BATCH_SIZE}", flush=True)
    print("loading embedding model...", flush=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

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
        print(f"embedding progress: {end}/{total} chunks", flush=True)

    if vectorstore is None:
        raise RuntimeError("vectorstore build did not create an index")

    vectorstore.save_local(TEMP_VECTORSTORE_DIR)
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    shutil.move(TEMP_VECTORSTORE_DIR, VECTORSTORE_DIR)
    print(f"semantic child vectorstore built: {VECTORSTORE_DIR}", flush=True)


if __name__ == "__main__":
    main()
