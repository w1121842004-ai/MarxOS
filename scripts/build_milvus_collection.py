from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from marxos_embeddings import (
    HuggingFaceEmbeddings,
    create_sparse_encoder,
    embedding_encode_kwargs,
    resolve_cached_model_path,
)
from rag.semantic_retrieval import build_semantic_child_documents


DEFAULT_COLLECTION = "marxos_me_passages"
DEFAULT_DIM = 1024
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_TEXT_CHARS = 2000
DEFAULT_LOG_EVERY = 1000
DEFAULT_PARAGRAPH_CACHE = ROOT_DIR / "data" / "paragraph_cache_core.jsonl"
ME_SERIES_MARKERS = ("马克思恩格斯全集", "马克思恩格斯文集", "马克思恩格斯选集")


def require_pymilvus():
    env_uri = os.getenv("MILVUS_URI")
    clear_env_uri = bool(env_uri and "://" not in env_uri)
    if clear_env_uri:
        os.environ.pop("MILVUS_URI", None)
    try:
        from pymilvus import DataType, MilvusClient
    except ImportError as exc:
        raise RuntimeError("pymilvus is not installed. Run: pip install -r requirements.txt") from exc
    finally:
        if clear_env_uri and env_uri is not None:
            os.environ["MILVUS_URI"] = env_uri
    return DataType, MilvusClient


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_me_record(record: dict) -> bool:
    book = str(record.get("book") or "")
    source = str(record.get("source") or "")
    if any(marker in book for marker in ME_SERIES_MARKERS):
        return True
    return source.startswith(("me", "mea", "mes")) and source.endswith(".pdf")


def series_from_book(book: str) -> str:
    for marker in ME_SERIES_MARKERS:
        if marker in book:
            return marker
    return "马克思恩格斯"


def volume_from_book(book: str) -> str:
    import re

    match = re.search(r"(第\d+卷[AB]?)", book or "")
    if not match:
        return ""
    return match.group(1).replace("A", "(上)").replace("B", "(下)")


def as_int(value, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def stable_id(record: dict) -> str:
    paragraph_id = str(record.get("paragraph_id") or "")
    if paragraph_id:
        return paragraph_id
    raw = "|".join(
        str(record.get(key) or "")
        for key in ["source", "pdf_page_start", "paragraph_index", "paragraph_text"]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def clip_text(text: str, limit: int) -> str:
    text = str(text or "")
    if not limit or limit <= 0 or len(text) <= limit:
        return text
    head_limit = max(int(limit * 0.75), 1)
    tail_limit = max(limit - head_limit, 0)
    tail = text[-tail_limit:] if tail_limit else ""
    return text[:head_limit] + ("\n……\n" + tail if tail else "")


def row_from_record(
    record: dict,
    embedding: list[float],
    text: str,
    sparse_embedding: dict[int, float] | None = None,
) -> dict:
    book = str(record.get("book") or "")
    paragraph_id = stable_id(record)
    is_letter = bool(record.get("is_letter") or record.get("no_page_citation"))
    citation_mode = str(record.get("citation_mode") or ("letter_title_only" if is_letter else "page"))
    row = {
        "id": paragraph_id,
        "corpus": "marx_engels",
        "author": "马克思恩格斯",
        "series": series_from_book(book),
        "book": book,
        "volume": volume_from_book(book),
        "article": str(record.get("article") or ""),
        "section": str(record.get("section") or ""),
        "source": str(record.get("source") or ""),
        "source_file": str(record.get("source") or ""),
        "paragraph_id": paragraph_id,
        "parent_paragraph_id": str(record.get("parent_paragraph_id") or paragraph_id),
        "chunk_id": paragraph_id,
        "retrieval_unit": str(record.get("retrieval_unit") or "paragraph"),
        "pdf_page_start": as_int(record.get("pdf_page_start")),
        "pdf_page_end": as_int(record.get("pdf_page_end")),
        "printed_page_start": as_int(record.get("printed_page_start")),
        "printed_page_end": as_int(record.get("printed_page_end")),
        "citation_page_start": as_int(record.get("citation_page_start")),
        "citation_page_end": as_int(record.get("citation_page_end")),
        "citation_page_type": str(record.get("citation_page_type") or ""),
        "page_type": str(record.get("page_type") or ""),
        "is_letter": is_letter,
        "citation_mode": citation_mode,
        "child_chunk_index": as_int(record.get("child_chunk_index"), 1),
        "child_chunk_total": as_int(record.get("child_chunk_total"), 1),
        "child_char_start": as_int(record.get("child_char_start")),
        "child_char_end": as_int(record.get("child_char_end")),
        "child_chunk_size": as_int(record.get("child_chunk_size") or len(text)),
        "text_hash": text_hash(text),
        "text": text,
    }
    row["embedding"] = embedding
    if sparse_embedding is not None:
        row["sparse_embedding"] = sparse_embedding
    return row


def create_schema(
    client,
    collection_name: str,
    dim: int,
    drop_existing: bool = False,
    enable_sparse: bool = False,
) -> None:
    DataType, _MilvusClient = require_pymilvus()
    if client.has_collection(collection_name):
        if not drop_existing:
            return
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
    for field in [
        "corpus",
        "author",
        "series",
        "book",
        "volume",
        "article",
        "section",
        "source",
        "source_file",
        "paragraph_id",
        "parent_paragraph_id",
        "chunk_id",
        "retrieval_unit",
        "citation_page_type",
        "page_type",
        "citation_mode",
        "text_hash",
    ]:
        schema.add_field(field, DataType.VARCHAR, max_length=1024)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    for field in [
        "pdf_page_start",
        "pdf_page_end",
        "printed_page_start",
        "printed_page_end",
        "citation_page_start",
        "citation_page_end",
        "child_chunk_index",
        "child_chunk_total",
        "child_char_start",
        "child_char_end",
        "child_chunk_size",
    ]:
        schema.add_field(field, DataType.INT64)
    schema.add_field("is_letter", DataType.BOOL)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    if enable_sparse:
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    if enable_sparse:
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def batched(items, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def records_for_unit(records, unit: str, child_chunk_size: int, child_chunk_overlap: int):
    records = list(records)
    if unit == "paragraph":
        return records
    if unit != "semantic_child":
        raise ValueError(f"unsupported retrieval unit: {unit}")

    child_records = []
    docs = build_semantic_child_documents(
        records,
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )
    for doc in docs:
        metadata = dict(doc.metadata or {})
        parent_id = str(metadata.get("parent_paragraph_id") or metadata.get("paragraph_id") or "")
        child_index = as_int(metadata.get("child_chunk_index"), 1)
        metadata["paragraph_text"] = str(doc.page_content or "")
        metadata["parent_paragraph_id"] = parent_id
        metadata["paragraph_id"] = f"{parent_id}#c{child_index:03d}"
        metadata["retrieval_unit"] = "semantic_child"
        child_records.append(metadata)
    return child_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Milvus collection for Marx/Engels passages.")
    parser.add_argument("--paragraph-cache", default=str(DEFAULT_PARAGRAPH_CACHE))
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--token", default=os.getenv("MILVUS_TOKEN", ""))
    parser.add_argument("--embedding-model", default=os.getenv("MARXOS_EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--dim", type=int, default=int(os.getenv("MILVUS_DIM", str(DEFAULT_DIM))))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("MILVUS_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))))
    parser.add_argument("--device", default=os.getenv("MARXOS_EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--sparse-provider", default=os.getenv("MILVUS_SPARSE_PROVIDER", "none"))
    parser.add_argument("--max-text-chars", type=int, default=int(os.getenv("MILVUS_MAX_TEXT_CHARS", str(DEFAULT_MAX_TEXT_CHARS))))
    parser.add_argument("--unit", choices=["paragraph", "semantic_child"], default=os.getenv("MILVUS_RETRIEVAL_UNIT", "paragraph"))
    parser.add_argument("--child-chunk-size", type=int, default=int(os.getenv("SEMANTIC_CHILD_CHUNK_SIZE", "220")))
    parser.add_argument("--child-chunk-overlap", type=int, default=int(os.getenv("SEMANTIC_CHILD_CHUNK_OVERLAP", "50")))
    parser.add_argument("--log-every", type=int, default=int(os.getenv("MILVUS_LOG_EVERY", str(DEFAULT_LOG_EVERY))))
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    _DataType, MilvusClient = require_pymilvus()
    cache_path = Path(args.paragraph_cache)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    print(f"collection: {args.collection}", flush=True)
    print(f"milvus uri: {args.uri}", flush=True)
    print(f"embedding model: {args.embedding_model}", flush=True)
    embedding_model_path = resolve_cached_model_path(args.embedding_model)
    if embedding_model_path != args.embedding_model:
        print(f"embedding local path: {embedding_model_path}", flush=True)
    print(f"embedding device: {args.device}", flush=True)
    print(f"sparse provider: {args.sparse_provider}", flush=True)
    print(f"max text chars: {args.max_text_chars}", flush=True)
    print(f"retrieval unit: {args.unit}", flush=True)
    print(f"child chunk size: {args.child_chunk_size}", flush=True)
    print(f"child chunk overlap: {args.child_chunk_overlap}", flush=True)
    print(f"log every: {args.log_every}", flush=True)
    print(f"offset: {args.offset}", flush=True)
    print(f"limit: {args.limit}", flush=True)
    print(f"paragraph cache: {cache_path}", flush=True)
    sparse_encoder = create_sparse_encoder(args.sparse_provider, args.embedding_model, device=args.device)
    if sparse_encoder is not None:
        print("sparse encoder loaded", flush=True)
    combined_encoder = getattr(sparse_encoder, "embed_dense_and_sparse_documents", None)
    embeddings = None
    if not callable(combined_encoder):
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": args.device},
            encode_kwargs={
                **embedding_encode_kwargs(args.embedding_model),
                "batch_size": args.batch_size,
            },
        )
        print("embedding model loaded", flush=True)
    else:
        print("combined dense+sparse encoder enabled", flush=True)

    client_kwargs = {"uri": args.uri}
    if args.token:
        client_kwargs["token"] = args.token
    client = MilvusClient(**client_kwargs)
    create_schema(
        client,
        args.collection,
        args.dim,
        drop_existing=args.drop_existing,
        enable_sparse=sparse_encoder is not None,
    )

    records = (record for record in read_jsonl(cache_path) if is_me_record(record))
    if args.offset:
        import itertools

        records = itertools.islice(records, args.offset, None)
    if args.limit:
        import itertools

        records = itertools.islice(records, args.limit)
    records = records_for_unit(
        records,
        unit=args.unit,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
    )

    inserted = 0
    for batch_records in batched(records, args.batch_size):
        texts = [clip_text(record.get("paragraph_text") or "", args.max_text_chars) for record in batch_records]
        if callable(combined_encoder):
            vectors, sparse_vectors = combined_encoder(texts)
        else:
            vectors = embeddings.embed_documents(texts)
            sparse_vectors = sparse_encoder.embed_documents(texts) if sparse_encoder is not None else [None] * len(texts)
        rows = [
            row_from_record(record, vector, text, sparse_vector)
            for record, vector, text, sparse_vector in zip(batch_records, vectors, texts, sparse_vectors)
        ]
        client.upsert(collection_name=args.collection, data=rows)
        previous_inserted = inserted
        inserted += len(rows)
        if args.log_every <= 0:
            continue
        crossed_log_boundary = inserted // args.log_every > previous_inserted // args.log_every
        if crossed_log_boundary:
            print(f"upserted: {inserted}", flush=True)

    client.load_collection(args.collection)
    print(f"done: {inserted} rows upserted into {args.collection}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
