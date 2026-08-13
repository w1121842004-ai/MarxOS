from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from typing import Any, Protocol
import warnings

from langchain_core.documents import Document
from marxos.config import get_settings


SETTINGS = get_settings()


@dataclass
class VectorSearchResult:
    document: Document
    score: float | None = None
    backend: str = ""


class VectorBackend(Protocol):
    def search(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[VectorSearchResult]:
        ...


DEFAULT_OUTPUT_FIELDS = [
    # Legacy scalar fields (marxos_text_layer_* collections).
    "id",
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
    "pdf_page_start",
    "pdf_page_end",
    "printed_page_start",
    "printed_page_end",
    "citation_page_start",
    "citation_page_end",
    "citation_page_type",
    "page_type",
    "is_letter",
    "citation_mode",
    "child_chunk_index",
    "child_chunk_total",
    "child_char_start",
    "child_char_end",
    "child_chunk_size",
    "text_hash",
    # Corpus-v2 contract fields (marxos_passages_v2 collection).
    "work_id",
    "edition_id",
    "publisher",
    "publication_year",
    "record_id",
    "document_record_version",
    "retrieval_record_version",
    "metadata_schema_version",
    "build_id",
    "content_class",
    "quality_status",
    "text_source",
    "source_text_hash",
    "indexed_text_hash",
    "text_was_clipped",
    "retrievable",
    "text",
]


def milvus_filter_expr(filters: dict[str, Any] | None) -> str:
    """Convert simple equality/list filters to a Milvus boolean expression."""
    filters = filters or {}
    parts = []
    for key, value in filters.items():
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple, set)):
            values = [item for item in value if item not in (None, "")]
            if not values:
                continue
            rendered = ", ".join(_quote_filter_value(item) for item in values)
            parts.append(f"{key} in [{rendered}]")
        else:
            parts.append(f"{key} == {_quote_filter_value(value)}")
    return " and ".join(parts)


def _quote_filter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class MilvusVectorBackend:
    """Thin Milvus search adapter used after the collection has been built."""

    def __init__(
        self,
        client,
        collection_name: str,
        embedding_model,
        sparse_embedding_model=None,
        vector_field: str = "embedding",
        sparse_vector_field: str = "sparse_embedding",
        output_fields: list[str] | None = None,
        overfetch_factor: int | None = None,
        hybrid_enabled: bool | None = None,
        collection_loaded: bool = False,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.vector_field = vector_field
        self.sparse_vector_field = sparse_vector_field
        self._loaded = collection_loaded
        self._has_sparse_vector_field: bool | None = None
        self._hybrid_warning_emitted = False
        self.overfetch_factor = overfetch_factor or int(os.getenv("MILVUS_OVERFETCH_FACTOR", "4"))
        self.query_vector_cache_size = max(int(os.getenv("MILVUS_QUERY_VECTOR_CACHE_SIZE", "128")), 0)
        self._dense_query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._sparse_query_cache: OrderedDict[str, dict[int, float]] = OrderedDict()
        self._hybrid_query_cache: OrderedDict[str, tuple[list[float], dict[int, float]]] = OrderedDict()
        self.hybrid_enabled = (
            hybrid_enabled
            if hybrid_enabled is not None
            else os.getenv(
                "MILVUS_HYBRID_SEARCH",
                os.getenv("MARXOS_MILVUS_HYBRID", "1" if SETTINGS.index.milvus_hybrid_search else "0"),
            ).lower()
            in {"1", "true", "yes", "on"}
        )
        self.output_fields = output_fields or DEFAULT_OUTPUT_FIELDS
        self._output_fields_reconciled = bool(output_fields)

    def _reconcile_output_fields(self) -> list[str]:
        """Drop output fields the target collection does not define.

        The v2 corpus collection replaces ``text_hash`` with dual source/index
        hashes and adds bibliography fields; the legacy collection has none of
        those. Requesting a missing field makes Milvus reject the whole search,
        so reconcile once against the live schema.
        """
        if self._output_fields_reconciled:
            return self.output_fields
        try:
            description = self.client.describe_collection(self.collection_name)
            fields = description.get("fields", []) if isinstance(description, dict) else []
            actual = {field.get("name") for field in fields if isinstance(field, dict) and field.get("name")}
            if actual:
                kept = [field for field in self.output_fields if field in actual]
                if kept:
                    self.output_fields = kept
        except Exception:
            pass
        self._output_fields_reconciled = True
        return self.output_fields

    def search(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[VectorSearchResult]:
        if not self._loaded:
            self.client.load_collection(self.collection_name)
            self._loaded = True
        self._reconcile_output_fields()
        search_limit = max(k, min(max(k * self.overfetch_factor, k + 12), 500))
        hybrid_capable = self._can_hybrid_search()
        if hybrid_capable:
            query_vector, sparse_vector = self._cached_hybrid_query_vectors(query)
        else:
            query_vector = self._cached_dense_query_vector(query)
            sparse_vector = None
        results = self._search_results(query_vector, sparse_vector, search_limit, filters)
        hits = results[0] if results else []
        converted = [self._hit_to_result(hit) for hit in hits]
        filtered = [
            result for result in converted
            if not self._is_noise_document(result.document)
        ]
        return (filtered or converted)[:k]

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        filters = kwargs.get("filters") or kwargs.get("filter")
        return [result.document for result in self.search(query, k=k, filters=filters)]

    def prewarm(self, query: str = "马克思主义", search: bool = False) -> None:
        if not self._loaded:
            self.client.load_collection(self.collection_name)
            self._loaded = True
        if self._can_hybrid_search():
            self._cached_hybrid_query_vectors(query)
        else:
            self._cached_dense_query_vector(query)
        if search:
            self.search(query, k=1)

    def _search_results(
        self,
        query_vector: list[float],
        sparse_vector: dict[int, float] | None,
        search_limit: int,
        filters: dict[str, Any] | None,
    ):
        filter_expr = milvus_filter_expr(filters) or ""
        if sparse_vector:
            try:
                return self._hybrid_search_results(query_vector, sparse_vector, search_limit, filter_expr)
            except Exception as exc:
                if not self._hybrid_warning_emitted:
                    warnings.warn(f"Milvus hybrid search failed; falling back to dense search: {exc}")
                    self._hybrid_warning_emitted = True

        return self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field=self.vector_field,
            limit=search_limit,
            filter=filter_expr,
            output_fields=self.output_fields,
        )

    def _hybrid_search_results(
        self,
        query_vector: list[float],
        sparse_vector: dict[int, float],
        search_limit: int,
        filter_expr: str,
    ):
        from pymilvus import AnnSearchRequest, RRFRanker

        dense_request = AnnSearchRequest(
            data=[query_vector],
            anns_field=self.vector_field,
            param={"metric_type": "COSINE", "params": {"ef": max(search_limit, 64)}},
            limit=search_limit,
            filter=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_vector],
            anns_field=self.sparse_vector_field,
            param={"metric_type": "IP", "params": {}},
            limit=search_limit,
            filter=filter_expr,
        )
        return self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(),
            limit=search_limit,
            output_fields=self.output_fields,
        )

    def _cached_dense_query_vector(self, query: str) -> list[float]:
        cached = self._dense_query_cache.get(query)
        if cached is not None:
            self._dense_query_cache.move_to_end(query)
            return cached
        vector = self.embedding_model.embed_query(query)
        self._cache_put(self._dense_query_cache, query, vector)
        return vector

    def _cached_sparse_query_vector(self, query: str) -> dict[int, float]:
        cached = self._sparse_query_cache.get(query)
        if cached is not None:
            self._sparse_query_cache.move_to_end(query)
            return cached
        vector = self.sparse_embedding_model.embed_query(query)
        self._cache_put(self._sparse_query_cache, query, vector)
        return vector

    def _cached_hybrid_query_vectors(self, query: str) -> tuple[list[float], dict[int, float]]:
        cached = self._hybrid_query_cache.get(query)
        if cached is not None:
            self._hybrid_query_cache.move_to_end(query)
            return cached
        embed_pair = getattr(self.sparse_embedding_model, "embed_dense_and_sparse_query", None)
        if callable(embed_pair):
            pair = embed_pair(query)
        else:
            pair = (self._cached_dense_query_vector(query), self._cached_sparse_query_vector(query))
        self._cache_put(self._hybrid_query_cache, query, pair)
        return pair

    def _cache_put(self, cache: OrderedDict, key: str, value) -> None:
        if self.query_vector_cache_size <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.query_vector_cache_size:
            cache.popitem(last=False)

    def _can_hybrid_search(self) -> bool:
        if not self.hybrid_enabled or self.sparse_embedding_model is None:
            return False
        if self._has_sparse_vector_field is None:
            self._has_sparse_vector_field = self._collection_has_field(self.sparse_vector_field)
        return bool(self._has_sparse_vector_field)

    def _collection_has_field(self, field_name: str) -> bool:
        try:
            description = self.client.describe_collection(self.collection_name)
        except Exception:
            return False
        fields = description.get("fields", []) if isinstance(description, dict) else []
        return any(field.get("name") == field_name for field in fields if isinstance(field, dict))

    def _hit_to_result(self, hit) -> VectorSearchResult:
        if isinstance(hit, dict):
            entity = hit.get("entity") or {}
            score = hit.get("distance")
        else:
            entity = getattr(hit, "entity", None) or {}
            score = getattr(hit, "score", None)
        if hasattr(entity, "to_dict"):
            entity = entity.to_dict()
        entity = dict(entity)
        metadata = {key: value for key, value in entity.items() if key != "text"}
        metadata["milvus_id"] = metadata.get("id")
        metadata["retrieval_backend"] = "milvus"
        if score is not None:
            metadata["vector_score"] = score
        self._add_legacy_metadata(metadata)
        page_content = entity.get("text") or ""
        return VectorSearchResult(
            document=Document(page_content=page_content, metadata=metadata),
            score=score,
            backend="milvus",
        )

    @staticmethod
    def _add_legacy_metadata(metadata: dict[str, Any]) -> None:
        pdf_page = metadata.get("pdf_page_start")
        printed_page = metadata.get("printed_page_start")
        citation_page = metadata.get("citation_page_start")
        if metadata.get("pdf_page") is None:
            metadata["pdf_page"] = pdf_page
        if metadata.get("page") is None:
            metadata["page"] = pdf_page
        if metadata.get("printed_page") is None:
            metadata["printed_page"] = None if printed_page in (-1, None) else printed_page
        if metadata.get("citation_page") is None:
            metadata["citation_page"] = None if citation_page in (-1, None) else citation_page
        if metadata.get("parent_paragraph_id") is None:
            metadata["parent_paragraph_id"] = metadata.get("paragraph_id")
        if metadata.get("retrieval_unit") is None:
            metadata["retrieval_unit"] = "paragraph_child"
        metadata.setdefault("child_chunk_index", 1)
        metadata.setdefault("child_chunk_total", 1)
        metadata.setdefault("child_chunk_size", metadata.get("paragraph_char_count"))
        if metadata.get("page_span") is None and pdf_page not in (-1, None):
            metadata["page_span"] = [pdf_page]

    @staticmethod
    def _is_noise_document(doc: Document) -> bool:
        metadata = doc.metadata or {}
        content = str(doc.page_content or "")
        article = str(metadata.get("article") or metadata.get("section") or "")

        if metadata.get("page_type") in {"toc", "title_page"}:
            return True
        stripped = content.strip()
        if stripped.startswith(("说明本卷", "本卷收入", "本卷是", "第一卷说明", "第二卷说明", "第三卷说明", "第四卷说明", "第五卷说明", "第六卷说明", "第七卷说明", "第八卷说明", "第九卷说明", "第十卷说明")):
            return True
        if article == metadata.get("book") and (metadata.get("citation_page") or 9999) <= 10:
            return True
        if any(marker in article for marker in ["目录", "目次", "人名索引", "名目索引", "文献索引", "报刊索引", "地名索引"]):
            return True
        if any(marker in article for marker in ["本卷中引用和提到的著作索引", "本书中引用和提到的著作索引"]):
            return True
        if len(content.strip()) < 24:
            return True
        dash_entry_count = content.count("——") + content.count("---")
        if content.count("———") >= 4 or content.count("---") >= 4:
            return True
        if dash_entry_count >= 2 and len(content) < 320:
            return True
        if "并见" in content and dash_entry_count >= 1:
            return True
        punctuation_count = sum(1 for char in content if char in ".。!！?？,，;；:：…·•-—_[]()（）")
        if content and punctuation_count / max(len(content), 1) > 0.38:
            return True
        return False
