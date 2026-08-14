from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on"}

DEFAULT_CORPUS_PROFILE = "me_full_v2_2"
DEFAULT_RETRIEVAL_PROFILE = "milvus_bgem3_v2_2"
DEFAULT_ANSWER_PROFILE = "deepseek_default"


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in TRUE_VALUES


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def existing_path_or_fallback(primary: str, fallback: str) -> str:
    return primary if Path(primary).exists() else fallback


def first_existing_path(*paths: str) -> str:
    for path in paths:
        if Path(path).exists():
            return path
    return paths[0]


CORPUS_PROFILES = {
    "me_full": {
        "article_map_path": "rag/article_map.json",
        "article_map_extra_paths": "rag/article_map_core.json",
        "topic_catalog_path": "rag/topic_catalog.json",
        "paragraph_cache_path": existing_path_or_fallback("data/paragraph_cache.jsonl", "data/paragraph_cache_core.jsonl"),
        "semantic_parent_cache_path": existing_path_or_fallback(
            "data/semantic_parent_cache.jsonl",
            "data/semantic_parent_cache_core.jsonl",
        ),
        "ocr_cache_dir": "data/ocr_cache",
        "page_map_path": "data/page_map.json",
        "preferred_editions": ("me", "wenji", "xuanji"),
    },
    "me_full_v2": {
        # Corpus-v2 bypass rebuild (文集10卷 + 选集4卷): authoritative paragraph
        # artifacts under data/artifacts/corpus_v2 feed both parent expansion
        # and the Milvus corpus_v2 collection.
        "article_map_path": "rag/article_map.json",
        "article_map_extra_paths": "rag/article_map_core.json",
        "topic_catalog_path": "rag/topic_catalog.json",
        "paragraph_cache_path": "data/artifacts/corpus_v2/paragraph_records_enriched_v2_1.jsonl",
        "semantic_parent_cache_path": "data/artifacts/corpus_v2/paragraph_records_enriched_v2_1.jsonl",
        "ocr_cache_dir": "data/ocr_cache_text_layer",
        "page_map_path": "data/page_map.json",
        "preferred_editions": ("me", "wenji", "xuanji"),
    },
    "me_full_v2_2": {
        # Corpus v2.2 (全集75卷 + 文集10卷 + 选集4卷)：全集页码 V2 已写回缓存，
        # 权威段落产物在 data/artifacts/corpus_v2_2，父段落扩展与检索同源。
        "article_map_path": "rag/article_map.json",
        "article_map_extra_paths": "rag/article_map_core.json",
        "topic_catalog_path": "rag/topic_catalog.json",
        "paragraph_cache_path": "data/artifacts/corpus_v2_2/paragraph_records_enriched.jsonl",
        "semantic_parent_cache_path": "data/artifacts/corpus_v2_2/paragraph_records_enriched.jsonl",
        "ocr_cache_dir": "data/ocr_cache_text_layer",
        "page_map_path": "data/page_map.json",
        "preferred_editions": ("me", "wenji", "xuanji"),
    },
    "core_test": {
        "article_map_path": "rag/article_map_core.json",
        "article_map_extra_paths": "",
        "topic_catalog_path": "rag/topic_catalog.json",
        "paragraph_cache_path": "data/paragraph_cache_core.jsonl",
        "semantic_parent_cache_path": "data/semantic_parent_cache_core.jsonl",
        "ocr_cache_dir": "data/ocr_cache",
        "page_map_path": "data/page_map.json",
        "preferred_editions": ("wenji", "xuanji", "me"),
    },
}


RETRIEVAL_PROFILES = {
    "milvus_bgem3_stable": {
        "vector_backend_default": "milvus",
        "milvus_hybrid_search": True,
        "milvus_sparse_provider": "lexical",
        "semantic_parent_window": 1,
        "semantic_child_parent_window": 0,
        "semantic_child_chunk_size": 180,
        "semantic_child_chunk_overlap": 40,
        "milvus_retrieval_unit": "paragraph",
        "milvus_uri": "./data/milvus_lite/marxos_text_layer_bgem3.db",
        "milvus_collection": "marxos_text_layer_bgem3",
        "bm25_stats_path": "",
    },
    "milvus_bgem3_fast": {
        "vector_backend_default": "milvus",
        "milvus_hybrid_search": False,
        "milvus_sparse_provider": "none",
        "semantic_parent_window": 1,
        "semantic_child_parent_window": 0,
        "semantic_child_chunk_size": 180,
        "semantic_child_chunk_overlap": 40,
        "milvus_retrieval_unit": "paragraph",
        "milvus_uri": "./data/milvus_lite/marxos_text_layer_bgem3.db",
        "milvus_collection": "marxos_text_layer_bgem3",
        "bm25_stats_path": "",
    },
    "milvus_bgem3_hybrid": {
        "vector_backend_default": "milvus",
        "milvus_hybrid_search": True,
        "milvus_sparse_provider": "bge-m3",
        "semantic_parent_window": 1,
        "semantic_child_parent_window": 0,
        "semantic_child_chunk_size": 180,
        "semantic_child_chunk_overlap": 40,
        "milvus_retrieval_unit": "semantic_child",
        "milvus_uri": "./data/milvus_lite/marxos_text_layer_bgem3.db",
        "milvus_collection": "marxos_text_layer_bgem3",
        "bm25_stats_path": "",
    },
    "faiss_semantic": {
        "vector_backend_default": "faiss",
        "milvus_hybrid_search": False,
        "milvus_sparse_provider": "none",
        "semantic_parent_window": 1,
        "semantic_child_parent_window": 0,
        "semantic_child_chunk_size": 180,
        "semantic_child_chunk_overlap": 40,
        "milvus_retrieval_unit": "semantic_child",
        "milvus_uri": "./data/milvus_lite/marxos_text_layer_bgem3.db",
        "milvus_collection": "marxos_text_layer_bgem3",
        "bm25_stats_path": "",
    },
    "milvus_bgem3_v2": {
        # Promoted corpus-v2 baseline: side-by-side collection built from
        # semantic child chunks (320/64) with corpus-fitted BM25 sparse.
        "vector_backend_default": "milvus",
        "milvus_hybrid_search": True,
        "milvus_sparse_provider": "bm25",
        "semantic_parent_window": 1,
        "semantic_child_parent_window": 0,
        "semantic_child_chunk_size": 320,
        "semantic_child_chunk_overlap": 64,
        "milvus_retrieval_unit": "semantic_child",
        "milvus_uri": "./data/milvus_lite/marxos_corpus_v2.db",
        "milvus_collection": "marxos_passages_v2",
        "bm25_stats_path": "data/artifacts/corpus_v2/bm25_stats_v2_1.json",
    },
    "milvus_bgem3_v2_2": {
        # Corpus v2.2（全集 75 卷并入）：191,792 semantic child rows。
        "vector_backend_default": "milvus",
        "milvus_hybrid_search": True,
        "milvus_sparse_provider": "bm25",
        "semantic_parent_window": 1,
        "semantic_child_parent_window": 0,
        "semantic_child_chunk_size": 320,
        "semantic_child_chunk_overlap": 64,
        "milvus_retrieval_unit": "semantic_child",
        "milvus_uri": "./data/milvus_lite/marxos_corpus_v2_2.db",
        "milvus_collection": "marxos_passages_v2_2",
        "bm25_stats_path": "data/artifacts/corpus_v2_2/bm25_stats.json",
    },
}


ANSWER_PROFILES = {
    # deepseek-v4-pro: 深度分析/学术问答（deep 模式）。
    # deepseek-v4-flash: 快速/标准问答与辅助任务（定位、验证、域外问答）。
    "deepseek_default": {
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-v4-pro",
        "default_performance_mode": "deep",
        "citation_audit_enabled": True,
    },
    "deepseek_fast": {
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-v4-flash",
        "default_performance_mode": "fast",
        "citation_audit_enabled": True,
    },
    "deepseek_standard": {
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-v4-flash",
        "default_performance_mode": "standard",
        "citation_audit_enabled": True,
    },
}


def active_profile_name(env_name: str, default: str) -> str:
    return os.getenv(env_name, default).strip() or default


def profile_data(profiles: dict[str, dict], active_name: str, fallback_name: str) -> dict:
    if active_name in profiles:
        return dict(profiles[active_name])
    return dict(profiles[fallback_name])


@dataclass(frozen=True)
class ProfileSettings:
    active_corpus_profile: str
    active_retrieval_profile: str
    active_answer_profile: str


@dataclass(frozen=True)
class CorpusSettings:
    profile_name: str
    article_map_path: str
    article_map_extra_paths: str
    topic_catalog_path: str
    paragraph_cache_path: str
    semantic_parent_cache_path: str
    ocr_cache_dir: str
    page_map_path: str
    preferred_editions: tuple[str, ...]


@dataclass(frozen=True)
class ModelSettings:
    embedding_model: str
    embedding_device: str
    deepseek_base_url: str
    deepseek_model: str
    deepseek_flash_model: str = "deepseek-v4-flash"


@dataclass(frozen=True)
class IndexSettings:
    vectorstore_dir: str
    paragraph_vectorstore_dir: str
    milvus_uri: str
    milvus_collection: str
    milvus_dim: int
    milvus_batch_size: int
    milvus_sparse_provider: str
    milvus_hybrid_search: bool
    default_milvus_paragraph_cache: str
    milvus_retrieval_unit: str
    milvus_bm25_stats_path: str = ""


@dataclass(frozen=True)
class RetrievalSettings:
    profile_name: str
    semantic_parent_window: int
    semantic_child_parent_window: int
    semantic_child_chunk_size: int
    semantic_child_chunk_overlap: int
    vector_backend_default: str
    hybrid_retrieval_env: str = "MARXOS_HYBRID_RETRIEVAL"
    dual_retrieval_env: str = "MARXOS_DUAL_RETRIEVAL"
    vector_backend_env: str = "MARXOS_VECTOR_BACKEND"
    rerank_debug_env: str = "MARXOS_DEBUG_RERANK"


@dataclass(frozen=True)
class AnswerSettings:
    profile_name: str
    deepseek_base_url: str
    deepseek_model: str
    default_performance_mode: str
    citation_audit_enabled: bool


@dataclass(frozen=True)
class WebSettings:
    host: str
    port: int
    metrics_log_path: str


@dataclass(frozen=True)
class AppSettings:
    profiles: ProfileSettings
    corpus: CorpusSettings
    models: ModelSettings
    index: IndexSettings
    retrieval: RetrievalSettings
    answer: AnswerSettings
    web: WebSettings
    trace_env: str = "MARXOS_TRACE"
    trace_only_env: str = "MARXOS_TRACE_ONLY"
    dev_mode_env: str = "MARXOS_DEV_MODE"
    dev_token_env: str = "MARXOS_DEV_TOKEN"
    dev_token_input_env: str = "MARXOS_DEV_TOKEN_INPUT"


def build_corpus_settings(profile_name: str) -> CorpusSettings:
    profile = profile_data(CORPUS_PROFILES, profile_name, DEFAULT_CORPUS_PROFILE)
    return CorpusSettings(
        profile_name=profile_name if profile_name in CORPUS_PROFILES else DEFAULT_CORPUS_PROFILE,
        article_map_path=env_str("ARTICLE_MAP_PATH", profile["article_map_path"]),
        article_map_extra_paths=env_str("ARTICLE_MAP_EXTRA_PATHS", profile["article_map_extra_paths"]),
        topic_catalog_path=env_str("TOPIC_CATALOG_PATH", profile["topic_catalog_path"]),
        paragraph_cache_path=env_str("PARAGRAPH_CACHE_PATH", profile["paragraph_cache_path"]),
        semantic_parent_cache_path=env_str("SEMANTIC_PARENT_CACHE_PATH", profile["semantic_parent_cache_path"]),
        ocr_cache_dir=env_str("OCR_CACHE_DIR", profile["ocr_cache_dir"]),
        page_map_path=env_str("PAGE_MAP_PATH", profile["page_map_path"]),
        preferred_editions=env_tuple("MARXOS_PREFERRED_EDITIONS", profile["preferred_editions"]),
    )


def build_retrieval_settings(profile_name: str) -> RetrievalSettings:
    profile = profile_data(RETRIEVAL_PROFILES, profile_name, DEFAULT_RETRIEVAL_PROFILE)
    return RetrievalSettings(
        profile_name=profile_name if profile_name in RETRIEVAL_PROFILES else DEFAULT_RETRIEVAL_PROFILE,
        semantic_parent_window=env_int("SEMANTIC_PARENT_WINDOW", profile["semantic_parent_window"]),
        semantic_child_parent_window=env_int("SEMANTIC_CHILD_PARENT_WINDOW", profile["semantic_child_parent_window"]),
        semantic_child_chunk_size=env_int("SEMANTIC_CHILD_CHUNK_SIZE", profile["semantic_child_chunk_size"]),
        semantic_child_chunk_overlap=env_int("SEMANTIC_CHILD_CHUNK_OVERLAP", profile["semantic_child_chunk_overlap"]),
        vector_backend_default=env_str("MARXOS_VECTOR_BACKEND_DEFAULT", profile["vector_backend_default"]),
    )


def build_answer_settings(profile_name: str) -> AnswerSettings:
    profile = profile_data(ANSWER_PROFILES, profile_name, "deepseek_default")
    return AnswerSettings(
        profile_name=profile_name if profile_name in ANSWER_PROFILES else "deepseek_default",
        deepseek_base_url=env_str("DEEPSEEK_BASE_URL", profile["deepseek_base_url"]),
        deepseek_model=env_str("DEEPSEEK_MODEL", profile["deepseek_model"]),
        default_performance_mode=env_str("MARXOS_PERFORMANCE_MODE_DEFAULT", profile["default_performance_mode"]),
        citation_audit_enabled=env_flag(
            "MARXOS_CITATION_AUDIT_ENABLED",
            "1" if profile["citation_audit_enabled"] else "0",
        ),
    )


def build_model_settings(answer: AnswerSettings) -> ModelSettings:
    return ModelSettings(
        embedding_model=env_str("MARXOS_EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_device=env_str("MARXOS_EMBEDDING_DEVICE", "cpu"),
        deepseek_base_url=answer.deepseek_base_url,
        deepseek_model=answer.deepseek_model,
        deepseek_flash_model=env_str("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
    )


def build_index_settings(corpus: CorpusSettings, retrieval: RetrievalSettings) -> IndexSettings:
    profile = profile_data(RETRIEVAL_PROFILES, retrieval.profile_name, DEFAULT_RETRIEVAL_PROFILE)
    return IndexSettings(
        vectorstore_dir=env_str(
            "VECTORSTORE_DIR",
            existing_path_or_fallback("vectorstore/marx_reader", "vectorstore/marx_reader_core"),
        ),
        paragraph_vectorstore_dir=env_str("PARAGRAPH_VECTORSTORE_DIR", "vectorstore/marx_reader_paragraph"),
        milvus_uri=env_str("MILVUS_URI", profile["milvus_uri"]),
        milvus_collection=env_str("MILVUS_COLLECTION", profile["milvus_collection"]),
        milvus_dim=env_int("MILVUS_DIM", 1024),
        milvus_batch_size=env_int("MILVUS_BATCH_SIZE", 4),
        milvus_sparse_provider=env_str("MILVUS_SPARSE_PROVIDER", profile["milvus_sparse_provider"]),
        milvus_hybrid_search=env_flag(
            "MILVUS_HYBRID_SEARCH",
            "1" if profile["milvus_hybrid_search"] else "0",
        ),
        default_milvus_paragraph_cache=first_existing_path(
            corpus.semantic_parent_cache_path,
            corpus.paragraph_cache_path,
            "data/semantic_parent_cache_core.jsonl",
            "data/paragraph_cache_core.jsonl",
        ),
        milvus_retrieval_unit=env_str(
            "MILVUS_RETRIEVAL_UNIT",
            profile["milvus_retrieval_unit"],
        ),
        milvus_bm25_stats_path=env_str("MARXOS_BM25_STATS_PATH", profile.get("bm25_stats_path", "")),
    )


def build_web_settings() -> WebSettings:
    return WebSettings(
        host=env_str("MARXOS_WEB_HOST", "127.0.0.1"),
        port=env_int("MARXOS_WEB_PORT", 7860),
        metrics_log_path=env_str("MARXOS_METRICS_LOG", "logs/api_ask_metrics.jsonl"),
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    active_corpus = active_profile_name("MARXOS_CORPUS_PROFILE", DEFAULT_CORPUS_PROFILE)
    active_retrieval = active_profile_name("MARXOS_RETRIEVAL_PROFILE", DEFAULT_RETRIEVAL_PROFILE)
    active_answer = active_profile_name("MARXOS_ANSWER_PROFILE", DEFAULT_ANSWER_PROFILE)

    profiles = ProfileSettings(
        active_corpus_profile=active_corpus,
        active_retrieval_profile=active_retrieval,
        active_answer_profile=active_answer,
    )
    corpus = build_corpus_settings(active_corpus)
    retrieval = build_retrieval_settings(active_retrieval)
    answer = build_answer_settings(active_answer)
    models = build_model_settings(answer)
    index = build_index_settings(corpus, retrieval)
    web = build_web_settings()

    return AppSettings(
        profiles=profiles,
        corpus=corpus,
        models=models,
        index=index,
        retrieval=retrieval,
        answer=answer,
        web=web,
    )
