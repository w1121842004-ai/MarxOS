from __future__ import annotations

import hashlib
import math
import os
import warnings
from collections import Counter
from pathlib import Path


try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:
    try:
        from langchain_core._api.deprecation import LangChainDeprecationWarning
    except ImportError:
        LangChainDeprecationWarning = DeprecationWarning

    from langchain_community.embeddings import HuggingFaceEmbeddings as _CommunityHuggingFaceEmbeddings

    warnings.filterwarnings(
        "ignore",
        message=r".*HuggingFaceEmbeddings.*deprecated in LangChain 0\.2\.2.*",
        category=LangChainDeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*HuggingFaceEmbeddings.*deprecated in LangChain 0\.2\.2.*",
        category=DeprecationWarning,
    )

    HuggingFaceEmbeddings = _CommunityHuggingFaceEmbeddings


def embedding_encode_kwargs(model_name: str) -> dict:
    """Return encode kwargs that match the selected embedding family."""
    if "bge" in (model_name or "").lower():
        return {"normalize_embeddings": True}
    return {}


def resolve_cached_model_path(model_name: str) -> str:
    """Prefer a local HuggingFace snapshot when one is already cached."""
    if not model_name or "/" not in model_name or Path(model_name).exists():
        return model_name

    model_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    refs_main = model_cache / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = model_cache / "snapshots" / revision
        if (snapshot / "config.json").exists():
            return str(snapshot)

    snapshots_dir = model_cache / "snapshots"
    if snapshots_dir.exists():
        for snapshot in sorted(snapshots_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if (snapshot / "config.json").exists():
                return str(snapshot)

    return model_name


class BGEM3SparseEncoder:
    """BGE-M3 sparse lexical weights generated through FlagEmbedding."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "BGE-M3 sparse vectors require FlagEmbedding. "
                "Install it or use MILVUS_SPARSE_PROVIDER=lexical."
            ) from exc

        model_path = resolve_cached_model_path(model_name)
        try:
            self.model = BGEM3FlagModel(model_path, use_fp16=False, devices=device)
        except TypeError:
            self.model = BGEM3FlagModel(model_path, use_fp16=False, device=device)
        self.max_length = self._max_length_from_env()

    @staticmethod
    def _max_length_from_env() -> int | None:
        value = os.getenv("BGE_M3_MAX_LENGTH", "").strip()
        if not value:
            return None
        try:
            max_length = int(value)
        except ValueError:
            return None
        return max_length if max_length > 0 else None

    def _encode_kwargs(self) -> dict:
        return {"max_length": self.max_length} if self.max_length else {}

    def embed_documents(self, texts: list[str]) -> list[dict[int, float]]:
        outputs = self.model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            **self._encode_kwargs(),
        )
        lexical_weights = outputs.get("lexical_weights") or outputs.get("sparse_vecs") or []
        return [self._normalize_sparse_vector(vector) for vector in lexical_weights]

    def embed_query(self, text: str) -> dict[int, float]:
        return self.embed_documents([text])[0]

    def embed_dense_and_sparse_query(self, text: str) -> tuple[list[float], dict[int, float]]:
        outputs = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            **self._encode_kwargs(),
        )
        dense_vecs = outputs.get("dense_vecs")
        dense_vector = dense_vecs[0].tolist() if hasattr(dense_vecs[0], "tolist") else list(dense_vecs[0])
        lexical_weights = outputs.get("lexical_weights") or outputs.get("sparse_vecs") or [{}]
        return dense_vector, self._normalize_sparse_vector(lexical_weights[0])

    def embed_dense_and_sparse_documents(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        outputs = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            **self._encode_kwargs(),
        )
        dense_vecs = outputs.get("dense_vecs")
        if dense_vecs is None:
            dense_vecs = []
        dense_vectors = [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in dense_vecs
        ]
        lexical_weights = outputs.get("lexical_weights") or outputs.get("sparse_vecs") or []
        sparse_vectors = [self._normalize_sparse_vector(vector) for vector in lexical_weights]
        return dense_vectors, sparse_vectors

    @staticmethod
    def _normalize_sparse_vector(vector) -> dict[int, float]:
        if isinstance(vector, dict):
            items = vector.items()
        else:
            items = vector or []
        normalized = {}
        for key, value in items:
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if weight > 0:
                normalized[int(key)] = weight
        return normalized


class LexicalSparseEncoder:
    """Small local sparse encoder for Milvus sparse-vector indexing."""

    def embed_documents(self, texts: list[str]) -> list[dict[int, float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> dict[int, float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> dict[int, float]:
        from rag.semantic_retrieval import sparse_query_tokens

        counts = Counter(sparse_query_tokens(text or ""))
        vector: dict[int, float] = {}
        for term, count in counts.items():
            term_id = self._term_id(term)
            vector[term_id] = 1.0 + math.log(float(count))
        return vector

    @staticmethod
    def _term_id(term: str) -> int:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
        # Milvus sparse vector ids must be positive and lower than 2^32 - 1.
        return (int.from_bytes(digest, "big") % ((1 << 32) - 2)) + 1


def create_sparse_encoder(
    provider: str,
    model_name: str,
    device: str = "cpu",
    bm25_stats_path: str | None = None,
):
    provider = (provider or "none").strip().lower().replace("_", "-")
    if provider in {"", "0", "false", "off", "none"}:
        return None
    if provider in {"bge-m3", "bgem3", "bge"}:
        return BGEM3SparseEncoder(model_name=model_name, device=device)
    if provider in {"lexical", "light", "bm25-lite"}:
        return LexicalSparseEncoder()
    if provider == "bm25":
        if not bm25_stats_path:
            raise ValueError("bm25 sparse provider requires a fitted BM25 stats path")
        from marxos.indexing.bm25_sparse_v2 import BM25SparseEncoderV2

        return BM25SparseEncoderV2.load(bm25_stats_path)
    raise ValueError(f"Unknown sparse provider: {provider}")


__all__ = [
    "BGEM3SparseEncoder",
    "HuggingFaceEmbeddings",
    "LexicalSparseEncoder",
    "create_sparse_encoder",
    "embedding_encode_kwargs",
    "resolve_cached_model_path",
]
