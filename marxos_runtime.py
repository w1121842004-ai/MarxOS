from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from marxos_embeddings import HuggingFaceEmbeddings, create_sparse_encoder, embedding_encode_kwargs
from marxos_vector_backend import MilvusVectorBackend

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
except ImportError:
    LangChainDeprecationWarning = DeprecationWarning


@dataclass
class RuntimeState:
    embedding_model: str
    vectorstore_dir: str
    paragraph_vectorstore_dir: str
    dev_mode_env: str
    dev_token_env: str
    dev_token_input_env: str
    trace_env: str
    trace_only_env: str
    dual_retrieval_env: str
    vector_backend_env: str = "MARXOS_VECTOR_BACKEND"
    milvus_uri: str = "./data/milvus_lite/marxos_bgem3_sparse.db"
    milvus_collection: str = "marxos_me_passages"
    milvus_embedding_device: str = "cpu"
    embeddings_instance: HuggingFaceEmbeddings | None = None
    vectorstore_instance: FAISS | None = None
    paragraph_vectorstore_instance: FAISS | None = None
    milvus_client_instance: object | None = None
    milvus_vectorstore_instance: MilvusVectorBackend | None = None
    sparse_embeddings_instance: object | None = None

    @staticmethod
    def env_flag(name: str) -> bool:
        return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}

    def dev_mode_enabled(self) -> bool:
        if not self.env_flag(self.dev_mode_env):
            return False

        expected_token = os.getenv(self.dev_token_env)
        if not expected_token:
            return True

        return os.getenv(self.dev_token_input_env) == expected_token

    def trace_enabled(self) -> bool:
        return self.dev_mode_enabled() and self.env_flag(self.trace_env)

    def trace_only_enabled(self) -> bool:
        return self.dev_mode_enabled() and self.env_flag(self.trace_only_env)

    def dual_retrieval_enabled(self) -> bool:
        return self.dev_mode_enabled() and self.env_flag(self.dual_retrieval_env)

    def vector_backend(self) -> str:
        configured = os.getenv(self.vector_backend_env, "").strip().lower()
        if configured:
            return configured
        if self.milvus_uri and Path(self.milvus_uri).exists():
            return "milvus"
        return "faiss"

    def _resolve_cached_model_path(self, model_name: str) -> str:
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

    def load_embeddings(self) -> HuggingFaceEmbeddings:
        if self.embeddings_instance is None:
            model_name = self._resolve_cached_model_path(self.embedding_model)
            model_kwargs = {}
            if self.vector_backend() == "milvus":
                model_kwargs["device"] = self.milvus_embedding_device
            with warnings.catch_warnings():
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
                try:
                    self.embeddings_instance = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs=model_kwargs,
                        encode_kwargs=embedding_encode_kwargs(self.embedding_model),
                    )
                except Exception as exc:
                    # SentenceTransformers may issue a small HuggingFace Hub
                    # metadata request even when the model is already cached.
                    # In offline/sandboxed runs, retry against the local cache.
                    if not any(
                        marker in str(exc)
                        for marker in [
                            "huggingface.co",
                            "Cannot send a request",
                            "WinError 10013",
                            "Max retries exceeded",
                        ]
                    ):
                        raise
                    self.embeddings_instance = HuggingFaceEmbeddings(
                        model_name=model_name,
                        encode_kwargs=embedding_encode_kwargs(self.embedding_model),
                        model_kwargs={**model_kwargs, "local_files_only": True},
                    )
        return self.embeddings_instance

    def load_milvus_vectorstore(self) -> MilvusVectorBackend:
        if self.milvus_vectorstore_instance is None:
            MilvusClient = self._import_milvus_client()

            self.milvus_client_instance = MilvusClient(uri=self.milvus_uri)
            self.milvus_client_instance.load_collection(self.milvus_collection)
            self.milvus_vectorstore_instance = MilvusVectorBackend(
                client=self.milvus_client_instance,
                collection_name=self.milvus_collection,
                embedding_model=self.load_embeddings(),
                sparse_embedding_model=self.load_sparse_embeddings(),
                collection_loaded=True,
            )
            if os.getenv("MILVUS_PREWARM_QUERY_ENCODER", "1").lower() in {"1", "true", "yes", "on"}:
                self.milvus_vectorstore_instance.prewarm(
                    query=os.getenv("MILVUS_PREWARM_QUERY", "马克思主义"),
                    search=os.getenv("MILVUS_PREWARM_SEARCH", "0").lower() in {"1", "true", "yes", "on"},
                )
        return self.milvus_vectorstore_instance

    def load_sparse_embeddings(self):
        provider = os.getenv("MILVUS_SPARSE_PROVIDER", "none")
        if provider.strip().lower() in {"", "0", "false", "off", "none"}:
            return None
        if self.sparse_embeddings_instance is None:
            self.sparse_embeddings_instance = create_sparse_encoder(
                provider,
                self.embedding_model,
                device=self.milvus_embedding_device,
            )
        return self.sparse_embeddings_instance

    def _import_milvus_client(self):
        # pymilvus also reads MILVUS_URI at import time and expects an HTTP URI.
        # MarxOS uses MILVUS_URI for Milvus Lite .db paths too, so hide local
        # paths during import and pass them explicitly to MilvusClient instead.
        env_uri = os.getenv("MILVUS_URI")
        clear_env_uri = bool(env_uri and "://" not in env_uri)
        if clear_env_uri:
            os.environ.pop("MILVUS_URI", None)
        try:
            from pymilvus import MilvusClient
        finally:
            if clear_env_uri and env_uri is not None:
                os.environ["MILVUS_URI"] = env_uri
        return MilvusClient

    def load_vectorstore(self):
        if self.vector_backend() == "milvus":
            return self.load_milvus_vectorstore()
        if self.vectorstore_instance is None:
            self.vectorstore_instance = FAISS.load_local(
                self.vectorstore_dir,
                self.load_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return self.vectorstore_instance

    def paragraph_vectorstore_exists(self) -> bool:
        if self.vector_backend() == "milvus":
            return False
        return os.path.exists(os.path.join(self.paragraph_vectorstore_dir, "index.faiss"))

    def load_paragraph_vectorstore(self) -> FAISS:
        if self.paragraph_vectorstore_instance is None:
            self.paragraph_vectorstore_instance = FAISS.load_local(
                self.paragraph_vectorstore_dir,
                self.load_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return self.paragraph_vectorstore_instance
