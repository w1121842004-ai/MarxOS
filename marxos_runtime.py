from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from marxos_embeddings import HuggingFaceEmbeddings

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
    embeddings_instance: HuggingFaceEmbeddings | None = None
    vectorstore_instance: FAISS | None = None
    paragraph_vectorstore_instance: FAISS | None = None

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

    def load_embeddings(self) -> HuggingFaceEmbeddings:
        if self.embeddings_instance is None:
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
                self.embeddings_instance = HuggingFaceEmbeddings(model_name=self.embedding_model)
        return self.embeddings_instance

    def load_vectorstore(self) -> FAISS:
        if self.vectorstore_instance is None:
            self.vectorstore_instance = FAISS.load_local(
                self.vectorstore_dir,
                self.load_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return self.vectorstore_instance

    def paragraph_vectorstore_exists(self) -> bool:
        return os.path.exists(os.path.join(self.paragraph_vectorstore_dir, "index.faiss"))

    def load_paragraph_vectorstore(self) -> FAISS:
        if self.paragraph_vectorstore_instance is None:
            self.paragraph_vectorstore_instance = FAISS.load_local(
                self.paragraph_vectorstore_dir,
                self.load_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return self.paragraph_vectorstore_instance
