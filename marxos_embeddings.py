from __future__ import annotations

import warnings


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


__all__ = ["HuggingFaceEmbeddings"]
