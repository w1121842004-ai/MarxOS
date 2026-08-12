"""Centralized MarxOS configuration."""

from .retrieval_strategies import (
    IntentRetrievalStrategy,
    apply_strategy,
    get_intent_strategy,
)
from .settings import (
    AppSettings,
    AnswerSettings,
    CorpusSettings,
    IndexSettings,
    ModelSettings,
    ProfileSettings,
    RetrievalSettings,
    WebSettings,
    get_settings,
)

__all__ = [
    "AppSettings",
    "AnswerSettings",
    "CorpusSettings",
    "IndexSettings",
    "IntentRetrievalStrategy",
    "ModelSettings",
    "ProfileSettings",
    "RetrievalSettings",
    "WebSettings",
    "apply_strategy",
    "get_intent_strategy",
    "get_settings",
]
