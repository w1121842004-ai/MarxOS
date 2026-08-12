"""
Lightweight binary relevance classifier — detects whether a query is
Marxism-related before routing it through the expensive RAG pipeline.

Architecture: LogisticRegression on BGE-M3 (1024-dim) embeddings.
Model size: ~3 KB.  Inference: <1 ms (after embedding).

When the classifier file is absent, ``is_marxism_relevant()`` returns
``True`` (pass-through) so existing behaviour is unchanged.

Training: ``scripts/build_relevance_classifier.py``
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL_PATH = Path(
    os.getenv("RELEVANCE_CLASSIFIER_PATH", "data/relevance_classifier.pkl")
)


class RelevanceClassifier:
    """Binary classifier:  embedding → float in [0, 1] (Marxism relevance)."""

    def __init__(self, model: Any, config: dict | None = None):
        self.model = model  # sklearn estimator with predict_proba
        self.config = config or {}

    def predict_proba(self, embedding: np.ndarray) -> float:
        """Return P(Marxism-relevant) for a single embedding vector."""
        probs = self.model.predict_proba(embedding.reshape(1, -1))[0]
        # predict_proba returns [P(0), P(1)] — return P(1)
        if len(probs) >= 2:
            return float(probs[1])
        return float(probs[0])

    def predict(self, embedding: np.ndarray, threshold: float = 0.5) -> bool:
        """Return True if the query is Marxism-relevant."""
        return self.predict_proba(embedding) >= threshold

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> RelevanceClassifier | None:
        """Load classifier from disk.  Returns ``None`` when unavailable."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return cls(
                model=data["model"],
                config=data.get("config"),
            )
        except (OSError, pickle.UnpicklingError, KeyError):
            return None

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Persist classifier to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "config": self.config,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


# ---------------------------------------------------------------------------
# Quick rule-based pre-check — handles obvious cases in 0ms
# ---------------------------------------------------------------------------

# These markers are deliberately short and high-precision.
# They act as a fast-path bypass: if any hit, we skip the ML model entirely.

STRONG_MARXISM_MARKERS: tuple[str, ...] = (
    "马克思",
    "恩格斯",
    "资本论",
    "共产党宣言",
    "剩余价值",
    "阶级斗争",
    "唯物史观",
    "唯物辩证法",
    "异化劳动",
    "商品拜物教",
    "哥达纲领",
    "费尔巴哈",
    "德意志意识形态",
    "政治经济学批判",
    "科学社会主义",
    "空想社会主义",
    "无产阶级专政",
    "巴黎公社",
    "反杜林论",
    "自然辩证法",
    "家庭私有制",
    "黑格尔法哲学",
    "1844年经济学",
    "关于费尔巴哈",
    "社会主义从空想到科学",
    "雇佣劳动与资本",
    "工资价格和利润",
    "法兰西内战",
    "路易波拿巴",
    "共产主义原理",
    "马恩",
    "马列",
    "马克思主义",
)

OBVIOUS_OUT_OF_DOMAIN_MARKERS: tuple[str, ...] = (
    "天气",
    "股票",
    "房价",
    "游戏攻略",
    "编程",
    "Python",
    "Java",
    "React",
    "Vue",
    "Docker",
    "Kubernetes",
    "量子力学",
    "相对论",
    "DNA",
    "基因编辑",
    "明朝",
    "唐朝",
    "秦始皇",
    "世界杯",
    "NBA",
    "欧冠",
    "减肥",
    "菜谱",
    "化妆",
    "穿搭",
    "考研英语",
    "托福",
    "雅思",
    "微积分",
    "线性代数",
    "电路",
    "芯片",
    "iPhone",
    "特斯拉股票",
    "比特币",
)


def quick_relevance_check(query: str) -> bool | None:
    """Rule-based pre-check for obvious cases.

    Returns:
        True  — obviously Marxism-related, skip ML
        False — obviously out-of-domain, skip ML
        None  — uncertain, needs ML
    """
    if not query or not query.strip():
        return False

    # Check strong Marxism markers first (higher priority)
    for marker in STRONG_MARXISM_MARKERS:
        if marker in query:
            return True

    # Then check out-of-domain markers
    for marker in OBVIOUS_OUT_OF_DOMAIN_MARKERS:
        if marker in query:
            return False

    return None  # needs ML model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_relevance_classifier: RelevanceClassifier | None = None
_relevance_load_attempted: bool = False


def _get_relevance_classifier() -> RelevanceClassifier | None:
    """Lazy-load the relevance classifier. Returns None if unavailable."""
    global _relevance_classifier, _relevance_load_attempted
    if _relevance_load_attempted:
        return _relevance_classifier
    _relevance_load_attempted = True
    try:
        _relevance_classifier = RelevanceClassifier.load()
    except Exception:
        _relevance_classifier = None
    return _relevance_classifier


def is_marxism_relevant(
    query: str,
    embedding: np.ndarray | None = None,
    threshold: float = 0.5,
) -> bool:
    """Check whether *query* is Marxism-related.

    Uses a three-tier strategy:
    1. Rule-based quick check (0 ms)
    2. ML classifier if available (<1 ms)
    3. Pass-through (return True) if classifier unavailable

    Parameters:
        query: The user query string.
        embedding: Pre-computed BGE-M3 embedding (optional — if omitted,
                   classifier is used for rules-only mode).
        threshold: Decision threshold for ML classifier (default 0.5).
    """
    # Tier 1: rule-based quick check
    quick = quick_relevance_check(query)
    if quick is not None:
        return quick

    # Tier 2: ML classifier
    classifier = _get_relevance_classifier()
    if classifier is not None and embedding is not None:
        try:
            return classifier.predict(embedding, threshold=threshold)
        except Exception:
            pass

    # Tier 3: pass-through (no classifier available)
    return True
