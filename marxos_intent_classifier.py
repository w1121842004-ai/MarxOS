"""
Lightweight intent classifier — SetFit-style classification head atop the
existing sentence-transformer embedding model.

**Why this approach:**

- Reuses the embedding model already loaded by ``marxos_runtime.py`` (zero
  additional model bytes).
- Trains a tiny logistic-regression head (~10 KB on disk) that adds <1 ms
  inference overhead after embedding.
- Falls back gracefully to the rule-based ``marxos_query_intent`` system when
  the classifier file is absent.

**Training:** ``scripts/build_intent_classifier.py``

**Integration:** ``marxos_query_intent.classify_query_v2()`` auto-detects the
classifier and blends its output with rule-based scores.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL_PATH = Path(os.getenv("INTENT_CLASSIFIER_PATH", "data/intent_classifier.pkl"))

# 7 intent labels in canonical order
INTENT_LABELS: tuple[str, ...] = (
    "bibliographic_lookup",
    "quote_lookup",
    "concept_explain",
    "comparison",
    "deep_analysis",
    "theory_analysis",
    "rag_answer",
)


class IntentClassifier:
    """Scikit-learn classifier that maps a 384-dim embedding → 7 intent probs."""

    def __init__(self, model: Any, label_encoder: Any, config: dict | None = None):
        self.model = model  # sklearn estimator with predict_proba
        self.label_encoder = label_encoder  # sklearn LabelEncoder
        self.config = config or {}

    def predict_proba(self, embedding: np.ndarray) -> dict[str, float]:
        """Return ``{intent: probability}`` for a single embedding vector."""
        probs = self.model.predict_proba(embedding.reshape(1, -1))[0]
        labels = self.label_encoder.inverse_transform(range(len(probs)))
        return {str(label): float(p) for label, p in zip(labels, probs)}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> IntentClassifier | None:
        """Load classifier from disk.  Returns ``None`` when unavailable."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return cls(
                model=data["model"],
                label_encoder=data["label_encoder"],
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
                    "label_encoder": self.label_encoder,
                    "config": self.config,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


# ---------------------------------------------------------------------------
# Blended prediction — combines classifier scores with rule-based scores
# ---------------------------------------------------------------------------


def blend_predictions(
    query: str,
    embedding: np.ndarray | None,
    rule_scores: dict[str, float],
    classifier: IntentClassifier | None = None,
    blend_weight: float = 0.6,
) -> dict[str, float]:
    """Blend classifier probabilities with rule-based scores.

    Parameters:
        blend_weight: Weight given to the classifier (0–1).  0 = rules only,
                      1 = classifier only.  Default 0.6 gives classifier the
                      edge but lets rules break ties for edge cases.
    """
    if classifier is None or embedding is None:
        return rule_scores

    try:
        ml_probs = classifier.predict_proba(embedding)
    except Exception:
        return rule_scores

    blended: dict[str, float] = {}
    for intent in set(list(rule_scores.keys()) + list(ml_probs.keys())):
        rule = rule_scores.get(intent, 0.0)
        ml = ml_probs.get(intent, 0.0)
        blended[intent] = blend_weight * ml + (1.0 - blend_weight) * rule

    return blended
