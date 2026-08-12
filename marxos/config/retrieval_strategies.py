"""Intent-specific retrieval strategy configuration.

Each intent maps to an ``IntentRetrievalStrategy`` that overrides specific
knobs in the base ``performance_settings()`` dict.  Strategies compose with
(rather than replace) the global performance mode so that the preset
(fast / standard / deep) still governs defaults, while intent tunes the
retrieval behaviour where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Strategy dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntentRetrievalStrategy:
    """Per-intent overrides for the retrieval / generation pipeline.

    Every field defaults to ``None`` (or a no-op value), meaning "use the
    base performance-mode setting".  Only set fields where the intent
    genuinely needs different behaviour.
    """

    # -- retrieval volume ---------------------------------------------------
    retrieve_k_override: int | None = None
    """Override ``retrieve_k`` (or ``rag_retrieve_k`` when intent is
    ``rag_answer``)."""

    # -- feature toggles ----------------------------------------------------
    force_hybrid: bool | None = None
    """Force hybrid (dense + sparse RRF) on / off regardless of perf mode."""

    force_paragraph: bool | None = None
    """Force paragraph-level retrieval on / off."""

    force_planner: bool | None = None
    """Force multi-query planner decomposition on / off."""

    allow_exact_quote: bool = True
    """When ``False``, skip the OCR exact-quote shortcut entirely."""

    # -- BM25 / sparse ------------------------------------------------------
    sparse_first: bool = False
    """When ``True``, try BM25 sparse retrieval *before* falling through to
    dense vector search.  Used by ``quote_lookup`` to prioritise lexical
    matching."""

    sparse_weight_multiplier: float = 1.0
    """Multiplier applied to the BM25 RRF weight in ``_hybrid_merge_candidates``.
    Values > 1.0 make sparse hits rank higher."""

    # -- reranking ----------------------------------------------------------
    rerank_boost: dict[str, float] = field(default_factory=dict)
    """Additive bonuses applied to specific ``rerank_documents`` score
    dimensions (e.g. ``{"hybrid_signal": 20}``)."""

    rerank_penalty: dict[str, float] = field(default_factory=dict)
    """Additive penalties (negative values) applied to specific
    ``rerank_documents`` score dimensions.  Clamped so the dimension
    never goes below 0."""

    # -- quality assessment -------------------------------------------------
    crag_threshold_override: int | None = None
    """Override the quality threshold in ``assess_retrieval_quality()``."""

    # -- context window -----------------------------------------------------
    context_doc_char_multiplier: float = 1.0
    """Multiplier for ``context_doc_char_limit`` and
    ``context_total_char_limit``.  > 1.0 = longer context for
    deep-analysis intents; < 1.0 = shorter for quote lookups."""


# ---------------------------------------------------------------------------
# Per-intent strategies
# ---------------------------------------------------------------------------

INTENT_STRATEGIES: dict[str, IntentRetrievalStrategy] = {
    # ── exact quote: BM25 first, OCR exact match, lighter context ──────
    "quote_lookup": IntentRetrievalStrategy(
        retrieve_k_override=3,
        sparse_first=True,
        sparse_weight_multiplier=2.0,
        force_hybrid=True,
        force_paragraph=False,
        force_planner=False,
        allow_exact_quote=True,
        rerank_boost={"hybrid_signal": 20},
        context_doc_char_multiplier=0.5,
    ),

    # ── bibliographic: skip vector entirely (handled by local lookup) ──
    "bibliographic_lookup": IntentRetrievalStrategy(
        allow_exact_quote=False,
        force_hybrid=False,
        force_paragraph=False,
        force_planner=False,
    ),

    # ── concept explain: prefer concept-term matches over source ───────
    "concept_explain": IntentRetrievalStrategy(
        force_hybrid=True,
        force_paragraph=True,
        rerank_boost={"concept_focus": 15, "concept_source": 10},
    ),

    # ── comparison: wide retrieval, multi-work, no single-source bias ──
    "comparison": IntentRetrievalStrategy(
        retrieve_k_override=8,
        force_hybrid=True,
        force_paragraph=True,
        force_planner=True,
        sparse_weight_multiplier=1.3,
        rerank_penalty={"source_match": -30},
        crag_threshold_override=40,
    ),

    # ── deep analysis: widest retrieval, longest context ───────────────
    "deep_analysis": IntentRetrievalStrategy(
        retrieve_k_override=12,
        force_hybrid=True,
        force_paragraph=True,
        force_planner=True,
        context_doc_char_multiplier=1.5,
    ),

    # ── theory analysis: similar to deep but slightly narrower ─────────
    "theory_analysis": IntentRetrievalStrategy(
        retrieve_k_override=10,
        force_hybrid=True,
        force_paragraph=True,
        force_planner=True,
        sparse_weight_multiplier=1.2,
    ),

    # ── default / rag_answer: no overrides — use base perf mode ────────
    "rag_answer": IntentRetrievalStrategy(),
}

# Fallback for unknown / future intents.
_DEFAULT_STRATEGY = IntentRetrievalStrategy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_intent_strategy(intent: str) -> IntentRetrievalStrategy:
    """Return the retrieval strategy for an intent.

    Unknown intents receive the default (no-override) strategy so the
    system behaves exactly as before.
    """
    if isinstance(intent, str):
        return INTENT_STRATEGIES.get(intent, _DEFAULT_STRATEGY)
    # Handle IntentResult objects (they compare equal to their .primary string)
    return INTENT_STRATEGIES.get(str(intent), _DEFAULT_STRATEGY)


def apply_strategy(
    base_perf: dict,
    strategy: IntentRetrievalStrategy,
) -> dict:
    """Merge intent-strategy overrides into a base performance dict.

    Returns a **new** dict — neither input is mutated.
    """
    perf = dict(base_perf)

    # -- retrieve_k ---------------------------------------------------------
    if strategy.retrieve_k_override is not None:
        perf["retrieve_k"] = strategy.retrieve_k_override
        # Also bump rag_retrieve_k proportionally for rag_answer
        if perf.get("rag_retrieve_k", 0) > 0:
            perf["rag_retrieve_k"] = max(
                strategy.retrieve_k_override,
                perf["rag_retrieve_k"],
            )

    # -- feature toggles ----------------------------------------------------
    if strategy.force_hybrid is not None:
        perf["hybrid_retrieval"] = strategy.force_hybrid
    if strategy.force_paragraph is not None:
        perf["paragraph_retrieval"] = strategy.force_paragraph
    if strategy.force_planner is not None:
        perf["planner_multi_query"] = strategy.force_planner

    # -- allow_exact_quote is threaded through ctx, not perf dict -----------
    #    (handled in app.py / _retrieval_ctx)

    # -- BM25 / sparse ------------------------------------------------------
    if strategy.sparse_weight_multiplier != 1.0:
        perf["sparse_weight_multiplier"] = strategy.sparse_weight_multiplier
    if strategy.sparse_first:
        perf["sparse_first"] = True

    # -- context window -----------------------------------------------------
    if strategy.context_doc_char_multiplier != 1.0:
        mul = strategy.context_doc_char_multiplier
        if "context_doc_char_limit" in perf and perf["context_doc_char_limit"] is not None:
            perf["context_doc_char_limit"] = int(perf["context_doc_char_limit"] * mul)
        if "context_total_char_limit" in perf and perf["context_total_char_limit"] is not None:
            perf["context_total_char_limit"] = int(perf["context_total_char_limit"] * mul)

    # -- CRAG threshold (handled in orchestration) --------------------------
    if strategy.crag_threshold_override is not None:
        perf["crag_threshold_override"] = strategy.crag_threshold_override

    # -- rerank adjustments are threaded separately via strategy object -----
    #    (passed through to rerank_documents)

    return perf
