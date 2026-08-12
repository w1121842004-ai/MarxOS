"""Tests for intent-specific retrieval strategies."""

import unittest

from marxos.config.retrieval_strategies import (
    INTENT_STRATEGIES,
    IntentRetrievalStrategy,
    apply_strategy,
    get_intent_strategy,
)


class TestIntentStrategies(unittest.TestCase):
    """Verify strategy mapping and merging logic."""

    def test_get_known_intent_returns_strategy(self):
        for intent in ["quote_lookup", "bibliographic_lookup", "concept_explain",
                       "comparison", "deep_analysis", "theory_analysis", "rag_answer"]:
            with self.subTest(intent=intent):
                s = get_intent_strategy(intent)
                self.assertIsInstance(s, IntentRetrievalStrategy)

    def test_get_unknown_intent_returns_default(self):
        s = get_intent_strategy("nonexistent_intent")
        self.assertIsInstance(s, IntentRetrievalStrategy)
        # Default strategy has all None / no-op values
        self.assertIsNone(s.retrieve_k_override)
        self.assertFalse(s.sparse_first)
        self.assertEqual(s.sparse_weight_multiplier, 1.0)

    def test_quote_lookup_sparse_first(self):
        s = get_intent_strategy("quote_lookup")
        self.assertTrue(s.sparse_first)
        self.assertEqual(s.sparse_weight_multiplier, 2.0)
        self.assertTrue(s.force_hybrid)
        self.assertFalse(s.force_paragraph)
        self.assertFalse(s.force_planner)
        self.assertDictEqual(s.rerank_boost, {"hybrid_signal": 20})

    def test_comparison_multi_work(self):
        s = get_intent_strategy("comparison")
        self.assertEqual(s.retrieve_k_override, 8)
        self.assertTrue(s.force_hybrid)
        self.assertTrue(s.force_paragraph)
        self.assertTrue(s.force_planner)
        self.assertDictEqual(s.rerank_penalty, {"source_match": -30})
        self.assertEqual(s.crag_threshold_override, 40)

    def test_deep_analysis_long_context(self):
        s = get_intent_strategy("deep_analysis")
        self.assertEqual(s.retrieve_k_override, 12)
        self.assertTrue(s.force_hybrid)
        self.assertTrue(s.force_paragraph)
        self.assertTrue(s.force_planner)
        self.assertEqual(s.context_doc_char_multiplier, 1.5)

    def test_concept_explain_boost(self):
        s = get_intent_strategy("concept_explain")
        self.assertTrue(s.force_hybrid)
        self.assertTrue(s.force_paragraph)
        self.assertDictEqual(s.rerank_boost, {"concept_focus": 15, "concept_source": 10})

    def test_bibliographic_skips_retrieval(self):
        s = get_intent_strategy("bibliographic_lookup")
        self.assertFalse(s.allow_exact_quote)
        self.assertFalse(s.force_hybrid)
        self.assertFalse(s.force_paragraph)
        self.assertFalse(s.force_planner)

    def test_rag_answer_is_default(self):
        s = get_intent_strategy("rag_answer")
        self.assertIsNone(s.retrieve_k_override)
        self.assertIsNone(s.force_hybrid)
        self.assertIsNone(s.force_paragraph)
        self.assertIsNone(s.force_planner)
        self.assertFalse(s.sparse_first)
        self.assertEqual(s.sparse_weight_multiplier, 1.0)

    def test_intent_result_object_str_coercion(self):
        """IntentResult objects compare equal to their .primary string."""
        s = INTENT_STRATEGIES.get("quote_lookup")
        self.assertIsNotNone(s)
        self.assertTrue(s.sparse_first)


class TestApplyStrategy(unittest.TestCase):
    """Verify strategy merging into performance dicts."""

    def test_apply_strategy_preserves_base(self):
        base = {"retrieve_k": 5, "hybrid_retrieval": True, "paragraph_retrieval": False}
        s = IntentRetrievalStrategy()
        result = apply_strategy(base, s)
        self.assertEqual(result["retrieve_k"], 5)
        self.assertEqual(result["hybrid_retrieval"], True)
        self.assertEqual(result["paragraph_retrieval"], False)

    def test_apply_strategy_overrides_k(self):
        base = {"retrieve_k": 5, "rag_retrieve_k": 12}
        s = IntentRetrievalStrategy(retrieve_k_override=8)
        result = apply_strategy(base, s)
        self.assertEqual(result["retrieve_k"], 8)
        self.assertEqual(result["rag_retrieve_k"], 12)  # bumped to at least override

    def test_apply_strategy_forces_hybrid(self):
        base = {"hybrid_retrieval": False}
        s = IntentRetrievalStrategy(force_hybrid=True)
        result = apply_strategy(base, s)
        self.assertTrue(result["hybrid_retrieval"])

    def test_apply_strategy_context_multiplier(self):
        base = {"context_doc_char_limit": 4000, "context_total_char_limit": 16000}
        s = IntentRetrievalStrategy(context_doc_char_multiplier=0.5)
        result = apply_strategy(base, s)
        self.assertEqual(result["context_doc_char_limit"], 2000)
        self.assertEqual(result["context_total_char_limit"], 8000)

    def test_apply_strategy_sparse_first(self):
        base = {}
        s = IntentRetrievalStrategy(sparse_first=True)
        result = apply_strategy(base, s)
        self.assertTrue(result["sparse_first"])

    def test_apply_strategy_crag_threshold(self):
        base = {}
        s = IntentRetrievalStrategy(crag_threshold_override=40)
        result = apply_strategy(base, s)
        self.assertEqual(result["crag_threshold_override"], 40)

    def test_apply_strategy_does_not_mutate_input(self):
        base = {"retrieve_k": 5}
        original = dict(base)
        s = IntentRetrievalStrategy(retrieve_k_override=10)
        result = apply_strategy(base, s)
        self.assertEqual(base["retrieve_k"], 5)  # unchanged
        self.assertEqual(original["retrieve_k"], 5)
        self.assertEqual(result["retrieve_k"], 10)

    def test_apply_strategy_none_unchanged(self):
        base = {"retrieve_k": 4, "hybrid_retrieval": None}
        s = IntentRetrievalStrategy()  # all None
        result = apply_strategy(base, s)
        self.assertEqual(result["retrieve_k"], 4)
        self.assertIsNone(result["hybrid_retrieval"])

    def test_full_quote_lookup_strategy_merge(self):
        base = {"retrieve_k": 5, "hybrid_retrieval": True, "paragraph_retrieval": True,
                "context_doc_char_limit": 4000}
        s = get_intent_strategy("quote_lookup")
        result = apply_strategy(base, s)
        self.assertEqual(result["retrieve_k"], 3)
        self.assertTrue(result["hybrid_retrieval"])
        self.assertFalse(result["paragraph_retrieval"])
        self.assertTrue(result["sparse_first"])
        self.assertEqual(result["sparse_weight_multiplier"], 2.0)
        self.assertEqual(result["context_doc_char_limit"], 2000)  # 0.5x


if __name__ == "__main__":
    unittest.main()
