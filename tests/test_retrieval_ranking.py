import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from retrieval.ranking import rerank_documents


class IntentPenaltyRankingTests(unittest.TestCase):
    def test_penalty_keeps_negative_scores_instead_of_clamping_them_to_zero(self):
        disallowed = Document(page_content="bad", metadata={"topic_allowed": False})
        allowed = Document(page_content="good", metadata={"topic_allowed": True})
        strategy = SimpleNamespace(
            rerank_boost={},
            rerank_penalty={"topic_title": -50},
        )
        ctx = {
            "normalize_for_match": lambda value: str(value or "").lower(),
            "clean_text": lambda value, default="": value or default,
            "score_concept_focus": lambda query, metadata, content: 0,
            "score_concept_source_priority": lambda query, metadata: 0,
            "score_document_quality": lambda metadata, content: 0,
            "is_noisy_article_title": lambda article: False,
            "RERANK_DEBUG_ENV": "MARXOS_TEST_RERANK_DEBUG",
        }

        with (
            patch("retrieval.ranking.metadata_matches_constraints", return_value=False),
            patch("retrieval.ranking.page_in_expected_range", return_value=False),
            patch(
                "retrieval.ranking.topic_title_allowed",
                side_effect=lambda metadata, constraints, ctx: metadata["topic_allowed"],
            ),
        ):
            ranked = rerank_documents(
                "query",
                [disallowed, allowed],
                {"topic_id": "topic"},
                ctx,
                strategy=strategy,
            )

        self.assertIs(ranked[0], allowed)


if __name__ == "__main__":
    unittest.main()
