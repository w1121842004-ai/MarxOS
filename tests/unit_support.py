from __future__ import annotations

import unittest
from unittest.mock import patch


class OfflineAppTestCase(unittest.TestCase):
    """Keep application unit tests independent of local services and caches."""

    def setUp(self) -> None:
        super().setUp()
        self._offline_patchers = [
            patch(
                "marxos.generation.llm_client.OpenAI",
                side_effect=AssertionError(
                    "unit test attempted to construct a real LLM client; inject a fake client"
                ),
            ),
            patch(
                "app.load_vectorstore",
                side_effect=AssertionError(
                    "unit test attempted to load the machine vector index; inject a fake store"
                ),
            ),
            patch("app.sparse_index_ready", return_value=False),
            patch("app.sparse_retrieve_documents", return_value=[]),
            patch("app.book_locator_constraints", return_value={}),
        ]
        for patcher in self._offline_patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
