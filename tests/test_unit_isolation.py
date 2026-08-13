from __future__ import annotations

from unittest.mock import patch

import app
from tests.unit_support import OfflineAppTestCase


class UnitIsolationTests(OfflineAppTestCase):
    def test_real_llm_client_is_blocked_by_default(self):
        with self.assertRaisesRegex(AssertionError, "real LLM client"):
            app.create_deepseek_client()

    def test_machine_vectorstore_is_blocked_by_default(self):
        with self.assertRaisesRegex(AssertionError, "machine vector index"):
            app.load_vectorstore()

    def test_machine_sparse_cache_is_disabled_by_default(self):
        self.assertFalse(app.sparse_index_ready())
        self.assertEqual(app.sparse_retrieve_documents("测试"), [])

    def test_llm_book_locator_is_disabled_by_default(self):
        self.assertEqual(app.book_locator_constraints("未匹配标题"), {})

    def test_explicit_fake_llm_can_override_offline_guard(self):
        sentinel = object()
        with patch("marxos.generation.llm_client.OpenAI", return_value=sentinel):
            self.assertIs(app.create_deepseek_client(), sentinel)
