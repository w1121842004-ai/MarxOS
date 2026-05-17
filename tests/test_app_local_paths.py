import unittest
import io
from unittest.mock import patch

from langchain_core.documents import Document

import app
from rag.exact_quote_lookup import exact_quote_lookup


class AppLocalPathTests(unittest.TestCase):
    def test_app_import_is_side_effect_free(self):
        self.assertTrue(callable(app.run_query))
        self.assertTrue(callable(app.main))

    def test_bibliographic_query_intent_for_where_wording(self):
        query = "\u300a\u5171\u4ea7\u515a\u5ba3\u8a00\u300b\u6536\u5f55\u5728\u54ea\u91cc\uff1f"

        self.assertEqual(app.classify_query(query), "bibliographic_lookup")
        self.assertEqual(app.extract_bibliographic_title(query), "\u5171\u4ea7\u515a\u5ba3\u8a00")

    def test_bibliographic_query_returns_without_vectorstore_or_api(self):
        query = "\u300a\u5171\u4ea7\u515a\u5ba3\u8a00\u300b\u6536\u5f55\u5728\u54ea\u91cc\uff1f"

        with patch("app.load_vectorstore") as load_vectorstore, patch("app.OpenAI") as openai:
            answer = app.run_query(query)

        load_vectorstore.assert_not_called()
        openai.assert_not_called()
        self.assertIn("\u5171\u4ea7\u515a\u5ba3\u8a00", answer)
        self.assertIn("376-435", answer)

    def test_unknown_bibliographic_title_returns_no_trusted_answer_locally(self):
        query = "\u300a\u4e00\u4e2a\u4e0d\u5b58\u5728\u7684\u9a6c\u514b\u601d\u8457\u4f5c\u6807\u9898\u300b\u6536\u5f55\u5728\u54ea\u91cc\uff1f"

        with patch("app.load_vectorstore") as load_vectorstore, patch("app.OpenAI") as openai:
            answer = app.run_query(query)

        load_vectorstore.assert_not_called()
        openai.assert_not_called()
        self.assertIn("\u672a\u80fd\u5728\u5f53\u524d\u6838\u5fc3\u4e66\u76ee\u8868\u4e2d\u786e\u8ba4", answer)

    def test_exact_quote_metadata_has_confidence_and_classic_fields(self):
        query = "\u4e00\u4e2a\u5e7d\u7075\uff0c\u5171\u4ea7\u4e3b\u4e49\u7684\u5e7d\u7075\uff0c\u5728\u6b27\u6d32\u6e38\u8361\u3002\u51fa\u81ea\u54ea\u91cc\uff1f"

        docs = exact_quote_lookup(query, limit=1)

        self.assertTrue(docs)
        metadata = docs[0].metadata
        self.assertEqual(metadata.get("match_type"), "exact_quote")
        self.assertEqual(metadata.get("confidence"), 1.0)
        self.assertEqual(metadata.get("citation_page_type"), "pdf_page")
        self.assertEqual(metadata.get("lookup_scope"), "core_classic")
        self.assertEqual(metadata.get("classic_title"), "\u5171\u4ea7\u515a\u5ba3\u8a00")
        self.assertEqual(metadata.get("classic_work_year"), "1848")

    def test_exact_quote_query_returns_deterministic_answer_without_vectorstore_or_api(self):
        query = "\u4e00\u4e2a\u5e7d\u7075\uff0c\u5171\u4ea7\u4e3b\u4e49\u7684\u5e7d\u7075\uff0c\u5728\u6b27\u6d32\u6e38\u8361\u3002\u51fa\u81ea\u54ea\u91cc\uff1f"

        with patch("app.load_vectorstore") as load_vectorstore, patch("app.OpenAI") as openai:
            answer = app.run_query(query)

        load_vectorstore.assert_not_called()
        openai.assert_not_called()
        self.assertIn("\u5171\u4ea7\u515a\u5ba3\u8a00", answer)
        self.assertIn("PDF\u7b2c", answer)
        self.assertNotIn("vector_candidate", answer)

    def test_exact_quote_global_fallback_prefers_query_classic_metadata(self):
        query = "\u54f2\u5b66\u5bb6\u4eec\u53ea\u662f\u7528\u4e0d\u540c\u7684\u65b9\u5f0f\u89e3\u91ca\u4e16\u754c\uff0c\u95ee\u9898\u5728\u4e8e\u6539\u53d8\u4e16\u754c\u3002\u51fa\u81ea\u54ea\u91cc\uff1f"

        docs = exact_quote_lookup(query, limit=1)

        self.assertTrue(docs)
        self.assertEqual(docs[0].metadata.get("classic_id"), "theses_feuerbach")
        self.assertIn("\u5173\u4e8e\u8d39\u5c14\u5df4\u54c8\u7684\u63d0\u7eb2", docs[0].metadata.get("article"))

    def test_exact_quote_suppresses_reference_hits_when_core_classic_is_known(self):
        query = "\u5404\u5c3d\u6240\u80fd\uff0c\u6309\u9700\u5206\u914d\u3002\u51fa\u81ea\u54ea\u91cc\uff1f"

        docs = exact_quote_lookup(query, limit=5)

        self.assertTrue(docs)
        self.assertTrue(all(doc.metadata.get("classic_id") == "critique_gotha_programme" for doc in docs))

    def test_unconfirmed_quote_query_returns_no_trusted_answer_without_vectorstore_or_api(self):
        query = "\u8bf7\u7ed9\u51fa\u201c\u8fd9\u662f\u4e00\u53e5\u968f\u4fbf\u7f16\u9020\u7684\u5f15\u6587\u201d\u7684\u51c6\u786e\u9875\u7801\u3002"

        with patch("app.load_vectorstore") as load_vectorstore, patch("app.OpenAI") as openai:
            answer = app.run_query(query)

        load_vectorstore.assert_not_called()
        openai.assert_not_called()
        self.assertEqual(answer, "\u672a\u80fd\u5728\u5f53\u524d OCR \u7f13\u5b58\u4e2d\u786e\u8ba4\u8be5\u5f15\u6587\u7684\u7cbe\u786e\u51fa\u5904\u3002")

    def test_quote_vector_candidates_are_marked_and_warned(self):
        query = "\u8bf7\u7ed9\u51fa\u201c\u8fd9\u662f\u4e00\u53e5\u968f\u4fbf\u7f16\u9020\u7684\u5f15\u6587\u201d\u7684\u51c6\u786e\u9875\u7801\u3002"
        fake_doc = Document(
            page_content="\u5047\u5019\u9009\u7247\u6bb5",
            metadata={
                "book": "\u6d4b\u8bd5\u4e66",
                "article": "\u6d4b\u8bd5\u7bc7",
                "page": 1,
                "pdf_page": 1,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [fake_doc]

        with patch("app.exact_quote_lookup", return_value=[]):
            docs = app.retrieve_documents(query, FakeDb(), k=1)

        self.assertEqual(docs[0].metadata.get("match_type"), "vector_candidate")
        self.assertEqual(docs[0].metadata.get("confidence"), 0.0)

        context = app.build_context(docs, "quote_lookup")
        self.assertIn("No exact quote match was found", context)
        self.assertIn("vector candidates only", context)
        self.assertIn("CTX-1", context)
        self.assertNotIn("\u3010\u8d44\u6599", context)

    def test_normalize_metadata_adds_standard_fields_without_dropping_old_fields(self):
        metadata = {
            "book": "马克思恩格斯全集 第46卷A",
            "source": "me46a.pdf",
            "page": 12,
            "pdf_page": 34,
            "custom_old_field": "keep-me",
        }

        normalized = app.normalize_metadata(metadata)

        self.assertEqual(normalized.get("series"), "马克思恩格斯全集")
        self.assertEqual(normalized.get("publisher"), "人民出版社")
        self.assertEqual(normalized.get("source_file"), "me46a.pdf")
        self.assertEqual(normalized.get("custom_old_field"), "keep-me")

    def test_normalize_metadata_does_not_fill_article_map_from_pdf_page_only(self):
        metadata = {
            "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u5168\u96c6\u8865\u5377 \u7b2c1\u5377",
            "article": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u5168\u96c6\u8865\u5377 \u7b2c1\u5377",
            "source": "mea01.pdf",
            "page": 210,
            "pdf_page": 210,
        }

        normalized = app.normalize_metadata(metadata)

        self.assertEqual(normalized.get("article"), metadata["article"])

    def test_normalize_metadata_marks_mea_printed_page_low_trust(self):
        metadata = {
            "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u5168\u96c6\u8865\u5377 \u7b2c1\u5377",
            "article": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u5168\u96c6\u8865\u5377 \u7b2c1\u5377",
            "source": "mea01.pdf",
            "page": 210,
            "printed_page": 210,
            "pdf_page": 240,
        }

        normalized = app.normalize_metadata(metadata)

        self.assertEqual(normalized.get("article"), metadata["book"])
        self.assertEqual(normalized.get("printed_page_trust"), "low")
        self.assertEqual(normalized.get("citation_page"), 240)
        self.assertEqual(normalized.get("citation_page_type"), "pdf_page")

    def test_format_citation_uses_pdf_label_when_only_pdf_page_exists(self):
        citation = app.format_citation(
            {
                "book": "测试书",
                "article": "测试篇",
                "pdf_page": 9,
                "source": "test.pdf",
            }
        )

        self.assertIn("PDF第9页", citation)
        self.assertNotIn("，第9页", citation)

    def test_query_routing_enhanced_concept_and_analysis_patterns(self):
        self.assertEqual(app.classify_query("\u4eba\u7684\u672c\u8d28\u662f\u4ec0\u4e48\uff1f"), "concept_explain")
        self.assertEqual(app.classify_query("如何理解剩余价值这个概念？"), "concept_explain")
        self.assertEqual(app.classify_query("结合现实怎么看待资本逻辑？"), "theory_analysis")

    def test_build_prompt_dispatches_to_small_prompt_builders(self):
        quote_prompt = app.build_prompt("quote_lookup", "问题", "材料")
        analysis_prompt = app.build_prompt("theory_analysis", "问题", "材料")

        self.assertIn("只输出出处", quote_prompt)
        self.assertIn("生产力与生产关系", analysis_prompt)

    def test_prompt_forbids_internal_context_labels(self):
        analysis_prompt = app.build_prompt("theory_analysis", "question", "context")

        self.assertIn("\u4e0d\u8981\u5199", analysis_prompt)
        self.assertIn("\u8d44\u65991", analysis_prompt)
        self.assertIn("\u7247\u6bb51", analysis_prompt)

    def test_trace_mode_prints_internal_quote_lookup_details(self):
        query = "\u4e00\u4e2a\u5e7d\u7075\uff0c\u5171\u4ea7\u4e3b\u4e49\u7684\u5e7d\u7075\uff0c\u5728\u6b27\u6d32\u6e38\u8361\u3002\u51fa\u81ea\u54ea\u91cc\uff1f"

        with patch.dict("os.environ", {"MARXOS_DEV_MODE": "1", "MARXOS_TRACE": "1"}), patch("sys.stderr", new_callable=io.StringIO) as stderr:
            answer = app.run_query(query)

        trace = stderr.getvalue()
        self.assertIn("MarxOS Trace", trace)
        self.assertIn("intent: quote_lookup", trace)
        self.assertIn("exact_quote_docs", trace)
        self.assertIn("sentence_citation", trace)
        self.assertIn("\u5171\u4ea7\u515a\u5ba3\u8a00", answer)

    def test_trace_mode_is_locked_in_user_mode(self):
        query = "\u4e00\u4e2a\u5e7d\u7075\uff0c\u5171\u4ea7\u4e3b\u4e49\u7684\u5e7d\u7075\uff0c\u5728\u6b27\u6d32\u6e38\u8361\u3002\u51fa\u81ea\u54ea\u91cc\uff1f"

        with patch.dict("os.environ", {"MARXOS_TRACE": "1"}, clear=True), patch("sys.stderr", new_callable=io.StringIO) as stderr:
            answer = app.run_query(query)

        self.assertNotIn("MarxOS Trace", stderr.getvalue())
        self.assertIn("\u5171\u4ea7\u515a\u5ba3\u8a00", answer)

    def test_trace_mode_requires_token_when_configured(self):
        with patch.dict(
            "os.environ",
            {
                "MARXOS_DEV_MODE": "1",
                "MARXOS_DEV_TOKEN": "secret",
                "MARXOS_DEV_TOKEN_INPUT": "wrong",
                "MARXOS_TRACE": "1",
            },
            clear=True,
        ):
            self.assertFalse(app.trace_enabled())

        with patch.dict(
            "os.environ",
            {
                "MARXOS_DEV_MODE": "1",
                "MARXOS_DEV_TOKEN": "secret",
                "MARXOS_DEV_TOKEN_INPUT": "secret",
                "MARXOS_TRACE": "1",
            },
            clear=True,
        ):
            self.assertTrue(app.trace_enabled())

    def test_trace_only_returns_debug_answer_without_calling_llm(self):
        fake_doc = Document(
            page_content="\u4eba\u7684\u672c\u8d28\u6d4b\u8bd5\u7247\u6bb5",
            metadata={
                "book": "\u6d4b\u8bd5\u4e66",
                "article": "\u6d4b\u8bd5\u7bc7",
                "page": 7,
                "pdf_page": 9,
                "source": "test.pdf",
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [fake_doc]

        with (
            patch.dict("os.environ", {"MARXOS_DEV_MODE": "1", "MARXOS_TRACE_ONLY": "1"}, clear=False),
            patch("app.load_vectorstore", return_value=FakeDb()) as load_vectorstore,
            patch("app.OpenAI") as openai,
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            answer = app.run_query("\u4eba\u7684\u672c\u8d28\u662f\u4ec0\u4e48\uff1f")

        load_vectorstore.assert_called_once()
        openai.assert_not_called()
        self.assertIn("TRACE_ONLY", answer)
        self.assertIn("intent: concept_explain", answer)
        self.assertIn("source=test.pdf", answer)


if __name__ == "__main__":
    unittest.main()
