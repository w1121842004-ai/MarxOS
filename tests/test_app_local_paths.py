import unittest
import io
from unittest.mock import patch

from langchain_core.documents import Document

import app
import web_app
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

    def test_unreadable_cli_query_does_not_call_vectorstore_or_api(self):
        with patch("app.load_vectorstore") as load_vectorstore, patch("app.OpenAI") as openai:
            answer = app.run_query("?????????????")

        load_vectorstore.assert_not_called()
        openai.assert_not_called()
        self.assertIn("\u672a\u80fd\u8bfb\u53d6\u5230\u53ef\u7528\u7684\u4e2d\u6587\u95ee\u9898", answer)

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

    def test_exact_quote_returns_empty_when_gotha_quote_is_not_confirmed_in_current_scope(self):
        query = "\u5404\u5c3d\u6240\u80fd\uff0c\u6309\u9700\u5206\u914d\u3002\u51fa\u81ea\u54ea\u91cc\uff1f"

        docs = exact_quote_lookup(query, limit=5)

        self.assertEqual(docs, [])

    def test_exact_quote_prefers_manifesto_for_workers_of_world_slogan(self):
        query = "\u201c\u5168\u4e16\u754c\u65e0\u4ea7\u8005\uff0c\u8054\u5408\u8d77\u6765\uff01\u201d\u51fa\u81ea\u54ea\u91cc\uff1f"

        docs = exact_quote_lookup(query, limit=5)

        self.assertTrue(docs)
        self.assertTrue(all(doc.metadata.get("classic_id") == "communist_manifesto" for doc in docs))
        self.assertEqual(docs[0].metadata.get("source"), "mes01.pdf")
        self.assertEqual(docs[0].metadata.get("pdf_page"), 451)

    def test_exact_quote_prefers_body_hit_over_annotation(self):
        query = "\u201c\u56fd\u5bb6\u662f\u793e\u4f1a\u5728\u4e00\u5b9a\u53d1\u5c55\u9636\u6bb5\u4e0a\u7684\u4ea7\u7269\u3002\u201d\u51fa\u81ea\u54ea\u91cc\uff1f"

        docs = exact_quote_lookup(query, limit=5)

        self.assertTrue(docs)
        self.assertEqual(docs[0].metadata.get("source"), "mea04.pdf")
        self.assertEqual(docs[0].metadata.get("pdf_page"), 206)
        self.assertNotEqual(docs[0].metadata.get("pdf_page"), 693)

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

    def test_paragraph_retrieval_does_not_use_exact_quote_shortcut(self):
        query = "\u201c\u56fd\u5bb6\u662f\u793e\u4f1a\u5728\u4e00\u5b9a\u53d1\u5c55\u9636\u6bb5\u4e0a\u7684\u4ea7\u7269\u3002\u201d\u51fa\u81ea\u54ea\u91cc\uff1f"
        paragraph_doc = Document(
            page_content="\u56fd\u5bb6\u662f\u793e\u4f1a\u5728\u4e00\u5b9a\u53d1\u5c55\u9636\u6bb5\u4e0a\u7684\u4ea7\u7269\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c4\u5377",
                "article": "\u5bb6\u5ead\u3001\u79c1\u6709\u5236\u548c\u56fd\u5bb6\u7684\u8d77\u6e90",
                "source": "mea04.pdf",
                "page": 180,
                "pdf_page": 206,
                "retrieval_unit": "paragraph",
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [paragraph_doc]

        with patch("app.exact_quote_lookup") as exact_quote:
            docs = app.retrieve_paragraph_documents(query, FakeDb(), k=1)

        exact_quote.assert_not_called()
        self.assertEqual(docs[0].metadata.get("retrieval_unit"), "paragraph")
        self.assertEqual(docs[0].metadata.get("match_type"), "paragraph_vector_candidate")

    def test_paragraph_retrieval_refines_weak_concept_article_from_content(self):
        paragraph_doc = Document(
            page_content="\u5f02\u5316\u52b3\u52a8\uff0c\u7531\u4e8e\u4f7f\u81ea\u7136\u754c\u540c\u4eba\u76f8\u5f02\u5316\uff0c\u4e5f\u5c31\u4f7f\u7c7b\u540c\u4eba\u76f8\u5f02\u5316\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c1\u5377",
                "article": "1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f",
                "section": "1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f",
                "source": "mea01.pdf",
                "page": 161,
                "printed_page": 161,
                "pdf_page": 182,
                "retrieval_unit": "paragraph",
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [paragraph_doc]

        docs = app.retrieve_paragraph_documents("\u5f02\u5316\u52b3\u52a8\u662f\u4ec0\u4e48\uff1f", FakeDb(), k=1)

        self.assertEqual(docs[0].metadata.get("article"), "\u5f02\u5316\u52b3\u52a8")
        self.assertEqual(docs[0].metadata.get("section"), "\u5f02\u5316\u52b3\u52a8")
        self.assertEqual(docs[0].metadata.get("raw_article"), "1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f")

    def test_retrieve_documents_prefers_capital_concept_core_sources(self):
        loose_doc = Document(
            page_content="\u8d44\u672c\u662f\u5546\u4e1a\u89c2\u5ff5\uff0c\u5e76\u4e14\u8fd9\u91cc\u6536\u5f55\u4e86\u5f88\u591a\u8d44\u672c\u5b9a\u4e49\u6458\u5f55\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c8\u5377",
                "article": "\u8d44\u672c\u6458\u5f55",
                "source": "mea08.pdf",
                "page": 393,
                "printed_page": 393,
                "pdf_page": 497,
            },
        )
        core_doc = Document(
            page_content="\u8fd9\u91cc\u8bf4\u660e\u8d44\u672c\u4ef7\u503c\u4f5c\u4e3a\u8d44\u672c\u4ef7\u503c\u800c\u5b58\u5728\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c7\u5377",
                "article": "\u8d44\u672c\u5173\u7cfb",
                "source": "mea07.pdf",
                "page": 445,
                "printed_page": 445,
                "pdf_page": 522,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [loose_doc, core_doc]

        docs = app.retrieve_documents("\u8d44\u672c\u662f\u4ec0\u4e48\uff1f", FakeDb(), k=1)

        self.assertEqual(docs[0].metadata.get("source"), "mea07.pdf")

    def test_retrieve_documents_demotes_index_like_chunks(self):
        noisy_index_doc = Document(
            page_content="\u5386\u53f2\u552f\u7269\u4e3b\u4e49\u8fd9\u4e00\u672f\u8bed---509\u3001609\u3001637\u3001641\u3002---\u6069\u683c\u65af\u5173\u4e8e\u5386\u53f2\u552f\u7269\u4e3b\u4e49\u7684\u4e66\u4fe1---592-595\u3001612-614\u3002\u7d22\u5f15\u6761\u76ee\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u4e13\u9898\u8d44\u6599 \u7b2c4\u5377",
                "article": "\u540d\u76ee\u7d22\u5f15",
                "source": "mes04.pdf",
                "page": 953,
                "pdf_page": 953,
                "page_type": "body",
            },
        )
        body_doc = Document(
            page_content="\u5386\u53f2\u552f\u7269\u4e3b\u4e49\u8981\u4ece\u793e\u4f1a\u7269\u8d28\u751f\u4ea7\u51fa\u53d1\u8bf4\u660e\u5386\u53f2\u53d1\u5c55\uff0c\u628a\u751f\u4ea7\u65b9\u5f0f\u548c\u793e\u4f1a\u7ed3\u6784\u770b\u4f5c\u653f\u6cbb\u548c\u7cbe\u795e\u5386\u53f2\u7684\u57fa\u7840\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u4e13\u9898\u8d44\u6599 \u7b2c1\u5377",
                "article": "\u5171\u4ea7\u515a\u5ba3\u8a00",
                "source": "mes01.pdf",
                "page": 401,
                "pdf_page": 401,
                "page_type": "body",
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [noisy_index_doc, body_doc]

        docs = app.retrieve_documents("\u4ec0\u4e48\u662f\u5386\u53f2\u552f\u7269\u4e3b\u4e49\uff1f", FakeDb(), k=2)

        self.assertEqual(docs[0].metadata.get("source"), "mes01.pdf")

    def test_retrieve_documents_boosts_concept_focus_terms(self):
        loose_doc = Document(
            page_content="\u8fd9\u91cc\u8ba8\u8bba\u8d44\u672c\u5468\u8f6c\u548c\u4f01\u4e1a\u4e3b\u6536\u5165\uff0c\u5076\u7136\u63d0\u5230\u52b3\u52a8\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c7\u5377",
                "article": "\u8d44\u672c\u5468\u8f6c",
                "source": "mea07.pdf",
                "page": 1,
                "pdf_page": 1,
            },
        )
        focused_doc = Document(
            page_content="\u52b3\u52a8\u8fc7\u7a0b\u9996\u5148\u8981\u64c5\u5f00\u6bcf\u4e00\u79cd\u7279\u5b9a\u7684\u793e\u4f1a\u7684\u5f62\u5f0f\u6765\u52a0\u4ee5\u8003\u5bdf\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c5\u5377",
                "article": "\u7b2c\u4e94\u7ae0\u52b3\u52a8\u8fc7\u7a0b\u548c\u4ef7\u503c\u589e\u6b96\u8fc7\u7a0b",
                "source": "mea05.pdf",
                "page": 221,
                "pdf_page": 271,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [loose_doc, focused_doc]

        docs = app.retrieve_documents("\u52b3\u52a8\u8fc7\u7a0b\u662f\u4ec0\u4e48\uff1f", FakeDb(), k=2)

        self.assertEqual(docs[0].metadata.get("source"), "mea05.pdf")

    def test_retrieve_documents_does_not_exact_quote_match_plain_concept_query(self):
        vector_doc = Document(
            page_content="\u8d44\u672c\u4e0d\u662f\u7269\uff0c\u800c\u662f\u4e00\u5b9a\u7684\u793e\u4f1a\u751f\u4ea7\u5173\u7cfb\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c7\u5377",
                "article": "\u7b2c\u56db\u5341\u516b\u7ae0\u4e09\u4f4d\u4e00\u4f53\u7684\u516c\u5f0f",
                "source": "mea07.pdf",
                "page": 940,
                "pdf_page": 940,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [vector_doc]

        with patch("app.exact_quote_lookup") as exact_quote:
            docs = app.retrieve_documents("\u8d44\u672c\u662f\u4ec0\u4e48\uff1f", FakeDb(), k=1)

        exact_quote.assert_not_called()
        self.assertEqual(docs[0].metadata.get("source"), "mea07.pdf")

    def test_retrieve_documents_boosts_definition_style_concept_passage(self):
        broad_doc = Document(
            page_content="\u8d44\u672c\u5728\u4e0d\u540c\u751f\u4ea7\u9886\u57df\u4e4b\u95f4\u8f6c\u79fb\uff0c\u8ffd\u9010\u66f4\u9ad8\u7684\u5229\u6da6\u7387\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c8\u5377",
                "article": "\u8d44\u672c\u8bba\u624b\u7a3f\u6458\u9009",
                "source": "mea08.pdf",
                "page": 590,
                "pdf_page": 590,
            },
        )
        definition_doc = Document(
            page_content="\u4ec0\u4e48\u662f\u8d44\u672c\uff1f\u8d44\u672c\u662f\u79ef\u84c4\u7684\u52b3\u52a8\uff0c\u53ea\u6709\u5f53\u5b83\u7ed9\u6240\u6709\u8005\u5e26\u6765\u6536\u5165\u6216\u5229\u6da6\u7684\u65f6\u5019\u624d\u53eb\u505a\u8d44\u672c\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c1\u5377",
                "article": "\u8d44\u672c\u7684\u5229\u6da6",
                "source": "mea01.pdf",
                "page": 130,
                "pdf_page": 151,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [broad_doc, definition_doc]

        docs = app.retrieve_documents("\u8d44\u672c\u662f\u4ec0\u4e48\uff1f", FakeDb(), k=2)

        self.assertEqual(docs[0].metadata.get("source"), "mea01.pdf")

    def test_retrieve_documents_replaces_weak_concept_article_with_classic_title(self):
        dirty_doc = Document(
            page_content="\u56fd\u5bb6\u548c\u65e7\u7684\u6c0f\u65cf\u7ec4\u7ec7\u4e0d\u540c\u7684\u5730\u65b9\uff0c\u7b2c\u4e00\u70b9\u5c31\u662f\u5b83\u6309\u5730\u533a\u6765\u5212\u5206\u5b83\u7684\u56fd\u6c11\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c4\u5377",
                "article": "*\u672a\u6765\u7684\u610f\u5927\u5229\u9769\u547d\u548c\u793e\u4f1a\u515a..............468\u2022",
                "section": "*\u672a\u6765\u7684\u610f\u5927\u5229\u9769\u547d\u548c\u793e\u4f1a\u515a..............468\u2022",
                "source": "mea04.pdf",
                "page": 77,
                "printed_page": 77,
                "pdf_page": 89,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [dirty_doc]

        docs = app.retrieve_documents("\u56fd\u5bb6\u7684\u8d77\u6e90\u662f\u4ec0\u4e48\uff1f", FakeDb(), k=1)

        self.assertEqual(docs[0].metadata.get("article"), "\u5bb6\u5ead\u3001\u79c1\u6709\u5236\u548c\u56fd\u5bb6\u7684\u8d77\u6e90")
        self.assertEqual(docs[0].metadata.get("section"), "\u5bb6\u5ead\u3001\u79c1\u6709\u5236\u548c\u56fd\u5bb6\u7684\u8d77\u6e90")
        self.assertEqual(docs[0].metadata.get("classic_id"), "origin_family_private_property_state")
        self.assertEqual(docs[0].metadata.get("raw_article"), "*\u672a\u6765\u7684\u610f\u5927\u5229\u9769\u547d\u548c\u793e\u4f1a\u515a..............468\u2022")

    def test_retrieve_documents_keeps_precise_concept_section_title(self):
        precise_doc = Document(
            page_content="\u52b3\u52a8\u8fc7\u7a0b\u662f\u5236\u9020\u4f7f\u7528\u4ef7\u503c\u7684\u6709\u76ee\u7684\u7684\u6d3b\u52a8\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c5\u5377",
                "article": "\u7b2c\u4e94\u7ae0\u52b3\u52a8\u8fc7\u7a0b\u548c\u4ef7\u503c\u589e\u6b96\u8fc7\u7a0b",
                "section": "\u7b2c\u4e94\u7ae0\u52b3\u52a8\u8fc7\u7a0b\u548c\u4ef7\u503c\u589e\u6b96\u8fc7\u7a0b",
                "source": "mea05.pdf",
                "page": 221,
                "printed_page": 221,
                "pdf_page": 271,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [precise_doc]

        docs = app.retrieve_documents("\u52b3\u52a8\u8fc7\u7a0b\u662f\u4ec0\u4e48\uff1f", FakeDb(), k=1)

        self.assertEqual(docs[0].metadata.get("article"), "\u7b2c\u4e94\u7ae0\u52b3\u52a8\u8fc7\u7a0b\u548c\u4ef7\u503c\u589e\u6b96\u8fc7\u7a0b")
        self.assertEqual(docs[0].metadata.get("classic_id"), "capital_vol1")

    def test_retrieve_documents_cleans_concept_article_dot_leaders(self):
        dotted_doc = Document(
            page_content="\u5269\u4f59\u4ef7\u503c\u7387\u7531\u5269\u4f59\u4ef7\u503c\u540c\u53ef\u53d8\u8d44\u672c\u7684\u6bd4\u7387\u6765\u51b3\u5b9a\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c5\u5377",
                "article": "2.\u4ef7\u503c\u589e\u6b96\u8fc7\u7a0b.........\u2026.......\u2026...................\u2026",
                "section": "2.\u4ef7\u503c\u589e\u6b96\u8fc7\u7a0b.........\u2026.......\u2026...................\u2026",
                "source": "mea05.pdf",
                "page": 237,
                "printed_page": 237,
                "pdf_page": 287,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [dotted_doc]

        docs = app.retrieve_documents("\u5269\u4f59\u4ef7\u503c\u662f\u4ec0\u4e48\uff1f", FakeDb(), k=1)

        self.assertEqual(docs[0].metadata.get("article"), "2.\u4ef7\u503c\u589e\u6b96\u8fc7\u7a0b")
        self.assertEqual(docs[0].metadata.get("classic_id"), "capital_vol1")
        self.assertIn(".........", docs[0].metadata.get("raw_article"))

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
            "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c1\u5377",
            "article": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c1\u5377",
            "source": "mea01.pdf",
            "page": 210,
            "pdf_page": 210,
        }

        normalized = app.normalize_metadata(metadata)

        self.assertEqual(normalized.get("article"), metadata["article"])

    def test_normalize_metadata_marks_mea_printed_page_low_trust(self):
        metadata = {
            "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c1\u5377",
            "article": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c1\u5377",
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

    def test_normalize_metadata_corrects_mea_mes_collection_names_from_source(self):
        mea = app.normalize_metadata(
            {
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u5168\u96c6\u8865\u5377 \u7b2c2\u5377",
                "source": "mea02.pdf",
                "pdf_page": 86,
            }
        )
        mes = app.normalize_metadata(
            {
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u4e13\u9898\u8d44\u6599 \u7b2c1\u5377",
                "source": "mes01.pdf",
                "pdf_page": 451,
            }
        )

        self.assertEqual(mea.get("series"), "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6")
        self.assertEqual(mea.get("volume"), "\u7b2c2\u5377")
        self.assertEqual(mes.get("series"), "\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6")
        self.assertEqual(mes.get("volume"), "\u7b2c1\u5377")

    def test_normalize_metadata_cleans_or_suppresses_noisy_article_titles(self):
        cleaned = app.normalize_metadata(
            {
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c1\u5377",
                "article": "\u79c1\u6709\u8d22\u4ea7\u548c\u5171\u4ea7\u4e3b\u4e49].........................",
                "source": "mea01.pdf",
                "pdf_page": 210,
            }
        )
        suppressed = app.normalize_metadata(
            {
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c6\u5377",
                "article": "Yy 3uc",
                "source": "mea06.pdf",
                "pdf_page": 348,
            }
        )
        fragment = app.normalize_metadata(
            {
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6 \u7b2c3\u5377",
                "article": "\u51e0\u4e4e\u53c8\u88ab\u300a\u6279\u5224\u53f2\u300b\u4e2d\u4ee5\u201c\u5386\u53f2\u773c\u5149\u7684\u5e7f\u535a\u8fdc\u5927\u201d\u81ea\u8be9\u7684\u65e0\u77e5\u6240\u8d85",
                "source": "mes03.pdf",
                "pdf_page": 684,
            }
        )
        unmatched_bracket = app.normalize_metadata(
            {
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c7\u5377",
                "article": "\u8d44\u672c\u8bba\u300b\u7b2c\u4e09\u518c\u589e\u8865",
                "source": "mea07.pdf",
                "pdf_page": 933,
            }
        )

        self.assertEqual(cleaned.get("article"), "\u79c1\u6709\u8d22\u4ea7\u548c\u5171\u4ea7\u4e3b\u4e49")
        self.assertIsNone(suppressed.get("article"))
        self.assertEqual(suppressed.get("raw_article"), "Yy 3uc")
        self.assertIsNone(fragment.get("article"))
        self.assertIsNone(unmatched_bracket.get("article"))

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

    def test_citation_page_label_uses_trusted_printed_page(self):
        metadata = {
            "book": "测试书",
            "article": "测试篇",
            "printed_page": 123,
            "pdf_page": 456,
            "source": "test.pdf",
        }

        self.assertIn("第123页", app.format_citation(metadata))
        self.assertEqual(app.source_page_label(metadata), "第123页（PDF第456页）")

    def test_citation_page_label_uses_pdf_for_low_trust_printed_page(self):
        metadata = {
            "book": "马克思恩格斯文集 第1卷",
            "article": "马克思恩格斯文集 第1卷",
            "source": "mea01.pdf",
            "page": 210,
            "printed_page": 210,
            "pdf_page": 240,
        }

        citation = app.format_citation(metadata)

        self.assertIn("PDF第240页", citation)
        self.assertNotIn("第210页", citation)
        self.assertEqual(app.source_page_label(metadata), "PDF第240页（印刷页210低信任）")

    def test_build_context_uses_normalized_citation_page_label(self):
        doc = Document(
            page_content="测试段落",
            metadata={
                "book": "马克思恩格斯文集 第1卷",
                "article": "马克思恩格斯文集 第1卷",
                "source": "mea01.pdf",
                "page": 210,
                "printed_page": 210,
                "pdf_page": 240,
            },
        )

        context = app.build_context([doc], "rag_answer")

        self.assertIn("来源：《马克思恩格斯文集 第1卷》马克思恩格斯文集 第1卷，PDF第240页（印刷页210低信任）", context)
        self.assertNotIn("来源：《马克思恩格斯文集 第1卷》马克思恩格斯文集 第1卷，第210页（PDF第240页）", context)

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
        self.assertIn("\u3010\u539f\u8457\u5185\u5bb9\u3011", analysis_prompt)
        self.assertIn("CTX-1", analysis_prompt)

    def test_build_context_citation_formats_are_not_numbered(self):
        doc = Document(
            page_content="\u8d44\u672c\u662f\u79ef\u84c4\u7684\u52b3\u52a8\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c1\u5377",
                "article": "\u8d44\u672c\u7684\u5229\u6da6",
                "source": "mea01.pdf",
                "pdf_page": 151,
            },
        )

        context = app.build_context([doc], "concept_explain")

        self.assertIn("\u53e5\u5b50\u5f15\u6587\u683c\u5f0f\uff1a\u300a", context)
        self.assertIn("\u6bb5\u843d\u5177\u4f53\u51fa\u5904\u683c\u5f0f\uff1a\u300a", context)
        self.assertNotIn("\u53e5\u5b50\u5f15\u6587\u683c\u5f0f\uff1a(1)", context)
        self.assertNotIn("\u6bb5\u843d\u5177\u4f53\u51fa\u5904\u683c\u5f0f\uff1a(1)", context)

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

    def test_query_with_interrogative_wording_is_not_routed_to_quote_lookup(self):
        query = "\u5f53\u524d\uff0cAI\u65f6\u4ee3\u7684\u201c\u70bc\u4e39\u201d\u5176\u80cc\u540e\u7684\u672c\u8d28\u662f\u4ec0\u4e48"

        self.assertEqual(app.classify_query(query), "concept_explain")

    def test_query_about_communism_realization_routes_to_theory_analysis(self):
        query = "\u5171\u4ea7\u4e3b\u4e49\u662f\u4e0d\u662f\u4e00\u5b9a\u4f1a\u5b9e\u73b0\uff1f"

        self.assertEqual(app.classify_query(query), "theory_analysis")

    def test_retrieve_documents_prefers_manifesto_over_malthus_for_communism(self):
        malthus_doc = Document(
            page_content="\u5173\u4e8e\u9a6c\u5c14\u8428\u65af\u4eba\u53e3\u8bba\u7684\u8ba8\u8bba\u6761\u76ee\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c10\u5377",
                "article": "\u9a6c\u5c14\u8428\u65af\u4eba\u53e3\u8bba",
                "source": "mea10.pdf",
                "page": 120,
                "printed_page": 120,
                "pdf_page": 120,
            },
        )
        manifesto_doc = Document(
            page_content="\u5171\u4ea7\u4e3b\u4e49\u4e0d\u662f\u5e94\u5f53\u5b9e\u73b0\u7684\u72b6\u6001\uff0c\u800c\u662f\u6d88\u706d\u73b0\u5b58\u72b6\u6001\u7684\u73b0\u5b9e\u8fd0\u52a8\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6 \u7b2c1\u5377",
                "article": "\u5171\u4ea7\u515a\u5ba3\u8a00",
                "source": "mes01.pdf",
                "page": 401,
                "printed_page": 401,
                "pdf_page": 401,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [malthus_doc, manifesto_doc]

        docs = app.retrieve_documents("\u5171\u4ea7\u4e3b\u4e49\u662f\u4e0d\u662f\u4e00\u5b9a\u4f1a\u5b9e\u73b0\uff1f", FakeDb(), k=1)

        self.assertEqual(docs[0].metadata.get("source"), "mes01.pdf")

    def test_title_constrained_query_prefers_candidates_in_expected_page_range(self):
        out_of_range_doc = Document(
            page_content="\u524d\u8a00\u4e2d\u63d0\u5230\u54e5\u8fbe\u7eb2\u9886\u6279\u5224\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6 \u7b2c4\u5377",
                "article": "\u9a6c\u514b\u601d\u4e3b\u4e49\u7406\u8bba\u7814\u7a76\u548c\u5efa\u8bbe\u5de5\u7a0b\u91cd\u70b9\u9879\u76ee",
                "source": "mes04.pdf",
                "page": 17,
                "printed_page": 17,
                "pdf_page": 17,
            },
        )
        in_range_doc = Document(
            page_content="\u5728\u8fd9\u90e8\u8457\u4f5c\u4e2d\u5206\u6790\u4e86\u5171\u4ea7\u4e3b\u4e49\u793e\u4f1a\u4e24\u4e2a\u53d1\u5c55\u9636\u6bb5\u3002",
            metadata={
                "book": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6 \u7b2c4\u5377",
                "article": "\u5361\u00b7\u9a6c\u514b\u601d\u54e5\u8fbe\u7eb2\u9886\u6279\u5224",
                "source": "mes04.pdf",
                "page": 615,
                "printed_page": 615,
                "pdf_page": 615,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [out_of_range_doc, in_range_doc]

        docs = app.retrieve_documents("\u9a6c\u514b\u601d\u5728\u300a\u54e5\u8fbe\u7eb2\u9886\u6279\u5224\u300b\u4e2d\u5982\u4f55\u5206\u6790\u5171\u4ea7\u4e3b\u4e49\u7684", FakeDb(), k=1)

        self.assertEqual(docs[0].metadata.get("page"), 615)

    def test_farmer_cooperative_query_infers_fadeng_farmer_problem_title(self):
        constraints = app.constraints_from_query("\u8bf7\u5217\u51fa\u5341\u6bb5\u9a6c\u514b\u601d\u5173\u4e8e\u519c\u6c11\u5408\u4f5c\u793e\u7684\u89c2\u70b9")

        self.assertEqual(constraints.get("topic_id"), "peasant_cooperative")
        self.assertGreaterEqual(len(constraints.get("sources") or set()), 3)
        self.assertIn("mea04.pdf", constraints.get("sources") or set())

    def test_diversify_documents_can_enforce_distinct_sources(self):
        docs = [
            Document(page_content="a", metadata={"source": "mea04.pdf", "article": "法德农民问题"}),
            Document(page_content="b", metadata={"source": "mea04.pdf", "article": "法德农民问题（二）"}),
            Document(page_content="c", metadata={"source": "mea02.pdf", "article": "德国农民战争"}),
            Document(page_content="d", metadata={"source": "mea05.pdf", "article": "土地国有化"}),
        ]

        result = app.diversify_documents(docs, k=3, min_distinct_sources=3)

        self.assertEqual(len(result), 3)
        self.assertEqual(len({doc.metadata.get("source") for doc in result}), 3)

    def test_normalize_topic_title_removes_author_prefix_and_noise(self):
        self.assertEqual(
            app.normalize_topic_title("弗·恩格斯法德农民问题"),
            "法德农民问题",
        )
        self.assertEqual(
            app.normalize_topic_title("卡·马克思论土地国有化.....................................................23•"),
            "论土地国有化",
        )

    def test_topic_catalog_includes_six_popular_marx_themes(self):
        topic_ids = [topic.get("id") for topic in app.TOPIC_CATALOG]

        self.assertEqual(
            topic_ids,
            [
                "social_analysis",
                "capitalism_critique",
                "alienation_liberation",
                "state_revolution",
                "peasant_cooperative",
                "socialism_communism",
            ],
        )

    def test_topic_seed_queries_expand_with_topic_work_titles(self):
        constraints = app.constraints_from_query("请列出十段马克思关于农民合作社的观点")

        seeds = app.topic_seed_queries("请列出十段马克思关于农民合作社的观点", constraints)

        self.assertEqual(seeds[0], "请列出十段马克思关于农民合作社的观点")
        self.assertIn("法德农民问题", seeds)
        self.assertIn("德国农民战争", seeds)

    def test_explicit_work_query_takes_priority_over_broad_topic(self):
        constraints = app.constraints_from_query("请概括共产党宣言关于阶级斗争的观点")

        self.assertTrue(constraints.get("strict_title"))
        self.assertEqual(constraints.get("title"), "共产党宣言")
        self.assertNotEqual(constraints.get("topic_id"), "social_analysis")

    def test_format_topic_viewpoint_removes_leading_punctuation_and_work_prefix(self):
        item = {
            "article": "法德农民问题",
            "section": "法德农民问题",
            "excerpt": "。法德农民问题 要把这财产从现在就已经压在它身上的重担下解放出来z，把f田农变成自由的所有者。",
        }
        constraints = {
            "topic_id": "peasant_cooperative",
            "topic_markers": ["农民合作社", "小农", "社会帮助", "土地问题"],
        }

        viewpoint = app.format_topic_viewpoint(item, constraints)

        self.assertFalse(viewpoint.startswith("。"))
        self.assertNotIn("法德农民问题 要", viewpoint)
        self.assertNotIn("z", viewpoint)
        self.assertIn("解放农民", viewpoint)

    def test_strict_title_list_query_can_answer_without_openai(self):
        docs = [
            Document(
                page_content="。到目前为止的一切社会的历史都是阶级斗争的历史。",
                metadata={
                    "book": "马克思恩格斯选集 第1卷",
                    "article": "共产党宣言",
                    "section": "共产党宣言",
                    "source": "mes01.pdf",
                    "printed_page": 376,
                    "pdf_page": 451,
                },
            ),
            Document(
                page_content="。我们的时代，资产阶级时代，却有一个特点：它使阶级对立简单化了。",
                metadata={
                    "book": "马克思恩格斯选集 第1卷",
                    "article": "共产党宣言",
                    "section": "共产党宣言",
                    "source": "mes01.pdf",
                    "printed_page": 377,
                    "pdf_page": 452,
                },
            ),
        ]

        class FakeDb:
            def similarity_search(self, _query, k):
                return docs

        with patch("app.load_vectorstore", return_value=FakeDb()), patch("app.OpenAI") as openai:
            answer = app.run_query("请概括共产党宣言关于阶级斗争的观点")

        openai.assert_not_called()
        self.assertIn("《共产党宣言》", answer)
        self.assertIn("阶级斗争", answer)
        self.assertIn("观点：", answer)

    def test_filter_evidence_keeps_only_matched_items_when_citations_match(self):
        answer = (
            "\u8fd9\u662f\u7b54\u6848\u3002\n\n"
            "*\u5f15\u7528\u6ce8\u91ca*\n"
            "1. \u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c1\u5377\uff0c\u7b2c401\u9875\u3002"
        )
        evidence = [
            {
                "citation": "\u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c1\u5377\uff0c\u7b2c401\u9875",
                "source": "mes01.pdf",
                "printed_page": 401,
                "excerpt": "matched",
            },
            {
                "citation": "\u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c1\u5377\uff0c\u7b2c402\u9875",
                "source": "mes01.pdf",
                "printed_page": 402,
                "excerpt": "unmatched",
            },
        ]

        result = app.filter_evidence_to_answer(answer, evidence, fallback_limit=2)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "E1")
        self.assertEqual(result[0]["excerpt"], "matched")
        self.assertEqual(result[0]["answer_citation"], "\u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c1\u5377\uff0c\u7b2c401\u9875\u3002")

    def test_filter_evidence_falls_back_when_citation_lines_exist_but_no_match(self):
        answer = (
            "\u8fd9\u662f\u7b54\u6848\u3002\n\n"
            "*\u5f15\u7528\u6ce8\u91ca*\n"
            "1. \u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c9\u5377\uff0c\u7b2c999\u9875\u3002"
        )
        evidence = [
            {"citation": "\u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c1\u5377\uff0c\u7b2c401\u9875", "source": "mes01.pdf", "printed_page": 401, "excerpt": "A"},
            {"citation": "\u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c1\u5377\uff0c\u7b2c402\u9875", "source": "mes01.pdf", "printed_page": 402, "excerpt": "B"},
            {"citation": "\u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c1\u5377\uff0c\u7b2c403\u9875", "source": "mes01.pdf", "printed_page": 403, "excerpt": "C"},
        ]

        result = app.filter_evidence_to_answer(answer, evidence, fallback_limit=2)

        self.assertEqual(len(result), 2)
        self.assertEqual([item["id"] for item in result], ["E1", "E2"])
        self.assertTrue(all("answer_citation" not in item for item in result))

    def test_filter_evidence_returns_empty_when_no_evidence_available(self):
        answer = (
            "\u8fd9\u662f\u7b54\u6848\u3002\n\n"
            "*\u5f15\u7528\u6ce8\u91ca*\n"
            "1. \u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c9\u5377\uff0c\u7b2c999\u9875\u3002"
        )

        result = app.filter_evidence_to_answer(answer, [], fallback_limit=3)

        self.assertEqual(result, [])

    def test_web_ask_metrics_marks_fallback_usage(self):
        answer = (
            "\u8fd9\u662f\u7b54\u6848\u3002\n\n"
            "*\u5f15\u7528\u6ce8\u91ca*\n"
            "1. \u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c9\u5377\uff0c\u7b2c999\u9875\u3002"
        )
        evidence = [
            {"id": "E1", "citation": "\u300a\u9a6c\u514b\u601d\u6069\u683c\u65af\u9009\u96c6\u300b\u7b2c1\u5377\uff0c\u7b2c401\u9875", "source": "mes01.pdf", "printed_page": 401}
        ]
        citation_audit = {"ok": False, "issues": [{"type": "citation_not_in_evidence"}], "evidence_count": 1}

        metrics = web_app.MarxOSHandler._build_ask_metrics(
            query="\u8fd9\u53e5\u8bdd\u51fa\u81ea\u54ea\u91cc",
            intent="quote_lookup",
            history=[{"role": "user", "text": "q"}],
            answer=answer,
            evidence=evidence,
            citation_audit=citation_audit,
            elapsed_ms=123,
        )

        self.assertEqual(metrics["evidence_count"], 1)
        self.assertEqual(metrics["citation_lines_count"], 1)
        self.assertEqual(metrics["matched_count"], 0)
        self.assertTrue(metrics["fallback_used"])
        self.assertFalse(metrics["audit_ok"])


if __name__ == "__main__":
    unittest.main()
