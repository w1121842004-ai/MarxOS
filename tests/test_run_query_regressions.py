import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

import app
from marxos.app import orchestration
from tests.unit_support import OfflineAppTestCase


class RunQueryRegressionTests(OfflineAppTestCase):
    def test_collect_retrieval_materials_uses_corrective_retrieval_for_low_quality_initial_docs(self):
        weak_doc = Document(
            page_content="定位提示：请在原文范围内核对。",
            metadata={
                "source": "mes01.pdf",
                "article": "关于费尔巴哈的提纲",
                "match_type": "locator_backstop",
                "page": 135,
            },
        )
        strong_doc = Document(
            page_content="人的本质不是单个人所固有的抽象物，在其现实性上，它是一切社会关系的总和。",
            metadata={
                "source": "mes01.pdf",
                "article": "关于费尔巴哈的提纲",
                "printed_page": 135,
                "citation_page": 135,
                "pdf_page": 151,
                "match_type": "vector_candidate",
            },
        )
        calls = {"count": 0}

        def fake_retrieve_documents(_query, _db, k=5):
            calls["count"] += 1
            return [weak_doc] if calls["count"] == 1 else [strong_doc]

        def fake_evidence_from_docs(docs):
            items = []
            for index, doc in enumerate(docs, start=1):
                items.append(
                    {
                        "id": f"E{index}",
                        "source": doc.metadata.get("source"),
                        "article": doc.metadata.get("article"),
                        "printed_page": doc.metadata.get("printed_page"),
                        "citation_page": doc.metadata.get("citation_page"),
                        "excerpt": doc.page_content[:80],
                    }
                )
            return items

        state = orchestration.collect_retrieval_materials(
            query="人的本质是什么？",
            route_query="人的本质是什么？",
            query_intent="concept_explain",
            constraints={},
            paragraph_vectorstore_dir="vectorstore/marx_reader_paragraph",
            trace=False,
            trace_only=False,
            topic_info_from_constraints=lambda constraints: constraints,
            set_last_topic_info=lambda _info: None,
            print_trace_line=lambda _text: None,
            print_constraints_trace=lambda _constraints: None,
            load_vectorstore=lambda: object(),
            retrieve_documents=fake_retrieve_documents,
            paragraph_vectorstore_exists=lambda: False,
            load_paragraph_vectorstore=lambda: None,
            filter_paragraph_docs_by_text_overlap=lambda query, docs, limit=None: docs[:limit],
            merge_prefer_paragraph_docs=lambda paragraph_docs, chunk_docs, limit: (paragraph_docs + chunk_docs)[:limit],
            refine_docs_citation_pages_for_query=lambda docs, query: docs,
            evidence_from_docs=fake_evidence_from_docs,
            is_topic_view_list_query=lambda query, constraints: False,
        )

        self.assertEqual(calls["count"], 2)
        self.assertEqual(state["crag_report"].get("path"), "corrective")
        self.assertGreaterEqual(state["crag_report"].get("score", 0), state["crag_report"].get("threshold", 0))
        self.assertEqual(state["docs"][0].metadata.get("printed_page"), 135)

    def test_run_query_rejects_unsupported_slogan_without_vector_or_llm(self):
        query = "以人民为中心是否出自马克思原著？"

        with patch("app.load_vectorstore") as load_vectorstore, patch("marxos.generation.llm_client.OpenAI") as openai:
            answer = app.run_query(query)

        load_vectorstore.assert_not_called()
        openai.assert_not_called()
        self.assertIn("不是马克思原著中的原文表达", answer)

    def test_run_query_answers_chitchat_without_vector_or_llm(self):
        with patch("app.load_vectorstore") as load_vectorstore, patch("app.create_deepseek_client") as openai:
            answer = app.run_query("你好")

        load_vectorstore.assert_not_called()
        openai.assert_not_called()
        self.assertEqual(app.classify_query("你好"), "chitchat")
        self.assertIn("我是 MarxOS", answer)

    def test_run_query_can_build_topic_view_answer_without_llm(self):
        docs = [
            Document(
                page_content="对小农不能采取暴力剥夺，而应通过示范和社会帮助引导其逐步走向合作化。",
                metadata={
                    "book": "马克思恩格斯文集 第4卷",
                    "article": "法德农民问题",
                    "section": "法德农民问题",
                    "source": "mea04.pdf",
                    "printed_page": 320,
                    "pdf_page": 337,
                },
            ),
            Document(
                page_content="分散的小块土地应结合起来实行较大规模经营，并通过合作社方式重新组织生产。",
                metadata={
                    "book": "马克思恩格斯文集 第4卷",
                    "article": "法德农民问题",
                    "section": "法德农民问题",
                    "source": "mea04.pdf",
                    "printed_page": 321,
                    "pdf_page": 338,
                },
            ),
        ]

        class FakeDb:
            def similarity_search(self, _query, k):
                return docs

        def fake_retrieve_documents(_query, _db, k=5):
            return docs[:k]

        with (
            patch("app.load_vectorstore", return_value=FakeDb()),
            patch("app.retrieve_documents", side_effect=fake_retrieve_documents),
            patch("app.paragraph_vectorstore_exists", return_value=False),
            patch("marxos.generation.llm_client.OpenAI") as openai,
        ):
            answer = app.run_query("请列出马克思关于农民合作社的观点")

        openai.assert_not_called()
        self.assertIn("原著材料", answer)
        self.assertIn("合作", answer)
        self.assertIn("观点", answer)

    def test_run_query_uses_llm_path_and_keeps_verified_citation(self):
        doc = Document(
            page_content="人的本质不是单个人所固有的抽象物，在其现实性上，它是一切社会关系的总和。",
            metadata={
                "book": "马克思恩格斯选集 第1卷",
                "article": "关于费尔巴哈的提纲",
                "section": "关于费尔巴哈的提纲",
                "source": "mes01.pdf",
                "printed_page": 135,
                "pdf_page": 151,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [doc]

        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "人的本质不是脱离社会关系的孤立抽象物，而是在现实社会关系中形成和理解的。"
                        )
                    )
                )
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with (
            patch("app.load_vectorstore", return_value=FakeDb()),
            patch("app.retrieve_documents", return_value=[doc]),
            patch("app.paragraph_vectorstore_exists", return_value=False),
            patch("marxos.generation.llm_client.OpenAI", return_value=fake_client) as openai,
        ):
            answer = app.run_query("人的本质是什么？")

        self.assertGreaterEqual(openai.call_count, 1)  # may trigger recovery loop
        self.assertIn("人的本质", answer)
        self.assertTrue(app.LAST_CITATION_AUDIT.get("ok"))
        self.assertGreaterEqual(len(app.LAST_EVIDENCE), 1)
        self.assertEqual(app.LAST_EVIDENCE[0].get("printed_page"), 135)

    def test_run_query_normalizes_pdf_page_label_from_llm_output(self):
        doc = Document(
            page_content="人的本质不是单个人所固有的抽象物，在其现实性上，它是一切社会关系的总和。",
            metadata={
                "book": "马克思恩格斯选集 第1卷",
                "article": "关于费尔巴哈的提纲",
                "section": "关于费尔巴哈的提纲",
                "source": "mes01.pdf",
                "printed_page": 135,
                "pdf_page": 151,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [doc]

        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "人的本质是现实社会关系中的人的本质。\n\n"
                            "*引用注释*\n"
                            "1. 《马克思恩格斯选集》第1卷，PDF第135页。"
                        )
                    )
                )
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with (
            patch("app.load_vectorstore", return_value=FakeDb()),
            patch("app.paragraph_vectorstore_exists", return_value=False),
            patch("marxos.generation.llm_client.OpenAI", return_value=fake_client),
        ):
            answer = app.run_query("人的本质是什么？")

        self.assertNotIn("PDF第", answer)
        self.assertIn("第135页", answer)
        self.assertTrue(app.LAST_CITATION_AUDIT.get("ok"))

    def test_run_query_trace_only_still_skips_llm(self):
        doc = Document(
            page_content="测试片段",
            metadata={
                "book": "测试书",
                "article": "测试篇",
                "source": "test.pdf",
                "printed_page": 7,
                "pdf_page": 9,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [doc]

        with (
            patch.dict(
                "os.environ",
                {
                    "MARXOS_DEV_MODE": "1",
                    "MARXOS_TRACE_ONLY": "1",
                },
                clear=False,
            ),
            patch("app.load_vectorstore", return_value=FakeDb()),
            patch("app.paragraph_vectorstore_exists", return_value=False),
            patch("sys.stderr", new_callable=io.StringIO),
            patch("marxos.generation.llm_client.OpenAI") as openai,
        ):
            answer = app.run_query("测试概念说明")

        openai.assert_not_called()
        self.assertIn("TRACE_ONLY", answer)
        self.assertIn("Top chunks", answer)

    def test_run_query_keeps_working_when_phoenix_enabled_without_packages(self):
        doc = Document(
            page_content="Phoenix fallback span test",
            metadata={
                "book": "test book",
                "article": "test article",
                "source": "test.pdf",
                "printed_page": 11,
                "pdf_page": 13,
            },
        )

        class FakeDb:
            def similarity_search(self, _query, k):
                return [doc]

        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Phoenix enabled fallback test."
                    )
                )
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        trace_manager = app.phoenix.trace_manager
        original_state = (
            trace_manager._initialized,
            trace_manager._tracer,
            trace_manager._enabled,
            trace_manager._init_error,
        )

        try:
            trace_manager._initialized = False
            trace_manager._tracer = None
            trace_manager._enabled = False
            trace_manager._init_error = ""
            with (
                patch.dict(
                    "os.environ",
                    {
                        "MARXOS_PHOENIX_ENABLED": "1",
                    },
                    clear=False,
                ),
                patch("app.load_vectorstore", return_value=FakeDb()),
                patch("app.paragraph_vectorstore_exists", return_value=False),
                patch("marxos.generation.llm_client.OpenAI", return_value=fake_client),
            ):
                answer = app.run_query("Phoenix fallback test?")
        finally:
            (
                trace_manager._initialized,
                trace_manager._tracer,
                trace_manager._enabled,
                trace_manager._init_error,
            ) = original_state

        self.assertTrue(answer)
        self.assertIn("Phoenix", answer)

    # ── P2 高频回归：版本尊重与专题越界 ─────────────────────────────

    def test_explicit_edition_request_respects_collection(self):
        wenji = app.constraints_from_query("《雇佣劳动与资本》收录在文集中哪一卷？")
        xuanji = app.constraints_from_query("《雇佣劳动与资本》收录在选集中哪一卷？")

        self.assertEqual(wenji.get("sources"), {"mea01.pdf"})
        self.assertEqual(xuanji.get("sources"), {"mes01.pdf"})

    def test_unversioned_query_prefers_configured_edition_order(self):
        constraints = app.constraints_from_query("《雇佣劳动与资本》收录在哪一卷？")

        # 全集 > 文集 > 选集（配置优先级 MARXOS_PREFERRED_EDITIONS）
        self.assertEqual(constraints.get("entries", [{}])[0].get("source"), "me06.pdf")
        self.assertEqual(
            constraints.get("sources"),
            {"me06.pdf", "mea01.pdf", "mes01.pdf"},
        )

    def test_topic_followup_does_not_leak_into_unrelated_question(self):
        from marxos.web import support as web_support
        from web_app import MarxOSHandler

        history = [
            {
                "role": "bot",
                "text": "马克思关于农民问题的论述……",
                "topic": {"topic_id": "peasant_cooperative", "topic_label": "农民问题与土地问题"},
            },
        ]
        unrelated = "如何理解剩余价值这个概念？"
        self.assertEqual(
            web_support.topic_scoped_query(
                unrelated,
                history,
                web_support.is_contextual_followup,
                MarxOSHandler._names_explicit_subject,
            ),
            unrelated,
        )
        contextual = "再展开说说小农的具体处境"
        self.assertEqual(
            web_support.topic_scoped_query(
                contextual,
                history,
                web_support.is_contextual_followup,
                MarxOSHandler._names_explicit_subject,
            ),
            "农民问题与土地问题：再展开说说小农的具体处境",
        )


if __name__ == "__main__":
    unittest.main()
