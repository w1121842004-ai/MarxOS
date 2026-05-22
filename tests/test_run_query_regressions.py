import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

import app


class RunQueryRegressionTests(unittest.TestCase):
    def test_run_query_rejects_unsupported_slogan_without_vector_or_llm(self):
        query = "以人民为中心是否出自马克思原著？"

        with patch("app.load_vectorstore") as load_vectorstore, patch("app.OpenAI") as openai:
            answer = app.run_query(query)

        load_vectorstore.assert_not_called()
        openai.assert_not_called()
        self.assertIn("不是马克思原著中的原文表达", answer)

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

        with patch("app.load_vectorstore", return_value=FakeDb()), patch("app.OpenAI") as openai:
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
            patch("app.paragraph_vectorstore_exists", return_value=False),
            patch("app.OpenAI", return_value=fake_client) as openai,
        ):
            answer = app.run_query("人的本质是什么？")

        openai.assert_called_once()
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
            patch("app.OpenAI", return_value=fake_client),
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
            patch("app.OpenAI") as openai,
        ):
            answer = app.run_query("测试概念说明")

        openai.assert_not_called()
        self.assertIn("TRACE_ONLY", answer)
        self.assertIn("Top chunks", answer)


if __name__ == "__main__":
    unittest.main()
