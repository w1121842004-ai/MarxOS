import unittest

from langchain_core.documents import Document

import app


class RetrievalFrontMatterTests(unittest.TestCase):
    def test_front_matter_candidates_are_detected_and_demoted(self):
        constraints = {
            "title": "共产党宣言",
            "strict_title": True,
            "sources": {"mea02.pdf"},
        }
        body_doc = Document(
            page_content=(
                "资产阶级在历史上曾经起过非常革命的作用。"
                "至今一切社会的历史都是阶级斗争的历史。"
            ),
            metadata={
                "book": "马克思恩格斯文集 第2卷",
                "article": "共产党宣言",
                "section": "共产党宣言",
                "source": "mea02.pdf",
                "printed_page": 4,
                "citation_page": 4,
            },
        )
        preface_doc = Document(
            page_content=(
                "1883年德文版序言。本版序言不幸只能由我一个人署名了。"
                "马克思这位比其他任何人都更应受到欧美整个工人阶级感谢的人物，已经长眠于海格特公墓。"
            ),
            metadata={
                "book": "马克思恩格斯选集 第1卷",
                "article": "共产党宣言",
                "section": "共产党宣言",
                "raw_article": "1883年德文版序言",
                "source": "mes01.pdf",
                "printed_page": 380,
                "citation_page": 380,
            },
        )

        self.assertFalse(app.is_front_matter_candidate(body_doc.metadata, body_doc.page_content, constraints))
        self.assertTrue(app.is_front_matter_candidate(preface_doc.metadata, preface_doc.page_content, constraints))

        ranked = app.rerank_documents("共产党宣言讲了什么", [preface_doc, body_doc], constraints)
        self.assertEqual(ranked[0].metadata.get("source"), "mea02.pdf")
        self.assertEqual(ranked[0].metadata.get("printed_page"), 4)


if __name__ == "__main__":
    unittest.main()
