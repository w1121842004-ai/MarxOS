import unittest

from langchain_core.documents import Document

from rag.paragraph_cache import (
    is_incomplete_paragraph,
    merge_records,
    paragraph_record,
    paragraph_record_to_document,
    split_page_paragraphs,
)


class ParagraphCacheTests(unittest.TestCase):
    def test_split_page_paragraphs_joins_visual_lines_and_filters_repeated_title(self):
        metadata = {
            "article": "家庭、私有制和国家的起源",
            "section": "家庭、私有制和国家的起源",
        }
        text = (
            "ω\n"
            "家庭、私有制和国家的起源\n"
            "国家是社会在一定发展阶段上的产物，\n"
            "国家是承认这个社会陷入了不可解决的矛盾。\n"
            "第二段开始，\n"
            "这里结束。"
        )

        paragraphs = split_page_paragraphs(text, metadata, min_chars=12)

        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(paragraphs[0], "国家是社会在一定发展阶段上的产物，国家是承认这个社会陷入了不可解决的矛盾。")
        self.assertEqual(paragraphs[1], "第二段开始，这里结束。")

    def test_incomplete_paragraph_detection(self):
        self.assertTrue(is_incomplete_paragraph("这是一段跨页开头"))
        self.assertFalse(is_incomplete_paragraph("这是一段完整文字。"))

    def test_merge_records_keeps_page_span(self):
        doc_a = Document(
            page_content="",
            metadata={"pdf_page": 10, "printed_page": 8, "citation_page": 8, "citation_page_type": "printed_page"},
        )
        doc_b = Document(
            page_content="",
            metadata={"pdf_page": 11, "printed_page": 9, "citation_page": 9, "citation_page_type": "printed_page"},
        )
        left = paragraph_record("test.pdf", doc_a, "这一段在上一页没有结束", 1)
        right = paragraph_record("test.pdf", doc_b, "下一页继续结束。", 1)

        merged = merge_records(left, right)

        self.assertEqual(merged["paragraph_text"], "这一段在上一页没有结束下一页继续结束。")
        self.assertEqual(merged["page_span"], [10, 11])
        self.assertEqual(merged["pdf_page_start"], 10)
        self.assertEqual(merged["pdf_page_end"], 11)
        self.assertTrue(merged["cross_page"])

    def test_paragraph_record_to_document_adds_retrieval_metadata(self):
        record = {
            "source": "mea04.pdf",
            "book": "马克思恩格斯文集 第4卷",
            "article": "家庭、私有制和国家的起源",
            "section": "家庭、私有制和国家的起源",
            "paragraph_text": "国家是社会在一定发展阶段上的产物。",
            "pdf_page_start": 206,
            "pdf_page_end": 207,
            "printed_page_start": 180,
            "printed_page_end": 181,
            "citation_page_start": 180,
            "citation_page_end": 181,
            "citation_page_type": "printed_page",
            "paragraph_id": "mea04.pdf#p000410",
        }

        doc = paragraph_record_to_document(record)

        self.assertEqual(doc.page_content, "国家是社会在一定发展阶段上的产物。")
        self.assertEqual(doc.metadata["retrieval_unit"], "paragraph")
        self.assertEqual(doc.metadata["page"], 180)
        self.assertEqual(doc.metadata["pdf_page"], 206)
        self.assertEqual(doc.metadata["printed_page"], 180)
        self.assertEqual(doc.metadata["page_range"], "180-181")


if __name__ == "__main__":
    unittest.main()
