import unittest

from rag.build_vectorstore_from_cache import infer_page_metadata
from rag.build_vectorstore_from_cache import infer_page_metadata_from_layout
from rag.build_vectorstore_from_cache import infer_page_from_sequence


class PageMetadataInferenceTests(unittest.TestCase):
    def test_keeps_embedded_line_end_number_untrusted(self):
        text = "\u89e3\u7b54\u3002103\n\u6b63\u6587\u5185\u5bb9"

        printed_page, article = infer_page_metadata(text, "fallback", pdf_page=207)

        self.assertIsNone(printed_page)
        self.assertEqual(article, "fallback")

    def test_keeps_annotation_page_ranges_untrusted(self):
        text = (
            "\u6b63\u6587\u5185\u5bb9\n"
            "\u300a\u6587\u96c6\u300b1843\u5e74\u7248\u7b2c84-85\u9875\u3002"
        )

        printed_page, article = infer_page_metadata(text, "fallback", pdf_page=215)

        self.assertIsNone(printed_page)
        self.assertEqual(article, "fallback")

    def test_keeps_year_like_header_untrusted(self):
        text = "1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f\n\u6b63\u6587\u5185\u5bb9"

        printed_page, article = infer_page_metadata(text, "fallback", pdf_page=209)

        self.assertIsNone(printed_page)
        self.assertEqual(article, "fallback")

    def test_layout_metadata_prefers_header_footer_over_body_notes(self):
        cleaned_page = {
            "cleaned_text": "\u6b63\u6587\u6ce8\u91ca\u63d0\u5230\u7b2c84-85\u9875\u3002",
            "header_text": "\u7b2c\u4e00\u7ae0\u5546\u54c191",
            "footer_text": "",
        }

        printed_page, article, source = infer_page_metadata_from_layout(
            cleaned_page,
            "fallback",
            pdf_page=103,
        )

        self.assertEqual(printed_page, 91)
        self.assertEqual(article, "fallback")
        self.assertEqual(source, "ocr_layout")

    def test_layout_metadata_rejects_body_only_note_page_number(self):
        cleaned_page = {
            "cleaned_text": "\u6b63\u6587\u6ce8\u91ca\u63d0\u5230\u7b2c84-85\u9875\u3002",
            "header_text": "",
            "footer_text": "",
        }

        printed_page, article, source = infer_page_metadata_from_layout(
            cleaned_page,
            "fallback",
            pdf_page=215,
        )

        self.assertIsNone(printed_page)
        self.assertEqual(article, "fallback")
        self.assertEqual(source, "text_margin")

    def test_sequence_fills_short_gap_after_trusted_page(self):
        context = {"mea01.pdf": {"pdf_page": 203, "printed_page": 182}}

        printed_page, source = infer_page_from_sequence(
            "mea01.pdf",
            204,
            None,
            "text_margin",
            context,
        )

        self.assertEqual(printed_page, 183)
        self.assertEqual(source, "page_sequence")
        self.assertEqual(context["mea01.pdf"]["printed_page"], 183)

    def test_sequence_does_not_fill_large_gap(self):
        context = {"mea01.pdf": {"pdf_page": 203, "printed_page": 182}}

        printed_page, source = infer_page_from_sequence(
            "mea01.pdf",
            210,
            None,
            "text_margin",
            context,
        )

        self.assertIsNone(printed_page)
        self.assertEqual(source, "text_margin")

    def test_sequence_corrects_outlier_after_stable_run(self):
        context = {
            "mea01.pdf": {
                "pdf_page": 58,
                "printed_page": 37,
                "run_length": 8,
            }
        }

        printed_page, source = infer_page_from_sequence(
            "mea01.pdf",
            59,
            8,
            "ocr_layout",
            context,
        )

        self.assertEqual(printed_page, 38)
        self.assertEqual(source, "page_sequence_corrected")

    def test_sequence_keeps_outlier_before_stable_run(self):
        context = {
            "mea01.pdf": {
                "pdf_page": 20,
                "printed_page": 5,
                "run_length": 1,
            }
        }

        printed_page, source = infer_page_from_sequence(
            "mea01.pdf",
            23,
            3,
            "ocr_layout",
            context,
        )

        self.assertEqual(printed_page, 3)
        self.assertEqual(source, "ocr_layout")


if __name__ == "__main__":
    unittest.main()
