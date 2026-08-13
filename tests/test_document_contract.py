import unittest

from marxos.data.document_contract import (
    DOCUMENT_RECORD_VERSION,
    audit_document_records,
    inherit_record_metadata,
    normalize_document_record,
    stable_paragraph_id,
)


class DocumentContractTests(unittest.TestCase):
    def test_stable_paragraph_id_ignores_sequence_but_tracks_spans_and_text(self):
        base = {
            "source": "mea01.pdf", "paragraph_index": 1,
            "spans": [{"page_id": "mea01.pdf#pdf10", "char_start": 3, "char_end": 9}],
            "paragraph_text": "社会存在。",
        }
        first = stable_paragraph_id(base)
        self.assertEqual(first, stable_paragraph_id({**base, "paragraph_index": 999}))
        self.assertNotEqual(first, stable_paragraph_id({**base, "paragraph_text": "社会意识。"}))
        self.assertNotEqual(first, stable_paragraph_id({**base, "spans": [{**base["spans"][0], "char_start": 4}]}))
    def test_normalize_paragraph_adds_version_lineage_and_volume(self):
        record = normalize_document_record(
            {
                "source": "mea04.pdf",
                "book": "马克思恩格斯文集 第4卷",
                "article": "家庭、私有制和国家的起源",
                "paragraph_id": "mea04.pdf#p000001",
                "paragraph_text": "正文。",
                "pdf_page_start": 10,
                "pdf_page_end": 10,
                "citation_page_start": 8,
                "citation_page_end": 8,
                "citation_page_type": "printed_page",
                "page_span": [10],
                "page_type": "body",
            },
            retrieval_unit="paragraph",
        )

        self.assertEqual(record["document_record_version"], DOCUMENT_RECORD_VERSION)
        self.assertEqual(record["volume"], "第4卷")
        self.assertEqual(record["parent_paragraph_id"], "mea04.pdf#p000001")
        self.assertEqual(record["retrieval_unit"], "paragraph")
        self.assertEqual(record["source_page_ids"], ["mea04.pdf#pdf10"])

    def test_normalize_quarantines_index_and_mojibake_records(self):
        index_record = normalize_document_record(
            {
                "source": "mea01.pdf", "paragraph_id": "p1", "article": "人名索引",
                "paragraph_text": "索引条目", "page_type": "body", "page_span": [10],
            },
            retrieval_unit="paragraph",
        )
        mojibake_record = normalize_document_record(
            {
                "source": "mea01.pdf", "paragraph_id": "p2", "article": "资本论",
                "paragraph_text": "G+ÂG", "page_type": "body", "page_span": [11],
            },
            retrieval_unit="paragraph",
        )

        self.assertEqual(index_record["page_type"], "person_index")
        self.assertFalse(index_record["retrievable"])
        self.assertEqual(index_record["page_type_source"], "article_policy")
        self.assertFalse(mojibake_record["retrievable"])
        self.assertIn("mojibake", mojibake_record["quality_flags"])

    def test_normalize_quarantines_editorial_front_matter(self):
        for record in (
            {"article": "马克思主义理论研究和建设工程重点项目", "paragraph_text": "一、编写说明正文"},
            {"article": "马克思恩格斯文集 第1卷", "paragraph_text": "编辑说明"},
        ):
            normalized = normalize_document_record(
                {"source": "mea01.pdf", "paragraph_id": "front", "page_type": "body", **record},
                retrieval_unit="paragraph",
            )
            self.assertEqual("preface_editorial", normalized["page_type"])
            self.assertFalse(normalized["retrievable"])

    def test_child_inherits_all_citation_and_provenance_fields(self):
        parent = normalize_document_record(
            {
                "source": "mes01.pdf",
                "book": "马克思恩格斯选集 第1卷",
                "article": "共产党宣言",
                "section": "共产党宣言",
                "paragraph_id": "mes01.pdf#p000010",
                "paragraph_text": "全世界无产者，联合起来！",
                "pdf_page_start": 451,
                "pdf_page_end": 451,
                "printed_page_start": 435,
                "printed_page_end": 435,
                "citation_page_start": 435,
                "citation_page_end": 435,
                "citation_page_type": "printed_page",
                "page_span": [451],
                "page_type": "body",
            },
            retrieval_unit="paragraph",
        )

        child = inherit_record_metadata(parent, {"paragraph_text": "联合起来！"}, "semantic_child")

        for field in (
            "source", "book", "volume", "article", "section", "pdf_page_start",
            "pdf_page_end", "printed_page_start", "printed_page_end",
            "citation_page_start", "citation_page_end", "citation_page_type",
            "page_span", "source_page_ids", "document_record_version",
        ):
            self.assertEqual(child[field], parent[field])
        self.assertEqual(child["parent_paragraph_id"], parent["paragraph_id"])

    def test_audit_reports_stable_machine_readable_quality_codes(self):
        records = [
            {
                "paragraph_id": "x#1", "source": "x.pdf", "retrieval_unit": "paragraph",
                "paragraph_text": "目录……1", "page_type": "toc", "article": "目录",
                "pdf_page_start": 5, "pdf_page_end": 5, "page_span": [5],
            },
            {
                "paragraph_id": "x#2", "source": "x.pdf", "retrieval_unit": "paragraph",
                "paragraph_text": "正文", "page_type": "body", "article": "",
                "pdf_page_start": 9, "pdf_page_end": 8, "page_span": [9],
            },
        ]

        report = audit_document_records(records)

        self.assertEqual(report["schema_version"], "document-audit/v1")
        self.assertFalse(report["summary"]["passed"])
        self.assertEqual(
            [issue["code"] for issue in report["issues"]],
            ["FRONT_MATTER_LEAK", "ARTICLE_MISSING", "PAGE_RANGE_REVERSED"],
        )

    def test_audit_classifies_polluted_articles_and_unknown_article(self):
        records = [
            {
                "paragraph_id": "x#1", "source": "x.pdf", "retrieval_unit": "paragraph",
                "paragraph_text": "阿贝尔，卡尔……", "page_type": "body",
                "article": "人 名 索 引 ……", "pdf_page_start": 10, "pdf_page_end": 10,
            },
            {
                "paragraph_id": "x#2", "source": "x.pdf", "retrieval_unit": "paragraph",
                "paragraph_text": "尚未映射篇名的正文。", "page_type": "body",
                "article": "未知篇名", "pdf_page_start": 11, "pdf_page_end": 11,
            },
        ]

        report = audit_document_records(records)

        self.assertEqual(report["summary"]["issues_by_code"], {
            "ARTICLE_POLLUTION": 1,
            "ARTICLE_UNKNOWN": 1,
        })
        pollution = report["issues"][0]
        self.assertEqual(pollution["severity"], "error")
        self.assertEqual(pollution["policy"], "exclude_from_body_retrieval")
        self.assertFalse(pollution["exempted"])

    def test_notes_are_classified_but_exempt_from_body_pollution_and_footnote_policy(self):
        report = audit_document_records([{
            "paragraph_id": "x#note", "source": "x.pdf", "retrieval_unit": "paragraph",
            "paragraph_text": "［12］马克思在此处引用了……", "page_type": "notes",
            "article": "注 释", "pdf_page_start": 20, "pdf_page_end": 20,
        }])

        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["summary"]["exemptions_by_code"], {
            "ARTICLE_POLLUTION": 1,
            "FOOTNOTE_ORPHAN": 1,
        })
        self.assertEqual(report["issues"], [])

    def test_footnote_policy_avoids_numbered_body_paragraph_false_positive(self):
        report = audit_document_records([
            {
                "paragraph_id": "x#short", "source": "x.pdf", "retrieval_unit": "paragraph",
                "paragraph_text": "[8] 参见本卷第20页。", "page_type": "body", "article": "资本论",
                "pdf_page_start": 30, "pdf_page_end": 30,
            },
            {
                "paragraph_id": "x#long", "source": "x.pdf", "retrieval_unit": "paragraph",
                "paragraph_text": "（一）" + "这是正常的编号正文。" * 80, "page_type": "body", "article": "资本论",
                "pdf_page_start": 31, "pdf_page_end": 31,
            },
            {
                "paragraph_id": "x#page-anchor", "source": "x.pdf", "retrieval_unit": "paragraph",
                "paragraph_text": "［７７２］这是印刷页锚点，不是脚注。", "page_type": "body", "article": "资本论",
                "pdf_page_start": 32, "pdf_page_end": 32,
            },
        ])

        self.assertEqual(report["summary"]["issues_by_code"], {"FOOTNOTE_ORPHAN": 1})
        self.assertTrue(report["summary"]["passed"])

    def test_duplicate_text_distinguishes_same_page_record_and_cross_page_copy(self):
        base = {
            "source": "x.pdf", "retrieval_unit": "paragraph", "paragraph_text": "完全相同且足够长的重复正文文本。",
            "page_type": "body", "article": "资本论",
        }
        report = audit_document_records([
            {**base, "paragraph_id": "x#1", "pdf_page_start": 40, "pdf_page_end": 40},
            {**base, "paragraph_id": "x#2", "pdf_page_start": 40, "pdf_page_end": 40},
            {**base, "paragraph_id": "x#3", "pdf_page_start": 41, "pdf_page_end": 41},
        ])

        self.assertEqual(report["summary"]["issues_by_code"], {
            "DUPLICATE_PAGE_TEXT": 1,
            "DUPLICATE_TEXT": 1,
        })
        self.assertEqual(report["summary"]["issues_by_severity"], {"warning": 2})
        self.assertEqual(report["summary"]["blocking_issues"], 0)
        self.assertIn("2 warning", report["summary"]["readable"])

    def test_audit_handles_invalid_and_legacy_page_values_without_crashing(self):
        base = {
            "source": "x.pdf", "retrieval_unit": "milvus_passage", "paragraph_text": "正文内容。",
            "page_type": "body", "article": "资本论",
        }
        report = audit_document_records([
            {**base, "paragraph_id": "x#bad", "pdf_page_start": "abc", "pdf_page_end": "abc"},
            {**base, "paragraph_id": "x#legacy", "pdf_page": 12},
            {"id": "manifest", "retrieval_unit": "manifest", "text": "metadata"},
        ])

        self.assertEqual(report["summary"]["issues_by_code"], {"PAGE_REQUIRED": 1})
        self.assertEqual(report["issues"][0]["record_id"], "x#bad")


if __name__ == "__main__":
    unittest.main()
