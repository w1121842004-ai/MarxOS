import unittest

from marxos.data.bibliography_v2 import BibliographyIndex


WORK_CATALOG = {
    "$schema": "work_catalog_v2",
    "editions": {
        "wenji": {
            "name": "马克思恩格斯文集",
            "publisher": "人民出版社",
            "year": 2009,
            "volumes": 10,
            "volume_map": {
                "1": {"source": "mea01.pdf", "cover": "1843—1848年著作"},
            },
        },
        "xuanji": {
            "name": "马克思恩格斯选集",
            "publisher": "人民出版社",
            "year": 2012,
            "edition": "第3版",
            "volumes": 4,
            "volume_map": {
                "1": {"source": "mes01.pdf", "cover": "1843—1859年著作"},
            },
        },
    },
    "works": [
        {
            "work_id": "communist-manifesto",
            "title": "共产党宣言",
            "aliases": ["共产主义宣言"],
            "editions": {
                "wenji_v1": {
                    "source": "mea01.pdf",
                    "article_title": "共产党宣言",
                    "start_page": 500,
                    "end_page": 550,
                },
                "xuanji_v1": {
                    "source": "mes01.pdf",
                    "article_title": "共产党宣言",
                    "start_page": 376,
                    "end_page": 435,
                },
            },
        },
        {
            "work_id": "unrelated-work",
            "title": "另一篇文章",
            "aliases": [],
            "editions": {
                "wenji_v1": {
                    "source": "mea01.pdf",
                    "article_title": "另一篇文章",
                    "start_page": 520,
                    "end_page": 530,
                }
            },
        },
    ],
}

ARTICLE_MAP = {
    "mea01.pdf": {
        "book": "马克思恩格斯文集 第1卷",
        "entries": [
            {
                "title": "共产党宣言",
                "start_printed_page": 500,
                "end_printed_page": 550,
                "level": 1,
                "parent": None,
            },
            {
                "title": "未收入作品目录的短文",
                "start_printed_page": 700,
                "end_printed_page": 704,
                "level": 1,
                "parent": None,
            },
        ],
    }
}


class BibliographyV2Tests(unittest.TestCase):
    def setUp(self):
        self.index = BibliographyIndex(WORK_CATALOG, ARTICLE_MAP)

    def test_enriches_known_edition_and_work_from_source_page_and_article(self):
        original = {
            "source": "mea01.pdf",
            "article": "《共产党宣言》",
            "page_type": "body",
            "printed_page_start": 512,
            "citation_page_start": 512,
            "citation_page_type": "printed_page",
        }

        enriched = self.index.enrich(original)

        self.assertIsNot(enriched, original)
        self.assertNotIn("series", original)
        self.assertEqual(enriched["series"], "马克思恩格斯文集")
        self.assertEqual(enriched["volume"], "第1卷")
        self.assertEqual(enriched["edition_id"], "wenji-2009")
        self.assertEqual(enriched["publisher"], "人民出版社")
        self.assertEqual(enriched["publication_year"], 2009)
        self.assertEqual(enriched["work_id"], "communist-manifesto")
        self.assertTrue(enriched["article_id"].startswith("article_v2_"))
        self.assertEqual(enriched["bibliography_confidence"]["work_id"], 1.0)
        self.assertEqual(
            enriched["bibliography_sources"]["work_id"],
            "work_catalog:source+printed_page+article",
        )

    def test_article_title_disambiguates_overlapping_page_ranges(self):
        enriched = self.index.enrich(
            {
                "source": "mea01.pdf",
                "article": "另一篇文章",
                "page_type": "body",
                "citation_page_start": 525,
                "citation_page_type": "printed_page",
            }
        )

        self.assertEqual(enriched["work_id"], "unrelated-work")

    def test_unique_page_range_can_match_work_when_ocr_title_is_noisy(self):
        enriched = self.index.enrich(
            {
                "source": "mea01.pdf",
                "article": "被OCR污染的篇名",
                "page_type": "body",
                "printed_page_start": 510,
            }
        )

        self.assertEqual(enriched["work_id"], "communist-manifesto")
        self.assertEqual(enriched["bibliography_confidence"]["work_id"], 0.9)
        self.assertEqual(enriched["bibliography_sources"]["work_id"], "work_catalog:unique_source_page")

    def test_overlapping_page_ranges_do_not_guess_work_without_title(self):
        enriched = self.index.enrich(
            {
                "source": "mea01.pdf",
                "article": "被OCR污染的篇名",
                "page_type": "body",
                "printed_page_start": 525,
            }
        )

        self.assertEqual(enriched["work_id"], "")

    def test_alias_does_not_change_canonical_article_id(self):
        base = {
            "source": "mea01.pdf",
            "page_type": "body",
            "printed_page_start": 512,
        }

        canonical = self.index.enrich({**base, "article": "共产党宣言"})
        alias = self.index.enrich({**base, "article": "共产主义宣言"})

        self.assertEqual(alias["work_id"], "communist-manifesto")
        self.assertEqual(alias["article_id"], canonical["article_id"])

    def test_article_map_generates_stable_article_id_without_inventing_work(self):
        record = {
            "source": "mea01.pdf",
            "article": "未收入作品目录的短文",
            "page_type": "body",
            "printed_page_start": 701,
        }

        first = self.index.enrich(record)
        second = self.index.enrich({**record, "paragraph_text": "不同正文不应改变篇目ID"})

        self.assertEqual(first["article_id"], second["article_id"])
        self.assertEqual(first["work_id"], "")
        self.assertEqual(first["bibliography_confidence"]["article_id"], 0.95)
        self.assertEqual(first["bibliography_sources"]["article_id"], "article_map:exact_range")

    def test_notes_and_index_pages_never_receive_work_or_article_ids(self):
        for page_type in ("notes", "person_index", "subject_index", "toc"):
            with self.subTest(page_type=page_type):
                enriched = self.index.enrich(
                    {
                        "source": "mea01.pdf",
                        "article": "共产党宣言",
                        "page_type": page_type,
                        "printed_page_start": 512,
                    }
                )
                self.assertEqual(enriched["work_id"], "")
                self.assertEqual(enriched["article_id"], "")
                self.assertEqual(enriched["bibliography_sources"]["work_id"], "policy:non_body")
                self.assertEqual(enriched["bibliography_confidence"]["work_id"], 0.0)

        polluted = self.index.enrich(
            {
                "source": "mea01.pdf",
                "article": "人名索引",
                "page_type": "body",
                "printed_page_start": 512,
            }
        )
        self.assertEqual(polluted["work_id"], "")
        self.assertEqual(polluted["bibliography_sources"]["work_id"], "policy:non_body")

    def test_unknown_source_has_explicit_safe_fallback(self):
        enriched = self.index.enrich(
            {"source": "unknown.pdf", "article": "未知篇目", "page_type": "body"}
        )

        self.assertEqual(enriched["series"], "")
        self.assertEqual(enriched["volume"], "")
        self.assertEqual(enriched["edition_id"], "")
        self.assertEqual(enriched["publisher"], "")
        self.assertIsNone(enriched["publication_year"])
        self.assertEqual(enriched["work_id"], "")
        self.assertEqual(enriched["article_id"], "")
        self.assertEqual(enriched["bibliography_sources"]["edition_id"], "unknown")
        self.assertEqual(enriched["bibliography_confidence"]["edition_id"], 0.0)

    def test_xuanji_edition_id_includes_declared_edition(self):
        enriched = self.index.enrich(
            {
                "source": "mes01.pdf",
                "article": "共产党宣言",
                "page_type": "body",
                "citation_page_start": 400,
                "citation_page_type": "printed_page",
            }
        )

        self.assertEqual(enriched["edition_id"], "xuanji-2012-3e")
        self.assertEqual(enriched["series"], "马克思恩格斯选集")
        self.assertEqual(enriched["volume"], "第1卷")


if __name__ == "__main__":
    unittest.main()
