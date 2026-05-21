import unittest

from rag.core_classics import load_core_bibliography


class CoreBibliographyTests(unittest.TestCase):
    def test_core_bibliography_is_grouped_into_three_sections(self):
        sections = load_core_bibliography()

        self.assertEqual(
            [section.get("id") for section in sections],
            [
                "marxist_philosophy",
                "marxist_political_economy",
                "scientific_socialism",
            ],
        )

    def test_each_section_contains_expected_core_work_ids(self):
        sections = {section.get("id"): section for section in load_core_bibliography()}

        philosophy_ids = [work.get("classic_id") for work in sections["marxist_philosophy"]["works"]]
        political_economy_ids = [work.get("classic_id") for work in sections["marxist_political_economy"]["works"]]
        socialism_ids = [work.get("classic_id") for work in sections["scientific_socialism"]["works"]]

        self.assertIn("theses_feuerbach", philosophy_ids)
        self.assertIn("capital_vol1", political_economy_ids)
        self.assertIn("communist_manifesto", socialism_ids)


if __name__ == "__main__":
    unittest.main()
