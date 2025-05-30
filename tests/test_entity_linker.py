import unittest
from koios.ontology.linker import EntityLinker


class TestEntityLinker(unittest.TestCase):

    def setUp(self):
        self.linker = EntityLinker(["Phenomena", "Storm", "Hurricane"])

    def test_exact_match_finds_span(self):
        text = "A major Hurricane caused damage."
        matches = self.linker.exact_match(text)
        self.assertTrue(any("Hurricane" == m[1] for m in matches))

    def test_exact_match_case_insensitive(self):
        text = "a hurricane hit"
        matches = self.linker.exact_match(text)
        self.assertTrue(any("Hurricane" == m[1] for m in matches))

    def test_fuzzy_match_hits(self):
        match = self.linker.fuzzy_match("Huricane")  # Intentional typo
        self.assertEqual(match, "Hurricane")

    def test_fuzzy_match_miss(self):
        match = self.linker.fuzzy_match("Banana", cutoff=0.95)
        self.assertIsNone(match)

    def test_link_text_exact_mode(self):
        text = "Storm and Hurricane are major phenomena."
        links = self.linker.link_text(text, method="exact", return_tokens=False)
        mentions = [l["mention"].lower() for l in links]
        print("EXACT mentions:", mentions)
        self.assertIn("storm", mentions)
        self.assertIn("hurricane", mentions)

    def test_link_text_fuzzy_mode(self):
        text = "The hurican and strom intensified."
        links = self.linker.link_text(text, method="fuzzy", cutoff=0.4, return_tokens=False)
        classes = [l["class"] for l in links]
        print("FUZZY classes:", classes)
        self.assertIn("Hurricane", classes)
        self.assertIn("Storm", classes)


if __name__ == "__main__":
    unittest.main()
