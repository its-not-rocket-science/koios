import unittest
import importlib.util
from pathlib import Path
import pytest
from koios.ontology.parser import SWEETOntology
from koios.ontology.generator import OntologyEmbeddingGenerator


def is_node2vec_available():
    return importlib.util.find_spec("node2vec") is not None


class TestOntologyEmbeddingGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a small test ontology relative to this file's location
        base_dir = Path(__file__).parent
        path = base_dir / "data" / "ontology" / "test_mini.owl"
        # Convert to string for owlready2 compatibility
        cls.ontology = SWEETOntology(str(path))
        cls.generator = OntologyEmbeddingGenerator(cls.ontology.hierarchy)

    def test_build_graph(self):
        G = self.generator.build_graph()
        self.assertTrue(G.has_node("Hurricane"))
        self.assertTrue(G.has_edge("Hurricane", "Storm"))

    def test_generate_onehot(self):
        vectors = self.generator.generate_onehot()
        self.assertIn("Hurricane", vectors)
        self.assertEqual(len(vectors["Hurricane"]), len(vectors))
        self.assertEqual(vectors["Hurricane"].sum(), 1)

    @pytest.mark.skipif(not is_node2vec_available(), reason="node2vec is not installed")
    def test_generate_node2vec(self):
        vectors = self.generator.generate_node2vec(dim=8)
        assert "Hurricane" in vectors
        assert len(vectors["Hurricane"]) == 8

    def test_generate_transe_raises(self):
        with self.assertRaises(NotImplementedError):
            self.generator.generate_transe()


if __name__ == "__main__":
    unittest.main()
