import unittest
from koios.embeddings.ontology_embedding_generator import OntologyEmbeddingGenerator
# from koios.ontology.parser import SWEETOntology


class DummyOntology:
    def get_all_classes(self):
        return ["A", "B", "C"]

    @property
    def hierarchy(self):
        return {"A": ["B"], "B": ["C"]}

    def get_relations(self):
        class Dummy:
            def __init__(self, name):
                self.name = name
        return [("relatedTo", [Dummy("A")], [Dummy("C")])]


class TestOntologyEmbeddingGenerator(unittest.TestCase):

    def setUp(self):
        self.dummy_ontology = DummyOntology()

    def test_random_embedding(self):
        generator = OntologyEmbeddingGenerator(
            self.dummy_ontology, embedding_dim=10, method="random")
        embeddings = generator.generate()
        self.assertEqual(len(embeddings), 3)
        for tensor in embeddings.values():
            self.assertEqual(tensor.shape, (10,))

    def test_onehot_embedding(self):
        generator = OntologyEmbeddingGenerator(
            self.dummy_ontology, method="onehot")
        embeddings = generator.generate()
        self.assertEqual(len(embeddings), 3)
        for _label, tensor in embeddings.items():
            self.assertEqual(tensor.sum().item(), 1.0)
            self.assertEqual(tensor.shape[0], 3)

    def test_node2vec_embedding(self):
        generator = OntologyEmbeddingGenerator(
            self.dummy_ontology, embedding_dim=8, method="node2vec")
        embeddings = generator.generate()
        self.assertIn("A", embeddings)
        self.assertEqual(embeddings["A"].shape[0], 8)

    def test_transe_embedding(self):
        generator = OntologyEmbeddingGenerator(
            self.dummy_ontology, embedding_dim=6, method="transe")
        embeddings = generator.generate()
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(embeddings["A"].shape, (6,))

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            OntologyEmbeddingGenerator(
                self.dummy_ontology, method="invalid").generate()

    def test_empty_classes(self):
        class EmptyOntology:
            def get_all_classes(self):
                return []

            @property
            def hierarchy(self):
                return {}

            def get_relations(self):
                return []

        generator = OntologyEmbeddingGenerator(
            EmptyOntology(), method="random")
        embeddings = generator.generate()
        self.assertEqual(len(embeddings), 0)


if __name__ == '__main__':
    unittest.main()
