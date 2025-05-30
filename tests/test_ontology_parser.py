import unittest
from pathlib import Path
import torch
from koios.ontology.parser import SWEETOntology
from koios.embeddings import EmbeddingIndex


class TestSWEETOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = Path(__file__).parent
        json_path = base_dir / "data" / "ontology" / "test_mini.json"
        cls.ontology = SWEETOntology(str(json_path))

    def test_class_extraction(self):
        classes = self.ontology.get_all_classes()
        self.assertIn("Hurricane", classes)
        self.assertIn("Storm", classes)

    def test_generate_embedding_index(self):
        class_names = self.ontology.get_all_classes()
        embeddings = {name: torch.eye(len(class_names))[i] for i, name in enumerate(class_names)}
        index = EmbeddingIndex(embeddings)
        self.assertIn("Hurricane", index.index)

    def test_get_relations_returns_empty(self):
        self.assertEqual(self.ontology.get_relations(), [])

    def test_get_superclasses(self):
        parents = list(self.ontology.get_superclasses("Hurricane"))
        self.assertIn("Storm", parents)

    def test_export_to_json_and_load(self):
        temp_path = Path("temp_ontology.json")
        self.ontology.export_to_json(temp_path)
        loaded = SWEETOntology(str(temp_path))
        self.assertEqual(set(loaded.labels.keys()),
                         set(self.ontology.labels.keys()))
        temp_path.unlink(missing_ok=True)

    def test_hierarchy_structure(self):
        self.assertIsInstance(self.ontology.hierarchy, dict)
        # Hurricane is likely a subclass of Storm
        children = self.ontology.hierarchy.get("Storm", [])
        self.assertIn("Hurricane", children)

    def test_print_summary_does_not_crash(self):
        try:
            print("Classes:", self.ontology.get_all_classes())
        except RuntimeError as e:
            self.fail(f"Summary printing crashed: {e}")


if __name__ == "__main__":
    unittest.main()
