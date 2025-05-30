"""
parser.py

Provides functionality to load and parse OWL ontologies using Owlready2,
extracting class hierarchies, labels, and object properties. Specifically
designed for integrating modular ontologies such as SWEET into the Koios model
pipeline.
"""

import json
# import os
from collections import defaultdict
from pathlib import Path
from owlready2 import get_ontology
from koios.ontology.linker import EntityLinker


class SWEETOntology:
    """
    Represents a parsed OWL ontology, such as a subset of SWEET, with methods
    to extract classes, hierarchy, and relation structure.

    Attributes:
        onto: The loaded OWL ontology object.
        classes: A list of OWL classes found in the ontology.
        relations: A list of object properties (relations).
        hierarchy: A dictionary mapping superclass names to subclass names.
        labels: A dictionary mapping class names to readable labels.
    """

    def __init__(self, owl_path_or_url: str):
        """
        Load and parse an OWL ontology from a local file path or URL.

        Args:
            owl_path_or_url: Path or web URL pointing to a valid OWL file.
        """
        path = Path(owl_path_or_url)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.onto = None
            self.classes = data.get("classes", [])
            self.labels = data.get("labels", {})
            self.hierarchy = defaultdict(list, data.get("hierarchy", {}))
            self.relations = []  # Optionally populate from data.get("relations")
        else:
            self.onto = get_ontology(str(path)).load()
            self.classes = list(self.onto.classes())
            self.relations = list(self.onto.object_properties())
            self.hierarchy = defaultdict(list)
            self.labels = {}
            self._extract_structure()

        self.linker = EntityLinker(self.get_all_classes())

    def is_valid_class(self, class_id: int) -> bool:
        """
        Checks if a class ID is valid in the ontology label space.

        Args:
            class_id (int): Integer ID assigned to an ontology class.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.linker.is_valid_class(class_id)

    def _extract_structure(self):
        """
        Build class label map and subclass hierarchy from the loaded ontology.
        """
        for cls in self.classes:
            cls_name = cls.name
            self.labels[cls_name] = cls.label.first() or cls_name
            for parent in cls.is_a:
                if hasattr(parent, "name"):
                    self.hierarchy[parent.name].append(cls_name)

    def get_all_classes(self):
        """
        Return a sorted list of all known class names in the ontology.

        Returns:
            List[str]: Sorted list of class identifiers.
        """
        return sorted(self.labels.keys())

    def get_superclasses(self, cls_name: str):
        """
        Yield the names of immediate superclasses for a given class.

        Args:
            cls_name (str): The name of the class.

        Yields:
            str: Superclass name(s) of the given class.
        """
        for parent, children in self.hierarchy.items():
            if cls_name in children:
                yield parent

    def get_relations(self):
        """
        Return a list of object property triples (name, domain, range).

        Returns:
            List[Tuple[str, list, list]]: Relation name and its domain/range.
        """
        triples = []
        for rel in self.relations:
            domain = getattr(rel, "domain", [])
            range_ = getattr(rel, "range", [])
            triples.append((rel.name, domain, range_))
        return triples

    def export_to_json(self, path="ontology_structure.json"):
        """
        Save the ontology structure (classes, hierarchy, relations) to a JSON file.
        """
        data = {
            "classes": self.get_all_classes(),
            "labels": self.labels,
            "hierarchy": dict(self.hierarchy),
            "relations": [(r, [d.name for d in dlist], [r_.name for r_ in rlist])
                          for r, dlist, rlist in self.get_relations()]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[✓] Exported ontology structure to {path}")

    def generate_embedding_index(self):
        """
        Generate a dictionary mapping class/relation names to unique indices.
        """
        index = {}
        all_items = set(self.get_all_classes())
        for r, _dlist, _rlist in self.get_relations():
            all_items.add(r)
        for i, item in enumerate(sorted(all_items)):
            index[item] = i
        return index

    def print_summary(self, max_items=10):
        """
        Print a short summary of ontology content for debugging or inspection.

        Args:
            max_items (int): Number of example classes/relations to show.
        """
        print("Total classes:", len(self.classes))
        print("Total object properties:", len(self.relations))
        print("Sample classes:", list(self.labels.items())[:max_items])
        print("Sample relations:", self.get_relations()[:max_items])


if __name__ == "__main__":
    # Example usage for testing or inspection
    sweet_url = "https://sweetontology.net/phenomena.owl"
    ontology = SWEETOntology(sweet_url)
    ontology.print_summary()
