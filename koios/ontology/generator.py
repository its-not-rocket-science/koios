"""
Embedding generator for ontology concepts and relations.

Supports symbolic (one-hot) and geometric (TransE/node2vec) embeddings.
"""

import networkx as nx
import numpy as np

class OntologyEmbeddingGenerator:
    def __init__(self, hierarchy, relations=None):
        """
        Args:
            hierarchy (dict): Mapping of parent class -> list of subclasses.
            relations (list of tuples): Optional (head, relation, tail) triples.
        """
        self.hierarchy = hierarchy
        self.relations = relations or []

    def build_graph(self):
        """Converts class hierarchy into a NetworkX graph."""
        G = nx.DiGraph()
        for parent, children in self.hierarchy.items():
            for child in children:
                G.add_edge(parent, child)
        return G

    def generate_node2vec(self, dim=64):
        """
        Generates node embeddings using Node2Vec.

        Returns:
            dict: {concept_name: np.array(dim)}
        """
        from node2vec import Node2Vec
        G = self.build_graph()
        n2v = Node2Vec(G, dimensions=dim, quiet=True)
        model = n2v.fit()
        return {str(node): model.wv[str(node)] for node in G.nodes()}

    def generate_onehot(self):
        """
        Returns one-hot encoding of classes.

        Returns:
            dict: {concept_name: np.array}
        """
        concepts = sorted(set(self.hierarchy.keys()) | 
                          {c for children in self.hierarchy.values() for c in children})
        index = {c: i for i, c in enumerate(concepts)}
        vecs = {c: np.eye(len(index))[i] for c, i in index.items()}
        return vecs

    def generate_transe(self, dim=50):
        """
        Placeholder for TransE or other KG-based embedding methods.
        """
        raise NotImplementedError("TransE embedding not yet implemented.")


if __name__ == "__main__":
    # Example usage (stub):
    from koios.ontology.parser import SWEETOntology
    onto = SWEETOntology("data/ontology/test_mini.owl")
    gen = OntologyEmbeddingGenerator(onto.hierarchy)
    vectors = gen.generate_onehot()
    print("Hurricane:", vectors.get("Hurricane"))
