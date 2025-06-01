"""
Provides functionality to generate ontology-aware embeddings from an OWL ontology
(e.g., SWEET). Supports one-hot and dummy/random initialization schemes, with
extensibility for advanced embedding methods (e.g., node2vec, TransE).

Classes:
    OntologyEmbeddingGenerator: Handles the creation of structured embeddings from ontology classes.
"""

from typing import Dict
import torch
import networkx as nx
from node2vec import Node2Vec
from koios.ontology.parser import SWEETOntology


class OntologyEmbeddingGenerator:
    """
    A utility class to generate embeddings for ontology classes.

    This class provides ontology-aware initialization schemes such as one-hot encoding
    or random embeddings, and can be extended to include more sophisticated
    graph-based embedding strategies like node2vec or TransE.

    Attributes:
        ontology (SWEETOntology): The parsed ontology object containing class labels.
        embedding_dim (int): The dimensionality of generated embeddings.
        method (str): The embedding generation method ('onehot' or 'random').
    """

    def __init__(self, ontology: SWEETOntology, embedding_dim: int = 50, method: str = "random"):
        """
        Initialize the embedding generator with a given ontology and configuration.

        Args:
            ontology (SWEETOntology): The loaded and parsed ontology object.
            embedding_dim (int, optional): Dimensionality for embeddings. Defaults to 50.
            method (str, optional): Embedding method ('random' or 'onehot'). Defaults to 'random'.
        """
        self.ontology = ontology
        self.embedding_dim = embedding_dim
        self.method = method

    def generate(self) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for all ontology classes based on the selected method.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping class names to embedding tensors.
        """
        if self.method == "random":
            return self._generate_random()
        elif self.method == "onehot":
            return self._generate_onehot()
        elif self.method == "node2vec":
            return self._generate_node2vec()
        elif self.method == "transe":
            return self._generate_transe()
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")

    def _generate_random(self) -> Dict[str, torch.Tensor]:
        """
        Generate random embeddings for each ontology class.

        Returns:
            Dict[str, torch.Tensor]: Mapping of class names to random vectors.
        """
        return {
            label: torch.randn(self.embedding_dim)
            for label in self.ontology.get_all_classes()
        }

    def _generate_onehot(self) -> Dict[str, torch.Tensor]:
        """
        Generate one-hot embeddings for each ontology class.

        Returns:
            Dict[str, torch.Tensor]: Mapping of class names to one-hot vectors.
        """
        classes = self.ontology.get_all_classes()
        identity = torch.eye(len(classes))
        return {
            label: identity[i]
            for i, label in enumerate(classes)
        }



    def _generate_node2vec(self) -> Dict[str, torch.Tensor]:
        """
        Generate node2vec embeddings using class hierarchy and relations.
        Requires the node2vec package: pip install node2vec
        """
        G = nx.DiGraph()

        # Add class hierarchy edges
        for parent, children in self.ontology.hierarchy.items():
            for child in children:
                G.add_edge(parent, child)

        # Add relation edges
        for _rel, domains, ranges in self.ontology.get_relations():
            for d in domains:
                for r in ranges:
                    G.add_edge(d.name, r.name)

        node2vec = Node2Vec(G, dimensions=self.embedding_dim, quiet=True)
        model = node2vec.fit()

        return {
            node: torch.tensor(model.wv[node], dtype=torch.float32)
            for node in model.wv.index_to_key
        }

    
    def _generate_transe(self) -> Dict[str, torch.Tensor]:
        """
        Generate TransE-style embeddings by learning class and relation vectors using simple translation-based scoring.
        
        Returns:
            Dict[str, torch.Tensor]: Mapping of class names to learned embeddings.
        """
        classes = self.ontology.get_all_classes()
        relations = [rel[0] for rel in self.ontology.get_relations()]

        # Initialize embeddings with gradient tracking
        entity_emb = {c: torch.randn(self.embedding_dim, requires_grad=True) for c in classes}
        relation_emb = {r: torch.randn(self.embedding_dim, requires_grad=True) for r in relations}

        optimizer = torch.optim.Adam(list(entity_emb.values()) + list(relation_emb.values()), lr=0.01)

        # Build (head, relation, tail) triples
        triples = []
        for r, domains, ranges in self.ontology.get_relations():
            for d in domains:
                for t in ranges:
                    if d.name in entity_emb and t.name in entity_emb:
                        triples.append((d.name, r, t.name))

        # Train with margin ranking loss
        margin = 1.0
        for _ in range(100):
            optimizer.zero_grad()
            total_loss = 0.0
            for h, r, t in triples:
                h_vec = entity_emb[h]
                r_vec = relation_emb[r]
                t_vec = entity_emb[t]

                # Negative sample: corrupt tail
                corrupted_t = t
                while corrupted_t == t:
                    corrupted_t = classes[torch.randint(len(classes), (1,)).item()]
                t_neg_vec = entity_emb[corrupted_t]

                # L1 norm for TransE scoring
                pos_score = torch.norm(h_vec + r_vec - t_vec, p=1)
                neg_score = torch.norm(h_vec + r_vec - t_neg_vec, p=1)

                # Margin loss
                loss = torch.relu(pos_score - neg_score + margin)
                total_loss += loss

            total_loss.backward()
            optimizer.step()

        # Return only entity embeddings (detached from graph)
        return {k: v.detach() for k, v in entity_emb.items()}
