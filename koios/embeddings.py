"""
koios/embeddings.py

This module defines the EmbeddingIndex class used to wrap and manage
ontology-derived embedding vectors. It supports mapping entity names
to indices, stacking vectors into tensors, and providing vocab size
information for integration with PyTorch models.
"""
import torch
import numpy as np


class EmbeddingIndex:
    """
    A lightweight utility class to store and lookup integer indices
    for ontology-derived entities, types, or relations.

    This is used to map symbolic identifiers (e.g., 'Hurricane', 'Rainfall')
    into rows in a torch.nn.Embedding matrix, enabling structured integration
    of ontological knowledge into neural models like Koios.

    Attributes:
        index (dict): A mapping from symbolic names to integer indices.

    Example:
        idx = EmbeddingIndex({'Hurricane': 0, 'Rainfall': 1})
        row = embedding[idx['Hurricane']]  # Get vector for 'Hurricane'
    """

    def __init__(self, embeddings: dict):
        self.embeddings = embeddings
        self.index = {k: i for i, k in enumerate(embeddings.keys())}
        tensor_list = [torch.tensor(v, dtype=torch.float32) if isinstance(
            v, np.ndarray) else v for v in embeddings.values()]
        self.tensor = torch.stack(tensor_list)
        self.size = self.tensor.shape[0]

    def __getitem__(self, key):
        return self.index[key]

    def __contains__(self, key):
        return key in self.index

    def __len__(self):
        return self.size
    
    def get(self, key, default=None):
        """s
        Dictionary-like safe lookup for an entity index.

        Args:
            key (str): Entity name.
            default (int, optional): Default value if not found.

        Returns:
            int: Index of the entity, or default.
        """
        return self.index.get(key, default)

