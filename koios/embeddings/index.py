"""
Provides a lightweight wrapper to look up class indices from ontology embeddings.

Classes:
    EmbeddingIndex: Maps ontology class labels to indices and optionally stores the embedding matrix.
"""

import torch


class EmbeddingIndex:
    """
    Maps class labels to indices and optionally holds an embedding tensor.

    Attributes:
        index (dict): A mapping from class labels to indices.
        index_to_label (list): A list of labels ordered by their index.
        tensor (torch.Tensor): The stacked embedding vectors.
    """

    def __init__(self, embeddings: dict[str, torch.Tensor]):
        self.index = {label: i for i, label in enumerate(embeddings)}
        self.index_to_label = list(embeddings.keys())
        self.tensor = torch.stack(list(embeddings.values()))

    def __getitem__(self, label: str) -> int:
        return self.index[label]

    def get(self, label: str, default=None):
        return self.index.get(label, default)

    def __len__(self):
        return len(self.index)

