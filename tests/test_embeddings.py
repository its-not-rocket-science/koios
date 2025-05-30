import torch
from koios.embeddings import EmbeddingIndex


def test_index_lookup():
    idx = EmbeddingIndex({
        "Rainfall": torch.tensor([1.0, 0.0]),
        "Hurricane": torch.tensor([0.0, 1.0])
    })
    assert idx["Hurricane"] == 1


def test_missing():
    idx = EmbeddingIndex({"A": torch.tensor([0.5, 0.5])})
    assert idx.get("Z", 0) == 0
