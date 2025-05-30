import torch
from koios.model import KoiosModel


def test_model_forward():
    batch_size, seq_len = 2, 5
    model = KoiosModel(
        vocab_size=100,
        hidden_size=32,
        entity_vocab_size=10,
        entity_emb_dim=8,
        num_labels=5
    )
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    entity_ids = torch.randint(0, 10, (batch_size, seq_len))
    output = model(input_ids, entity_ids)
    assert output.shape == (batch_size, seq_len, 5)
