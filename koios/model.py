"""
Defines the Koios model architecture, extending a Transformer encoder with
ontology-aware entity, type, and relation embeddings.
"""
import torch
import torch.nn as nn


class KoiosModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, entity_vocab_size, entity_emb_dim, num_labels):
        """
        Args:
            vocab_size (int): Size of the token vocabulary.
            embed_dim (int): Dimensionality of token embeddings.
            ontology_dims (dict): Dimensions for 'entity', 'type', 'relation' embeddings.
        """
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.entity_emb = nn.Embedding(entity_vocab_size, entity_emb_dim)
        self.proj = nn.Linear(hidden_size + entity_emb_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, entity_ids):
        """
        Combines embeddings and feeds through transformer.

        Returns:
            logits over vocabulary
        """
        token_vecs = self.token_emb(input_ids)
        entity_vecs = self.entity_emb(entity_ids)
        x = torch.cat([token_vecs, entity_vecs], dim=-1)
        x = self.proj(x)
        x = self.encoder(x)
        return self.classifier(x)
