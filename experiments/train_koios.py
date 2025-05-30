"""
Train Koios model on a dataset with entity linking and ontology embeddings.
"""

import torch
import json
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from koios.config import load_config
from koios.ontology.parser import SWEETOntology
from koios.ontology.linker import EntityLinker
from koios.ontology.generator import OntologyEmbeddingGenerator
from koios.embeddings import EmbeddingIndex
from koios.model import KoiosModel
from koios.utils.metrics import log_predictions, traceability_score, classification_metrics, plot_confusion_matrix



class KoiosDataset(Dataset):
    """
    Custom dataset for Koios model training, loading samples from a JSONL file.
    Each sample contains text, label, and entity linking information.
    Supports tokenization and padding to a maximum length.
    """
    def __init__(self, jsonl_path, linker, max_len=32):
        self.linker = linker
        self.max_len = max_len
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        label = item["label"]

        tokens = self.linker.link_text(text, return_tokens=True) or []

        if not tokens:
            tokens = [{
                "token_id": 0,
                "text": "<PAD>",
                "entity_id": 0
            }]

        ids = torch.tensor([t["token_id"] for t in tokens[:self.max_len]], dtype=torch.long)
        entity_ids = torch.tensor([t["entity_id"] for t in tokens[:self.max_len]], dtype=torch.long)

        return {
            "input_ids": ids,
            "entity_ids": entity_ids,
            "label": torch.tensor(label, dtype=torch.long)
        }



def train():
    print("🔧 Loading config...")
    config = load_config("koios/config.yaml")

    print("🧠 Loading ontology...")
    ontology = SWEETOntology(config["ontology_path"])
    print("🔢 Label ID mapping:", ontology.linker.label_to_id)

    print("🔗 Linking entities...")
    linker = EntityLinker(ontology.hierarchy)

    print("🔁 Generating ontology embeddings...")
    generator = OntologyEmbeddingGenerator(ontology.hierarchy)
    embeddings = generator.generate_node2vec(dim=config["embedding_dim"])
    emb_index = EmbeddingIndex(embeddings)

    print("📐 Building model...")
    model = KoiosModel(
        vocab_size=32000,
        hidden_size=128,
        entity_vocab_size=len(emb_index),
        entity_emb_dim=config["embedding_dim"],
        num_labels=config["num_labels"]
    )

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    model.train()

    print("📦 Loading dataset...")
    dataset = KoiosDataset("data/processed/dev.jsonl", linker)
    loader = DataLoader(dataset, batch_size=4)

    print("🚀 Training...")
    for epoch in range(config["epochs"]):
        for batch in loader:
            if batch["input_ids"].shape[1] == 0:
                continue  # Skip empty input sequences

            outputs = model(
                input_ids=batch["input_ids"],
                entity_ids=batch["entity_ids"]
            )
            loss = torch.nn.functional.cross_entropy(
                outputs[:, 0, :], batch["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"✅ Epoch {epoch+1} done")

    print("✅ Training done")
    print("📊 Evaluating model...")
    
    print("🔍 Classification metrics:")
    classification_metrics(model, loader)
    
    print("🧪 Logging outputs...")
    log_predictions(model, loader, ontology)

    print("🧠 Traceability score:")
    traceability_score(model, loader, ontology)
    
    print("📊 Confusion matrix:")
    plot_confusion_matrix(model, loader, ontology)

if __name__ == "__main__":
    train()
