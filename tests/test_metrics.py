import os
import json
import torch
from koios.utils.metrics import traceability_score, classification_metrics, log_predictions


class DummyModel:
    def eval(self): pass

    def __call__(self, input_ids, entity_ids):
        return torch.randn(input_ids.shape[0], input_ids.shape[1], 10)


class DummyLinker:
    label_to_id = {"Hurricane": 0, "Storm": 1}


class DummyOntology:
    linker = DummyLinker()

    def is_valid_class(self, class_id):
        return class_id in {1, 2}


def test_classification_metrics():
    model = DummyModel()
    batch = {
        "input_ids": torch.tensor([[1]]),
        "entity_ids": torch.tensor([[1]]),
        "label": torch.tensor([1])
    }
    loader = [batch]
    scores = classification_metrics(model, loader)
    assert "accuracy" in scores


def test_traceability_score():
    dummy_model = DummyModel()
    dummy_batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "entity_ids": torch.tensor([[1, 1, 1]]),
        "label": torch.tensor([1])
    }
    dummy_loader = [dummy_batch]
    score = traceability_score(dummy_model, dummy_loader, DummyOntology())
    assert 0.0 <= score <= 1.0


def test_log_predictions(tmp_path):
    model = DummyModel()
    batch = {
        "input_ids": torch.randint(0, 10, (2, 5)),
        "entity_ids": torch.randint(0, 10, (2, 5)),
        "label": torch.tensor([1, 2])
    }
    dataloader = [batch]
    os.makedirs(tmp_path / "experiments/outputs", exist_ok=True)
    os.chdir(tmp_path)
    log_predictions(model, dataloader, ontology=DummyOntology())

    out_path = tmp_path / "experiments/outputs/predictions.json"
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert isinstance(data, list)
        assert all("predicted" in d and "label" in d for d in data)
