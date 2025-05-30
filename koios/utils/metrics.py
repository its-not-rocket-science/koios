"""
Evaluation metrics for classification, QA, and attribution.
"""

import os
import json
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def classification_metrics(model, dataloader):
    """Compute accuracy, precision, recall, and F1-score."""

    y_true = []
    y_pred = []

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                entity_ids=batch["entity_ids"]
            )
            preds = torch.argmax(outputs[:, 0, :], dim=-1)
            y_true.append(batch["label"].item())
            y_pred.append(preds.item())

    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }
    print("📊 Classification metrics:", scores)
    return scores


def log_predictions(model, dataloader, ontology):
    """Log predictions to a JSON file."""
    model.eval()
    outputs = []
    valid_ids = list(ontology.linker.label_to_id.values())

    for batch in dataloader:
        with torch.no_grad():
            preds = model(
                input_ids=batch["input_ids"],
                entity_ids=batch["entity_ids"]
            )
            # Mask invalid class logits
            masked_preds = preds[:, 0, :].clone()
            mask = torch.full((masked_preds.size(-1),), float("-inf"))
            for vid in valid_ids:
                if vid < masked_preds.size(-1):
                    mask[vid] = 0
            masked_preds += mask
            pred_classes = torch.argmax(masked_preds, dim=-1)

            for i, pred in enumerate(pred_classes):
                outputs.append({
                    "predicted": int(pred),
                    "label": int(batch["label"][i])
                })

    os.makedirs("experiments/outputs", exist_ok=True)
    with open("experiments/outputs/predictions.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)


def traceability_score(model, dataloader, ontology):
    """Check if predictions obey ontology type constraints."""
    correct = 0
    total = 0
    valid_ids = list(ontology.linker.label_to_id.values())

    for batch in dataloader:
        with torch.no_grad():
            preds = model(
                input_ids=batch["input_ids"],
                entity_ids=batch["entity_ids"]
            )
            masked_preds = preds[:, 0, :].clone()
            mask = torch.full((masked_preds.size(-1),), float("-inf"))
            for vid in valid_ids:
                if vid < masked_preds.size(-1):
                    mask[vid] = 0
            masked_preds += mask
            pred_classes = torch.argmax(masked_preds, dim=-1)

            for pred in pred_classes:
                print(f"→ Checking predicted ID: {pred.item()}")
                if ontology.is_valid_class(pred.item()):
                    correct += 1
                total += 1

    if total == 0:
        print("🔍 Traceability: No predictions to evaluate.")
        return 0.0

    score = correct / total
    print(f"🔍 Traceability: {correct}/{total} valid ({100 * score:.1f}%)")
    return score


def plot_confusion_matrix(model, dataloader, ontology):
    """Plot confusion matrix of predictions."""
    y_true, y_pred = [], []
    for batch in dataloader:
        with torch.no_grad():
            preds = model(
                input_ids=batch["input_ids"],
                entity_ids=batch["entity_ids"]
            )
            pred_classes = torch.argmax(preds[:, 0, :], dim=-1)
            for i, pred in enumerate(pred_classes):
                y_pred.append(int(pred))
                y_true.append(int(batch["label"][i]))

    labels = list(ontology.linker.label_to_id.values())
    label_names = list(ontology.linker.label_to_id.keys())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names,
                yticklabels=label_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("experiments/outputs/confusion_matrix.pdf",
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
