# Koios: Ontology-Grounded Transformer for Interpretable NLP

**Koios** is a knowledge-augmented transformer architecture that integrates structured embeddings from OWL ontologies into the input layer of a language model.

[![Tests](https://github.com/its-not-rocket-science/koios/actions/workflows/ci.yml/badge.svg)](https://github.com/its-not-rocket-science/koios/actions)

---

### ✨ Features

- Modular embedding layer with tokens, entities, types, relation embeddings
- OWL ontology support (via [Owlready2](https://owlready2.readthedocs.io/))
- Load ontology from either `.owl` or `.json`
- Embedding initialization from TransE, node2vec, or manual vectors
- Transformer-based backbone
- Built-in classification and traceability metrics
- Robust unit testing and >75% test coverage
- Interpretability and traceability of model predictions
- Fully test-covered pipeline with PyTorch + pytest

---

### 📦 Installation

```bash
git clone https://github.com/its-not-rocket-science/koios
cd koios
pip install -r requirements.txt
