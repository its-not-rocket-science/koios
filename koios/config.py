"""
Koios configuration module. Loads model, data, and ontology settings.
"""

import yaml
import argparse


def load_config(path=None):
    """
    Loads a YAML config file.
    
    Args:
        path (str): Path to the YAML config file. If None, defaults to "koios/config.yaml".

    Returns:
        dict: Configuration settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="koios/config.yaml", help="Path to YAML config")
    args, _ = parser.parse_known_args()
    path = path or args.config
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set default values if not present
    config.setdefault("embedding_dim", 128)
    config.setdefault("vocab_size", 32000)
    config.setdefault("num_labels", 10)
    config.setdefault("lr", 1e-3)
    config.setdefault("epochs", 3)
    config.setdefault("ontology_path", "data/ontology/test_mini.owl")

    return config