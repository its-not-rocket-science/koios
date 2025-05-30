
"""
Dataset loading and preprocessing utilities.
"""

import json


def load_jsonl(path):
    """
    Loads a JSONL file into a list of dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    """
    Saves a list of dicts to a JSONL file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
