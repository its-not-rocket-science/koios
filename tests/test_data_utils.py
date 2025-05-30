import tempfile
import os
from koios.utils.data import load_jsonl, save_jsonl


def test_jsonl_roundtrip():
    data = [{"id": 1, "text": "foo"}, {"id": 2, "text": "bar"}]

    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        path = f.name
        save_jsonl(data, path)

    loaded = load_jsonl(path)
    assert loaded == data

    os.remove(path)
