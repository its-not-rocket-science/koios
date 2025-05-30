import tempfile
import yaml
from koios.config import load_config


def test_load_config():
    dummy = {
        "vocab_size": 10,
        "embed_dim": 32,
        "ontology_dims": {"entity": 5, "type": 3, "relation": 2}
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(dummy, f)
        path = f.name

    cfg = load_config(path)
    assert cfg["vocab_size"] == 10
    assert cfg["ontology_dims"]["type"] == 3
