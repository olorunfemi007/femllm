import pytest
import os

# manual scripts require a running worker server; never collect them
collect_ignore_glob = ["test_*_manual.py"]

MODEL_DIR = "models/tinyllama"

@pytest.fixture(scope="session")
def model_dir():
    if not os.path.exists(MODEL_DIR):
        pytest.skip("TinyLlama not downloaded — run Task 1 setup first")
    return MODEL_DIR

@pytest.fixture(scope="session")
def shard_dir(model_dir):
    from tools.split_model import split_model
    out = "shards/tinyllama"
    if not os.path.exists(os.path.join(out, "manifest.json")):
        split_model(model_dir, out, num_workers=4, window_size=1)
    return out

@pytest.fixture(scope="session")
def shard_dir_w2(model_dir):
    from tools.split_model import split_model
    out = "shards/tinyllama_w2"
    if not os.path.exists(os.path.join(out, "manifest.json")):
        split_model(model_dir, out, num_workers=4, window_size=2)
    return out
