import pytest
import os

MODEL_DIR = "models/tinyllama"

@pytest.fixture(scope="session")
def model_dir():
    if not os.path.exists(MODEL_DIR):
        pytest.skip("TinyLlama not downloaded — run Task 1 setup first")
    return MODEL_DIR

@pytest.fixture(scope="session")
def shard_dir(tmp_path_factory, model_dir):
    from tools.split_model import split_model
    out = str(tmp_path_factory.mktemp("shards"))
    split_model(model_dir, out, num_workers=4, window_size=1)
    return out

@pytest.fixture(scope="session")
def shard_dir_w2(tmp_path_factory, model_dir):
    from tools.split_model import split_model
    out = str(tmp_path_factory.mktemp("shards_w2"))
    split_model(model_dir, out, num_workers=4, window_size=2)
    return out
