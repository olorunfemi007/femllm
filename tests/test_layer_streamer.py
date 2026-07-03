import time
import torch
from src.femllm.layer_streamer import LayerStreamer

def test_chunks_computed_correctly_window_1(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    assert streamer.chunks == [[0], [4], [8], [12], [16], [20]]

def test_chunks_computed_correctly_window_2(shard_dir_w2):
    streamer = LayerStreamer(f"{shard_dir_w2}/worker_0.safetensors", layer_indices=[0, 1, 8, 9, 16, 17], window_size=2)
    assert streamer.chunks == [[0, 1], [8, 9], [16, 17]]

def test_current_start_layer_begins_at_first_chunk(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    assert streamer.current_start_layer() == 0

def test_current_chunk_returns_weights_for_every_layer_in_it(shard_dir_w2):
    streamer = LayerStreamer(f"{shard_dir_w2}/worker_0.safetensors", layer_indices=[0, 1, 8, 9, 16, 17], window_size=2)
    chunk = streamer.current_chunk()
    assert set(chunk.keys()) == {0, 1}
    assert "self_attn.q_proj.weight" in chunk[0]
    assert "self_attn.q_proj.weight" in chunk[1]

def test_advance_moves_to_next_chunk(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    streamer.advance()
    assert streamer.current_start_layer() == 4
    assert set(streamer.current_chunk().keys()) == {4}

def test_advance_wraps_around_cycle(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    for _ in range(6):
        streamer.advance()
    assert streamer.current_start_layer() == 0

def test_advance_reloads_consistently_on_next_lap(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    first_weight = streamer.current_chunk()[0]["self_attn.q_proj.weight"]
    for _ in range(6):
        streamer.advance()
    second_weight = streamer.current_chunk()[0]["self_attn.q_proj.weight"]
    assert torch.allclose(first_weight, second_weight)

def test_advance_next_chunk_ready_without_extra_wait(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    time.sleep(0.3)  # let background prefetch for the second chunk finish
    start = time.time()
    streamer.advance()
    elapsed = time.time() - start
    assert elapsed < 0.1  # should already be resident, not a fresh disk read
