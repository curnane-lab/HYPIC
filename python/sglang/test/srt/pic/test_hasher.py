import torch
from sglang.srt.pic.segmenter import segment_hash


def test_hash_is_16_bytes():
    h = segment_hash([1, 2, 3])
    assert isinstance(h, bytes)
    assert len(h) == 16


def test_hash_deterministic():
    assert segment_hash([1, 2, 3]) == segment_hash([1, 2, 3])


def test_hash_distinguishes_order():
    assert segment_hash([1, 2, 3]) != segment_hash([3, 2, 1])


def test_hash_accepts_tensor():
    a = segment_hash([1, 2, 3])
    b = segment_hash(torch.tensor([1, 2, 3], dtype=torch.int64))
    assert a == b
