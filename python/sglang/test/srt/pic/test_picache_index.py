import torch
from sglang.srt.pic.picache import PICache, SegmentEntry
from sglang.srt.pic.segmenter import segment_hash


def _make_picache():
    class A:
        page_size = 1
        def available_size(self): return 4096
        def alloc(self, n): return torch.arange(n, dtype=torch.int64)
        def free(self, idx): pass
    class M:
        def available_size(self): return 64
        def free(self, idx): pass
    class R: pass
    return PICache(R(), A(), M(), page_size=1, disable=False)


def test_match_missing_returns_none():
    pc = _make_picache()
    assert pc._match_segment(b"\x00"*16, torch.tensor([1,2,3])) is None


def test_insert_then_match():
    pc = _make_picache()
    ids = torch.tensor([1, 2, 3], dtype=torch.int64)
    h = segment_hash(ids.tolist())
    entry = pc._insert_segment(
        seg_hash=h,
        token_ids=ids,
        full_kv_slots=torch.tensor([100, 101, 102], dtype=torch.int64),
        mamba_state_slot=5,
    )
    assert entry.last_access_time > 0
    assert entry.creation_time > 0
    matched = pc._match_segment(h, ids)
    assert matched is entry


def test_insert_idempotent_returns_existing(monkeypatch):
    pc = _make_picache()
    ids = torch.tensor([1, 2, 3], dtype=torch.int64)
    h = segment_hash(ids.tolist())
    a = pc._insert_segment(h, ids, torch.tensor([100,101,102], dtype=torch.int64), 5)
    b = pc._insert_segment(h, ids, torch.tensor([200,201,202], dtype=torch.int64), 6)
    assert a is b


def test_match_hash_collision_falls_back_to_token_ids():
    pc = _make_picache()
    ids1 = torch.tensor([1, 2, 3], dtype=torch.int64)
    ids2 = torch.tensor([7, 8, 9], dtype=torch.int64)
    h = segment_hash(ids1.tolist())
    pc._insert_segment(h, ids1, torch.tensor([10,11,12], dtype=torch.int64), 5)
    assert pc._match_segment(h, ids2) is None
