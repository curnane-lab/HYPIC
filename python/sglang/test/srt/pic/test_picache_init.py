import torch
import pytest
from sglang.srt.pic.picache import PICache

class FakeAllocator:
    page_size = 1
    def available_size(self): return 1024
    def alloc(self, n): return torch.arange(n, dtype=torch.int64)
    def free(self, idx): pass

class FakeReqPool:
    pass

class FakeMambaPool:
    def available_size(self): return 64
    def alloc_one(self): return 0
    def free(self, idx): pass

def test_picache_init_smoke():
    pc = PICache(
        req_to_token_pool=FakeReqPool(),
        token_to_kv_pool_allocator=FakeAllocator(),
        mamba_pool=FakeMambaPool(),
        page_size=1,
        disable=False,
        pic_mode="addition",
    )
    assert pc.supports_mamba() is True
    assert pc.is_tree_cache() is True
    assert pc.is_chunk_cache() is False
    assert pc.page_size == 1
    assert pc.disable is False
    assert pc.evictable_size() == 0

def test_picache_rejects_unknown_mode():
    with pytest.raises((AssertionError, NotImplementedError)):
        PICache(
            req_to_token_pool=FakeReqPool(),
            token_to_kv_pool_allocator=FakeAllocator(),
            mamba_pool=FakeMambaPool(),
            page_size=1, disable=False,
            pic_mode="bogus",
        )
