import pytest
from sglang.srt.managers.io_struct import GenerateReqInput


def test_pic_segments_field_default_none():
    req = GenerateReqInput(text="hi")
    assert req.pic_segments is None


def test_pic_segments_field_accepts_list_of_pairs():
    req = GenerateReqInput(text="hi", pic_segments=[(0, 2), (2, 4)])
    assert req.pic_segments == [(0, 2), (2, 4)]
