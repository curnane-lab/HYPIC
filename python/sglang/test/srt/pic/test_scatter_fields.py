import dataclasses

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)

_FIELDS = {"pic_scatter_single_seg", "pic_scatter_meta", "pic_combine"}


def test_tokenized_req_has_scatter_fields():
    f = {x.name for x in dataclasses.fields(TokenizedGenerateReqInput)}
    assert _FIELDS <= f


def test_generate_req_has_scatter_fields():
    f = {x.name for x in dataclasses.fields(GenerateReqInput)}
    assert _FIELDS <= f
