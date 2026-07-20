import pytest
from sglang.srt.server_args import ServerArgs

def test_pic_flags_exist_with_defaults():
    args = ServerArgs(model_path="dummy")
    assert args.pic_enable is False
    assert args.pic_separator_str == "<<PIC_SEP>>"
    assert args.pic_mode == "addition"
    assert args.pic_segment_min_tokens == -1

def test_pic_enable_requires_qwen3_5moe():
    with pytest.raises(AssertionError, match="Qwen3_5MoeForCausalLM"):
        ServerArgs(model_path="meta-llama/Llama-3-8B", pic_enable=True)

def test_pic_enable_requires_chunked_prefill_disabled():
    with pytest.raises(AssertionError, match="chunked_prefill_size"):
        ServerArgs(
            model_path="Qwen/Qwen3.5-35B-A3B",
            pic_enable=True,
            chunked_prefill_size=2048,
        )
