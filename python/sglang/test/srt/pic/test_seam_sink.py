import importlib.util
from pathlib import Path


def _pic_module():
    path = Path(__file__).parents[3] / "srt" / "pic" / "__init__.py"
    spec = importlib.util.spec_from_file_location("pic_init_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_seam_sink_tokens_ratio_and_count():
    module = _pic_module()
    assert module.SEAM_SINK_DEFAULT == 8
    assert module.resolve_seam_sink_tokens(0, 100) == 0
    assert module.resolve_seam_sink_tokens(0.25, 100) == 25
    assert module.resolve_seam_sink_tokens(0.01, 10) == 1
    assert module.resolve_seam_sink_tokens(1, 100) == 100
    assert module.resolve_seam_sink_tokens(8, 100) == 8
    assert module.resolve_seam_sink_tokens(200, 100) == 100
