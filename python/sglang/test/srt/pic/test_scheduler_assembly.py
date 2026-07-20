"""T14 sanity: PICache is importable alongside scheduler with no circular import."""


def test_picache_importable_from_scheduler_path():
    from sglang.srt.managers import scheduler  # noqa: F401
    from sglang.srt.pic.picache import PICache  # noqa: F401
