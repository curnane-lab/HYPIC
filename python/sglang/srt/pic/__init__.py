"""Position-Independent Cache (PIC) for hybrid linear+full-attention models.

See qianyou/2026-05-28-pic-sglang-design.md for design.
"""
import math as _math
import os as _os

# transition_rope_recompute: token seam to recompute in each hit segment.
# 0 = reuse whole hit segment; 0 < x <= 1 = ratio; x > 1 = token count.
SEAM_SINK_DEFAULT: float = float(_os.environ.get("PIC_SEAM_SINK", "8"))


def resolve_seam_sink_tokens(sink_spec: float, max_tokens: int) -> int:
    if sink_spec <= 0 or max_tokens <= 0:
        return 0
    if sink_spec <= 1:
        return min(max_tokens, _math.ceil(max_tokens * sink_spec))
    return min(max_tokens, int(sink_spec))


def pic_rope_positions(positions, q, forward_batch):
    """RoPE positions to use under PIC (else the original ``positions`` arg).

    The ``positions`` arg threaded into the model can be the stale pre-override
    contiguous fallback: ``compute_position`` yields ``prefix_len + i``, which is
    wrong when cache hits aren't contiguous (e.g. a SYS miss, then hit chunks,
    then a query miss). ``ForwardBatch.positions`` carries the corrected real
    per-segment positions (PIC override in ``forward_batch_info``); roping with
    those makes cached/miss K land at their true absolute positions. Non-PIC
    requests, or any row-count mismatch, fall back to the arg unchanged.
    """
    fbp = getattr(forward_batch, "positions", None)
    if (
        getattr(forward_batch, "pic_mode", None) is not None
        and fbp is not None
        and fbp.shape[0] == q.shape[0]
    ):
        return fbp
    return positions
