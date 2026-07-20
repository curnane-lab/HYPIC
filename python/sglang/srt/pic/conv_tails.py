"""PIC conv1d cross-segment state passing.

A miss segment's first `K-1` conv1d outputs need the conv history that the
preceding segment (hit or miss) would have left. We mirror vLLM's approach:

  - **Capture** at forward time: for each miss segment that will be cached
    (i.e. has a persist slot in the mamba pool), save its last `K-1` raw
    `mixed_qkv` input tokens into `conv_tails[persist_slot]`.
  - **Load** before `causal_conv1d_fn`: for each miss segment whose preceding
    segment has a known mamba slot with populated `conv_tails`, copy
    `conv_tails[prev_slot]` into `conv_states[seg_slot]` and set
    `has_initial_state[seg]=True`.

This matches `vllm/model_executor/layers/mamba/gdn_linear_attn.py:1156-1171`
where the external initial conv state is injected into `conv_state`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


def build_prev_tail_slots(
    *,
    batch_size: int,
    pic_hit_segments: Optional[Sequence],
    pic_hit_mamba_slots: Optional[Sequence],
    pic_miss_segments: Sequence,
    pic_miss_mamba_slots: Optional[Sequence],
    req_cache_indices: torch.Tensor,
) -> List[int]:
    """For each miss segment in scheduling order, return the mamba slot of the
    segment immediately preceding it in position order (or -1 if none).

    Position order = sort by `start` over all (hit + miss) segments of the
    request. The previous segment's slot tells us where to load conv history
    from before the miss segment's conv1d runs.

    The order in the returned list matches the order miss segments are appended
    to `_pic_seg_*` lists (request-major, then position order within each
    request). Caller is responsible for keeping this contract.
    """
    prev_slots: List[int] = []
    for req_idx in range(batch_size):
        hit_segs = pic_hit_segments[req_idx] if pic_hit_segments else []
        hit_slots_dict = pic_hit_mamba_slots[req_idx] if pic_hit_mamba_slots else {}
        miss_segs = pic_miss_segments[req_idx]
        miss_slots_dict = pic_miss_mamba_slots[req_idx] if pic_miss_mamba_slots else {}

        # Build (start, slot) for every segment in this request.
        position = []
        for (s, e, h) in hit_segs:
            position.append((s, hit_slots_dict[h]))
        for (s, e) in miss_segs:
            # Non-last miss has a persist slot; last miss uses the per-req
            # slot. For "prev" lookups it doesn't matter — we'll only check
            # this slot if a later miss references it (extremely unlikely for
            # last-miss). Use -1 to mark "not eligible as source".
            slot = miss_slots_dict.get((s, e), -1)
            position.append((s, slot))
        position.sort(key=lambda x: x[0])

        # For each miss segment (in original order), find its index in the
        # sorted list and read the preceding slot.
        for (s, e) in miss_segs:
            prev = -1
            for i, (ps, pslot) in enumerate(position):
                if ps == s:
                    if i > 0:
                        prev = position[i - 1][1]
                    break
            prev_slots.append(int(prev))
    return prev_slots


def load_conv_history(
    *,
    conv_states: torch.Tensor,                  # [num_req_slots, *conv_shape]
    conv_tails_layer: Optional[torch.Tensor],   # [num_mamba_slots, *conv_shape]
    seg_conv_indices: torch.Tensor,             # [num_segs] dst slots in conv_states
    prev_tail_slots: torch.Tensor,              # [num_segs] src slots in conv_tails
    has_initial_state: torch.Tensor,            # [num_segs] bool
    layout: str = "gdn",                        # "gdn" or "kda"
) -> None:
    """Copy conv_tails[prev_slot] into conv_states[seg_slot] for segments
    flagged `has_initial_state=True`. Mutates `conv_states` in place.

    layout="gdn":  conv_tails == conv_states layout, direct copy.
    layout="kda":  conv_tails is (slot, K-1, D); conv_states (passed transposed
                   to (slot, D, K-1) for causal_conv1d_fn) needs the loaded
                   tail transposed on the last two axes."""
    if conv_tails_layer is None:
        return
    if not bool(has_initial_state.any().item()):
        return
    mask = has_initial_state
    dst = seg_conv_indices[mask].to(torch.long)
    src = prev_tail_slots[mask].to(torch.long)
    # Guard against -1 sentinels that might still be present if caller
    # mis-aligned `has_initial_state` with `prev_tail_slots`.
    valid = src >= 0
    if not bool(valid.all().item()):
        dst = dst[valid]
        src = src[valid]
    if dst.numel() == 0:
        return
    if layout == "kda":
        conv_states[dst] = conv_tails_layer[src].transpose(-1, -2)
    else:
        conv_states[dst] = conv_tails_layer[src]


def capture_conv_tails(
    *,
    mixed_qkv: torch.Tensor,                    # [seq_len, qkv_dim] pre-conv1d input
    seg_cu_seqlens: torch.Tensor,               # [num_segs+1]
    persist_src_idx: torch.Tensor,              # [num_persist] seg indices (in seg order)
    persist_dst_slot: torch.Tensor,             # [num_persist] mamba slots to write
    conv_tails_layer: Optional[torch.Tensor],   # gdn: [slots, qkv_dim, K-1]; kda: [slots, K-1, qkv_dim]
    persist_src_end_offset: Optional[torch.Tensor] = None,  # [num_persist] subtract from seg end
    layout: str = "gdn",                        # "gdn" or "kda"
) -> None:
    """For each persisted (non-last) miss segment, write its last `K-1` raw
    `mixed_qkv` tokens into `conv_tails[dst_slot]`.

    Output layout depends on `layout`:
      - "gdn": [qkv_dim, K-1]  (transposed)
      - "kda": [K-1, qkv_dim]  (no transpose)

    `persist_src_end_offset[i]` (optional) shifts the capture window left by N
    tokens — used by transition_rope_recompute to capture at an interior tail
    instead of the segment's physical end. None means
    capture at the segment's physical end (default).

    If a segment is shorter than `K-1`, the leading slots are left as zeros
    (caller must have zero-initialized `conv_tails` at alloc time).

    Implementation note: this runs once per layer per batch, so we avoid any
    GPU→CPU sync (no `.tolist()`, no `.item()`). All work is expressed as
    fancy-index reads + a single scatter, so the kernel launches stay on the
    stream and the per-layer host overhead is the cost of building a few small
    index tensors."""
    if conv_tails_layer is None or persist_src_idx.numel() == 0:
        return
    # The K-1 axis lives at the last dim for kda layout, second-to-last for gdn.
    K_minus_1 = (
        conv_tails_layer.shape[-2] if layout == "kda" else conv_tails_layer.shape[-1]
    )
    device = mixed_qkv.device
    src_idx_long = persist_src_idx.to(torch.long)
    dst_slot_long = persist_dst_slot.to(torch.long)
    cu_long = seg_cu_seqlens.to(torch.long)

    starts = cu_long.index_select(0, src_idx_long)            # [N]
    ends = cu_long.index_select(0, src_idx_long + 1)          # [N]
    if persist_src_end_offset is not None:
        ends = ends - persist_src_end_offset.to(torch.long)
    seg_lens = ends - starts                                  # [N]
    takes = torch.clamp(seg_lens, max=K_minus_1)              # [N]
    # Per-(persist, slot-in-window) source row in mixed_qkv. slot j corresponds
    # to position (K_minus_1-1 - j) tokens before `end`. We use the inverted
    # axis ordering: row r is `end - K_minus_1 + r`. Rows where r < K_minus_1
    # - takes are out-of-segment and must be zero-padded.
    offsets = torch.arange(K_minus_1, device=device, dtype=torch.long)  # [K-1]
    src_rows = ends.unsqueeze(1) - K_minus_1 + offsets.unsqueeze(0)     # [N, K-1]
    valid = offsets.unsqueeze(0) >= (K_minus_1 - takes).unsqueeze(1)    # [N, K-1]
    # Clamp out-of-range src_rows to a safe row (0); their values get masked out.
    safe_rows = torch.where(valid, src_rows, torch.zeros_like(src_rows))
    gathered = mixed_qkv.index_select(0, safe_rows.reshape(-1)).reshape(
        src_rows.shape[0], K_minus_1, mixed_qkv.shape[-1]
    )                                                                  # [N, K-1, D]
    gathered = torch.where(valid.unsqueeze(-1), gathered, gathered.new_zeros(()))
    if layout == "kda":
        # KDA conv_tails layout is [slot, K-1, D] — no transpose needed.
        conv_tails_layer[dst_slot_long] = gathered.contiguous()
    else:
        # GDN conv_tails layout is [slot, qkv_dim, K-1] → transpose last two axes.
        conv_tails_layer[dst_slot_long] = gathered.transpose(1, 2).contiguous()
