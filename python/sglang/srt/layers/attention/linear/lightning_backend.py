import logging
import math
import os
from collections import namedtuple
from typing import Optional, Union

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.lightning_attn import (
    BailingLinearKernel,
    linear_decode_forward_triton,
)
from sglang.srt.layers.attention.linear.linear_metadata import (
    BailingLinearMetadata,
)
from sglang.srt.layers.attention.linear.seg_la import SegLaMeta, seg_la_fwd
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.pic import diag_layer_dump as _diag
from sglang.srt.pic.policy import POLICIES, PICCompose
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

logger = logging.getLogger(__name__)


RopeArgs = namedtuple("RopeArgs", ["needs_rope", "cos_sin_cache", "is_neox", "rotary_dim"])


class LightningAttentionBackend(MambaAttnBackendBase):
    """
    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # seg_la processes draft tokens as a chain -- it has no parent-indices
        # plumbing for tree-shaped drafts, so spec v2 tree verify (topk > 1) would
        # commit wrong mamba states silently. Fail fast instead of mis-decoding.
        if self.topk > 1:
            raise NotImplementedError(
                "Lightning (seg_la) linear-attention backend does not support "
                f"speculative decoding with topk > 1 (got topk={self.topk}); "
                "seg_la verifies a draft tree as a chain. Use "
                "--speculative-eagle-topk 1."
            )
        # lightning attn does not need conv cache, but to keep the interface for mamba cache
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.BLOCK = (
            model_runner.model_config.block
            if hasattr(model_runner.model_config, "block")
            else 256
        )
        total_num_heads = model_runner.model_config.hf_config.num_attention_heads
        num_hidden_layers = model_runner.model_config.hf_config.num_hidden_layers
        self.tp_slope = LightningAttentionBackend._build_slope_tensor(
            total_num_heads, num_hidden_layers, self.device
        )
        self.linear_backend = getattr(
            model_runner.model_config.hf_config, "linear_backend", "seg_la"
        )
        logger.info(
            f"linear_backend for linear attention in hybrid_linear_backend: {self.linear_backend}"
        )
        # Ling/Bailing applies RoPE to keys before the linear-attn op, so the
        # recurrent state carries position phase. PIC composition across
        # segments at different start positions requires a K-dim block-diagonal
        # rerotation regardless of pic_mode (addition / transition / transition_rope).
        self.pic_state_needs_rope_rerotate = True

        # SegLA has no conv state (conv_kernel=1), but the shared
        # _init_track_conv_indices path (used in extra_buffer mode for prefix
        # tracking) reads conv_states_shape[-1]. Setting it to (0, 0) makes
        # the conv_state_len fall back to 0, which is the correct semantics
        # for a no-conv linear-attn op.
        self.conv_states_shape = (0, 0)
        self._pic_seg_state_fp32 = None
        self._pic_seg_prefix_state_fp32 = None
        # Cross-layer plan cache: id(forward_batch) -> _PicPlan (supports
        # 4-tuple unpacking). The plan only depends on forward_batch.pic_*
        # metadata + pic_mode, which are identical across all 20 decoder
        # layers of one prefill step.
        self._pic_plan_cache_key = None
        self._pic_plan_cache_value = None
        self._model_runner = model_runner

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        # seq_lens_cpu is unused by the underlying _replay_metadata for
        # non-target-verify modes; pass it through for compatibility.
        bs = forward_batch.batch_size
        metadata = self._replay_metadata(
            bs,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
        )
        self.forward_metadata = BailingLinearMetadata.prepare_decode(
            metadata.query_start_loc,
            metadata.mamba_cache_indices,
            bs,
            forward_batch.seq_lens,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        bailing_meta = BailingLinearMetadata.prepare_mixed(
            metadata.query_start_loc,
            metadata.mamba_cache_indices,
            forward_batch,
        )
        # Carry over the prefix-cache snapshot tracking fields populated by
        # `MambaAttnBackendBase._forward_metadata` (track_ssm_final_src/dst,
        # has_mamba_track_mask, etc.) so `forward_extend` can perform the
        # SegLA-specific aligned-end snapshot copy.
        bailing_meta.track_ssm_h_src = getattr(metadata, "track_ssm_h_src", None)
        bailing_meta.track_ssm_h_dst = getattr(metadata, "track_ssm_h_dst", None)
        bailing_meta.track_ssm_final_src = getattr(metadata, "track_ssm_final_src", None)
        bailing_meta.track_ssm_final_dst = getattr(metadata, "track_ssm_final_dst", None)
        bailing_meta.target_chunk_idx = getattr(metadata, "target_chunk_idx", None)
        bailing_meta.track_ssm_h_src_packed = getattr(metadata, "track_ssm_h_src_packed", None)
        bailing_meta.track_conv_indices = getattr(metadata, "track_conv_indices", None)
        bailing_meta.has_mamba_track_mask = getattr(metadata, "has_mamba_track_mask", None)
        self.forward_metadata = bailing_meta

    @staticmethod
    def _build_slope_tensor(
        n_attention_heads: int, num_hidden_layers: int, device="cuda"
    ):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.tensor(
            get_slopes(n_attention_heads), dtype=torch.float32
        ).reshape(n_attention_heads, 1, 1)

        tp_heads = n_attention_heads // get_parallel().attn_tp_size
        tp_rank = get_parallel().attn_tp_rank
        if num_hidden_layers <= 1:
            slope_rate_list = [slopes * (1 + 1e-5)]
        else:
            slope_rate_list = [
                slopes * (1 - layer_id / (num_hidden_layers - 1) + 1e-5)
                for layer_id in range(num_hidden_layers)
            ]

        tp_slope = [
            slope_rate_list[layer_id][tp_rank * tp_heads : (tp_rank + 1) * tp_heads]
            .contiguous()
            .to(device)
            for layer_id in range(num_hidden_layers)
        ]

        return tp_slope

    def _prefill_and_mix_infer(
        self,
        q,
        k,
        v,
        kv_cache,
        state_indices_tensor,
        forward_batch,
        layer,
        metadata,
    ):
        hidden = []
        for _prefill_idx in range(metadata.num_prefills):
            if _prefill_idx >= forward_batch.extend_start_loc.shape[0]:
                break
            if _prefill_idx >= state_indices_tensor.shape[0]:
                break

            _start = forward_batch.extend_start_loc[_prefill_idx]

            if _prefill_idx + 1 < forward_batch.extend_start_loc.shape[0]:
                _end = forward_batch.extend_start_loc[_prefill_idx + 1]
            else:
                if (
                    forward_batch.extend_seq_lens is not None
                    and _prefill_idx < forward_batch.extend_seq_lens.shape[0]
                    and metadata.num_decodes > 0
                ):
                    seq_len = forward_batch.extend_seq_lens[_prefill_idx]
                    _end = _start + seq_len
                else:
                    _end = q.shape[0]

            slot_id = state_indices_tensor[_prefill_idx]
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            slice_layer_cache = kv_cache[slot_id, ...]
            out_slice = BailingLinearKernel.jit_linear_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                self.tp_slope[layer.layer_id],
                self.BLOCK,
                layer_idx=layer.layer_id,
            )
            hidden.append(out_slice.contiguous())
        if metadata.num_decodes > 0:
            hidden.append(
                self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, metadata, layer
                )
            )

        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)

        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, metadata, layer):
        num_prefill_tokens = metadata.num_prefill_tokens
        num_prefills = metadata.num_prefills
        q = q[num_prefill_tokens:].unsqueeze(2).contiguous()
        k = k[num_prefill_tokens:].unsqueeze(2).contiguous()
        v = v[num_prefill_tokens:].unsqueeze(2).contiguous()
        slot_id = state_indices_tensor[num_prefills:]

        assert slot_id.shape[0] == q.shape[0], (
            f"slot_id length {slot_id.shape[0]} does not match decode batch size {q.shape[0]}. "
            "This indicates a bug in the upstream logic that should be investigated."
        )
        hidden = linear_decode_forward_triton(
            q, k, v, kv_cache, self.tp_slope[layer.layer_id], slot_id, 32
        )
        return hidden

    def _linear_attention_entry(
        self,
        q,
        k,
        v,
        kv_cache,
        state_indices_tensor,
        metadata,
        layer,
        mask=None,
        temp_cache=None,
        intermediate_state_indices=None,
    ):
        q_offsets = metadata.query_start_loc

        seg_meta = SegLaMeta(
            batch_size=metadata.batch_size,
            q_offsets=metadata.query_start_loc,
            s_offsets=state_indices_tensor,
            q_lengths=q_offsets.diff(),
            s_scales=metadata.has_initial_states,
            max_q_length=None,
            mask=mask,
        )
        hidden = seg_la_fwd(
            q=q,
            k=k,
            v=v,
            s=kv_cache,
            decay_scales=self.tp_slope[layer.layer_id],
            meta=seg_meta,
            caches=temp_cache,
            cache_indices=intermediate_state_indices,
            decouple=True,
        )
        return hidden

    def _track_seg_la_unaligned_prefix_states(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        metadata,
        forward_batch: ForwardBatch,
        layer: RadixAttention,
        initial_states: Optional[torch.Tensor],
    ) -> None:
        if (
            not getattr(metadata, "has_mamba_track_mask", False)
            or metadata.track_ssm_h_dst is None
            or metadata.track_ssm_h_dst.numel() == 0
            or initial_states is None
        ):
            return

        track_mask = forward_batch.mamba_track_mask
        if track_mask is None:
            return
        req_indices = track_mask.nonzero(as_tuple=False).squeeze(-1)
        if req_indices.numel() == 0:
            return

        chunk_size = get_global_server_args().mamba_cache_chunk_size
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        masked_lens = lens_to_track.index_select(0, req_indices)
        unaligned = (masked_lens % chunk_size) != 0
        if not bool(unaligned.any().item()):
            return

        req_indices = req_indices[unaligned]
        target_lens = (masked_lens[unaligned] // chunk_size) * chunk_size
        dst_slots = metadata.track_ssm_h_dst
        assert dst_slots.numel() == req_indices.numel(), (
            f"SegLA track dst/request mismatch: {dst_slots.numel()} vs "
            f"{req_indices.numel()}"
        )

        state_shape = ssm_states.shape[1:]
        for i, req_idx_t in enumerate(req_indices):
            target_len = int(target_lens[i].item())
            if target_len <= 0:
                continue
            req_idx = int(req_idx_t.item())
            start = int(metadata.query_start_loc[req_idx].item())
            q_slice = q[start : start + target_len].contiguous()
            k_slice = k[start : start + target_len].contiguous()
            v_slice = v[start : start + target_len].contiguous()

            temp_state = ssm_states.new_empty((1, *state_shape))
            temp_state[0].copy_(initial_states[i])
            q_offsets = torch.tensor([0, target_len], dtype=torch.int32, device=q.device)
            meta = SegLaMeta(
                batch_size=1,
                q_offsets=q_offsets,
                s_offsets=torch.zeros(1, dtype=torch.int32, device=q.device),
                q_lengths=torch.tensor([target_len], dtype=torch.int32, device=q.device),
                s_scales=torch.ones(1, dtype=torch.bool, device=q.device),
                max_q_length=None,
                mask=None,
            )
            seg_la_fwd(
                q=q_slice,
                k=k_slice,
                v=v_slice,
                s=temp_state,
                decay_scales=self.tp_slope[layer.layer_id],
                meta=meta,
                decouple=True,
            )
            ssm_states[dst_slots[i]] = temp_state[0]

    def _pic_state_decay(self, layer: RadixAttention, length: int, state: torch.Tensor):
        slope = self.tp_slope[layer.layer_id].to(device=state.device, dtype=torch.float32)
        return torch.exp(-slope * float(length))

    def _pic_decay_for_length(self, layer, length):
        """Cached exp(-slope * length) for SegLA decay. Key by (layer_id, length).
        Returned tensor is fp32 on the current device; safe to use as multiplier on
        fp32 accumulators.
        """
        length = int(length)
        cache = getattr(self, "_pic_decay_cache", None)
        if cache is None:
            cache = {}
            self._pic_decay_cache = cache
        key = (layer.layer_id, length)
        decay = cache.get(key)
        if decay is None:
            slope = self.tp_slope[layer.layer_id].to(
                device=self.req_to_token_pool.mamba2_layer_cache(layer.layer_id).temporal.device,
                dtype=torch.float32,
            )
            decay = torch.exp(-slope * float(length))
            cache[key] = decay
        return decay

    def _pic_compose_state(
        self,
        mode: str,
        layer: RadixAttention,
        accum: torch.Tensor,
        segment_state: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        accum_f = accum.to(torch.float32)
        segment_f = segment_state.to(torch.float32)
        if POLICIES[mode].compose is PICCompose.ADDITION:
            return accum_f + segment_f
        # SegLA's recurrent state advances the previous state by exponential
        # decay before adding the next segment. This is the transition operator
        # for Ring/Bailing SegLA and is what distinguishes transition modes
        # from the raw addition baseline.
        return accum_f * self._pic_state_decay(layer, length, accum_f) + segment_f

    def _pic_rotate_state_key_axis(
        self,
        state: torch.Tensor,
        delta: int,
        cos_sin_cache: Optional[torch.Tensor],
        is_neox_style: bool,
        rotary_dim: int,
    ) -> torch.Tensor:
        """Rotate the SegLA state along its key dimension by a constant RoPE delta.

        SegLA state is [heads, key_dim, value_dim], representing Σ k[:, None] * v[None, :].
        Moving a whole cached segment by delta positions left-multiplies the key axis by R(delta).
        """
        if (
            delta == 0
            or cos_sin_cache is None
            or rotary_dim <= 0
            or state.shape[-2] < rotary_dim
        ):
            return state

        abs_delta = abs(int(delta))
        if abs_delta >= cos_sin_cache.shape[0]:
            raise ValueError(
                f"PIC transition_rope delta {delta} exceeds RoPE cache length "
                f"{cos_sin_cache.shape[0]}"
            )

        cos_sin = cos_sin_cache[abs_delta].to(device=state.device, dtype=torch.float32)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if delta < 0:
            sin = -sin

        state_f = state.to(torch.float32)
        rot_part = state_f[..., :rotary_dim, :]
        pass_part = state_f[..., rotary_dim:, :]
        cos = cos.view(1, -1, 1)
        sin = sin.view(1, -1, 1)
        if is_neox_style:
            half = rotary_dim // 2
            x1 = rot_part[..., :half, :]
            x2 = rot_part[..., half:rotary_dim, :]
            rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-2)
        else:
            x1 = rot_part[..., ::2, :]
            x2 = rot_part[..., 1::2, :]
            rotated = torch.stack(
                (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-2
            ).flatten(-3, -2)
        if pass_part.numel() == 0:
            return rotated
        return torch.cat([rotated, pass_part], dim=-2)

    def _pic_build_rope_args(self, kwargs):
        """Build RopeArgs from forward kwargs. Centralizes needs_rope decision."""
        rope_cos_sin_cache = kwargs.get("pic_rope_cos_sin_cache")
        rope_is_neox = kwargs.get("pic_rope_is_neox", True)
        rope_rotary_dim = int(kwargs.get("pic_rope_rotary_dim") or 0)
        needs_rope = (
            getattr(self, "pic_state_needs_rope_rerotate", False)
            and rope_cos_sin_cache is not None
            and rope_rotary_dim > 0
        )
        return RopeArgs(needs_rope, rope_cos_sin_cache, rope_is_neox, rope_rotary_dim)

    def _pic_rotate_for_replay(self, state, start, rope_args):
        """Rotate a cached canonical-pos=0 state forward to real_pos=start for replay.
        No-op when rope is off or start=0.
        """
        if not rope_args.needs_rope or start == 0:
            return state
        delta = int(start)
        return self._pic_rotate_state_key_axis(
            state, delta, rope_args.cos_sin_cache, rope_args.is_neox, rope_args.rotary_dim,
        )

    def _pic_derotate_to_canonical(self, state, start, rope_args):
        """Rotate a freshly-computed state at real_pos=start back to canonical pos=0
        for caching. Inverse of _pic_rotate_for_replay (delta=-start, no env sign
        flip — sign flip is replay-only).
        No-op when rope is off or start=0.
        """
        if not rope_args.needs_rope or start == 0:
            return state
        return self._pic_rotate_state_key_axis(
            state, -int(start), rope_args.cos_sin_cache, rope_args.is_neox, rope_args.rotary_dim,
        )

    def _pic_load_hit_state(
        self, slot, start, seg_state_fp32, ssm_states, rope_args,
        rotated_hit_cache=None, cap=0,
    ):
        """Load a hit segment's fp32 state, applying rope rotate to current pos.

        slot: int mamba slot
        start: int absolute position the hit segment starts at
        seg_state_fp32: dict[int, Tensor] fp32 mirror of canonical-pos cached states
        ssm_states: bf16 SSM cache (fallback when seg_state_fp32 missing)
        rope_args: RopeArgs from _pic_build_rope_args
        rotated_hit_cache: optional dict[(slot_i, delta_signed), Tensor] for
            memoizing rotated tensors across hits in the same batch
        cap: max entries to insert into rotated_hit_cache (0 = no caching)

        Returns: fp32 state at real_pos=start
        """
        slot_i = int(slot)
        cached = seg_state_fp32.get(slot_i)
        if cached is not None:
            state = cached
        else:
            state = ssm_states[slot].to(torch.float32)
        if not rope_args.needs_rope or start == 0:
            return state
        delta = int(start)
        cache_key = (slot_i, delta)
        if rotated_hit_cache is not None:
            rotated = rotated_hit_cache.get(cache_key)
            if rotated is not None:
                return rotated
        rotated = self._pic_rotate_for_replay(state, start, rope_args)
        if (
            rotated_hit_cache is not None and cap > 0
            and delta != 0 and len(rotated_hit_cache) < cap
        ):
            rotated_hit_cache[cache_key] = rotated
        return rotated

    def _pic_ensure_meta_pool(self, device, max_group):
        """Allocate per-backend SegLaMeta tensor pool on first use.
        Pool fields:
          _pic_meta_pool_q_offsets: int32 (max_group+1,) — per-batch copy_
          _pic_meta_pool_q_lengths: int32 (max_group,)   — per-batch copy_
          _pic_meta_pool_s_offsets: int32 (max_group,)   — constant arange(max_group)
          _pic_meta_pool_s_scales:  int8  (max_group,)   — per-call copy_
            (0 = cold / ignore h0, 1 = seed h0 from s[s_offset]; kernel does
            `s_scale > 0` mask, so int8 acts as bool with mutable per-row state)
        """
        if getattr(self, "_pic_meta_pool_max_group", 0) >= max_group and \
                getattr(self, "_pic_meta_pool_q_offsets", None) is not None and \
                self._pic_meta_pool_q_offsets.device == device:
            return
        self._pic_meta_pool_q_offsets = torch.empty(max_group + 1, dtype=torch.int32, device=device)
        self._pic_meta_pool_q_lengths = torch.empty(max_group, dtype=torch.int32, device=device)
        self._pic_meta_pool_s_offsets = torch.arange(max_group, dtype=torch.int32, device=device)
        self._pic_meta_pool_s_scales = torch.zeros(max_group, dtype=torch.int8, device=device)
        self._pic_meta_pool_max_group = max_group

    def _pic_run_segla_batch(
        self, q_tokens, k_tokens, v_tokens, init_states,
        q_offsets_cpu, q_lengths_cpu, group_size, layer,
        *, s_scales_per_row,
    ):
        """Run SegLA forward on a batch of segments sharing a token window.

        q_tokens/k_tokens/v_tokens: pre-sliced [q_base:q_limit] views
        init_states: (group_size, *state_shape) fp32 buffer, updated in-place to
            final states by the kernel
        q_offsets_cpu: list of len group_size+1, last entry = total token count
        q_lengths_cpu: list of len group_size
        group_size: must be <= max_group_size used at pool init
        s_scales_per_row: list[int]|Tensor of len group_size. Per-row seeding
            flag passed to the SegLA kernel: 0 = cold (ignore h0 in init_states,
            start state from zero), 1 = seed (use h0 = init_states[s_offset]).

        Returns: out_group with shape (num_tokens, num_q_heads, head_dim)
        """
        assert len(s_scales_per_row) == group_size, (
            f"s_scales_per_row len {len(s_scales_per_row)} != group_size {group_size}"
        )
        device = init_states.device
        self._pic_ensure_meta_pool(device, max(group_size, 1))
        q_off = self._pic_meta_pool_q_offsets[: group_size + 1]
        q_len = self._pic_meta_pool_q_lengths[: group_size]
        s_scales = self._pic_meta_pool_s_scales[: group_size]
        q_off.copy_(torch.tensor(q_offsets_cpu, dtype=torch.int32))
        q_len.copy_(torch.tensor(q_lengths_cpu, dtype=torch.int32))
        s_scales.copy_(torch.tensor(s_scales_per_row, dtype=torch.int8))
        meta = SegLaMeta(
            batch_size=group_size,
            q_offsets=q_off,
            s_offsets=self._pic_meta_pool_s_offsets[: group_size],
            q_lengths=q_len,
            s_scales=s_scales,
            max_q_length=None,
            mask=None,
        )
        return seg_la_fwd(
            q=q_tokens, k=k_tokens, v=v_tokens,
            s=init_states,
            decay_scales=self.tp_slope[layer.layer_id],
            meta=meta,
            decouple=True,
        )

    class _PicPlan:
        """Execution plan for one prefill step.

        Supports 4-element tuple unpacking for backward compatibility with
        the existing 4-pass consumer.  New recompute metadata lives as
        attributes consumed by the 3-pass rewrite (Task 4+).
        """

        __slots__ = (
            "req_plans",
            "miss_records",
            "abs_to_q_index_per_req",
            "abs_pos_flat",
            "pass_rows",
            "pass3_rows",
            "compose_program",
            "row_q_offsets_cpu",
            "row_q_lengths_cpu",
            "row_s_offsets_cpu",
            "num_rows",
            "kernel_h0_buf_shape",
        )

        def __init__(self, req_plans, miss_records, abs_to_q_index_per_req,
                     abs_pos_flat):
            self.req_plans = req_plans
            self.miss_records = miss_records
            self.abs_to_q_index_per_req = abs_to_q_index_per_req
            self.abs_pos_flat = abs_pos_flat
            self.pass_rows = []
            self.pass3_rows = []
            self.compose_program = []
            self.row_q_offsets_cpu = []
            self.row_q_lengths_cpu = []
            self.row_s_offsets_cpu = []
            self.num_rows = 0
            self.kernel_h0_buf_shape = None

        def __iter__(self):
            return iter((
                self.req_plans,
                self.miss_records,
                self.abs_to_q_index_per_req,
                self.abs_pos_flat,
            ))

    def _pic_get_or_build_plan(self, forward_batch, q_total, pic_mode_str, device=None):
        """Build (or fetch from cache) per-request PIC execution plan.

        Plan only depends on forward_batch.pic_* metadata + pic_mode_str,
        which are constant across all decoder layers of one prefill step,
        so we cache by id(forward_batch) and reuse across 20 layer calls.

        When `device` is provided and the mode is recompute, pre-builds
        sink_q_idx_gpu cuda tensors on each hit-with-seam
        step dict so emit_hit_seam can skip the per-layer CPU->GPU sync.

        Returns a _PicPlan supporting 4-tuple unpacking for backward compat:
          (req_plans, miss_records, abs_to_q_index_per_req, abs_pos_flat).
        Recompute mode adds: pass_rows, compose_program, row_q_offsets_cpu,
        row_q_lengths_cpu, row_s_offsets_cpu, num_rows.
        """
        cache_key = (id(forward_batch), pic_mode_str, q_total)
        if self._pic_plan_cache_key == cache_key:
            return self._pic_plan_cache_value

        pic_hit_segments = forward_batch.pic_hit_segments or []
        pic_miss_segments = forward_batch.pic_miss_segments or []
        pic_hit_mamba_slots = forward_batch.pic_hit_mamba_slots or []
        pic_miss_mamba_slots = forward_batch.pic_miss_mamba_slots or []
        is_recompute = pic_mode_str == "transition_rope_recompute"
        rope_meta = (
            getattr(forward_batch, "pic_rope_meta", None) if is_recompute else None
        )

        req_plans = []
        miss_records = []
        abs_to_q_index_per_req = []
        q_cursor = 0
        for req_idx in range(forward_batch.batch_size):
            hit_segments = pic_hit_segments[req_idx] if req_idx < len(pic_hit_segments) else []
            miss_segments = (
                pic_miss_segments[req_idx] if req_idx < len(pic_miss_segments) else []
            )
            hit_slots = (
                pic_hit_mamba_slots[req_idx]
                if req_idx < len(pic_hit_mamba_slots)
                else {}
            )
            miss_slots = (
                pic_miss_mamba_slots[req_idx]
                if req_idx < len(pic_miss_mamba_slots)
                else {}
            )

            hit_seam_positions = {}
            abs_to_q_index = None
            req_q_start = q_cursor
            if is_recompute and rope_meta is not None and req_idx < len(rope_meta):
                seam_info = rope_meta[req_idx].get("seam") or {}
                hit_seam_positions = seam_info.get("hit_seam", {}) or {}
                req_positions = []
                for s, e in miss_segments:
                    req_positions.extend(range(s, e))
                for (_s, _e), sink_pos in hit_seam_positions.items():
                    req_positions.extend(sink_pos)
                req_positions.sort()
                abs_to_q_index = {
                    int(pos): req_q_start + i for i, pos in enumerate(req_positions)
                }

            ordered = []
            for start, end, seg_hash in hit_segments:
                ordered.append((start, end, "hit", seg_hash))
            for start, end in miss_segments:
                ordered.append((start, end, "miss", None))
            ordered.sort(key=lambda x: x[0])
            last_segment = (ordered[-1][0], ordered[-1][1]) if ordered else None

            steps = []
            for start, end, kind, seg_hash in ordered:
                length = end - start
                if length <= 0:
                    continue
                if kind == "hit":
                    seam = (
                        hit_seam_positions.get((start, end)) if is_recompute else None
                    )
                    steps.append(
                        {
                            "kind": "hit",
                            "start": start,
                            "end": end,
                            "length": length,
                            "seg_hash": seg_hash,
                            "hit_slots": hit_slots,
                            "seam": seam,
                        }
                    )
                    continue

                miss_id = len(miss_records)
                if abs_to_q_index is not None:
                    q_start = abs_to_q_index[int(start)]
                    q_end = abs_to_q_index[int(end - 1)] + 1
                    if q_end - q_start != length:
                        raise RuntimeError(
                            f"PIC recompute: miss segment [{start},{end}) is not "
                            f"contiguous in abs_to_q_index "
                            f"(got {q_end - q_start} != {length})"
                        )
                else:
                    q_start = q_cursor
                    q_end = q_cursor + length
                    q_cursor = q_end
                miss_slot = miss_slots.get((start, end))
                cache_segment = miss_slot is not None and (start, end) != last_segment
                miss_records.append(
                    {
                        "req_idx": req_idx,
                        "start": start,
                        "end": end,
                        "length": length,
                        "q_start": q_start,
                        "q_end": q_end,
                        "miss_slot": miss_slot,
                        "cache_segment": cache_segment,
                    }
                )
                steps.append(
                    {
                        "kind": "miss",
                        "start": start,
                        "end": end,
                        "length": length,
                        "miss_id": miss_id,
                    }
                )
            req_plans.append(steps)
            abs_to_q_index_per_req.append(abs_to_q_index)
            if abs_to_q_index is not None:
                q_cursor = req_q_start + len(abs_to_q_index)

        assert q_cursor == q_total, (
            f"PIC Lightning batched consumed {q_cursor} tokens, expected {q_total}"
        )

        abs_pos_flat = []
        for req_idx in range(forward_batch.batch_size):
            if abs_to_q_index_per_req[req_idx] is not None:
                inv = sorted(
                    abs_to_q_index_per_req[req_idx].items(), key=lambda kv: kv[1]
                )
                abs_pos_flat.extend(p for p, _ in inv)
            else:
                ms = pic_miss_segments[req_idx] if req_idx < len(pic_miss_segments) else []
                for (s, e) in ms:
                    abs_pos_flat.extend(range(s, e))
        abs_pos_flat = abs_pos_flat if len(abs_pos_flat) == q_total else None

        plan = self._PicPlan(req_plans, miss_records, abs_to_q_index_per_req, abs_pos_flat)
        if is_recompute and device is not None:
            for req_idx, steps in enumerate(req_plans):
                a2q = abs_to_q_index_per_req[req_idx]
                if a2q is None:
                    continue
                for step in steps:
                    if step["kind"] != "hit" or step.get("seam") is None:
                        continue
                    sink_pos = step["seam"]
                    if sink_pos:
                        step["sink_q_idx_gpu"] = torch.tensor(
                            [a2q[int(p)] for p in sink_pos],
                            dtype=torch.long, device=device,
                        )

            # ---- Task 3: per-row recompute metadata ----
            pass_rows = []
            compose_program = []
            for req_idx, steps in enumerate(req_plans):
                seg_id = 0
                for step in steps:
                    if step["kind"] == "miss":
                        miss_id = step["miss_id"]
                        rec = miss_records[miss_id]
                        length = rec["length"]
                        q_s = rec["q_start"]
                        # Single row per miss segment (no seam split).
                        # Matches non-recompute path's chunk processing to
                        # avoid fp32 non-associativity noise from different
                        # kernel chunk boundaries.
                        row_m = {
                            "kind": "miss",
                            "req_id": req_idx,
                            "seg_id": seg_id,
                            "q_idx": torch.arange(
                                q_s, q_s + length,
                                dtype=torch.long, device=device,
                            ),
                            "length": length,
                            "slot": rec.get("miss_slot"),
                            "pool_write": True,
                        }
                        pass_rows.append(row_m)
                        compose_program.append(
                            ("seed_local", len(pass_rows) - 1)
                        )
                        seg_id += 1
                    elif step["kind"] == "hit":
                        seam = step.get("seam")
                        if seam is not None:
                            row_base = len(pass_rows)
                            sink_q_idx = step.get("sink_q_idx_gpu")
                            sink_len = (
                                int(sink_q_idx.numel())
                                if sink_q_idx is not None else 0
                            )
                            hit_pool_slot = step["hit_slots"][step["seg_hash"]]
                            if sink_len > 0:
                                row_s = {
                                    "kind": "hit_sink",
                                    "req_id": req_idx,
                                    "seg_id": seg_id,
                                    "q_idx": sink_q_idx,
                                    "length": sink_len,
                                    "slot": hit_pool_slot,
                                    "pool_write": False,
                                }
                                pass_rows.append(row_s)
                            if len(pass_rows) > row_base:
                                compose_program.append((
                                    "seed_pool_then_local",
                                    row_base,
                                    len(pass_rows) - 1,
                                    hit_pool_slot,
                                    step["start"],
                                    step["length"],
                                ))
                            else:
                                compose_program.append((
                                    "hit_pool_only",
                                    hit_pool_slot,
                                    req_idx,
                                    step["length"],
                                    step["start"],
                                    -1,
                                ))
                        else:
                            compose_program.append((
                                "hit_pool_only",
                                step["hit_slots"][step["seg_hash"]],
                                req_idx,
                                step["length"],
                                step["start"],
                                -1,
                            ))
                        seg_id += 1
                compose_program.append(("writeback", req_idx))

            # SegLaMeta offset/length arrays
            row_q_offsets_cpu = [0]
            row_q_lengths_cpu = []
            for row in pass_rows:
                row_q_lengths_cpu.append(row["length"])
                row_q_offsets_cpu.append(
                    row_q_offsets_cpu[-1] + row["length"]
                )

            plan.pass_rows = pass_rows
            plan.compose_program = compose_program
            plan.row_q_offsets_cpu = row_q_offsets_cpu
            plan.row_q_lengths_cpu = row_q_lengths_cpu
            plan.row_s_offsets_cpu = list(range(len(pass_rows)))
            plan.num_rows = len(pass_rows)
            plan.pass3_rows = pass_rows
        self._pic_plan_cache_key = cache_key
        self._pic_plan_cache_value = plan
        return plan

    def _forward_extend_pic(
        self,
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        """Dispatch SegLA PIC to the GDN-isomorphic v2 path (non-recompute
        modes: addition / transition / transition_rope) or the dedicated
        3-pass recompute path (transition_rope_recompute).

        Env knobs:
        - PIC_LIGHTNING_BATCH_MISS_MAX_SEGMENTS (default 64): group_size cap
        - PIC_LIGHTNING_ROTATED_HIT_CACHE (default 0): rotated hit state memo cap
        - PIC_SEAM_SINK (default 8): hit-segment recompute window (ratio if <=1, else token count)
        """
        if self.linear_backend != "seg_la":
            raise NotImplementedError(
                f"PIC for LightningAttentionBackend currently supports "
                f"linear_backend='seg_la', got {self.linear_backend!r}"
            )

        if kwargs.get("pic_mode") != "transition_rope_recompute":
            return self._forward_extend_pic_v2(
                mode, q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        # ---- 3-pass recompute path ----
        if self.kv_cache_dtype_str != "auto" and layer.k_scale is not None:
            q = q.to(self.kv_cache_dtype)
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        ssm_states = mamba_cache_params.temporal
        state_shape = ssm_states.shape[1:]
        out = q.new_empty(q.shape[0], layer.tp_q_head_num, layer.v_head_dim)
        if self._pic_seg_state_fp32 is None:
            self._pic_seg_state_fp32 = {}
            logger.info("PIC fp32 segment_state sparse mirror enabled")
        if self._pic_seg_prefix_state_fp32 is None:
            self._pic_seg_prefix_state_fp32 = {}
        mamba_layer_idx = self.req_to_token_pool.mamba_map[layer.layer_id]
        (_rp, _mr, _a2q, abs_pos_flat_cached) = self._pic_get_or_build_plan(
            forward_batch, q.shape[0], kwargs.get("pic_mode"), device=q.device
        )
        self._pic_plan_cache_value.kernel_h0_buf_shape = (
            self._pic_plan_cache_value.num_rows, *state_shape
        )
        self._pic_abs_pos_flat = abs_pos_flat_cached
        return self._forward_extend_pic_recompute_3pass(
            mode=mode, q=q, k=k, v=v, layer=layer,
            forward_batch=forward_batch,
            plan=self._pic_plan_cache_value,
            state_shape=state_shape, ssm_states=ssm_states,
            mamba_layer_idx=mamba_layer_idx,
            out=out, save_kv_cache=save_kv_cache, **kwargs,
        )

    def _forward_extend_pic_v2(
        self,
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        """GDN-isomorphic SegLA PIC for non-recompute modes (addition /
        transition / transition_rope).

        Mirrors gdn_backend.forward_extend_pic_transition: a single unified
        per-request compose loop ``h = h*decay + seg_contrib`` over segments
        sorted by start position, with a per-miss h0 snapshot for the seeded
        Pass3 rerun.

          - hit : seg_contrib = R(start) @ S_canonical, loaded from the fp32
                  mirror (fp32 ssm_states pool fallback).
          - miss: seg_contrib = fresh h0=0 state at its real positions
                  (== R(start) @ derotate(fresh) to ~6e-8; no rope round-trip on
                  the hot path). Cached misses ALSO store derotate(fresh)=canonical
                  to the pool so a later hit rotates a state consistent with the
                  miss that produced it (hit-state == miss-state).

        Pass1 (miss h0=0) -> Pass2 (unified compose, snapshot h0, cache canonical)
        -> Pass3 (seeded rerun of misses -> o). Recompute mode is handled by
        _forward_extend_pic_recompute_3pass.
        """
        if self.linear_backend != "seg_la":
            raise NotImplementedError(
                f"PIC for LightningAttentionBackend currently supports "
                f"linear_backend='seg_la', got {self.linear_backend!r}"
            )
        if self.kv_cache_dtype_str != "auto" and layer.k_scale is not None:
            q = q.to(self.kv_cache_dtype)

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        ssm_states = mamba_cache_params.temporal
        req_cache_indices = self.forward_metadata.mamba_cache_indices
        state_shape = ssm_states.shape[1:]
        out = q.new_empty(q.shape[0], layer.tp_q_head_num, layer.v_head_dim)

        if self._pic_seg_state_fp32 is None:
            self._pic_seg_state_fp32 = {}
            logger.info("PIC fp32 segment_state sparse mirror enabled (v2)")
        if self._pic_seg_prefix_state_fp32 is None:
            self._pic_seg_prefix_state_fp32 = {}
        mamba_layer_idx = self.req_to_token_pool.mamba_map[layer.layer_id]
        seg_state_fp32 = self._pic_seg_state_fp32.setdefault(mamba_layer_idx, {})

        rope_args = self._pic_build_rope_args(kwargs)
        pic_mode_str = kwargs.get("pic_mode")

        (req_plans, miss_records, _abs_to_q, abs_pos_flat_cached) = (
            self._pic_get_or_build_plan(
                forward_batch, q.shape[0], pic_mode_str, device=q.device
            )
        )

        from sglang.srt.pic import diag_layer_dump as _diag_dump
        _diag_on = _diag_dump.gdn_enabled(layer.layer_id)
        if _diag_on:
            self._pic_abs_pos_flat = abs_pos_flat_cached
            _diag_dump.dump_qkvgb(
                layer.layer_id, "pic", q, k, v, None, None,
                abs_pos=self._pic_abs_pos_flat,
            )

        max_segments = max(
            1, int(os.environ.get("PIC_LIGHTNING_BATCH_MISS_MAX_SEGMENTS", "64"))
        )
        num_miss = len(miss_records)

        def finish_req(req_idx, accum):
            if req_idx < req_cache_indices.shape[0]:
                dst = req_cache_indices[req_idx]
                if int(dst.item()) >= 0:
                    ssm_states[dst] = accum.to(ssm_states.dtype)
            if _diag_on:
                _diag_dump.dump_ssm_state(layer.layer_id, "pic_ssm_final", accum)

        # ---- Pass1 + Pass1.5: miss h0=0 -> fresh-real state; cached misses
        # also derotated to canonical (pos=0) and scattered to the pool so a
        # later hit rotates the *identical* state (hit-state == miss-state). ----
        fresh_by_miss_id = {}
        init_states = None
        if num_miss > 0:
            init_states = torch.empty(
                (num_miss, *state_shape), dtype=torch.float32, device=ssm_states.device
            )
            for group_start in range(0, num_miss, max_segments):
                group = miss_records[group_start : group_start + max_segments]
                group_size = len(group)
                q_base = group[0]["q_start"]
                q_limit = group[-1]["q_end"]
                q_offsets_cpu = [rec["q_start"] - q_base for rec in group] + [
                    q_limit - q_base
                ]
                q_lengths_cpu = [rec["length"] for rec in group]
                zero_states = ssm_states.new_zeros((group_size, *state_shape))
                self._pic_run_segla_batch(
                    q[q_base:q_limit], k[q_base:q_limit], v[q_base:q_limit],
                    zero_states, q_offsets_cpu, q_lengths_cpu, group_size, layer,
                    s_scales_per_row=[1] * group_size,
                )
                for local_i, rec in enumerate(group):
                    mid = group_start + local_i
                    fresh_real = zero_states[local_i].to(torch.float32)
                    fresh_by_miss_id[mid] = fresh_real
                    if rec["cache_segment"]:
                        canonical = self._pic_derotate_to_canonical(
                            fresh_real, rec["start"], rope_args
                        )
                        mslot = rec["miss_slot"]
                        ssm_states[mslot] = canonical.to(ssm_states.dtype)
                        seg_state_fp32[int(mslot)] = canonical.detach().clone()

        # ---- Pass2: unified compose  h = h*decay + R(start)@S_canonical ----
        for req_idx, steps in enumerate(req_plans):
            accum = torch.zeros(
                state_shape, dtype=torch.float32, device=ssm_states.device
            )
            for step in steps:
                start = step["start"]
                if step["kind"] == "hit":
                    slot_i = int(step["hit_slots"][step["seg_hash"]])
                    canonical = seg_state_fp32.get(slot_i)
                    if canonical is None:
                        canonical = ssm_states[slot_i].to(torch.float32)
                    seg_contrib = self._pic_rotate_for_replay(canonical, start, rope_args)
                else:
                    mid = step["miss_id"]
                    # Snapshot pre-compose accum as Pass3's h0 (real-pos frame).
                    init_states[mid].copy_(accum)
                    # Miss contributes its fresh real-position state directly
                    # (== R(start)@canonical to ~6e-8); no rope round-trip.
                    seg_contrib = fresh_by_miss_id[mid]
                if POLICIES[mode].compose is PICCompose.ADDITION:
                    accum = accum + seg_contrib
                else:
                    accum = (
                        accum * self._pic_decay_for_length(layer, step["length"])
                        + seg_contrib
                    )
                if _diag_on:
                    _tag = "base_h_after_%s" % step["kind"]
                    if step["kind"] == "hit":
                        _seg_id = int(step["hit_slots"].get(step.get("seg_hash", 0), -1))
                    else:
                        _seg_id = step.get("miss_id", -1)
                    _diag_dump.dump_compose_intermediate(
                        layer.layer_id, _tag, _seg_id, accum.flatten()[:128],
                    )
            finish_req(req_idx, accum)

        # ---- Pass3: rerun misses with composed h0 -> correct o ----
        if num_miss > 0:
            for group_start in range(0, num_miss, max_segments):
                group = miss_records[group_start : group_start + max_segments]
                group_size = len(group)
                q_base = group[0]["q_start"]
                q_limit = group[-1]["q_end"]
                q_offsets_cpu = [rec["q_start"] - q_base for rec in group] + [
                    q_limit - q_base
                ]
                q_lengths_cpu = [rec["length"] for rec in group]
                temp_states = init_states[
                    group_start : group_start + group_size
                ].contiguous()
                out_group = self._pic_run_segla_batch(
                    q[q_base:q_limit], k[q_base:q_limit], v[q_base:q_limit],
                    temp_states, q_offsets_cpu, q_lengths_cpu, group_size, layer,
                    s_scales_per_row=[1] * group_size,
                )
                out[q_base:q_limit].copy_(out_group)

        if _diag_on:
            _diag_dump.dump_o(
                layer.layer_id, "pic", out,
                abs_pos=getattr(self, "_pic_abs_pos_flat", None),
            )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_extend_pic_recompute_3pass(
        self,
        mode: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        plan,
        state_shape,
        ssm_states: torch.Tensor,
        mamba_layer_idx: int,
        out: torch.Tensor,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """3-pass recompute rewrite (Pass1 + Pass2.5 + Pass3).

        Pass1: cold row-batched SegLA — one batched kernel call over all
        plan.pass_rows with s_scales=0 (h0=0).  Produces per-row final S
        in state_buf [num_rows, H, K, V] fp32.

        Pass2.5: write full-segment cold state to ssm_states + seg_state_fp32
        mirror for cached miss segments.

        Pass3: compose chain + seeded kernel.  Walk the compose
        program to build per-request h_accum, seed kernel_h0_buf, then
        run a second batched SegLA with s_scales=1 (seed h0 from buf).
        Scatter the output to the correct positions in out.
        """
        # ---- Pass1: cold row-batched SegLA ----
        if plan.num_rows == 0:
            return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        from sglang.srt.pic import diag_layer_dump as _diag_dump
        if _diag_dump.gdn_enabled(layer.layer_id):
            _diag_dump.dump_qkvgb(
                layer.layer_id, "pic", q, k, v, None, None,
                abs_pos=getattr(self, "_pic_abs_pos_flat", None),
            )

        all_q_idx = torch.cat(
            [row["q_idx"] for row in plan.pass_rows], dim=0
        )
        q_packed = q.index_select(0, all_q_idx)
        k_packed = k.index_select(0, all_q_idx)
        v_packed = v.index_select(0, all_q_idx)

        num_heads, k_dim, v_dim = plan.kernel_h0_buf_shape[1:]
        state_buf = torch.zeros(
            plan.num_rows, num_heads, k_dim, v_dim,
            dtype=torch.float32, device=q.device,
        )

        self._pic_run_segla_batch(
            q_packed, k_packed, v_packed,
            init_states=state_buf,
            q_offsets_cpu=plan.row_q_offsets_cpu,
            q_lengths_cpu=plan.row_q_lengths_cpu,
            group_size=plan.num_rows,
            layer=layer,
            s_scales_per_row=[0] * plan.num_rows,
        )

        # ---- Pass2.5: write full-segment cold state for cached misses ----
        # Derotate to canonical pos=0 and write to ssm_states[miss_slot]
        # + seg_state_fp32 mirror. hit_pool_only reads this with rotation.
        rope_args = self._pic_build_rope_args(kwargs)
        seg_state_fp32 = self._pic_seg_state_fp32.setdefault(mamba_layer_idx, {})
        seg_prefix_state_fp32 = self._pic_seg_prefix_state_fp32.setdefault(mamba_layer_idx, {})
        slope_h = self.tp_slope[layer.layer_id].view(-1)  # [H]
        _miss_row_map = {}  # miss_slot -> (row_idx, miss_id)
        _row_i = 0
        _miss_id = 0
        for _req_idx, _steps in enumerate(plan.req_plans):
            for _step in _steps:
                if _step["kind"] == "miss":
                    _rec = plan.miss_records[_miss_id]
                    if (
                        _row_i < plan.num_rows
                        and plan.pass_rows[_row_i]["kind"] == "miss"
                    ):
                        if _rec["cache_segment"] and _rec["miss_slot"] is not None:
                            _miss_row_map[_rec["miss_slot"]] = (_row_i, _miss_id)
                        _row_i += 1
                    _miss_id += 1
                elif _step["kind"] == "hit":
                    _seam = _step.get("seam")
                    if _seam is not None:
                        if (
                            _row_i < plan.num_rows
                            and plan.pass_rows[_row_i]["kind"] == "hit_sink"
                        ):
                            _row_i += 1

        for _mslot, (_row_idx, _mid) in _miss_row_map.items():
            # state_buf[row] IS the full segment cold state (single row per miss).
            _full_cold = state_buf[_row_idx]
            # Derotate to canonical pos=0 for caching (matches legacy Pass C).
            _rec_start = plan.miss_records[_mid]["start"]
            _seg_to_cache = self._pic_derotate_to_canonical(
                _full_cold, _rec_start, rope_args,
            )
            ssm_states[_mslot] = _seg_to_cache.to(ssm_states.dtype)
            seg_state_fp32[int(_mslot)] = _seg_to_cache.detach().clone()
            # No prefix state cache write for miss segments — hit segments'
            # prefix states were cached during their original miss forward.

        # ---- Pass3: compose chain + seeded kernel (Task 6) ----
        # Step 1: Allocate kernel_h0_buf and per-row S/decay tensors.
        kernel_h0_buf = torch.zeros_like(state_buf)
        row_S = torch.empty_like(state_buf)
        row_decay = torch.empty(
            plan.num_rows, slope_h.shape[0],
            dtype=torch.float32, device=q.device,
        )
        for row_idx, row in enumerate(plan.pass_rows):
            # All rows in pass_rows are processed by Pass1 (cold, h0=0),
            # so state_buf has the cold S for every row. Decay is always
            # exp(-slope * row_length). The pool is only used by
            # hit_pool_only ops (no kernel rows) in the compose chain.
            row_S[row_idx] = state_buf[row_idx]
            row_decay[row_idx] = torch.exp(-slope_h * row["length"])

        # Step 2: Execute compose program (host-Python over GPU tensors).
        req_cache_indices = self.forward_metadata.mamba_cache_indices
        h_accum_by_req = {}
        for op in plan.compose_program:
            tag = op[0]
            if tag == "seed_local":
                row_idx = op[1]
                row = plan.pass_rows[row_idx]
                rid = row["req_id"]
                if rid not in h_accum_by_req:
                    ps = req_cache_indices[rid]
                    h_accum_by_req[rid] = (
                        ssm_states[ps].clone().to(torch.float32)
                    )
                h = h_accum_by_req[rid]
                kernel_h0_buf[row_idx] = h
                h = h * row_decay[row_idx][:, None, None] + row_S[row_idx]
                h_accum_by_req[rid] = h
                # Diag: dump h_accum after miss compose
                from sglang.srt.pic import diag_layer_dump as _diag_dump
                if _diag_dump.gdn_enabled(layer.layer_id):
                    _diag_dump.dump_compose_intermediate(
                        layer.layer_id, "pic_h_after_miss", row_idx, h.flatten()[:128],
                    )
            elif tag == "seed_pool_then_local":
                sink_row, _last_row, hit_slot, seg_start, seg_len = op[1], op[2], op[3], op[4], op[5]
                rid = plan.pass_rows[sink_row]["req_id"]
                if rid not in h_accum_by_req:
                    ps = req_cache_indices[rid]
                    h_accum_by_req[rid] = (
                        ssm_states[ps].clone().to(torch.float32)
                    )
                h = h_accum_by_req[rid]
                kernel_h0_buf[sink_row] = h

                # Load full segment cold state (rotated to current position)
                full_state = self._pic_load_hit_state(
                    hit_slot, seg_start, seg_state_fp32, ssm_states, rope_args,
                ).to(torch.float32)

                # Advance h_accum through the full segment in one step:
                #   h_final = h_accum * decay(seg_len) + full_state
                full_decay = torch.exp(-slope_h * seg_len)
                h_final = h * full_decay[:, None, None] + full_state

                h_accum_by_req[rid] = h_final
                # Diag: dump h_accum after hit-with-seam compose
                from sglang.srt.pic import diag_layer_dump as _diag_dump
                if _diag_dump.gdn_enabled(layer.layer_id):
                    _diag_dump.dump_compose_intermediate(
                        layer.layer_id, "pic_h_after_hit_seam", hit_slot, h_final.flatten()[:128],
                    )
            elif tag == "hit_pool_only":
                hit_slot, rid, seg_len, seg_start, row_idx = op[1], op[2], op[3], op[4], op[5]
                if rid not in h_accum_by_req:
                    ps = req_cache_indices[rid]
                    h_accum_by_req[rid] = (
                        ssm_states[ps].clone().to(torch.float32)
                    )
                h = h_accum_by_req[rid]
                if row_idx >= 0:
                    kernel_h0_buf[row_idx] = h
                # ponytail: hit-no-seam (row_idx < 0) needs no output work here;
                # those tokens' outputs were produced in prior forward passes.
                # Full segment cold state, rotated to current position.
                full_state = self._pic_load_hit_state(
                    hit_slot, seg_start, seg_state_fp32, ssm_states, rope_args,
                ).to(torch.float32)
                full_decay = torch.exp(-slope_h * seg_len)
                h = h * full_decay[:, None, None] + full_state
                h_accum_by_req[rid] = h
                # Diag: dump h_accum after hit compose
                from sglang.srt.pic import diag_layer_dump as _diag_dump
                if _diag_dump.gdn_enabled(layer.layer_id):
                    _diag_dump.dump_compose_intermediate(
                        layer.layer_id, "pic_h_after_hit", hit_slot, h.flatten()[:128],
                    )
            elif tag == "writeback":
                rid = op[1]
                if rid in h_accum_by_req:
                    ps = req_cache_indices[rid]
                    ssm_states[ps] = h_accum_by_req[rid].to(ssm_states.dtype)

        # Dump per-request SSM final states for diag.
        if h_accum_by_req:
            from sglang.srt.pic import diag_layer_dump as _diag_dump
            if _diag_dump.gdn_enabled(layer.layer_id):
                _h_buf = [h_accum_by_req[i] for i in sorted(h_accum_by_req)]
                _diag_dump.dump_ssm_finals(layer.layer_id, _h_buf, len(_h_buf))

        # Step 3: Run Pass3 seeded kernel (s_scales=1 → seed h0 from kernel_h0_buf).
        out_pass3 = self._pic_run_segla_batch(
            q_packed, k_packed, v_packed,
            init_states=kernel_h0_buf,
            q_offsets_cpu=plan.row_q_offsets_cpu,
            q_lengths_cpu=plan.row_q_lengths_cpu,
            group_size=plan.num_rows,
            layer=layer,
            s_scales_per_row=[1] * plan.num_rows,
        )

        # Step 4: Scatter Pass3 output back to out tensor.
        out.index_copy_(0, all_q_idx, out_pass3)

        # Step 5: hit-no-seam outputs already produced in prior forward passes;
        # only h_accum advancement (done above) is needed for the state chain.

        from sglang.srt.pic import diag_layer_dump as _diag_dump
        if _diag_dump.gdn_enabled(layer.layer_id):
            _diag_dump.dump_o(layer.layer_id, "pic", out,
                abs_pos=getattr(self, "_pic_abs_pos_flat", None),
            )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_pic_metadata(self, forward_batch: ForwardBatch):
        # SegLA PIC builds per-segment metadata inside _forward_extend_pic
        # because each miss segment depends on the previous segment's composed state.
        return

    def forward_extend_pic_addition(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        save_kv_cache=True,
        **kwargs,
    ):
        return self._forward_extend_pic(
            "addition", q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend_pic_transition(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        save_kv_cache=True,
        **kwargs,
    ):
        return self._forward_extend_pic(
            "transition", q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]

        metadata = self.forward_metadata

        if self.kv_cache_dtype_str != "auto" and layer.k_scale is not None:
            q = q.to(self.kv_cache_dtype)

        cache_indices = self.forward_metadata.mamba_cache_indices
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        ssm_states = mamba_cache_params.temporal
        if self.linear_backend == "minimax":
            o = self._prefill_and_mix_infer(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k,
                v,
                ssm_states,
                cache_indices,
                forward_batch,
                layer,
                metadata,
            )
        elif self.linear_backend == "seg_la":
            track_initial_states = None
            if (
                getattr(metadata, "has_mamba_track_mask", False)
                and metadata.track_ssm_h_dst is not None
                and metadata.track_ssm_h_dst.numel() > 0
            ):
                track_mask = forward_batch.mamba_track_mask
                req_indices = track_mask.nonzero(as_tuple=False).squeeze(-1)
                lens_to_track = (
                    forward_batch.mamba_track_seqlens
                    - forward_batch.extend_prefix_lens
                )
                masked_lens = lens_to_track.index_select(0, req_indices)
                chunk_size = get_global_server_args().mamba_cache_chunk_size
                unaligned = (masked_lens % chunk_size) != 0
                if bool(unaligned.any().item()):
                    unaligned_req_indices = req_indices[unaligned]
                    track_initial_states = ssm_states[
                        cache_indices.index_select(0, unaligned_req_indices)
                    ].clone()
                    has_init = metadata.has_initial_states.index_select(
                        0, unaligned_req_indices
                    ).to(track_initial_states.device)
                    if not bool(has_init.all().item()):
                        track_initial_states = torch.where(
                            has_init.view(-1, *([1] * (track_initial_states.dim() - 1))),
                            track_initial_states,
                            torch.zeros_like(track_initial_states),
                        )
            intermediate_state_indices = (
                torch.arange(
                    cache_indices.shape[0],
                    dtype=torch.int32,
                    device=cache_indices.device,
                )
                if forward_batch.forward_mode.is_target_verify()
                else None
            )
            o = self._linear_attention_entry(
                q,
                k,
                v,
                ssm_states,
                cache_indices,
                metadata,
                layer,
                temp_cache=(
                    mamba_cache_params.intermediate_ssm
                    if forward_batch.forward_mode.is_target_verify()
                    else None
                ),
                intermediate_state_indices=intermediate_state_indices,
            )
            if __import__("sglang.srt.pic.diag_layer_dump", fromlist=["gdn_enabled"]).gdn_enabled(layer.layer_id):
                from sglang.srt.pic.diag_layer_dump import (
                    dump_compose_intermediate,
                    dump_ssm_state,
                    dump_qkvgb,
                    dump_o,
                )

                dump_qkvgb(layer.layer_id, "base", q, k, v, None, None)
                if o.numel() > 0:
                    dump_o(layer.layer_id, "base", o)
                    if cache_indices.numel() > 0:
                        _idx0 = int(cache_indices[0].item())
                        if _idx0 >= 0:
                            dump_ssm_state(
                                layer.layer_id, "base_ssm_final", ssm_states[_idx0]
                            )
                    dump_compose_intermediate(
                        layer.layer_id,
                        "base_segla_raw_out_last",
                        -1,
                        o.reshape(o.shape[0], -1)[-1],
                    )

                _diag_ranges = None
                if _diag_ranges is not None:
                    pass

                for idx in cache_indices.detach().cpu().tolist()[:1]:
                    if int(idx) >= 0:
                        dump_ssm_state(
                            layer.layer_id,
                            "base_segla_final_state",
                            ssm_states[int(idx)],
                        )
            # SegLA has no per-chunk h-buffer; the kernel only writes the FINAL
            # state of each extend segment into the working slot. For
            # extra_buffer prefix-cache to be CORRECT, we must snapshot that
            # final state into the ping-pong track buffer ONLY when the
            # scheduler asked to track at exactly the end-of-extend position
            # (aligned case in `_init_track_ssm_indices`). Schedule_batch
            # restricts mask=True to that case for SegLA, so here we just
            # honor the final_src/dst copy.
            if (
                getattr(metadata, "has_mamba_track_mask", False)
                and metadata.track_ssm_final_src is not None
                and metadata.track_ssm_final_src.numel() > 0
            ):
                ssm_states[metadata.track_ssm_final_dst] = ssm_states[
                    metadata.track_ssm_final_src
                ]
            self._track_seg_la_unaligned_prefix_states(
                q,
                k,
                v,
                ssm_states,
                cache_indices,
                metadata,
                forward_batch,
                layer,
                track_initial_states,
            )
        else:
            raise ValueError(
                f"linear backend: {self.linear_backend} is not support for now"
            )

        if (
            not forward_batch.forward_mode.is_target_verify()
            and forward_batch.mamba_track_mask is not None
        ):
            # save mamba cache for extra buffer
            mamba_track_mask = forward_batch.mamba_track_mask
            mamba_track_indices = forward_batch.mamba_track_indices
            dst_masked = mamba_track_indices[mamba_track_mask]
            src_masked = metadata.mamba_cache_indices[mamba_track_mask]
            ssm_states[dst_masked] = ssm_states[src_masked]

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ) -> torch.Tensor:
        layer_id = layer.layer_id if layer else kwargs["layer_id"]

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        if self.kv_cache_dtype_str != "auto":
            q = q.to(self.kv_cache_dtype)

        # Do linear attention
        cache_indices = self.forward_metadata.mamba_cache_indices
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        ssm_states = mamba_cache_params.temporal
        if self.linear_backend == "minimax":
            o = self._decode_infer(q, k, v, ssm_states, cache_indices, metadata, layer)
        elif self.linear_backend == "seg_la":
            o = self._linear_attention_entry(
                q, k, v, ssm_states, cache_indices, metadata, layer
            )
        else:
            raise ValueError(
                f"linear backend: {self.linear_backend} is not support for now"
            )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
