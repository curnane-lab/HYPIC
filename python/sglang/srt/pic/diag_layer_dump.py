"""Env-gated per-layer fingerprint dump for PIC ↔ baseline divergence bisect.

Set PIC_DIAG_DUMP=/abs/path/to/file before launching the engine. Every
decoder layer's last-token hidden state (after the layer's full forward,
i.e. residual stream) is appended as one JSONL line:

  {"layer": 0, "kind": "linear", "rank": 0, "norm": 12.34, "head": [...8 vals...]}

A second script (`tools/diag_layer_diff.py`) consumes baseline.jsonl and
pic.jsonl and prints the first layer where they diverge.

Designed for single-prompt single-rank capture (the example writes only
TP rank 0). For TP, ranks race on the same file — gate to rank 0 below.
"""

from __future__ import annotations

import json
import os
import threading

import torch

_PATH = os.environ.get("PIC_DIAG_DUMP", "")
_GDN_PATH = os.environ.get("PIC_DIAG_GDN", "")
_LOCK = threading.Lock()
_ENABLED = bool(_PATH)
_GDN_ENABLED = bool(_GDN_PATH)
def _parse_layer_filter(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return {0}
    if raw == "-1":
        return None  # None == all layers
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


_GDN_LAYERS = _parse_layer_filter(os.environ.get("PIC_DIAG_GDN_LAYER", "0"))
_TENSOR_DIR = os.environ.get("PIC_DIAG_TENSOR_DIR", "")
_TENSOR_COUNTS: dict[tuple[int, str, int], int] = {}


def enabled() -> bool:
    return _ENABLED


def gdn_enabled(layer_id: int) -> bool:
    if not _GDN_ENABLED:
        return False
    if _GDN_LAYERS is None:
        return True
    return int(layer_id) in _GDN_LAYERS


def _rank() -> int:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


def _append_jsonl(path: str, rec: dict) -> None:
    line = json.dumps(rec)
    with _LOCK:
        with open(path, "a") as f:
            f.write(line + "\n")


def dump_gdn_tokens(layer_id: int, tag: str, x: torch.Tensor,
                    abs_pos=None) -> None:
    """Dump per-token L2 norm of a tensor, shape [T, ...]. Reduces over all
    but the first dim. Used to compare post-conv1d / post-gating tensors
    token-by-token between baseline and PIC.

    `abs_pos` (optional): per-token abs position in the full sequence.
    When provided, the comparator can reindex BASE[abs_pos] to align with
    PIC's batch-order norms (needed when PIC's batch ≠ BASE's tail, e.g.
    recompute mode with seam>0 puts hit-seg sink tokens in batch)."""
    if not gdn_enabled(layer_id) or _rank() != 0:
        return
    t = x.detach().reshape(x.shape[0], -1).to(torch.float32)
    norms = t.norm(dim=-1).cpu().tolist()
    rec = {
        "layer": int(layer_id), "tag": tag, "T": int(x.shape[0]),
        "rank": _rank(),
        "norms": [float(n) for n in norms],
    }
    if abs_pos is not None:
        rec["abs_pos"] = [int(p) for p in abs_pos]
    _append_jsonl(_GDN_PATH, rec)


def dump_conv_slots(layer_id: int, tag: str, slots: torch.Tensor,
                    flags, conv_states: torch.Tensor) -> None:
    """Dump conv_states for the given slot indices + the has_initial_state
    flag vector. Used to verify whether the PIC miss segment loads any
    history (it should: the conv tail at end of preceding hit segment)."""
    if not gdn_enabled(layer_id) or _rank() != 0:
        return
    s = slots.detach().to(torch.int64).cpu().tolist()
    rec = {
        "layer": int(layer_id), "tag": tag, "rank": _rank(),
        "slot_indices": [int(x) for x in s],
    }
    if isinstance(flags, torch.Tensor):
        rec["has_initial_state"] = [bool(v) for v in flags.detach().cpu().tolist()]
    else:
        rec["has_initial_state"] = list(flags) if flags is not None else None
    norms = []
    for idx in s:
        if 0 <= idx < conv_states.shape[0]:
            v = conv_states[idx].detach().to(torch.float32)
            norms.append(float(v.norm().item()))
        else:
            norms.append(None)
    rec["conv_state_norms"] = norms
    _append_jsonl(_GDN_PATH, rec)


def dump_compose_intermediate(layer_id: int, tag: str, seg_i: int,
                              tensor: torch.Tensor) -> None:
    """Dump a per-segment compose intermediate. Stores fp32 norm + first-8
    flat values, plus checksum samples from later regions to detect layout
    drift (head[:8] alone could miss differences in tail). When
    PIC_DIAG_TENSOR_DIR is set, also persists the full flat tensor for
    offline element-wise diff."""
    if not gdn_enabled(layer_id) or _rank() != 0:
        return
    f = tensor.detach().to(torch.float32).flatten()
    n = int(f.numel())
    rec = {
        "layer": int(layer_id), "tag": tag, "seg_i": int(seg_i),
        "rank": _rank(),
        "numel": n,
        "norm": float(f.norm().item()),
        "head": [float(v) for v in f[:8].cpu().tolist()],
    }
    if n >= 128:
        rec["sample_64_72"] = [float(v) for v in f[64:72].cpu().tolist()]
        rec["sample_mid"] = [float(v) for v in f[n // 2 : n // 2 + 8].cpu().tolist()]
        rec["sample_tail"] = [float(v) for v in f[-8:].cpu().tolist()]
    if _TENSOR_DIR:
        os.makedirs(_TENSOR_DIR, exist_ok=True)
        key = (int(layer_id), str(tag), int(seg_i), _rank())
        with _LOCK:
            idx = _TENSOR_COUNTS.get(key, 0)
            _TENSOR_COUNTS[key] = idx + 1
        safe_tag = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(tag))
        path = os.path.join(
            _TENSOR_DIR,
            f"L{int(layer_id):03d}_S{int(seg_i):02d}_{safe_tag}_r{_rank()}_{idx}.pt",
        )
        torch.save(f.cpu(), path)
        rec["tensor_path"] = path
    _append_jsonl(_GDN_PATH, rec)


def dump_qkv_seg(layer_id: int, tag_prefix: str, seg_i: int,
                 q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Dump per-segment q/k/v fingerprints. Persists first-token & last-token
    full per-head fp32 slices when PIC_DIAG_TENSOR_DIR is set, so offline
    comparator can compute full-tensor rel_l2 per token (not biased head[:8])."""
    if not gdn_enabled(layer_id) or _rank() != 0:
        return
    for name, t in (("q", q), ("k", k), ("v", v)):
        f_all = t.detach().to(torch.float32)
        f = f_all.flatten()
        rec = {
            "layer": int(layer_id),
            "tag": f"{tag_prefix}_{name}",
            "seg_i": int(seg_i),
            "rank": _rank(),
            "T": int(t.shape[0]),
            "shape": [int(x) for x in t.shape],
            "norm": float(f.norm().item()),
            "head": [float(x) for x in f[:8].cpu().tolist()],
        }
        if _TENSOR_DIR and f_all.shape[0] >= 1:
            os.makedirs(_TENSOR_DIR, exist_ok=True)
            tag_first = f"{tag_prefix}_{name}_first"
            tag_last = f"{tag_prefix}_{name}_last"
            for slice_tag, slice_t in (
                (tag_first, f_all[0]),
                (tag_last, f_all[-1]),
            ):
                key = (int(layer_id), slice_tag, int(seg_i), _rank())
                with _LOCK:
                    idx = _TENSOR_COUNTS.get(key, 0)
                    _TENSOR_COUNTS[key] = idx + 1
                safe_tag = "".join(
                    c if c.isalnum() or c in "._-" else "_" for c in slice_tag
                )
                path = os.path.join(
                    _TENSOR_DIR,
                    f"L{int(layer_id):03d}_S{int(seg_i):02d}_{safe_tag}_r{_rank()}_{idx}.pt",
                )
                torch.save(slice_t.flatten().cpu(), path)
            rec["tensor_first"] = tag_first
            rec["tensor_last"] = tag_last
        _append_jsonl(_GDN_PATH, rec)


def dump_ssm_state(layer_id: int, tag: str, state: torch.Tensor) -> None:
    """Dump final SSM state for one slot. state shape: [H_v, V, K]."""
    if not gdn_enabled(layer_id) or _rank() != 0:
        return
    f = state.detach().to(torch.float32)
    rec = {
        "layer": int(layer_id), "tag": tag, "rank": _rank(),
        "norm": float(f.norm().item()),
        "shape": [int(x) for x in f.shape],
        "head": [float(v) for v in f.flatten()[:8].cpu().tolist()],
    }
    if _TENSOR_DIR:
        os.makedirs(_TENSOR_DIR, exist_ok=True)
        key = (int(layer_id), str(tag), _rank())
        with _LOCK:
            idx = _TENSOR_COUNTS.get(key, 0)
            _TENSOR_COUNTS[key] = idx + 1
        safe_tag = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(tag))
        path = os.path.join(
            _TENSOR_DIR, f"layer{int(layer_id):03d}_{safe_tag}_rank{_rank()}_{idx}.pt"
        )
        torch.save(f.cpu(), path)
        rec["tensor_path"] = path
    _append_jsonl(_GDN_PATH, rec)


# ---------------------------------------------------------------------------
# Shared bundle emitters
#
# Same shape on BASE and PIC, only the `prefix` differs ("base" vs "pic").
# Every helper is env-gated internally via the underlying primitives — calls
# are cheap no-ops when PIC_DIAG_GDN is unset.
# ---------------------------------------------------------------------------


def dump_qkvgb(layer_id, prefix, q, k, v, g, beta, abs_pos=None):
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        if t is None:
            continue
        dump_gdn_tokens(layer_id, f"{prefix}_{name}", t, abs_pos=abs_pos)


def dump_o(layer_id, prefix, core_attn_out, abs_pos=None):
    o = core_attn_out if core_attn_out.dim() != 4 else core_attn_out[0]
    dump_gdn_tokens(layer_id, f"{prefix}_o", o, abs_pos=abs_pos)


def dump_ssm_finals(layer_id, h_accum_buf, batch_size):
    """Per-req SSM final state dump for the PIC path. BASE writes a single
    `base_ssm_final` via dump_ssm_state directly."""
    for req_idx in range(batch_size):
        dump_ssm_state(layer_id, "pic_ssm_final", h_accum_buf[req_idx])


def dump(layer_id: int, kind: str, hidden: torch.Tensor) -> None:
    """Append a fingerprint line. `hidden` shape: [T, H] or [..., H]."""
    if not _ENABLED:
        return
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = 0
    if rank != 0:
        return

    h = hidden
    if h.dim() >= 2:
        h = h.reshape(-1, h.shape[-1])[-1]  # last token
    h32 = h.detach().to(torch.float32).cpu()
    rec = {
        "layer": int(layer_id),
        "kind": kind,
        "rank": int(rank),
        "T": int(hidden.reshape(-1, hidden.shape[-1]).shape[0])
        if hidden.dim() >= 2
        else 1,
        "norm": float(h32.norm().item()),
        "head": [float(x) for x in h32[:8].tolist()],
    }
    line = json.dumps(rec)
    with _LOCK:
        with open(_PATH, "a") as f:
            f.write(line + "\n")


def dump_topk(layer_id: int, kind: str, indices: torch.Tensor,
              weights: torch.Tensor = None) -> None:
    """Append top-K expert indices for the last token. indices shape [T, top_k]."""
    if not _ENABLED:
        return
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = 0
    if rank != 0:
        return
    idx = indices.detach()
    if idx.dim() >= 2:
        idx = idx.reshape(-1, idx.shape[-1])[-1]
    rec = {
        "layer": int(layer_id),
        "kind": kind,
        "rank": int(rank),
        "T": int(indices.reshape(-1, indices.shape[-1]).shape[0])
        if indices.dim() >= 2 else 1,
        "topk": [int(x) for x in idx.cpu().tolist()],
    }
    if weights is not None:
        w = weights.detach()
        if w.dim() >= 2:
            w = w.reshape(-1, w.shape[-1])[-1]
        rec["weights"] = [float(x) for x in w.to(torch.float32).cpu().tolist()]
    line = json.dumps(rec)
    with _LOCK:
        with open(_PATH, "a") as f:
            f.write(line + "\n")
