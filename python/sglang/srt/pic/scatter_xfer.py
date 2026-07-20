"""PIC distributed-scatter: combine-side prealloc control plane.

Allocates dst KV/mamba slots on the combine node and returns the local
nixl-agent metadata + buffer descriptors so the scatter (head) node can
push a segment's KV/state via WRITE. Reuses the engine's existing
prefill-role NixlKVManager nixl agent (no new agent).
"""

import base64
import json
import logging
import os
import time
import urllib.request
from urllib.parse import urlparse

import numpy as np
import torch

from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.pic.pic_alloc import _alloc_one_mamba
from sglang.srt.pic.segmenter import segment_hash
from sglang.srt.observability.req_time_stats import convert_time_to_realtime

logger = logging.getLogger(__name__)

_PIC_PERF = os.environ.get("PIC_PERF", "0") == "1"
# pp_tick fires every scheduler tick (dominates PICPERF volume ~30x and throttles
# throughput); gate it separately so PIC_PERF traces stay light enough to reflect
# real load. The perfetto converter discards pp_tick anyway.
_PIC_PERF_TICK = os.environ.get("PIC_PERF_TICK", "0") == "1"
_PIC_LEAK = os.environ.get("PIC_LEAK", "0") == "1"
_PIC_SYNC_WRITE = os.environ.get("PIC_SYNC_WRITE", "0") == "1"


def _perf(tag, room, **kw):
    if _PIC_PERF:
        extra = " ".join(f"{k}={v}" for k, v in kw.items())
        logger.info("PICPERF %s room=%s t=%.1f %s", tag, room, time.time() * 1000, extra)


def _leak_probe(scheduler, tag):
    """Env-gated (PIC_LEAK=1) mamba-slot leak probe. Logs the pool free count
    plus the counters that attribute a drop: pending dst slots (slot A) vs
    tree-cache entries stuck at lock_ref>0 that can never evict (slot B)."""
    if not _PIC_LEAK:
        return
    try:
        mp = scheduler.req_to_token_pool.mamba_allocator
        entries = scheduler.tree_cache._entries
        n_locked = sum(1 for e in entries.values() if getattr(e, "lock_ref", 0) > 0)
        pend = len(getattr(scheduler, "_pic_scatter_pending", {}) or {})
        logger.info(
            "PICLEAK %s mamba_avail=%s pending=%s entries=%s locked=%s",
            tag, mp.available_size(), pend, len(entries), n_locked,
        )
    except Exception as e:
        logger.warning("PICLEAK probe failed: %s", e)


def text_hash(s: str) -> int:
    """FNV-1a 64-bit hash of `s`. Byte-identical to the Rust router side
    (`sgl-model-gateway::policies::pic::text_hash`) — shared directory key.

    ponytail: routing hint only; a collision just degrades to a recompute.
    """
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _get_manager(scheduler):
    # Reuse the prefill-role NixlKVManager (bootstrap queue .kv_manager) so
    # callers reach send_kvcache / maybe_send_extra without a new agent.
    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
    mgr = getattr(queue, "kv_manager", None) if queue is not None else None
    if mgr is None:
        raise RuntimeError("scatter_xfer: prefill NixlKVManager not found")
    return mgr


def _alloc_dst(scheduler, room, seg_index, n_tokens, seg_hash, token_ids):
    """Allocate dst KV+mamba slots on the combine node for one scattered seg,
    stash pending (with seg_hash+token_ids for later inject), and return a
    handle dict organized by KVArgsRegisterInfo+TransferInfo fields. Rollback
    order frees kv_slots / mamba_slot / inflight on any mid-alloc failure."""
    tc = scheduler.tree_cache
    mgr = _get_manager(scheduler)
    agent = mgr.agent
    kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    mamba_pool = scheduler.req_to_token_pool.mamba_allocator
    kv_slots = alloc_token_slots(tc, n_tokens).clone()
    mamba_slot = None
    inflight = False
    stashed = False
    try:
        # I-1: coerce to Python int at alloc so the except block's
        # torch.tensor([mamba_slot]) can't hit a numpy scalar.
        mamba_slot = int(_alloc_one_mamba(mamba_pool, tc))
        tc.add_inflight(n_tokens, 1)
        inflight = True
        if not hasattr(scheduler, "_pic_scatter_pending"):
            scheduler._pic_scatter_pending = {}
        scheduler._pic_scatter_pending[(room, seg_index)] = {
            "kv_slots": kv_slots,
            "mamba_slot": mamba_slot,
            "n_tokens": n_tokens,
            "seg_hash": seg_hash,
            "token_ids": token_ids,
        }
        stashed = True
        kv_ptrs, _kvlen, kv_item = kv_pool.get_contiguous_buf_infos()
        st_ptrs, _stlen, st_item = kv_pool.get_state_buf_infos()
        st_dim = (
            kv_pool.get_state_dim_per_tensor()
            if hasattr(kv_pool, "get_state_dim_per_tensor")
            else None
        )
    except Exception:
        if stashed:
            scheduler._pic_scatter_pending.pop((room, seg_index), None)
        if inflight:
            tc.remove_inflight(n_tokens, 1)
        if mamba_slot is not None:
            mamba_pool.free(torch.tensor([mamba_slot], dtype=torch.int64, device=kv_slots.device))
        scheduler.token_to_kv_pool_allocator.free(kv_slots)
        raise
    return {
        "room": room,
        "seg_index": seg_index,
        "agent_meta_b64": base64.b64encode(agent.get_agent_metadata()).decode(),
        # M-2: buf-info arrays may be numpy → np.int64 elements break
        # json.dumps; force Python int (same as dst_kv_indices below).
        "dst_kv_ptrs": [int(x) for x in kv_ptrs],
        "dst_kv_item_len": [int(x) for x in kv_item],
        "dst_state_data_ptrs": [int(x) for x in st_ptrs],
        "dst_state_item_lens": [int(x) for x in st_item],
        "dst_state_dim_per_tensor": (
            [int(x) for x in st_dim] if st_dim is not None else None
        ),
        "dst_kv_indices": [int(x) for x in kv_slots.cpu().tolist()],
        "dst_state_indices": [int(mamba_slot)],
        "notif_kv": f"{room}|{seg_index}|kv",
        "notif_mamba": f"{room}|{seg_index}|mamba",
    }


def prealloc_and_push_handles(scheduler, req):
    """Combine side, on combine-request arrival (before self forward): batch
    pre-allocate dst slots for every scattered segment and POST the handle to
    each scatter worker, replacing the worker's per-seg HTTP prealloc round-trip.
    Local segments (worker == this engine) are skipped — cached by their own
    scatter sub-request. Any per-seg failure logs + continues (never crash)."""
    pic_combine = getattr(req, "pic_combine", None)
    if not pic_combine:
        return
    segments = getattr(req, "pic_segments", None)
    if not segments:
        return
    sa = scheduler.server_args
    for item in pic_combine:
        try:
            seg_index = item["seg_index"]
            worker_addr = item["worker_addr"]
            room = item["room"]
            s, t = segments[seg_index]
            token_ids = list(req.origin_input_ids[s:t])
            n_tokens = t - s
            # ponytail: relies on worker_addr carrying a scheme (http://...);
            # router guarantees it, so urlparse().port is populated.
            if urlparse(worker_addr).port == sa.port:
                # local segment: cached by its own scatter sub-request.
                continue
            seg_hash = segment_hash(token_ids)
            handle = _alloc_dst(
                scheduler, room, seg_index, n_tokens, seg_hash, token_ids
            )
            _http_post_json(
                worker_addr.rstrip("/") + "/pic_scatter/handle", handle
            )
            _perf("handle_post", room, seg=seg_index)
        except Exception as e:
            logger.warning(
                "scatter_xfer: prealloc_and_push_handles seg failed: %s", e
            )
            continue


def _wait_done(agent, handles):
    # ponytail: bounded so a dead remote can't hang the scheduler loop.
    deadline = time.monotonic() + 10.0
    while True:
        states = [agent.check_xfer_state(h) for h in handles]
        if any(s == "ERR" for s in states):
            raise RuntimeError("scatter_xfer: WRITE transfer error")
        if all(s == "DONE" for s in states):
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"scatter_xfer: WRITE wait timed out, handles={handles}")


def _http_post_json(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    # ponytail: /pic_scatter/handle is fire-and-forget on the worker (returns
    # ~immediately), so 5s is generous; a slow/dead worker just logs+continues.
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def _free_combine_pending(scheduler, req):
    """Task 3.5: free every still-pending dst slot belonging to a timed-out
    combine req (worker died before WRITE). Rooms come from req.pic_combine;
    segments already drained/injected are simply absent from pending → skipped.
    Per-item try/except so one bad free doesn't strand the rest."""
    pending = getattr(scheduler, "_pic_scatter_pending", None)
    if not pending:
        return
    # ponytail: frame by exact (room, seg_index) — same key alloc/inject use —
    # not room-only, so this never touches another req's pending even if a room
    # were ever reused. .get avoids a KeyError crashing the event loop.
    keys = {
        (item.get("room"), item.get("seg_index"))
        for item in getattr(req, "pic_combine", None) or []
    }
    if not keys:
        return
    seen = getattr(scheduler, "_pic_scatter_seen_notifs", None)
    tc = scheduler.tree_cache
    for (room, seg_index) in [k for k in pending if k in keys]:
        info = pending.get((room, seg_index))
        if info is None:
            continue
        try:
            # ponytail: the allocator cat's free_index onto its cuda free_pages,
            # so a cpu kv_slots (or one detached to cpu somewhere upstream) hits
            # "cpu vs cuda:0 (wrapper_CUDA_cat)". Coerce to the allocator's own
            # device — cheap no-op when already correct. Upgrade path: fix the
            # source device in _alloc_dst if this ever shows a real cpu clone.
            alloc = scheduler.token_to_kv_pool_allocator
            kv_slots = info["kv_slots"]
            dev = getattr(alloc, "device", None)
            if dev is not None and kv_slots.device != torch.device(dev):
                kv_slots = kv_slots.to(dev)
            alloc.free(kv_slots)
            scheduler.req_to_token_pool.mamba_allocator.free(
                torch.tensor([info["mamba_slot"]], dtype=torch.int64, device=info["kv_slots"].device)
            )
            tc.remove_inflight(info["n_tokens"], 1)
            if seen is not None:
                seen.discard(f"{room}|{seg_index}|kv".encode())
                seen.discard(f"{room}|{seg_index}|mamba".encode())
            logger.info(
                "PIC combine timeout free: room=%s seg=%s n_tok=%s",
                room, seg_index, info["n_tokens"],
            )
        except Exception as e:
            logger.warning(
                "scatter_xfer: combine timeout free failed for %s: %s",
                (room, seg_index), e,
            )
        pending.pop((room, seg_index), None)


def release_combine_holds(scheduler, req):
    """Drop the lock_ref each scattered segment took at inject/insert, now that
    the combine forward has consumed them (entries stay for directory reuse but
    become evictable). A PD combine transfers to decode and never hits the
    mixin's req.finished() release, so this is the ONLY release — called at
    combine prefill completion and on the timeout-abort path; without it every
    scattered seg leaks 1 lock_ref/request → mamba pool exhausts → OOM.
    ponytail: 并发同 hash 多 combine 共享 entry 时 refcount 可能少减；v1 可接受。"""
    if _PIC_PERF:
        _ts = req.time_stats
        _perf("combine_fwd", getattr(req, "bootstrap_room", None),
              s=convert_time_to_realtime(_ts.forward_entry_time) * 1000,
              e=convert_time_to_realtime(_ts.prefill_finished_time) * 1000)
    entries = scheduler.tree_cache._entries
    for h in getattr(req, "_pic_combine_hashes", []):
        ent = entries.get(h)
        if ent is not None and ent.lock_ref > 0:
            ent.lock_ref -= 1


def _try_release_combines(scheduler):
    """Scan parked combine requests; enqueue ready ones into the native-PD
    prefill path, abort timed-out ones. Safe to call repeatedly (commit-inject
    and local-scatter both trigger it)."""
    parked = getattr(scheduler, "_pic_combine_parked", None)
    if not parked:
        return
    entries = scheduler.tree_cache._entries
    for entry in list(parked):  # copy: we mutate `parked`
        req = entry["req"]
        if time.monotonic() > entry["deadline"]:
            from sglang.srt.managers.io_struct import AbortReq

            logger.warning(
                "scatter_xfer: combine req %s timed out waiting for segments", req.rid
            )
            # Task 3.5: worker(s) may have died before WRITE — free every pending
            # dst slot for this combine's rooms, else the whole batch of KV+mamba
            # slots + inflight leaks permanently. Rooms come from pic_combine.
            _free_combine_pending(scheduler, req)
            # injected-before-timeout segs also need their hold dropped (only the
            # still-pending ones are freed above), else the abort path leaks.
            release_combine_holds(scheduler, req)
            scheduler.abort_request(AbortReq(rid=req.rid))
            parked.remove(entry)
        elif all(h in entries for h in entry["hashes"]):
            # ready → reuse the PREFILL branch enqueue from _add_request_to_queue.
            from sglang.srt.disaggregation.utils import DisaggregationMode

            assert scheduler.disaggregation_mode == DisaggregationMode.PREFILL, (
                "PIC combine release expects PD-prefill mode"
            )
            # ponytail: scatter is PD-only; combine is always a native-PD prefill request.
            try:
                scheduler._prefetch_kvcache(req)
                scheduler.disagg_prefill_bootstrap_queue.add(
                    req, scheduler.model_config.num_key_value_heads
                )
                req.time_stats.set_prefill_bootstrap_queue_entry_time()
                logger.info(
                    "PIC combine RELEASED to bootstrap queue: rid=%s room=%s",
                    req.rid, getattr(req, "bootstrap_room", None),
                )
                _perf("combine_released", getattr(req, "bootstrap_room", None),
                      rid=req.rid, nseg=len(entry["hashes"]))
                _leak_probe(scheduler, "combine_release")
            except Exception as e:
                logger.error("PIC combine release FAILED: rid=%s err=%r", req.rid, e)
                raise
            parked.remove(entry)
        # else: still waiting — leave parked.


def drain_injects(scheduler):
    """Combine side, called every scheduler tick: poll nixl notifs and inject
    any segment whose kv+mamba WRITEs have both landed into PICache. token_ids
    and seg_hash come from the pending stash (combine self-computed them; the
    worker sends nothing back). Non-blocking: process whatever notifs arrived,
    leave not-yet-complete segments pending for the next tick. Per-seg failure
    logs + rolls back (free dst slots + remove_inflight), never crashes."""
    pending = getattr(scheduler, "_pic_scatter_pending", None)
    if not pending:
        return
    tc = scheduler.tree_cache
    # Task3 minor #2: agent lookup + notif poll must not bubble into the event
    # loop (a dead nixl agent would crash the scheduler); warn + retry next tick.
    try:
        agent = _get_manager(scheduler).agent
        if not hasattr(scheduler, "_pic_scatter_seen_notifs"):
            scheduler._pic_scatter_seen_notifs = set()
        seen = scheduler._pic_scatter_seen_notifs
        flags = scheduler.__dict__.setdefault("_pic_scatter_seg_recomputed", {})
        for _peer, messages in agent.get_new_notifs().items():
            for m in messages:
                parts = m.split(b"|")
                # remote mamba notif carries hit/miss: {room}|{seg}|mamba|r{0,1}
                # v0.5.14 maybe_send_extra appends a per-component suffix
                # ("_{i}") to the notif, so parts[3] is "r0_0"/"r1_0" not "r0"/
                # "r1" — match the r1 prefix, not the whole token.
                if len(parts) == 4 and parts[2] == b"mamba" and parts[3][:1] == b"r":
                    # room is a u64 int on the wire; key as int to match the
                    # pic_combine / pic_scatter_meta lookups (JSON number → int).
                    flags[(int(parts[0]), int(parts[1]))] = parts[3][:2] == b"r1"
                    seen.add(b"|".join(parts[:3]))  # base tag for exact match
                else:
                    seen.add(m)
    except Exception as e:
        logger.warning("scatter_xfer: drain_injects notif poll failed: %s", e)
        return

    for (room, seg_index), info in list(pending.items()):  # copy: we mutate
        want_kv = f"{room}|{seg_index}|kv".encode()
        want_mamba = f"{room}|{seg_index}|mamba".encode()
        if not (want_kv in seen and want_mamba in seen):
            continue
        try:
            seg_hash = info["seg_hash"]
            token_ids = torch.tensor(info["token_ids"], dtype=torch.int64)
            was_present = seg_hash in tc._entries
            tc.inject_received_segment(
                seg_hash, token_ids, info["kv_slots"], info["mamba_slot"]
            )
            if was_present:
                # dup hash: inject reused the existing entry; our dst orphaned.
                scheduler.token_to_kv_pool_allocator.free(info["kv_slots"])
                scheduler.req_to_token_pool.mamba_allocator.free(
                    torch.tensor([info["mamba_slot"]], dtype=torch.int64, device=info["kv_slots"].device)
                )
            tc.remove_inflight(info["n_tokens"], 1)
            logger.info(
                "PIC inject: room=%s seg=%s hash=%s n_tok=%s dup=%s entries=%s",
                room, seg_index, seg_hash.hex()[:12], len(info["token_ids"]),
                was_present, len(tc._entries),
            )
        except Exception as e:
            logger.warning(
                "scatter_xfer: inject failed for %s: %s", (room, seg_index), e
            )
            scheduler.token_to_kv_pool_allocator.free(info["kv_slots"])
            scheduler.req_to_token_pool.mamba_allocator.free(
                torch.tensor([info["mamba_slot"]], dtype=torch.int64, device=info["kv_slots"].device)
            )
            tc.remove_inflight(info["n_tokens"], 1)
        # Task3 minor #1: prune the notif tags so `seen` doesn't grow unbounded.
        seen.discard(want_kv)
        seen.discard(want_mamba)
        pending.pop((room, seg_index), None)

    _try_release_combines(scheduler)


def _fire_write(scheduler, h, entry, room, seg_index):
    """Fire (non-blocking) an RDMA WRITE of one segment's KV+mamba state to the
    combine dst described by handle `h`, reusing the NixlKVManager send
    primitives. send_kvcache/maybe_send_extra already post the transfer
    (agent.transfer internally), so this returns the live xfer handles WITHOUT
    waiting — the caller pins the source entry and records the handles in
    _pic_write_inflight for drain_writes to reap on a later tick."""
    mgr = _get_manager(scheduler)
    _perf("fire_enter", room, seg=seg_index)
    peer = mgr.agent.add_remote_agent(base64.b64decode(h.agent_meta_b64))

    n_tokens = len(entry.full_kv_slots)
    _perf("push_start", room, seg=seg_index, ntok=n_tokens)
    src_kv = np.asarray(
        entry.full_kv_slots.detach().cpu().to(torch.int64).numpy(),
        dtype=np.int32,
    )
    h_kv = mgr.send_kvcache(
        peer,
        src_kv,
        h.dst_kv_ptrs,
        np.asarray(h.dst_kv_indices, dtype=np.int32),
        0,
        h.notif_kv,
    )
    # true hit/miss for this seg rides the mamba notif so the combine side can
    # bill cached_tokens (default miss if retire didn't record it). pop: single use.
    recomputed = scheduler.__dict__.get("_pic_push_recomputed", {}).pop(
        (room, seg_index), True
    )
    mamba_notif = h.notif_mamba + ("|r1" if recomputed else "|r0")
    # v0.5.14 restructured maybe_send_extra to per-state-component nesting:
    # every state arg is List[List[int]] indexed by kv_args.state_types, and
    # maybe_send_extra does prefill_state_indices[i] then len(...). This model
    # has exactly one state component (StateType.MAMBA), so wrap each flat
    # handle list in a single-element outer list. Without this the inner value
    # is an int and len() raises -> the WRITE never fires (combine deadlocks).
    _dims = h.dst_state_dim_per_tensor
    h_st = mgr.maybe_send_extra(
        peer,
        [[int(entry.mamba_state_slot)]],
        [h.dst_state_data_ptrs],
        [[int(x) for x in h.dst_state_indices]],
        0,
        mamba_notif,
        decode_tp_size=mgr.attn_tp_size,
        dst_state_item_lens=[h.dst_state_item_lens],
        dst_state_dim_per_tensor=[_dims if _dims is not None else []],
    )
    # send_kvcache returns a single xfer handle; v0.5.14 maybe_send_extra
    # returns a LIST of handles (one per state component). Flatten so
    # drain_writes' check_xfer_state(h) always gets a single handle, not a
    # nested list ('list' object has no attribute '_handle').
    handles = []
    if h_kv:
        handles.append(h_kv)
    if h_st:
        handles.extend(h_st if isinstance(h_st, (list, tuple)) else [h_st])
    return handles


def _start_write(scheduler, h, entry, room, seg_index, seg_hash):
    """Pin the source entry for the RDMA read's duration, fire the WRITE, then:
    async (default) record it in _pic_write_inflight for drain_writes to reap;
    sync (PIC_SYNC_WRITE=1) poll to completion inline (old blocking semantics,
    kept for A/B). The pin is a dedicated lock_ref hold — independent of the
    cache_unfinished_req protect-lock — released on completion (drain_writes) or
    inline (sync path)."""
    entry.lock_ref += 1
    handles = _fire_write(scheduler, h, entry, room, seg_index)
    if _PIC_SYNC_WRITE:
        try:
            _wait_done(_get_manager(scheduler).agent, handles)
            _perf("write_done", room, seg=seg_index)
        finally:
            if entry.lock_ref > 0:
                entry.lock_ref -= 1
        return
    if not hasattr(scheduler, "_pic_write_inflight"):
        scheduler._pic_write_inflight = []
    # ponytail: unbounded in-flight list — bounded by segments-per-worker (small).
    # Add a cap here if a single request ever fans out to very many segments.
    scheduler._pic_write_inflight.append({
        "handles": handles,
        "seg_hash": seg_hash,
        "room": room,
        "seg_index": seg_index,
        "deadline": time.monotonic() + 10.0,
    })


def _reap_inflight(scheduler, inflight, item):
    """Release the source entry's in-flight WRITE hold and drop the item."""
    ent = scheduler.tree_cache._entries.get(item["seg_hash"])
    if ent is not None and ent.lock_ref > 0:
        ent.lock_ref -= 1
    inflight.remove(item)


def drain_writes(scheduler):
    """Worker side, every scheduler tick: poll fired scatter WRITEs and reap the
    completed / failed ones. All-DONE → emit write_done + drop the source entry's
    in-flight lock_ref hold. Any ERR or past deadline → warn + drop the hold
    (segment never reaches combine → combine times out → abort, existing path).
    Still-in-progress items stay for the next tick. Mirrors drain_injects;
    no-op when nothing is in flight; never crashes the loop."""
    inflight = getattr(scheduler, "_pic_write_inflight", None)
    if not inflight:
        return
    try:
        agent = _get_manager(scheduler).agent
    except Exception as e:
        logger.warning("scatter_xfer: drain_writes agent lookup failed: %s", e)
        return
    for item in list(inflight):  # copy: we mutate `inflight`
        try:
            states = [agent.check_xfer_state(h) for h in item["handles"]]
        except Exception as e:
            logger.warning(
                "scatter_xfer: check_xfer_state failed for %s: %s",
                (item["room"], item["seg_index"]), e,
            )
            if time.monotonic() > item["deadline"]:
                logger.warning(
                    "scatter_xfer: WRITE %s dropped (check_xfer_state kept raising)",
                    (item["room"], item["seg_index"]),
                )
                _reap_inflight(scheduler, inflight, item)
            continue
        err = any(s == "ERR" for s in states)
        timed_out = time.monotonic() > item["deadline"]
        done = all(s == "DONE" for s in states)
        if not (err or timed_out or done):
            continue  # still in progress → next tick
        if done:
            _perf("write_done", item["room"], seg=item["seg_index"])
        else:
            logger.warning(
                "scatter_xfer: WRITE %s dropped (err=%s timeout=%s states=%s)",
                (item["room"], item["seg_index"]), err, timed_out, states,
            )
        _reap_inflight(scheduler, inflight, item)


def maybe_push_after_prefill(req, scheduler):
    """After a scatter sub-request finishes prefill, push its segment to the
    combine worker. Non-blocking: if the combine's dst handle has already been
    POSTed (stashed in scheduler._pic_scatter_handles) WRITE now; otherwise
    register the segment in scheduler._pic_push_pending for try_push_pending to
    retry on later ticks (once the handle arrives via recv). We never poll here —
    the scheduler thread must stay free to recv the handle. On any failure: log +
    return (never crash the scheduler)."""
    meta = getattr(req, "pic_scatter_meta", None)
    if not meta:
        return
    try:
        combine_addr = meta["combine_addr"]
        room = meta["scatter_room"]
        seg_index = meta["seg_index"]
        if _PIC_PERF:
            _ts = req.time_stats
            _perf("scatter_fwd", room, seg=seg_index,
                  s=convert_time_to_realtime(_ts.forward_entry_time) * 1000,
                  e=convert_time_to_realtime(_ts.prefill_finished_time) * 1000)

        sa = scheduler.server_args
        # ponytail: match by port — engine binds --host 0.0.0.0 but the router
        # injects combine_addr as http://127.0.0.1:<port>, so a string compare
        # of host never matches. Port is unique per worker on this host.
        from urllib.parse import urlparse

        if urlparse(combine_addr).port == sa.port:
            # combine's own local segment — already cached by cache_unfinished_req.
            # Bump lock_ref (arrival hold) so it isn't evicted before the combine
            # claims it, then try to release any parked combine waiting on it.
            seg_hash = segment_hash(list(req.origin_input_ids))
            ent = scheduler.tree_cache._entries.get(seg_hash)
            if ent is not None:
                ent.lock_ref += 1
            _try_release_combines(scheduler)
            return

        token_ids = list(req.origin_input_ids)
        seg_hash = segment_hash(token_ids)
        key = (room, seg_index)

        h = getattr(scheduler, "_pic_scatter_handles", {}).get(key)
        if h is not None:
            # handle already here → fire WRITE (non-blocking) + pin source entry
            # so seg i+1 can compute while this WRITE drains on later ticks.
            entry = scheduler.tree_cache._entries[seg_hash]
            # pop first: the handle is single-use and the entry is about to be
            # retired, so drop it even if _start_write raises (PIC_SYNC_WRITE=1).
            scheduler._pic_scatter_handles.pop(key, None)
            _start_write(scheduler, h, entry, room, seg_index, seg_hash)
            return

        # handle not here yet → register for loop-driven retry. Store only
        # seg_hash (+ room/seg_index/deadline): try_push_pending re-fetches
        # full_kv_slots/mamba slot from _entries[seg_hash] rather than pinning
        # kv_slots here. Take a pending-hold (+1 lock_ref) so the entry can't be
        # evicted while it sits in _pic_push_pending — released exactly once in
        # try_push_pending when the WRITE fires or the item is timeout-dropped.
        ent = scheduler.tree_cache._entries.get(seg_hash)
        if ent is not None:
            ent.lock_ref += 1  # pending-hold: protect entry until try_push_pending fires/drops it
        if not hasattr(scheduler, "_pic_push_pending"):
            scheduler._pic_push_pending = []
        scheduler._pic_push_pending.append({
            "room": room,
            "seg_index": seg_index,
            "seg_hash": seg_hash,
            "deadline": time.monotonic() + sa.pic_scatter_timeout_s,
        })
        _perf("push_pend", room, seg=seg_index)
    except Exception as e:
        logger.warning("scatter_xfer: push after prefill failed: %s", e)


def pic_scatter_retire(scheduler, req, seg_hash, *, recomputed: bool):
    """Retire a scatter single-seg sub-request: push its (already-cached)
    segment state to combine, release the protect/hit lock_ref, optionally free
    the forward's mamba/req_pool slots, then finish the sub-request (no PD KV
    transfer to decode). Shared by the miss path (recomputed=True, after a
    forward) and the hit short-circuit (recomputed=False, no forward).

    Contract: on entry, _entries[seg_hash] holds a +1 protect/hit lock_ref for
    this request (miss: from cache_unfinished_req; hit: taken at enqueue). This
    fn releases exactly that +1. Net lock_ref per segment = 0. Must run in
    event-loop/tick context (uses stream_output)."""
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

    # Record ground-truth hit/miss for this segment so the combine engine can
    # bill cached_tokens correctly (exclude segments this request recomputed).
    # This is the fallback for a stale router directory: if it said "hit" but the
    # worker had evicted the seg, we recomputed here → recomputed=True → miss.
    meta = getattr(req, "pic_scatter_meta", None)
    if meta:
        from urllib.parse import urlparse

        room, seg_index = meta["scatter_room"], meta["seg_index"]
        if urlparse(meta["combine_addr"]).port == scheduler.server_args.port:
            # local segment: combine is this engine → record straight into the
            # combine-side dict (local segs carry no NIXL notif).
            scheduler.__dict__.setdefault("_pic_scatter_seg_recomputed", {})[
                (room, seg_index)
            ] = recomputed
        else:
            # remote segment: stash for _fire_write to append onto the notif.
            scheduler.__dict__.setdefault("_pic_push_recomputed", {})[
                (room, seg_index)
            ] = recomputed

    maybe_push_after_prefill(req, scheduler)
    ent = scheduler.tree_cache._entries.get(seg_hash)
    if ent is not None and ent.lock_ref > 0:
        ent.lock_ref -= 1
    if recomputed:
        if (
            getattr(req, "mamba_pool_idx", None) is not None
            and hasattr(scheduler.req_to_token_pool, "free_mamba_cache")
        ):
            scheduler.req_to_token_pool.free_mamba_cache(req)
        scheduler.req_to_token_pool.free(req)
    req.finished_reason = FINISH_LENGTH(length=0)
    scheduler.output_streamer.stream_output([req], req.return_logprob)


def pic_combine_cached_tokens(scheduler, req):
    """True cross-request cache-hit tokens for a combine req: Σ segment length
    over segments the executing worker did NOT recompute (a directory hit that
    actually held). The combine node's own match_prefix sees every scattered seg
    as cached (they were injected pre-combine), which over-counts; this rebills
    from the ground-truth per-seg recomputed flags. Missing flag → miss
    (conservative, never inflates). Pops consumed flags to bound the dict."""
    flags = scheduler.__dict__.get("_pic_scatter_seg_recomputed", {})
    segments = getattr(req, "pic_segments", None) or []
    total = 0
    for item in getattr(req, "pic_combine", None) or []:
        recomputed = flags.pop((item["room"], item["seg_index"]), True)
        if not recomputed:
            s, t = segments[item["seg_index"]]
            total += t - s
    return total


def try_push_pending(scheduler):
    """Scheduler-tick hook (worker side): drain _pic_push_pending — for each
    registered scatter segment whose combine dst handle has now arrived, WRITE
    it and drop the pending entry. Expired (deadline passed) or evicted
    (seg_hash gone from _entries) items are dropped with a warning. Per-item
    try/except so one bad push never strands the rest or crashes the loop."""
    pending = getattr(scheduler, "_pic_push_pending", None)
    if not pending:
        return
    handles = getattr(scheduler, "_pic_scatter_handles", {})
    if _PIC_PERF_TICK:
        _perf("pp_tick", 0, npend=len(pending), nh=len(handles))
    tc = scheduler.tree_cache
    for item in list(pending):  # copy: we mutate `pending`
        try:
            room = item["room"]
            seg_index = item["seg_index"]
            key = (room, seg_index)
            h = handles.get(key)
            if h is not None:
                entry = tc._entries.get(item["seg_hash"])
                if entry is None:
                    logger.warning(
                        "scatter_xfer: push-pending %s entry evicted; drop", key
                    )
                    pending.remove(item)
                    continue
                # Release the pending-hold BEFORE _start_write: it takes its own
                # write-pin, and single-threaded scheduler runs no eviction in the
                # gap, so a _start_write exception can't leak this +1 (the outer
                # except only removes the item, not the hold).
                if entry.lock_ref > 0:
                    entry.lock_ref -= 1  # release pending-hold; _start_write self-pins
                handles.pop(key, None)
                pending.remove(item)
                _start_write(scheduler, h, entry, room, seg_index, item["seg_hash"])
            elif time.monotonic() > item["deadline"]:
                logger.warning(
                    "scatter_xfer: handle for %s not received within timeout; "
                    "drop push", key,
                )
                ent = tc._entries.get(item["seg_hash"])
                if ent is not None and ent.lock_ref > 0:
                    ent.lock_ref -= 1  # release pending-hold on timeout-drop
                pending.remove(item)
        except Exception as e:
            logger.warning("scatter_xfer: try_push_pending item failed: %s", e)
            try:
                pending.remove(item)
            except ValueError:
                pass


if __name__ == "__main__":
    # M-1: rollback self-check — if _alloc_one_mamba raises, _alloc_dst must
    # free the kv_slots, not increment inflight, and leave pending empty.
    import sys
    import types

    _real_try_release = _try_release_combines  # saved before drain test stubs it

    class _KVAlloc:
        def __init__(self):
            self.freed = []

        def get_kvcache(self):
            return self

        def free(self, slots):
            self.freed.append(slots)

    class _TC:
        def __init__(self):
            self.inflight = 0

        def add_inflight(self, n, k):
            self.inflight += 1

        def remove_inflight(self, n, k):
            self.inflight -= 1

    class _ReqPool:
        class mamba_pool:  # noqa: N801
            pass

    class _Sched:
        def __init__(self):
            self.tree_cache = _TC()
            self.token_to_kv_pool_allocator = _KVAlloc()
            self.req_to_token_pool = _ReqPool()

    sched = _Sched()

    # stub module deps used by _alloc_dst before the raise.
    # ponytail: real kv_slots is a tensor; free paths read .device (see
    # abb9a16043), so the fake must carry one too. SimpleNamespace keeps
    # identity for M-1's `freed == [_sentinel]` check while exposing .device.
    _sentinel = types.SimpleNamespace(device=torch.device("cpu"))
    globals()["alloc_token_slots"] = lambda tc, n: types.SimpleNamespace(
        clone=lambda: _sentinel
    )
    globals()["_get_manager"] = lambda s: types.SimpleNamespace(agent=None)
    globals()["_alloc_one_mamba"] = lambda pool, tc: (_ for _ in ()).throw(
        RuntimeError("mamba pool exhausted")
    )

    raised = False
    try:
        _alloc_dst(sched, room=1, seg_index=0, n_tokens=4, seg_hash=b"h", token_ids=[1])
    except RuntimeError:
        raised = True

    assert raised, "expected _alloc_one_mamba failure to propagate"
    assert sched.token_to_kv_pool_allocator.freed == [_sentinel], "kv_slots not freed"
    assert sched.tree_cache.inflight == 0, "inflight leaked"
    assert not getattr(sched, "_pic_scatter_pending", {}), "pending not empty"
    print("scatter_xfer M-1 rollback self-check PASS")

    # Task 3: drain_injects self-check.
    class _TC2:
        def __init__(self):
            self._entries = {}
            self.injected = []
            self.inflight = 3  # two pending segs added inflight upstream
            self.released = 0

        def inject_received_segment(self, seg_hash, token_ids, kv, mamba):
            self.injected.append(seg_hash)

        def remove_inflight(self, n, k):
            self.inflight -= 1

    class _Agent:
        def __init__(self, notifs):
            self._notifs = notifs

        def get_new_notifs(self):
            n, self._notifs = self._notifs, {}
            return n

    class _Sched2:
        def __init__(self, notifs):
            self.tree_cache = _TC2()
            self.token_to_kv_pool_allocator = _KVAlloc()
            self.req_to_token_pool = _ReqPool()
            self.req_to_token_pool.mamba_pool = types.SimpleNamespace(
                free=lambda t: None
            )
            self._agent = _Agent(notifs)
            self._pic_scatter_pending = {
                (7, 0): {"kv_slots": _sentinel, "mamba_slot": 1,
                         "n_tokens": 4, "seg_hash": b"a", "token_ids": [1, 2, 3, 4]},
                (7, 1): {"kv_slots": _sentinel, "mamba_slot": 2,
                         "n_tokens": 5, "seg_hash": b"b", "token_ids": [5, 6, 7, 8, 9]},
            }

    released = {"n": 0}
    globals()["_try_release_combines"] = lambda s: released.__setitem__(
        "n", released["n"] + 1
    )
    both = {
        "p": [b"7|0|kv", b"7|0|mamba", b"7|1|kv", b"7|1|mamba"],
    }
    s2 = _Sched2(both)
    globals()["_get_manager"] = lambda s: types.SimpleNamespace(agent=s._agent)
    drain_injects(s2)
    assert not s2._pic_scatter_pending, "pending not cleared after both notifs"
    assert len(s2.tree_cache.injected) == 2, "inject not called twice"
    assert s2.tree_cache.inflight == 1, "remove_inflight not called twice"
    assert released["n"] == 1, "_try_release_combines not triggered"

    # notif not arrived → pending kept, inject not called, non-blocking.
    released["n"] = 0
    s3 = _Sched2({})  # empty notifs
    import time as _t
    t0 = _t.monotonic()
    drain_injects(s3)
    assert _t.monotonic() - t0 < 0.5, "drain blocked when notifs absent"
    assert len(s3._pic_scatter_pending) == 2, "pending dropped without notifs"
    assert len(s3.tree_cache.injected) == 0, "inject called without notifs"
    assert released["n"] == 1, "release should still run once"
    print("scatter_xfer Task 3 drain_injects self-check PASS")

    # Task 3.5: combine timeout frees all pending slots for its rooms.
    free_calls = {"kv": 0, "mamba": 0, "inflight": 0}

    class _KVAlloc35:
        def free(self, slots):
            free_calls["kv"] += 1

    class _TC35:
        def __init__(self):
            self.inflight = 2
            self._entries = {}

        def remove_inflight(self, n, k):
            free_calls["inflight"] += 1

    class _Sched35:
        def __init__(self):
            self.tree_cache = _TC35()
            self.token_to_kv_pool_allocator = _KVAlloc35()
            self.req_to_token_pool = types.SimpleNamespace(
                mamba_pool=types.SimpleNamespace(
                    free=lambda t: free_calls.__setitem__(
                        "mamba", free_calls["mamba"] + 1
                    )
                )
            )
            self._pic_scatter_seen_notifs = {
                b"9|0|kv", b"9|0|mamba", b"9|1|kv", b"9|1|mamba",
            }
            self._pic_scatter_pending = {
                (9, 0): {"kv_slots": _sentinel, "mamba_slot": 1,
                         "n_tokens": 4, "seg_hash": b"a", "token_ids": [1]},
                (9, 1): {"kv_slots": _sentinel, "mamba_slot": 2,
                         "n_tokens": 5, "seg_hash": b"b", "token_ids": [2]},
            }

    class _CombineReq:
        rid = "combine-1"
        pic_combine = [
            {"seg_index": 0, "worker_addr": "http://x:1", "room": 9},
            {"seg_index": 1, "worker_addr": "http://x:2", "room": 9},
        ]

    class _AbortReq:  # stub io_struct.AbortReq
        def __init__(self, rid):
            self.rid = rid

    aborted = {"n": 0}
    fake_io = types.ModuleType("sglang.srt.managers.io_struct")
    fake_io.AbortReq = _AbortReq
    sys.modules["sglang.srt.managers.io_struct"] = fake_io

    s4 = _Sched35()
    s4.abort_request = lambda r: aborted.__setitem__("n", aborted["n"] + 1)
    s4._pic_combine_parked = [
        {"req": _CombineReq(), "hashes": [], "deadline": _t.monotonic() - 1.0}
    ]
    # real _try_release_combines (drain test rebound the global to a stub).
    globals()["_try_release_combines"] = _real_try_release
    _real_try_release(s4)

    assert not s4._pic_scatter_pending, "pending not cleared on timeout"
    assert free_calls["kv"] == 2, f"kv free count {free_calls['kv']} != 2"
    assert free_calls["mamba"] == 2, f"mamba free count {free_calls['mamba']} != 2"
    assert free_calls["inflight"] == 2, "remove_inflight not called twice"
    assert aborted["n"] == 1, "abort_request not called"
    assert not s4._pic_scatter_seen_notifs, "seen tags not discarded"
    assert not s4._pic_combine_parked, "parked entry not removed"
    print("scatter_xfer Task 3.5 combine-timeout free self-check PASS")

    # Scatter opt A deadlock fix: worker-push must be non-blocking + loop-driven.
    class _Entry:
        full_kv_slots = list(range(4))
        mamba_state_slot = 5
        lock_ref = 0  # pending-hold taken/released by maybe_push/try_push_pending

    class _SchedPush:
        def __init__(self):
            self.server_args = types.SimpleNamespace(
                port=9000, pic_scatter_timeout_s=30.0
            )
            self.tree_cache = types.SimpleNamespace(_entries={b"seg": _Entry()})

    # stub segment_hash + _start_write so the test needs no torch/nixl.
    globals()["segment_hash"] = lambda ids: b"seg"
    _real_start_write = _start_write  # write-pipeline block below needs the real one back
    write_calls = {"n": 0}
    globals()["_start_write"] = lambda s, h, e, room, seg, sh: write_calls.__setitem__(
        "n", write_calls["n"] + 1
    )

    # (a) remote seg, handle absent → registers pending + returns immediately.
    sp = _SchedPush()
    req_a = types.SimpleNamespace(
        pic_scatter_meta={
            "combine_addr": "http://127.0.0.1:9999",  # different port → remote
            "scatter_room": 42,
            "seg_index": 0,
        },
        origin_input_ids=[1, 2, 3, 4],
    )
    t0 = _t.monotonic()
    maybe_push_after_prefill(req_a, sp)
    assert _t.monotonic() - t0 < 0.5, "maybe_push blocked without handle"
    assert len(sp._pic_push_pending) == 1, "pending not registered"
    assert write_calls["n"] == 0, "should not WRITE without handle"

    # (b) handle arrives → try_push_pending WRITEs + clears pending.
    sp._pic_scatter_handles = {(42, 0): object()}
    try_push_pending(sp)
    assert write_calls["n"] == 1, "try_push_pending did not WRITE"
    assert not sp._pic_push_pending, "pending not cleared after WRITE"
    assert (42, 0) not in sp._pic_scatter_handles, "handle not popped"

    # (c) deadline passed, no handle → dropped, no WRITE, no crash.
    sp2 = _SchedPush()
    sp2._pic_push_pending = [
        {"room": 1, "seg_index": 0, "seg_hash": b"seg",
         "deadline": _t.monotonic() - 1.0}
    ]
    sp2._pic_scatter_handles = {}
    write_calls["n"] = 0
    try_push_pending(sp2)
    assert not sp2._pic_push_pending, "expired pending not dropped"
    assert write_calls["n"] == 0, "expired item should not WRITE"

    # (d) handle present but entry evicted → dropped, no crash.
    sp3 = _SchedPush()
    sp3.tree_cache._entries = {}  # evicted
    sp3._pic_push_pending = [
        {"room": 2, "seg_index": 1, "seg_hash": b"seg",
         "deadline": _t.monotonic() + 30.0}
    ]
    sp3._pic_scatter_handles = {(2, 1): object()}
    try_push_pending(sp3)
    assert not sp3._pic_push_pending, "evicted-entry pending not dropped"
    print("scatter_xfer opt-A worker-push non-blocking self-check PASS")

    # ---- write-pipeline self-check (async reap + sync fallback) ----
    class _WEnt:
        def __init__(self):
            self.lock_ref = 0

    class _SchedW:
        def __init__(self):
            self.tree_cache = types.SimpleNamespace(_entries={})

    fire_agent = types.SimpleNamespace(_states=["DONE"])
    fire_agent.check_xfer_state = lambda h: fire_agent._states[0]
    globals()["_fire_write"] = lambda s, h, e, room, seg: ["H"]
    globals()["_get_manager"] = lambda s: types.SimpleNamespace(agent=fire_agent)
    globals()["_PIC_SYNC_WRITE"] = False
    globals()["_start_write"] = _real_start_write  # undo opt-A stub above

    # (async) fire records in-flight + pins entry, does not block.
    sw = _SchedW(); ent = _WEnt(); sw.tree_cache._entries[b"sh"] = ent
    _start_write(sw, object(), ent, room=1, seg_index=0, seg_hash=b"sh")
    assert ent.lock_ref == 1, "async: source entry not pinned"
    assert len(sw._pic_write_inflight) == 1, "async: not recorded in-flight"

    # all DONE → drop hold + reap.
    fire_agent._states = ["DONE"]
    drain_writes(sw)
    assert ent.lock_ref == 0, "async DONE: hold not released"
    assert not sw._pic_write_inflight, "async DONE: not reaped"

    # ERR → drop + reap.
    sw2 = _SchedW(); ent2 = _WEnt(); sw2.tree_cache._entries[b"sh"] = ent2
    _start_write(sw2, object(), ent2, 2, 0, b"sh")
    fire_agent._states = ["ERR"]
    drain_writes(sw2)
    assert ent2.lock_ref == 0 and not sw2._pic_write_inflight, "ERR not reaped/released"

    # timeout → drop + reap even while PROC.
    sw3 = _SchedW(); ent3 = _WEnt(); sw3.tree_cache._entries[b"sh"] = ent3
    _start_write(sw3, object(), ent3, 3, 0, b"sh")
    sw3._pic_write_inflight[0]["deadline"] = _t.monotonic() - 1.0
    fire_agent._states = ["PROC"]
    drain_writes(sw3)
    assert ent3.lock_ref == 0 and not sw3._pic_write_inflight, "timeout not reaped"

    # still in progress → kept, hold held.
    sw4 = _SchedW(); ent4 = _WEnt(); sw4.tree_cache._entries[b"sh"] = ent4
    _start_write(sw4, object(), ent4, 4, 0, b"sh")
    fire_agent._states = ["PROC"]
    drain_writes(sw4)
    assert ent4.lock_ref == 1 and len(sw4._pic_write_inflight) == 1, "PROC wrongly reaped"

    # (sync) PIC_SYNC_WRITE=1 → inline poll to DONE, no in-flight, hold released.
    globals()["_PIC_SYNC_WRITE"] = True
    fire_agent._states = ["DONE"]
    sw5 = _SchedW(); ent5 = _WEnt(); sw5.tree_cache._entries[b"sh"] = ent5
    _start_write(sw5, object(), ent5, 5, 0, b"sh")
    assert ent5.lock_ref == 0, "sync: hold not released after inline poll"
    assert not getattr(sw5, "_pic_write_inflight", []), "sync: must not record in-flight"
    globals()["_PIC_SYNC_WRITE"] = False

    # check_xfer_state keeps raising + deadline passed → reap anyway (Finding 1).
    fire_agent.check_xfer_state = lambda h: (_ for _ in ()).throw(RuntimeError("boom"))
    sw6 = _SchedW(); ent6 = _WEnt(); sw6.tree_cache._entries[b"sh"] = ent6
    _start_write(sw6, object(), ent6, 6, 0, b"sh")
    sw6._pic_write_inflight[0]["deadline"] = _t.monotonic() - 1.0
    drain_writes(sw6)
    assert ent6.lock_ref == 0 and not sw6._pic_write_inflight, "raise+timeout not reaped"
    fire_agent.check_xfer_state = lambda h: fire_agent._states[0]  # restore

    print("scatter_xfer write-pipeline self-check PASS")
    sys.exit(0)


def partition_combine_first(waiting_queue):
    """PIC Scatter ②: stable-partition so ready combine reqs sort ahead of
    scatter segment prefills in the prefill worker's waiting_queue.

    A combine only reaches waiting_queue after all its scattered segments have
    landed (until then it is parked in Scheduler._pic_combine_parked), so an
    is_pic_combine req seen here is always ready to run. Putting it first lets
    it run on this batch — draining its dst mamba slots and handing KV to decode
    ASAP — so the fixed router cap (PIC_MAX_COMBINE_INFLIGHT) pipeline turns over
    fast and workers stop idling. Segments backfill with the remaining budget.

    list.sort is stable, so relative order within each group is preserved.
    """
    waiting_queue.sort(key=lambda r: 0 if getattr(r, "is_pic_combine", False) else 1)
