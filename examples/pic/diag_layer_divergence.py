"""Per-layer divergence bisect between PIC and baseline (Qwen3.5-35B-A3B).

Uses the env-gated diag dump in sglang.srt.pic.diag_layer_dump. Both runs
write one JSONL line per decoder layer (rank 0, last-token residual).
We then diff the two streams to find the FIRST layer where the residual
diverges beyond a small threshold.

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 /opt/dynamo/venv/bin/python examples/pic/diag_layer_divergence.py

The script spawns two engines back-to-back (baseline, then a chosen PIC
mode) — set PIC_DIAG_MODE env to one of:
    addition, transition, transition_rope
Default: transition.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile

SEP = "<<PIC_SEP>>"

# --model config table. All models build chat-template prompts (SEP lives inside
# the user turn); PIC_DIAG_FORCE_SPLIT=1 makes the tokenizer strip SEP and
# tokenize per-segment so baseline/PIC token ids match. Folds the former
# qwen + ring_ variants into one --model script.
_RING_SEGS = [
    "All people living in the Thenum District work in the Chrysan Company. " * 900,
    "Derek is a single man living in the Thenum District. " * 900,
    ("Answer the question based on the given passages. "
     "Answer with the company name only. ") * 600,
]
_RING = dict(
    mode="transition_rope",
    template=True,
    system="You are a concise QA assistant. Answer as briefly as possible.",
    segs=_RING_SEGS,
    query="Which company does Derek work in? Reply with the company name only, no explanation.",
    warmups=[0, 2],
    post_template="\n</think>\n\nAnswer:",
)
_QWEN = dict(
    mode="transition",
    template=False,
    system="You are a helpful assistant.\n\n",
    segs=["Document A about cats.\n\n" * 60,
          "Document B about dogs.\n\n" * 60,
          "Document C about birds.\n\n" * 60],
    query="Question: which animal is in document B?",
    warmups=[0, 1, 2],
    # Close the think block so Qwen3.5 answers directly (thinking disabled).
    post="<|im_start|>assistant\n<think>\n\n</think>\n\n",
)
MODELS = {
    "qwen35b": dict(path="/workspace/models/Qwen3.5-35B-A3B", tp=2, **_QWEN),
    "qwen122b": dict(path="/workspace/models/Qwen3.5-122B-A10B", tp=4, **_QWEN),
    "ring_mini": dict(path="/workspace/models/Ring-mini-linear-2.0", tp=1, **_RING),
    "ring_flash": dict(path="/workspace/models/Ring-flash-linear-2.0", tp=4, **_RING),
}


def build_prompts(cfg: dict):
    """Prompt + warmups. SEP-joined segments; warmups reuse a subset (cfg
    ['warmups'] indices). template models wrap in chat template, raw models
    prepend the system text (original Qwen behavior)."""
    segs, q = cfg["segs"], cfg["query"]
    if cfg.get("template"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
        ph = "<<__USER_CONTENT__>>"
        templ = tok.apply_chat_template(
            [{"role": "system", "content": cfg["system"]},
             {"role": "user", "content": ph}],
            tokenize=False, add_generation_prompt=True,
        )
        templ += cfg.get("post_template", "")

        def chat(text: str) -> str:
            return templ.replace(ph, text)
    else:
        sys = cfg["system"]

        def chat(text: str) -> str:
            return sys + text

    prompt = chat(SEP + SEP.join(segs) + SEP + q) + cfg.get("post", "")
    warmups = [chat(SEP + segs[i] + SEP + q) + cfg.get("post", "") for i in cfg["warmups"]]
    return prompt, warmups


def run_once(dump_path: str, mode: str | None, prompt: str, warmup: list[str],
             model_path: str, tp: int, gdn_path: str = "") -> dict:
    """Spawn a subprocess Engine generation with PIC_DIAG_DUMP set.

    Returns {"input_len": int, "output_ids": list, "text": str}. Subprocess
    isolates the two engine boots (baseline vs PIC) for clean teardown across
    model sizes. Both runs use PIC_DIAG_FORCE_SPLIT=1 so the tokenizer strips
    SEP and does per-segment tokenization for both.
    """
    result_path = dump_path + ".result.json"
    input_path = dump_path + ".input.json"
    # Pass prompt/warmups via a file, not the command line: ring prompts are
    # huge (900x-repeated segments) and blow past ARG_MAX as a `-c` literal.
    json.dump({"prompt": prompt, "warmup": warmup}, open(input_path, "w"))
    script = f"""
import os, json
os.environ['PIC_DIAG_DUMP'] = {dump_path!r}
os.environ['PIC_DIAG_GDN'] = {gdn_path!r}
os.environ['PIC_DIAG_GDN_LAYER'] = '0'
os.environ['PIC_DIAG_FORCE_SPLIT'] = '1'
_inp = json.load(open({input_path!r}))
prompt, warmup = _inp['prompt'], _inp['warmup']
import sglang as sgl
common = dict(model_path={model_path!r}, tp_size={tp},
              disable_cuda_graph=True,
              log_level='error', trust_remote_code=True)
"""
    if mode is None:
        script += (
            "engine = sgl.Engine(**common, mem_fraction_static=0.80,"
            " chunked_prefill_size=-1,"
            " mamba_radix_cache_strategy='no_buffer', disable_radix_cache=True)\n"
        )
    else:
        script += (
            "engine = sgl.Engine(**common, page_size=1, chunked_prefill_size=-1,"
            f" mem_fraction_static=0.80, pic_enable=True, pic_mode={mode!r},"
            f" pic_separator_str={SEP!r})\n"
        )
    script += f"""
_no_warmup = {os.environ.get("PIC_DIAG_NO_WARMUP", "0")!r} == "1"
if not _no_warmup:
    for w in warmup:
        engine.generate(w, sampling_params={{'temperature': 0, 'max_new_tokens': 1}})
# Truncate dump files AFTER warmups so we capture only the final generate.
open({dump_path!r}, 'w').close()
if {gdn_path!r}:
    open({gdn_path!r}, 'w').close()
out = engine.generate(prompt, sampling_params={{'temperature': 0, 'max_new_tokens': 3}})
input_len = out.get('meta_info', {{}}).get('prompt_tokens', -1)
output_ids = out.get('output_ids', [])
print(f'INPUT_LEN={{input_len}}  OUTPUT_IDS={{output_ids}}  TEXT={{repr(out["text"][:60])}}', flush=True)
json.dump({{'input_len': input_len, 'output_ids': output_ids, 'text': out['text'][:60]}}, open({result_path!r}, 'w'))
engine.shutdown()
"""
    extra_env = {
        "PIC_DIAG_DUMP": dump_path,
        "PIC_DIAG_GDN": gdn_path,
        "PIC_DIAG_GDN_LAYER": "0",
        "PIC_DIAG_FORCE_SPLIT": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        # Ensure ninja (in venv) is on PATH so triton JIT doesn't SIGKILL.
        "PATH": "/opt/dynamo/venv/bin:" + os.environ.get("PATH", ""),
    }
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, **extra_env},
    )
    try:
        return json.load(open(result_path))
    except Exception:
        return {}


def diff(base_path: str, pic_path: str, rel_thresh: float = 1e-4) -> None:
    with open(base_path) as fb, open(pic_path) as fp:
        bl = [json.loads(l) for l in fb if "norm" in l]
        pl = [json.loads(l) for l in fp if "norm" in l]
    n = min(len(bl), len(pl))
    print(f"\n=== diff (baseline={len(bl)} layers, pic={len(pl)} layers, comparing {n}) ===")
    first_diverge = None
    for i in range(n):
        b, p = bl[i], pl[i]
        assert b["layer"] == p["layer"], f"layer mismatch at idx {i}"
        nb = b["norm"]
        diff_norm = sum((bx - px) ** 2 for bx, px in zip(b["head"], p["head"])) ** 0.5
        # rough cosine of head[:8] (cheap proxy)
        dot = sum(bx * px for bx, px in zip(b["head"], p["head"]))
        nh_b = sum(bx * bx for bx in b["head"]) ** 0.5
        nh_p = sum(px * px for px in p["head"]) ** 0.5
        cos = dot / (nh_b * nh_p + 1e-12)
        rel = abs(nb - p["norm"]) / max(nb, 1e-6)
        flag = ""
        if rel > rel_thresh or cos < 0.999:
            flag = "  <-- DIVERGE"
            if first_diverge is None:
                first_diverge = i
        print(f"layer {b['layer']:3d} {b['kind']:6s}  "
              f"norm base={nb:8.3f} pic={p['norm']:8.3f}  "
              f"rel={rel:7.1e}  cos8={cos:+.6f}{flag}")
    print(f"\nFIRST diverging layer: {first_diverge}")


def diff_gdn(base_path: str, pic_path: str) -> None:
    """Compare per-token GDN-input norms (post-conv1d k/v/g/beta at layer 0).

    If conv1d boundaries are the cause, base and PIC norms diverge at the
    first ~3 tokens of segments 2/3/4/5. If they match across all tokens
    but the final ssm_state still differs (per the residual diff), the bug
    is purely in compose math/cache.
    """
    if not (os.path.exists(base_path) and os.path.exists(pic_path)):
        return
    bl = [json.loads(l) for l in open(base_path) if l.strip()]
    pl = [json.loads(l) for l in open(pic_path) if l.strip()]
    # Aggregate ALL records per tag (multiple records when chunked prefill splits).
    def gather(records, prefix):
        out: dict[str, list[float]] = {}
        pos: dict[str, list[int]] = {}
        for r in records:
            t = r["tag"]
            if not t.startswith(prefix) or "norms" not in r:
                continue
            name = t[len(prefix):]
            if name not in ("q", "k", "v", "g", "beta"):
                continue
            out.setdefault(name, []).extend(r["norms"])
            if "abs_pos" in r:
                pos.setdefault(name, []).extend(r["abs_pos"])
        return out, pos
    btags, _ = gather(bl, "base_")
    ptags, ptags_pos = gather(pl, "pic_")
    print("\n=== GDN per-token norm diff (layer 0) ===")
    for name in ("q", "k", "v", "g", "beta"):
        b_full, p = btags.get(name), ptags.get(name)
        if b_full is None or p is None:
            print(f"{name}: missing (base={b_full is not None} pic={p is not None})")
            continue
        T = len(p)
        if len(b_full) < T:
            print(f"{name}: base has {len(b_full)} < pic {T}")
            continue
        ppos = ptags_pos.get(name)
        # abs_pos-aligned when available (recompute+seam>0); else tail-align.
        if ppos and len(ppos) == T:
            b = [b_full[pp] if 0 <= pp < len(b_full) else 0.0 for pp in ppos]
        else:
            b = b_full[-T:]
        max_rel = 0.0
        max_t = -1
        diverge_first = -1
        for t in range(T):
            bn, pn = b[t], p[t]
            denom = max(abs(bn), 1e-6)
            rel = abs(bn - pn) / denom
            if rel > max_rel:
                max_rel, max_t = rel, t
            if diverge_first < 0 and rel > 1e-4:
                diverge_first = t
        print(f"{name:5s} base_full={len(b_full)} pic={T} "
              f"max_rel={max_rel:.2e} @t={max_t}  "
              f"first_diverge(rel>1e-4)={diverge_first}  "
              f"base_tail[:4]={[f'{x:.3f}' for x in b[:4]]}  "
              f"pic[:4]={[f'{x:.3f}' for x in p[:4]]}")
    # Conv-state dumps: confirm has_initial_state + slot contents.
    print("\n=== Conv1d state at prefill entry (layer 0) ===")
    for r in bl:
        if r.get("tag") == "base_conv_pre":
            print(f"  baseline: slots={r['slot_indices']}  "
                  f"has_initial_state={r['has_initial_state']}  "
                  f"conv_state_norms={r['conv_state_norms']}")
    for r in pl:
        if r.get("tag") == "pic_trans_conv_pre":
            print(f"  PIC ({r['tag']}): slots={r['slot_indices']}  "
                  f"has_initial_state={r['has_initial_state']}  "
                  f"conv_state_norms={r['conv_state_norms']}")
    # Dump per-segment compose intermediates if present.
    base_ssm = [r for r in bl if r.get("tag") == "base_ssm_final"]
    pic_ssm  = [r for r in pl if r.get("tag") == "pic_ssm_final"]
    if base_ssm and pic_ssm:
        b, p = base_ssm[0], pic_ssm[0]
        bh, ph = b["head"], p["head"]
        diff = sum((x - y) ** 2 for x, y in zip(bh, ph)) ** 0.5
        denom = sum(x * x for x in bh) ** 0.5 + 1e-12
        rel_n = abs(b["norm"] - p["norm"]) / max(b["norm"], 1e-6)
        print(f"\n=== final SSM state (layer 0, req slot) ===")
        print(f"  base norm={b['norm']:.4f}  pic norm={p['norm']:.4f}  rel_norm={rel_n:.2e}")
        print(f"  head8 base={[f'{x:.3f}' for x in bh]}")
        print(f"  head8 pic ={[f'{x:.3f}' for x in ph]}")
        print(f"  head8 L2_diff/L2_base={diff/denom:.2e}")


def diff_compose_chain(base_gdn: str, pic_gdn: str) -> None:
    if not (os.path.exists(base_gdn) and os.path.exists(pic_gdn)):
        return
    bl = [json.loads(l) for l in open(base_gdn) if l.strip()]
    pl = [json.loads(l) for l in open(pic_gdn) if l.strip()]

    def by_tag(records, tag):
        return [r for r in records if r.get("tag") == tag]

    # [4] base_ssm_final vs pic_ssm_final
    b_ssm = by_tag(bl, "base_ssm_final")
    p_ssm = by_tag(pl, "pic_ssm_final")
    if b_ssm and p_ssm:
        b, p = b_ssm[0], p_ssm[0]
        rel_n = abs(b["norm"] - p["norm"]) / max(b["norm"], 1e-6)
        diff_h = sum((x - y) ** 2 for x, y in zip(b["head"], p["head"])) ** 0.5
        denom = sum(x * x for x in b["head"]) ** 0.5 + 1e-12
        print(f"\n=== [4] base_ssm_final vs pic_ssm_final ===")
        print(f"  norm  base={b['norm']:.6f}  pic={p['norm']:.6f}  rel_norm={rel_n:.3e}")
        print(f"  head8 base={[f'{x:.4f}' for x in b['head']]}")
        print(f"  head8 pic ={[f'{x:.4f}' for x in p['head']]}")
        print(f"  head8 L2_diff/L2_base={diff_h/denom:.3e}")

    # [5] core_attn_out per-token (abs_pos-aligned when available)
    b_o, p_o = by_tag(bl, "base_o"), by_tag(pl, "pic_o")
    if b_o and p_o:
        bn = []
        for r in b_o: bn.extend(r["norms"])
        pn, ppos = [], []
        for r in p_o:
            pn.extend(r["norms"])
            if "abs_pos" in r:
                ppos.extend(r["abs_pos"])
        T = min(len(bn), len(pn))
        if T > 0:
            if ppos and len(ppos) == len(pn):
                b_aln = [bn[p] if 0 <= p < len(bn) else 0.0 for p in ppos]
            else:
                b_aln = bn[-T:]
            max_rel, max_t = 0.0, -1
            for t in range(T):
                d = abs(b_aln[t] - pn[t]) / max(abs(b_aln[t]), 1e-6)
                if d > max_rel:
                    max_rel, max_t = d, t
            # magnitude-weighted (Frobenius over per-token norm diffs) — max_rel
            # above is a single-token outlier that a small ||o_base|| inflates;
            # this weights by magnitude so tiny-norm boundary tokens don't
            # dominate. NB: diff-of-norms, so understates directional error;
            # upgrade to norm-of-difference if this is ambiguous.
            frob_num = sum((b_aln[t] - pn[t]) ** 2 for t in range(T)) ** 0.5
            frob_den = max(sum(b_aln[t] ** 2 for t in range(T)) ** 0.5, 1e-12)
            frob_rel = frob_num / frob_den
            rels = sorted(abs(b_aln[t] - pn[t]) / max(abs(b_aln[t]), 1e-6) for t in range(T))
            median_rel = rels[len(rels) // 2]
            p99_rel = rels[min(len(rels) - 1, int(0.99 * len(rels)))]
            abs_err_at_max = abs(b_aln[max_t] - pn[max_t])
            onorm_at_max = abs(b_aln[max_t])
            print(f"\n=== [5] core_attn_out per-token norm  base_len={len(bn)} pic_len={len(pn)} ===")
            print(f"  max_rel={max_rel:.3e} @t={max_t}  (abs_err={abs_err_at_max:.3e}  ||o_base||={onorm_at_max:.3e})")
            print(f"  frob_rel(magnitude-weighted)={frob_rel:.3e}  median_rel={median_rel:.3e}  p99_rel={p99_rel:.3e}")
            print(f"  base_aln[:4]={[f'{x:.4f}' for x in b_aln[:4]]}  base_aln[-4:]={[f'{x:.4f}' for x in b_aln[-4:]]}")
            print(f"  pic [:4]={[f'{x:.4f}' for x in pn[:4]]}  pic [-4:]={[f'{x:.4f}' for x in pn[-4:]]}")


@dataclasses.dataclass
class TestResult:
    name: str
    actual: str
    reference: str
    status: str  # "PASS" | "FAIL" | "SKIP"


# Per-mode thresholds. Numbers calibrated on a clean Qwen3.5-35B-A3B run
# (TP=2, 5-seg 1096-tok prompt) — loose enough to absorb FP jitter, tight
# enough to catch real regressions.
TEST_THRESHOLDS = {
    "addition": dict(
        gdn_proj_max_rel=1e-5,
        core_attn_out_max_rel=1e-4,
        ssm_final_max_rel=1e-4,
        layer0_residual_rel=1e-4,
        check_first_diverge=False,
        check_all_3_tokens=False,
    ),
    "transition": dict(
        gdn_proj_max_rel=1e-5,
        core_attn_out_max_rel=1e-4,
        ssm_final_max_rel=1e-4,
        layer0_residual_rel=1e-4,
        check_first_diverge=True,
        first_diverge_min_layer=1,
        check_all_3_tokens=True,
    ),
    "transition_rope": dict(
        gdn_proj_max_rel=1e-5,
        core_attn_out_max_rel=1e-4,
        ssm_final_max_rel=1e-4,
        layer0_residual_rel=1e-4,
        check_first_diverge=False,
        check_all_3_tokens=False,
    ),
    "transition_rope_recompute": dict(
        gdn_proj_max_rel=1e-5,
        core_attn_out_max_rel=1e-4,
        ssm_final_max_rel=1e-4,
        layer0_residual_rel=1e-4,
        check_first_diverge=True,
        first_diverge_min_layer=1,
        check_all_3_tokens=True,
    ),
}


def _compute_metrics(base_path, pic_path, base_gdn, pic_gdn,
                     base_res, pic_res):
    m = {}
    m["base_output_ids"] = base_res.get("output_ids", []) or []
    m["pic_output_ids"] = pic_res.get("output_ids", []) or []
    m["base_input_len"] = base_res.get("input_len", -1)
    m["pic_input_len"] = pic_res.get("input_len", -1)

    bl = [json.loads(l) for l in open(base_path) if "norm" in l] if os.path.exists(base_path) else []
    pl = [json.loads(l) for l in open(pic_path) if "norm" in l] if os.path.exists(pic_path) else []
    n = min(len(bl), len(pl))
    m["n_layers_compared"] = n
    max_rel = 0.0
    first_diverge_flat = -1
    layer0_rels = []
    for i in range(n):
        b, p = bl[i], pl[i]
        if b["layer"] != p["layer"]:
            break
        nb, np_ = b["norm"], p["norm"]
        rel = abs(nb - np_) / max(nb, 1e-6)
        dot = sum(bx * px for bx, px in zip(b["head"], p["head"]))
        nh_b = sum(bx * bx for bx in b["head"]) ** 0.5
        nh_p = sum(px * px for px in p["head"]) ** 0.5
        cos = dot / (nh_b * nh_p + 1e-12)
        if (rel > 1e-4 or cos < 0.999) and first_diverge_flat < 0:
            first_diverge_flat = i
        max_rel = max(max_rel, rel)
        if b["layer"] == 0 and b.get("kind") == "linear" and b["T"] > 1:
            # Only prefill (T>1). Decode tokens follow divergent autoregressive
            # paths (different token IDs → different layer outputs), which
            # swamps the prefill comparison.
            layer0_rels.append(rel)
    m["max_layer_rel"] = max_rel
    m["first_diverge_layer_in_pass"] = (
        bl[first_diverge_flat]["layer"] if first_diverge_flat >= 0 else -1
    )
    m["layer0_max_rel"] = max(layer0_rels) if layer0_rels else None

    gdn_b = [json.loads(l) for l in open(base_gdn) if l.strip()] if os.path.exists(base_gdn) else []
    gdn_p = [json.loads(l) for l in open(pic_gdn) if l.strip()] if os.path.exists(pic_gdn) else []

    def _gather(records, prefix):
        out = {}
        pos = {}
        for r in records:
            t = r.get("tag", "")
            if not t.startswith(prefix) or "norms" not in r:
                continue
            name = t[len(prefix):]
            if name in ("q", "k", "v", "g", "beta"):
                out.setdefault(name, []).extend(r["norms"])
                if "abs_pos" in r:
                    pos.setdefault(name, []).extend(r["abs_pos"])
        return out, pos

    def _align(bv, pv, ppos):
        """Align BASE values to PIC by abs_pos when provided; else tail-align.
        Tail-align is only valid when PIC's batch IS the tail of BASE (e.g.
        transition with miss-only batch). With abs_pos available, this works
        for any subset (mini-seg recompute with seam>0)."""
        if ppos and len(ppos) == len(pv):
            return [bv[p] if 0 <= p < len(bv) else 0.0 for p in ppos]
        return bv[-len(pv):]

    bt, bt_pos = _gather(gdn_b, "base_")
    pt, pt_pos = _gather(gdn_p, "pic_")
    proj_rels = {}
    for name in ("q", "k", "v", "g", "beta"):
        bv, pv = bt.get(name), pt.get(name)
        if not bv or not pv or len(bv) < len(pv):
            proj_rels[name] = None
            continue
        b_aligned = _align(bv, pv, pt_pos.get(name))
        proj_rels[name] = max(
            abs(bn - pn) / max(abs(bn), 1e-6) for bn, pn in zip(b_aligned, pv)
        )
    m["gdn_proj_max_rels"] = proj_rels

    b_o, p_o, p_o_pos = [], [], []
    for r in gdn_b:
        if r.get("tag") == "base_o":
            b_o.extend(r["norms"])
    for r in gdn_p:
        if r.get("tag") == "pic_o":
            p_o.extend(r["norms"])
            if "abs_pos" in r:
                p_o_pos.extend(r["abs_pos"])
    T = min(len(b_o), len(p_o))
    if T > 0:
        b_aligned = _align(b_o, p_o, p_o_pos if p_o_pos else None)
        # Frobenius-relative over per-token norms — same style as ssm_final
        # (diff_h/denom). NOT a per-token max: a single low-norm token inflates
        # max_rel without affecting the residual/argmax. This aligns
        # core_attn_out with the other tensor metrics (ssm_final, residual).
        num = sum((bn - pn) ** 2 for bn, pn in zip(b_aligned, p_o)) ** 0.5
        den = sum(bn * bn for bn in b_aligned) ** 0.5 + 1e-12
        m["core_attn_out_max_rel"] = num / den
    else:
        m["core_attn_out_max_rel"] = None

    # SSM final state: BASE single-req vs PIC per-req. Compare BASE against
    # the LAST PIC record (the diag harness drives a single-req prompt).
    b_ssm = [r for r in gdn_b if r.get("tag") == "base_ssm_final"]
    p_ssm = [r for r in gdn_p if r.get("tag") == "pic_ssm_final"]
    if b_ssm and p_ssm:
        b, p = b_ssm[0], p_ssm[-1]
        rel_norm = abs(b["norm"] - p["norm"]) / max(b["norm"], 1e-6)
        diff_h = sum((x - y) ** 2 for x, y in zip(b["head"], p["head"])) ** 0.5
        denom = sum(x * x for x in b["head"]) ** 0.5 + 1e-12
        m["ssm_final_max_rel"] = max(rel_norm, diff_h / denom)
    else:
        m["ssm_final_max_rel"] = None
    return m


def _run_tests(m, mode):
    thr = TEST_THRESHOLDS.get(mode, TEST_THRESHOLDS["transition"])
    out = []

    def _ck(name, actual, ref, ok):
        out.append(TestResult(name, str(actual), str(ref), "PASS" if ok else "FAIL"))

    def _skip(name, why):
        out.append(TestResult(name, "n/a", why, "SKIP"))

    _ck("input_len match",
        f"base={m['base_input_len']} pic={m['pic_input_len']}",
        "equal & >0",
        m["base_input_len"] == m["pic_input_len"] and m["base_input_len"] > 0)

    bid, pid = m["base_output_ids"], m["pic_output_ids"]
    _ck("first-token argmax match",
        f"base={bid[:1]} pic={pid[:1]}", "equal",
        len(bid) >= 1 and len(pid) >= 1 and bid[0] == pid[0])

    if thr.get("check_all_3_tokens"):
        _ck("3-token argmax match",
            f"base={bid[:3]} pic={pid[:3]}", "equal",
            len(bid) >= 3 and len(pid) >= 3 and bid[:3] == pid[:3])
    else:
        _skip("3-token argmax match", f"{mode}: not strict")

    thr_v = thr["gdn_proj_max_rel"]
    for name in ("q", "k", "v", "g", "beta"):
        val = m["gdn_proj_max_rels"].get(name)
        if val is None:
            _skip(f"GDN {name} per-tok max_rel", "no data")
        else:
            _ck(f"GDN {name} per-tok max_rel",
                f"{val:.2e}", f"< {thr_v:.0e}", val < thr_v)

    val = m["core_attn_out_max_rel"]
    thr_v = thr["core_attn_out_max_rel"]
    if val is None:
        _skip("core_attn_out rel", "no data")
    else:
        _ck("core_attn_out rel",
            f"{val:.2e}", f"< {thr_v:.0e}", val < thr_v)

    val = m.get("ssm_final_max_rel")
    thr_v = thr.get("ssm_final_max_rel")
    if thr_v is None:
        _skip("ssm_final base vs pic", f"{mode}: N/A")
    elif val is None:
        _skip("ssm_final base vs pic", "no data")
    else:
        _ck("ssm_final base vs pic",
            f"{val:.2e}", f"< {thr_v:.0e}", val < thr_v)

    val = m["layer0_max_rel"]
    thr_v = thr["layer0_residual_rel"]
    if val is None:
        _skip("layer-0 residual rel (max over passes)", "no data")
    else:
        _ck("layer-0 residual rel (max over passes)",
            f"{val:.2e}", f"< {thr_v:.0e}", val < thr_v)

    if thr.get("check_first_diverge"):
        fd = m["first_diverge_layer_in_pass"]
        min_l = thr["first_diverge_min_layer"]
        _ck("first diverge layer (in-pass)",
            f"layer={fd if fd >= 0 else 'none'}",
            f">= {min_l}",
            fd < 0 or fd >= min_l)
    else:
        _skip("first diverge layer (in-pass)", f"{mode}: expected early drift")

    return out


def _print_test_summary(results):
    print("\n" + "=" * 92)
    print(f"  PIC numerical regression — {len(results)} checks")
    print("=" * 92)
    name_w = max(len(r.name) for r in results)
    act_w = max(len(r.actual) for r in results)
    ref_w = max(len(r.reference) for r in results)
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    for r in results:
        print(f"  [{r.status}] {r.name:<{name_w}}  "
              f"actual={r.actual:<{act_w}}  ref={r.reference:<{ref_w}}")
    print("-" * 92)
    print(f"  PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}  TOTAL={len(results)}")
    print("=" * 92)
    return n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS),
                    default=os.environ.get("PIC_DIAG_MODEL_KEY", "qwen35b"))
    args = ap.parse_args()
    cfg = dict(MODELS[args.model])
    cfg["path"] = os.environ.get("PIC_DIAG_MODEL", cfg["path"])
    tp = int(os.environ.get("PIC_DIAG_TP", cfg["tp"]))
    mode = os.environ.get("PIC_DIAG_MODE") or cfg["mode"]
    os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")

    prompt, warmups = build_prompts(cfg)
    tmpdir = tempfile.mkdtemp(prefix=f"pic_diag_{args.model}_")
    base_path = os.path.join(tmpdir, "baseline.jsonl")
    pic_path = os.path.join(tmpdir, "pic.jsonl")
    base_gdn = os.path.join(tmpdir, "baseline.gdn.jsonl")
    pic_gdn = os.path.join(tmpdir, "pic.gdn.jsonl")
    for p in (base_path, pic_path, base_gdn, pic_gdn):
        open(p, "w").close()
    print(f"=== [{args.model}] baseline (full_recompute) → {base_path} ===")
    base_res = run_once(base_path, None, prompt, warmups, cfg["path"], tp, gdn_path=base_gdn)
    print(f"\n=== [{args.model}] PIC mode={mode} → {pic_path} ===")
    pic_res = run_once(pic_path, mode, prompt, warmups, cfg["path"], tp, gdn_path=pic_gdn)

    print("\n=== token id comparison ===")
    print(f"  baseline: input_len={base_res.get('input_len')}  "
          f"output_ids={base_res.get('output_ids')}  text={base_res.get('text', '')!r}")
    print(f"  PIC:      input_len={pic_res.get('input_len')}  "
          f"output_ids={pic_res.get('output_ids')}  text={pic_res.get('text', '')!r}")
    if base_res.get('input_len') == pic_res.get('input_len'):
        print("  input_len MATCH ✓")
    else:
        print("  input_len MISMATCH ✗ — token ids differ, comparison is invalid")

    diff(base_path, pic_path)
    diff_gdn(base_gdn, pic_gdn)
    diff_compose_chain(base_gdn, pic_gdn)
    print(f"\nArtifacts: {tmpdir}")

    metrics = _compute_metrics(base_path, pic_path, base_gdn, pic_gdn,
                                base_res, pic_res)
    fail_count = _print_test_summary(_run_tests(metrics, mode))
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
