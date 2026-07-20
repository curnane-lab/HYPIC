"""Quick PIC test — 6-way offline (sgl.Engine) comparison, one script per --model.

Offline (in-process Engine) counterpart of quick_test_online.py. Folds Qwen +
Ring into one --model script: Qwen uses raw SYS-prefixed prompts, Ring wraps in
its chat template.

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 /opt/dynamo/venv/bin/python \\
  examples/pic/quick_test_offline.py --model ring_mini

Compares full_recompute / prefix_cache / pic_addition / pic_transition /
pic_transition_rope / pic_transition_rope_recompute.

All PIC prompts use SYS as the first segment to absorb attention sink:
  SYS + <<PIC_SEP>> + doc1 + <<PIC_SEP>> + ... + <<PIC_SEP>> + query
"""
import argparse
import os
import time

import sglang as sgl
from transformers import AutoTokenizer

SEP = "<<PIC_SEP>>"

_RING = dict(
    template=True,
    system="You are a concise QA assistant. Answer as briefly as possible.",
    c1=("All people living in the Thenum District work in Chrysan Company. "
        * int(os.environ.get("PIC_C1_REP", "360"))),
    c2=("Derek is a single man living in the Thenum District. "
        * int(os.environ.get("PIC_C2_REP", "180"))),
    c3=("Answer the question based on the given passages. "
        "Answer with the company name only. ") * int(os.environ.get("PIC_C3_REP", "120")),
    query=("Which company does Derek work in? Reply with the two-word company "
           "name only, no explanation. Start directly with the company name. "
           "Do not start with Derek."),
    close_think=True,
)
_QWEN = dict(
    template=False,
    system="You are a helpful assistant.",
    c1="Document A about cats. " * 800,
    c2="Document B about dogs. " * 800,
    c3="Document C about birds. " * 800,
    query="Question: which animal is in document B?",
    close_think=False,
    # Close the think block so Qwen3.5 answers directly (thinking disabled).
    post="<|im_start|>assistant\n<think>\n\n</think>\n\n",
)
MODELS = {
    "qwen35b": dict(path="/workspace/models/Qwen3.5-35B-A3B", tp=2, **_QWEN),
    "qwen122b": dict(path="/workspace/models/Qwen3.5-122B-A10B", tp=4, **_QWEN),
    "ring_mini": dict(path="/workspace/models/Ring-mini-linear-2.0", tp=1, **_RING),
    "ring_flash": dict(path="/workspace/models/Ring-flash-linear-2.0", tp=4, **_RING),
}

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", choices=list(MODELS),
                 default=os.environ.get("PIC_QT_MODEL", "qwen35b"))
_ARGS, _ = _ap.parse_known_args()
CFG = MODELS[_ARGS.model]
MODEL = os.environ.get("PIC_MODEL", CFG["path"])
TP = int(os.environ.get("PIC_TP", CFG["tp"]))

C1, C2, C3, Q = CFG["c1"], CFG["c2"], CFG["c3"], CFG["query"]
if CFG.get("template"):
    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    _PH = "<<__USER_CONTENT__>>"
    _TEMPL = _tok.apply_chat_template(
        [{"role": "system", "content": CFG["system"]},
         {"role": "user", "content": _PH}],
        tokenize=False, add_generation_prompt=True,
    )
    if CFG.get("close_think") and os.environ.get("PIC_CLOSE_THINK", "1") == "1":
        _TEMPL += "\n</think>\n\n" + os.environ.get("PIC_ANSWER_PREFIX", "Answer:")

    def chat(text: str) -> str:
        return _TEMPL.replace(_PH, text)
else:
    def chat(text: str) -> str:
        return CFG["system"] + text

# PIC prompts: SYS as first segment absorbs attention sink
_POST = CFG.get("post", "")
PIC_PROMPT = chat(f"{SEP}{C1}{SEP}{C2}{SEP}{C3}{SEP}{Q}") + _POST
PIC_W1 = chat(f"{SEP}{C1}{SEP}{Q}") + _POST
PIC_W2 = chat(f"{SEP}{C2}{SEP}{Q}") + _POST
PIC_W3 = chat(f"{SEP}{C3}{SEP}{Q}") + _POST

# Baseline prompts: same content concatenated (no separator needed)
PROMPT = chat(f"{C1}{C2}{C3}{Q}") + _POST
W1 = chat(f"{C1}{Q}") + _POST
W2 = chat(f"{C2}{Q}") + _POST
W3 = chat(f"{C3}{Q}") + _POST


def run_mode(label, engine_kwargs, prompt, warmup_prompts, pic_prompt=False):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    engine = sgl.Engine(**engine_kwargs)
    for wp in warmup_prompts:
        engine.generate(wp, sampling_params={"temperature": 0, "max_new_tokens": 4})
    t0 = time.perf_counter()
    out = engine.generate(prompt, sampling_params={"temperature": 0, "max_new_tokens": 1})
    ttft = time.perf_counter() - t0
    cached = out["meta_info"].get("cached_tokens", 0)
    total = out["meta_info"].get("prompt_tokens", len(prompt.split()))
    tok = out["output_ids"][0] if out.get("output_ids") else None
    out32 = engine.generate(prompt, sampling_params={"temperature": 0, "max_new_tokens": 32})
    text = out32["text"]
    ids32 = list(out32.get("output_ids") or [])[:32]
    engine.shutdown()
    print(f"  TTFT:    {ttft:.3f}s")
    print(f"  Cached:  {cached}/{total}")
    print(f"  1st tok: {tok}")
    print(f"  Output:  {text[:80]!r}")
    return {"ttft": ttft, "cached": cached, "total": total, "tok": tok,
            "text": text, "ids32": ids32}


def main():
    common = dict(model_path=MODEL, tp_size=TP, cuda_graph_backend_prefill="disabled", log_level="error", trust_remote_code=True)
    pic_common = dict(**common, page_size=1, chunked_prefill_size=-1, mem_fraction_static=0.80)
    results = {}
    only = os.environ.get("PIC_QT_ONLY", "").strip()  # comma-separated keys; empty=all

    # Warmup pattern: skip C2 so the test prompt produces a hit-after-miss
    # sequence (SYS hit, C1 hit, C2 miss, C3 hit, Q miss). This exercises the
    # transition_pool T persistence path during the Python hit chain.
    bl_warmups = [W1, W3]
    pic_warmups = [PIC_W1, PIC_W3]

    def want(key):
        return (not only) or (key in only)

    if want("full_recompute"):
        results["full_recompute"] = run_mode(
            "[1/6] Full-Recompute",
            dict(**common, mem_fraction_static=0.80, mamba_radix_cache_strategy="no_buffer",
                 disable_radix_cache=True),
            PROMPT, bl_warmups,
        )

    if want("prefix_cache"):
        results["prefix_cache"] = run_mode(
            "[2/6] Prefix-Cache (extra_buffer)",
            dict(**common, mem_fraction_static=0.80, mamba_radix_cache_strategy="extra_buffer"),
            PROMPT, bl_warmups,
        )

    if want("pic_addition"):
        results["pic_addition"] = run_mode(
            "[3/6] PIC addition (v1)",
            dict(**pic_common, pic_enable=True, pic_mode="addition"),
            PIC_PROMPT, pic_warmups, pic_prompt=True,
        )

    if want("pic_transition"):
        results["pic_transition"] = run_mode(
            "[4/6] PIC transition (v2)",
            dict(**pic_common, pic_enable=True, pic_mode="transition"),
            PIC_PROMPT, pic_warmups, pic_prompt=True,
        )

    if want("pic_transition_rope"):
        results["pic_transition_rope"] = run_mode(
            "[5/6] PIC transition_rope (pos=0)",
            dict(**pic_common, pic_enable=True, pic_mode="transition_rope"),
            PIC_PROMPT, pic_warmups, pic_prompt=True,
        )

    if want("pic_transition_rope_recompute"):
        results["pic_transition_rope_recompute"] = run_mode(
            "[6/6] PIC transition_rope_recompute (seam-window)",
            dict(**pic_common, pic_enable=True, pic_mode="transition_rope_recompute"),
            PIC_PROMPT, pic_warmups, pic_prompt=True,
        )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Mode':<30} {'TTFT':>8} {'Cached':>14} {'1st tok':>8}  {'Output[:32]'}")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 14} {'─' * 8}  {'─' * 20}")
    for name, r in results.items():
        cached_str = f"{r['cached']}/{r['total']}"
        text32 = repr(r['text'][:32])
        print(f"  {name:<30} {r['ttft']:>7.3f}s {cached_str:>14} {r['tok']:>8}  {text32}")
    print()

    if "full_recompute" not in results or "prefix_cache" not in results:
        return
    fr_ids = results["full_recompute"]["ids32"]
    pc_ids = results["prefix_cache"]["ids32"]
    pic_modes = ["pic_addition", "pic_transition", "pic_transition_rope", "pic_transition_rope_recompute"]

    def first_divergence_token(a, b):
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    pc_vs_fr = first_divergence_token(fr_ids, pc_ids)
    print(f"\n  bf16 noise floor FDT (prefix_cache vs full_recompute): {pc_vs_fr}/32")
    print("  PASS = FDT vs full_recompute >= noise floor.\n")
    print("  First Divergence Token (FDT) vs full_recompute (window=32):")
    print(f"  {'Mode':<30}  {'FDT / 32':>12}  {'verdict':>8}")
    print(f"  {'─' * 30}  {'─' * 12}  {'─' * 8}")
    print(f"  {'prefix_cache':<30}  {pc_vs_fr:>9}/32  {'[REF ]':>8}")
    all_pass = True
    for mode in pic_modes:
        if mode not in results:
            continue
        ids = results[mode]["ids32"]
        fdt = first_divergence_token(fr_ids, ids)
        verdict = "PASS" if fdt >= pc_vs_fr else "FAIL"
        if fdt < pc_vs_fr:
            all_pass = False
        print(f"  {mode:<30}  {fdt:>9}/32  [{verdict:>4}]")

    if all_pass:
        print("\n  OVERALL: PASS — all PIC modes reach bf16 noise floor")
    else:
        print("\n  OVERALL: FAIL — some PIC modes diverge below prefix_cache's noise floor")

    # Speedup
    if "full_recompute" in results:
        rc = results["full_recompute"]["ttft"]
        for mode in pic_modes:
            if mode in results:
                r = results[mode]
                print(f"  {mode} vs Full-Recompute: {rc / r['ttft']:.2f}x")


if __name__ == "__main__":
    main()
