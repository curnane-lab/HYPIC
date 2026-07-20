"""PIC quick test — 6-way HTTP-server comparison, one script per --model.

Folds the former quick_test_online.py (Qwen3.5) + ring_quick_test_online.py
(Ring) into one --model script. Every prompt is wrapped in the model's chat
template (SEP lives inside the user turn); each mode launches its own
sglang.launch_server, runs warmup + TTFT + FDT comparison, then shuts down.

Modes: full_recompute / prefix_cache / pic_addition / pic_transition /
       pic_transition_rope / pic_transition_rope_recompute

Run:
  CUDA_VISIBLE_DEVICES=0,1 /opt/dynamo/venv/bin/python \\
    examples/pic/quick_test_online.py --model ring_mini

Knobs: --model {qwen35b,ring_mini,ring_flash}; env PIC_MODEL/PIC_TP override
path/tp; PIC_QT_ONLY=csv to run a subset of modes; PIC_QT_WARMUP_SET=C1,C3;
PIC_PORT; PIC_READY_TIMEOUT; SGLANG_PY.
"""
import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer

SEP = "<<PIC_SEP>>"

# --model config table. Content + chat-template differ per model; everything
# else (launch/poll/compare) is shared.
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
    target="chrysan company",
    close_think=True,
)
_QWEN = dict(
    template=False,
    system="You are a helpful assistant.",
    c1="Document A about cats. " * 800,
    c2="Document B about dogs. " * 800,
    c3="Document C about birds. " * 800,
    query="Question: which animal is in document B?",
    target="dog",
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
SERVED_NAME = os.environ.get(
    "PIC_SERVED_NAME", os.path.basename(MODEL.rstrip("/")) or "model")
TP = int(os.environ.get("PIC_TP", CFG["tp"]))
PORT = int(os.environ.get("PIC_PORT", "9000"))
READY_TIMEOUT = int(os.environ.get("PIC_READY_TIMEOUT", "360"))
OBS_NEW = int(os.environ.get("PIC_OBS_NEW", "64"))
SGLANG_PY = os.environ.get("SGLANG_PY", sys.executable)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_PYTHON = os.path.join(REPO_ROOT, "python")
FORCE_SPLIT_BASELINE = os.environ.get("PIC_FORCE_SPLIT_BASELINE", "0") == "1"
ONLY = {x.strip() for x in os.environ.get("PIC_QT_ONLY", "").split(",") if x.strip()}
WARMUP_SET = {
    x.strip().upper()
    for x in os.environ.get("PIC_QT_WARMUP_SET", "C1,C3").split(",")
    if x.strip()
}


# ----------------------------- prompt build --------------------------------
C1, C2, C3, Q = CFG["c1"], CFG["c2"], CFG["c3"], CFG["query"]
TARGET = os.environ.get("PIC_TARGET", CFG["target"]).lower()

if CFG.get("template"):
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    PLACEHOLDER = "<<__USER_CONTENT__>>"
    TEMPLATED_WITH_PH = tok.apply_chat_template(
        [
            {"role": "system", "content": CFG["system"]},
            {"role": "user", "content": PLACEHOLDER},
        ],
        tokenize=False, add_generation_prompt=True,
    )
    assert PLACEHOLDER in TEMPLATED_WITH_PH, "placeholder vanished from chat template"
    if CFG.get("close_think") and os.environ.get("PIC_CLOSE_THINK", "1") == "1":
        TEMPLATED_WITH_PH += "\n</think>\n\n" + os.environ.get("PIC_ANSWER_PREFIX", "Answer:")

    def chat(text: str) -> str:
        return TEMPLATED_WITH_PH.replace(PLACEHOLDER, text)
else:
    # raw (original Qwen behavior): prepend system text, no chat template
    def chat(text: str) -> str:
        return CFG["system"] + text


_POST = CFG.get("post", "")
PIC_PROMPT = chat(f"{SEP}{C1}{SEP}{C2}{SEP}{C3}{SEP}{Q}") + _POST
PIC_W1 = chat(f"{SEP}{C1}{SEP}{Q}") + _POST
PIC_W2 = chat(f"{SEP}{C2}{SEP}{Q}") + _POST
PIC_W3 = chat(f"{SEP}{C3}{SEP}{Q}") + _POST

PROMPT = chat(f"{C1}{C2}{C3}{Q}") + _POST
W1 = chat(f"{C1}{Q}") + _POST
W2 = chat(f"{C2}{Q}") + _POST
W3 = chat(f"{C3}{Q}") + _POST


# --------------------------- launch / poll helpers --------------------------
def _wait_port_free(port, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        finally:
            s.close()
        time.sleep(0.5)
    raise RuntimeError(f"port {port} still bound")


def _wait_ready(port, proc, timeout):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, socket.timeout):
            pass
        time.sleep(2)
    raise RuntimeError(f"server not ready within {timeout}s")


def _post_generate(text, max_new):
    body = {"text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def _launch(extra_args, log_path):
    cmd = [SGLANG_PY, "-m", "sglang.launch_server", "--model-path", MODEL,
           "--served-model-name", SERVED_NAME, "--tp", str(TP),
           "--host", "0.0.0.0", "--port", str(PORT),
           "--disable-piecewise-cuda-graph",
           "--log-level", "warning", "--trust-remote-code", *extra_args]
    print(f"  launch: {' '.join(shlex.quote(c) for c in cmd)}")
    log = open(log_path, "w")
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_PYTHON + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
    if FORCE_SPLIT_BASELINE:
        env["PIC_DIAG_FORCE_SPLIT"] = "1"
    return subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=env)


def _shutdown(proc):
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=10)



# ------------------------------ per-mode runner -----------------------------
def run_mode(label, tag, extra_args, prompt, warmups):
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    _wait_port_free(PORT)
    log_path = f"/tmp/quick_test_{tag}.log"
    proc = _launch(extra_args, log_path)
    try:
        _wait_ready(PORT, proc, READY_TIMEOUT)
        for w in warmups:
            _post_generate(w, 4)
        n_samp = int(os.environ.get("PIC_TTFT_SAMPLES", "1"))
        novel = os.environ.get("PIC_NOVEL_QUERY", "0") == "1"
        samples = []
        first = None
        for i in range(n_samp):
            p = prompt + (f" (req {i})" if novel else "")
            t0 = time.perf_counter()
            first = _post_generate(p, 1)
            samples.append(time.perf_counter() - t0)
        samples.sort()
        ttft = samples[len(samples) // 2]
        ttft_min = samples[0]
        out = _post_generate(prompt, OBS_NEW)
    finally:
        _shutdown(proc)
        _wait_port_free(PORT)
    meta = first.get("meta_info", {})
    out_meta = out.get("meta_info", {})
    text = out.get("text", "")
    output_ids = out.get("output_ids") or []
    raw_path = f"/tmp/quick_test_{tag}_raw.txt"
    with open(raw_path, "w") as fh:
        fh.write(text)
    first_token = first["output_ids"][0] if first.get("output_ids") else None
    print(f"  prompt_tokens:     {meta.get('prompt_tokens')}")
    print(f"  cached_tokens:     {meta.get('cached_tokens')}")
    print(f"  ttft(med/min):     {ttft:.3f}s / {ttft_min:.3f}s")
    print(f"  first_token:       {first_token}")
    print(f"  finish_reason:     {out_meta.get('finish_reason')}")
    print(f"  completion_tokens: {out_meta.get('completion_tokens')}")
    print(f"  output[:80]:       {text[:80]!r}")
    print(f"  raw dump:          {raw_path}")
    return {
        "text": text,
        "ids32": list(output_ids)[:32],
        "meta": meta,
        "out_meta": out_meta,
        "first_token": first_token,
        "hit": TARGET in text.lower(),
        "finish_reason": out_meta.get("finish_reason"),
        "ttft": ttft,
        "ttft_min": ttft_min,
    }


def fdt(ref_ids, ids, window=32):
    m = min(len(ref_ids), len(ids), window)
    return next((i for i in range(m) if ref_ids[i] != ids[i]), m)


def _select_pic_warmups():
    sel = []
    if "C1" in WARMUP_SET:
        sel.append(PIC_W1)
    if "C2" in WARMUP_SET:
        sel.append(PIC_W2)
    if "C3" in WARMUP_SET:
        sel.append(PIC_W3)
    return sel


def main():
    pic_common = ["--page-size", "1", "--chunked-prefill-size", "-1",
                  "--mem-fraction-static", "0.80", "--pic-separator-str", SEP]
    baseline_prompt = PIC_PROMPT if FORCE_SPLIT_BASELINE else PROMPT
    baseline_warmups = [PIC_W1, PIC_W3] if FORCE_SPLIT_BASELINE else [W1, W3]
    mode_count = 6

    def want(mode):
        return not ONLY or mode in ONLY

    results = {}
    if want("full_recompute"):
        results["full_recompute"] = run_mode(
            f"[1/{mode_count}] full_recompute (chat)", "full_recompute",
            ["--mem-fraction-static", "0.80",
             "--mamba-scheduler-strategy", "no_buffer", "--disable-radix-cache"],
            baseline_prompt, baseline_warmups,
        )
    if want("prefix_cache"):
        results["prefix_cache"] = run_mode(
            f"[2/{mode_count}] prefix_cache (chat)", "prefix_cache",
            ["--mem-fraction-static", "0.80",
             "--mamba-scheduler-strategy", "extra_buffer"],
            baseline_prompt, baseline_warmups,
        )
    if want("pic_addition"):
        results["pic_addition"] = run_mode(
            f"[3/{mode_count}] pic_addition (chat)", "pic_addition",
            [*pic_common, "--pic-enable", "--pic-mode", "addition"],
            PIC_PROMPT, _select_pic_warmups(),
        )
    if want("pic_transition"):
        results["pic_transition"] = run_mode(
            f"[4/{mode_count}] pic_transition (chat)", "pic_transition",
            [*pic_common, "--pic-enable", "--pic-mode", "transition"],
            PIC_PROMPT, _select_pic_warmups(),
        )
    if want("pic_transition_rope"):
        results["pic_transition_rope"] = run_mode(
            f"[5/{mode_count}] pic_transition_rope (chat)", "pic_transition_rope",
            [*pic_common, "--pic-enable", "--pic-mode", "transition_rope"],
            PIC_PROMPT, _select_pic_warmups(),
        )
    if want("pic_transition_rope_recompute"):
        results["pic_transition_rope_recompute"] = run_mode(
            f"[6/{mode_count}] pic_transition_rope_recompute (chat)",
            "pic_transition_rope_recompute",
            [*pic_common, "--pic-enable", "--pic-mode", "transition_rope_recompute"],
            PIC_PROMPT, _select_pic_warmups(),
        )

    if "full_recompute" not in results or "prefix_cache" not in results:
        return

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Mode':<30} {'TTFT':>8} {'Cached':>14} {'1st tok':>8}  {'Output[:32]'}")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 14} {'─' * 8}  {'─' * 20}")
    for name, r in results.items():
        m = r["meta"]
        cached_str = f"{m.get('cached_tokens', 0)}/{m.get('prompt_tokens', 0)}"
        text32 = repr(r["text"][:32])
        print(f"  {name:<30} {r['ttft']:>7.3f}s {cached_str:>14} "
              f"{str(r['first_token']):>8}  {text32}")
    print()

    fr_ids = results["full_recompute"]["ids32"]
    pc_ids = results["prefix_cache"]["ids32"]
    pic_modes = [
        "pic_addition",
        "pic_transition",
        "pic_transition_rope",
        "pic_transition_rope_recompute",
    ]
    pc_vs_fr = fdt(fr_ids, pc_ids)
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
        f = fdt(fr_ids, results[mode]["ids32"])
        verdict_s = "PASS" if f >= pc_vs_fr else "FAIL"
        if f < pc_vs_fr:
            all_pass = False
        print(f"  {mode:<30}  {f:>9}/32  [{verdict_s:>4}]")

    if all_pass:
        print("\n  OVERALL: PASS — all PIC modes reach bf16 noise floor")
    else:
        print("\n  OVERALL: FAIL — some PIC modes diverge below prefix_cache's noise floor")

    rc = results["full_recompute"]["ttft"]
    for mode in pic_modes:
        if mode in results:
            print(f"  {mode} vs Full-Recompute: {rc / results[mode]['ttft']:.2f}x")


if __name__ == "__main__":
    main()
