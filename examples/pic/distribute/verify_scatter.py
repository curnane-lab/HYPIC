#!/usr/bin/env python3
# optA acceptance: distributed(pic_round_robin) vs single-GPU(pic transition).
# HIGH-CONFIDENCE prompt (determinate greedy answer) so output identity is not
# a low-confidence coin-flip. Embeds a codename fact; query asks it back.
#   /opt/dynamo/venv/bin/python verify_scatter.py <single_url> <pd_url> <N> [nchunk] [total]
import json, sys, time, urllib.request, statistics

SEP = "<<PIC_SEP>>"


def filler(tag, n):
    s = (f"In document section {tag} the notes describe subsystem "
         f"behavior under load latency and recovery over time").split()
    o = []
    while len(o) < n:
        o.extend(s)
    return " ".join(o[:n])


def build(total, salt, nchunk, codename, varlen=0):
    # sys states the fact plainly; chunks are filler; query recalls the fact.
    sys_seg = (f"You are a precise assistant. Important fact: the subsystem "
               f"codename is {codename}. Use only the provided documents." + salt)
    q = (f"Question: What is the subsystem codename mentioned above? "
         f"Answer in one word:")
    budget = total - 300
    if varlen:
        # 长短不一 + 对 round-robin 不利的排列:大小交替,大段落在同一残差类。
        # LPT 按长度重排后应显著压低 makespan。
        desc = sorted(range(1, nchunk + 1), reverse=True)  # [nchunk..1]
        weights, lo, hi = [], 0, len(desc) - 1
        while lo <= hi:
            weights.append(desc[lo]); lo += 1
            if lo <= hi:
                weights.append(desc[hi]); hi -= 1
        s = sum(weights)
        chunks = [filler(f"c{i}" + salt, max(50, budget * w // s))
                  for i, w in enumerate(weights)]
    else:
        per = budget // nchunk
        chunks = [filler(f"c{i}" + salt, per) for i in range(nchunk)]
    return SEP.join([sys_seg, *chunks, q])


def gen(u, p, mx=8):
    b = json.dumps({"text": p, "sampling_params": {"temperature": 0, "max_new_tokens": mx}}).encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        u.rstrip("/") + "/generate", data=b,
        headers={"Content-Type": "application/json"}, method="POST"), timeout=600).read())


def probe(base, prompt, mx=16):
    body = json.dumps({"text": prompt,
                       "sampling_params": {"temperature": 0, "max_new_tokens": mx},
                       "stream": True}).encode()
    r = urllib.request.Request(base.rstrip("/") + "/generate", data=body,
                               headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter(); tf = None; tl = None; n = 0; prev = 0
    with urllib.request.urlopen(r, timeout=600) as resp:
        for raw in resp:
            l = raw.decode("utf-8", "ignore").strip()
            if not l.startswith("data:"):
                continue
            p = l[5:].strip()
            if p == "[DONE]":
                break
            try:
                o = json.loads(p)
            except Exception:
                continue
            ids = o.get("output_ids"); now = time.perf_counter()
            cur = len(ids) if ids else prev
            if cur > prev:
                if tf is None:
                    tf = now
                tl = now; prev = cur; n = cur
    ttft = (tf - t0) if tf else time.perf_counter() - t0
    tpot = ((tl - tf) / (n - 1)) if (tf and n > 1) else 0.0
    return ttft, tpot


if __name__ == "__main__":
    single, pd, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nchunk = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    total = int(sys.argv[5]) if len(sys.argv) > 5 else 32000
    varlen = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    CODE = "ZEPHYR"
    b36 = lambda x: ((x == 0) and "0") or (b36(x // 36).lstrip("0") + "0123456789abcdefghijklmnopqrstuvwxyz"[x % 36])

    # identity on a fixed high-confidence prompt
    idp = build(total, " idcheck", nchunk, CODE, varlen)
    sj, pj = gen(single, idp), gen(pd, idp)
    si, di = sj.get("output_ids"), pj.get("output_ids")
    print(f"[{nchunk} chunk] prompt_tokens single={sj.get('meta_info',{}).get('prompt_tokens')} "
          f"pd={pj.get('meta_info',{}).get('prompt_tokens')}")
    print(f"OUTPUT identical={si==di}  single_text={sj.get('text')!r}  pd_text={pj.get('text')!r}")

    sp = []; tps = []; tpp = []
    for i in range(N):
        salt = " " + b36(time.time_ns() % 1000000)
        prompt = build(total, salt, nchunk, CODE, varlen)
        ts, ts_tp = probe(single, prompt)
        tp, tp_tp = probe(pd, prompt)
        sp.append(ts / tp); tps.append(ts_tp); tpp.append(tp_tp)
        print(f"run{i}: single={ts*1000:.0f}ms dist={tp*1000:.0f}ms speedup={ts/tp:.2f}x  "
              f"tpot s={ts_tp*1000:.2f} d={tp_tp*1000:.2f}")
        time.sleep(1)
    print(f"\n[{nchunk} chunk] MEDIAN speedup={statistics.median(sp):.2f}x  best={max(sp):.2f}x  "
          f"TPOT ratio={statistics.median(tpp)/statistics.median(tps):.2f}x  identity={si==di}")
