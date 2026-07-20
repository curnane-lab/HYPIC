# HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching

[![arXiv](https://img.shields.io/badge/arXiv-2607.01299-b31b1b.svg)](https://arxiv.org/abs/2607.01299)

HYPIC is the first serving system to bring position-independent caching (PIC) to hybrid-attention LLMs. Existing PIC reuses per-token KV cache, but hybrid stacks are mostly linear-attention layers that expose only a per-request recurrent state — so prior PIC primitives don't transfer. HYPIC closes this gap.

Key techniques:

- **Cached transition** — caches each segment's transition operator $T_C$ with its zero-start end-state, composing linear-attention states near-exactly in constant time.
- **Seam window** — recomputes a small window at each segment beginning to repair cross-segment attention in the full-attention layers.
- **Segment parallelism** — dispatches a request's cold segments across workers for parallel prefill, cutting long-cold-request tail TTFT.

Across four hybrid-attention models and five workloads, HYPIC cuts TTFT by **3.25×** and lifts QPS by **1.66×** over Prefix Cache, within **1.71** points of Full Recompute. Built on [SGLang](https://github.com/sgl-project/sglang) v0.5.14. See the [paper](https://arxiv.org/abs/2607.01299) for details.

## Install

```bash
git clone https://github.com/redai-infra/HYPIC.git && cd HYPIC
pip install -e "python[all]"
```

## Quickstart

Enable PIC on any launch by adding the PIC flags. Prompts are split into
reusable segments at a separator string (`<<PIC_SEP>>`):

```bash
python -m sglang.launch_server \
  --model-path /path/to/Qwen3.5-35B-A3B --tp 2 \
  --page-size 1 --chunked-prefill-size -1 --disable-piecewise-cuda-graph \
  --pic-enable \
  --pic-mode transition_rope_recompute \
  --pic-separator-str '<<PIC_SEP>>' \
  --enable-cache-report
```

```bash
curl localhost:30000/generate -H 'Content-Type: application/json' -d '{
  "text": "The Eiffel Tower is in Paris, France.",
  "sampling_params": {"temperature": 0, "max_new_tokens": 1}
}'
curl localhost:30000/generate -H 'Content-Type: application/json' -d '{
  "text": "Mount Fuji is the tallest mountain in Japan.",
  "sampling_params": {"temperature": 0, "max_new_tokens": 1}
}'
curl localhost:30000/generate -H 'Content-Type: application/json' -d '{
  "text": "You are a helpful assistant.<<PIC_SEP>>The Eiffel Tower is in Paris, France.<<PIC_SEP>>Mount Fuji is the tallest mountain in Japan.<<PIC_SEP>>Question: In which country is the Eiffel Tower?",
  "sampling_params": {"temperature": 0, "max_new_tokens": 512}
}'
```

Each `<<PIC_SEP>>`-delimited chunk is cached position-independently and reused across requests regardless of order. 
With `--enable-cache-report`, the final request's `meta_info` reports a cache hit (e.g. `"cached_tokens": 17`) for the two document chunks — even though neither is a shared prefix, where plain prefix caching would report zero.

### Self-contained examples (`examples/pic/`)

| Script | What it does |
|---|---|
| `quick_test_offline.py` | 6-way in-process (`sgl.Engine`) comparison — full_recompute / prefix_cache / pic_addition / pic_transition / pic_transition_rope / pic_transition_rope_recompute — on one model. |
| `quick_test_online.py` | Same 6-way comparison over an HTTP server: launches `sglang.launch_server` per mode, runs warmup + TTFT + first-decoded-token check. |
| `diag_layer_divergence.py` | Per-layer bisect: finds the first decoder layer where PIC's residual diverges from the baseline (accuracy debugging). |
| `distribute/` | Segment parallelism — 4-prefill + 1-decode PIC scatter cluster, with an LPT-vs-RR speedup/correctness harness. See `examples/pic/distribute/README.md`. |

```bash
# offline / online 6-way sweep on one model
python examples/pic/quick_test_offline.py --model ring_mini
python examples/pic/quick_test_online.py  --model qwen35b

# per-layer divergence bisect (PIC vs baseline)
PIC_DIAG_MODE=transition_rope python examples/pic/diag_layer_divergence.py

# distributed segment parallelism (scatter) — see distribute/README.md
cd examples/pic/distribute
export DISTPIC_UCX_ENV="UCX_NET_DEVICES=mlx5_1:1,... UCX_IB_GID_INDEX=7"  # required on RDMA/IB hosts (see distribute/README.md)
bash cluster.sh up          # 4 prefill + 1 decode + single-GPU PIC baseline (boots in background)
bash cluster.sh router lpt  # start the LPT scatter router
bash cluster.sh check       # wait until /health is green on every endpoint
bash cluster.sh accept lpt  # baseline vs scatter: OUTPUT identical + median speedup
bash cluster.sh down        # tear everything down
```

`--model` takes `qwen35b`, `qwen122b`, `ring_mini`, `ring_flash`; override the
model path / TP with `PIC_MODEL` / `PIC_TP`.

## PIC modes

`--pic-mode` selects how cached segment state is composed:

| Mode | Compose                | RoPE adjustment | Seam recompute | Use                                                           |
|---|------------------------|-----------------|---|---------------------------------------------------------------|
| `addition` | segment state add      | no               | no | naive linear-attention PIC baseline.                          |
| `transition` | cached transition operator | no              | no | Core HYPIC state compose.                                     |
| `transition_rope` | cached transition operator | yes             | no | + correct absolute-position RoPE for non-contiguous hits.     |
| `transition_rope_recompute` | cached transition operator | yes             | yes | **+ seam window** repairs full-attn cross-segment attention.  |

The seam width is a runtime knob independent of mode: `PIC_SEAM_SINK`
(default `8`). `0` = reuse the whole hit segment; `0 < x ≤ 1` = fraction of
each hit segment to recompute; `x > 1` = absolute token count.

## Supported models

Four hybrid linear+full-attention models are wired into the examples:

| `--model` | Path (default) | TP |
|---|---|---|
| `qwen35b` | `Qwen3.5-35B-A3B` | 2 |
| `qwen122b` | `Qwen3.5-122B-A10B` | 4 |
| `ring_mini` | `Ring-mini-linear-2.0` | 1 |
| `ring_flash` | `Ring-flash-linear-2.0` | 4 |

## Citation

```bibtex
@article{liu2026hypic,
  title={HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching},
  author={Liu, Yifei and Wu, Juntong and Liu, Yang and Hu, Junhao and Li, Minghao and Chen, Xiaoxu and Chen, Weihang},
  journal={arXiv preprint arXiv:2607.01299},
  year={2026}
}
```
