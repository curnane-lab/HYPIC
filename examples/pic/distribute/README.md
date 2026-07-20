# distribute — 4 卡 LPT vs RR 对拍（自包含）

只靠本目录两个文件即可完成「4 prefill + 1 decode PIC scatter 集群」的正确性 + 加速比对拍：

- `cluster.sh` — 起集群 / 起 router / 健康检查 / 跑对拍 / 清理（不依赖 pic_bench，只调 sglang 自带模块）。
- `verify_scatter.py` — 对拍器：单卡 PIC(transition) vs 分布式 PIC(scatter)，报 `OUTPUT identical` + median speedup。

## 拓扑（4-0 口径）

4 prefill GPU0-3（30001/03/05/07，boot 8998-9001）+ decode GPU4（30002）+ 单卡 PIC baseline GPU5（30010）。
router：`pic_lpt` @30020、`pic_round_robin` @30030。baseline 是**单卡 pic transition**（非 full-recompute），
两边同一 PIC kernel 路径，`identity=` 才有意义。

## 一次性前置（QS，pic_lpt 路由需 rust binding）

```bash
cd <sglang>/sgl-model-gateway/bindings/python && \
  PATH=$HOME/.cargo/bin:$PATH /usr/bin/python -m maturin develop --release -i /usr/bin/python
```

## 跑 B（QS）

```bash
cd /root/PIC/sglang/examples/pic/distribute
export DISTPIC_UCX_ENV="UCX_NET_DEVICES=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1 UCX_IB_GID_INDEX=7"

bash cluster.sh up                 # 起 4 prefill + decode + baseline（后台 nohup）
# 轮询 logs/*.log 直到各进程 'server is fired up'
bash cluster.sh router lpt         # 起 pic_lpt @30020
bash cluster.sh router rr          # 起 pic_round_robin @30030
bash cluster.sh check              # curl /health 所有端点（QS ss 坏，用 curl）

bash cluster.sh accept lpt         # baseline vs LPT 对拍（5 次，8-chunk varlen，32k）
bash cluster.sh accept rr          # baseline vs RR 对拍

bash cluster.sh down               # 清理（按 logs/pids 精确 kill，不 pkill -f）
```

`accept lpt|rr [N nchunk total]` 可调轮次/段数/总 token（varlen 固定为 1 = 4-2 负载）。

## 判据

- **正确性**：两条 accept 都应 `OUTPUT identical=True`（逐字一致）。
- **加速比**：LPT median ≈ **3.6×**、RR ≈ **2.4×**（LPT/RR ≈ 1.5×，LPT 压 scatter makespan ~37%）。

## 环境覆盖（默认 QS）

`PYTHON`（默认 `/usr/bin/python`）、`MODEL`（`/workspace/models/Qwen3.5-35B-A3B`）、`DISTPIC_UCX_ENV`、
`PIC_PW`（`tc_piecewise`|`off`）、`PIC_MAMBA`（384）、`PIC_MEMFRAC`（0.90）、`DECODE_GPU`（4）、`BASE_GPU`（5）。
H20 上改 `MODEL=/root/qianyou/models/Qwen3.5-35B-A3B PYTHON=/opt/dynamo/venv/bin/python`，UCX 通常可留空自动选。
