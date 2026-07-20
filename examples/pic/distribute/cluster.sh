#!/usr/bin/env bash
# Self-contained 4-prefill + 1-decode + 1-baseline PIC scatter cluster, sized
# for the verify_scatter.py LPT-vs-RR correctness+speedup check. Everything needed for
# "B: 真跑 4 卡 LPT vs RR 对拍" lives in this directory (this script + verify_scatter.py);
# it does NOT source pic_bench. Only sglang's own modules are invoked
# (sglang.launch_server, sglang_router.launch_router).
#
# Topology (4-0 口径): 4 prefill GPU0-3 + decode GPU4 + single-GPU PIC baseline GPU5.
#   baseline = single-GPU pic transition (verify_scatter <single_url>), NOT full-recompute,
#   so identity= comparison is PIC-vs-PIC (same kernel path, both greedy).
#
# Usage:
#   bash cluster.sh up                        # start 4 prefill + decode + baseline
#   bash cluster.sh router lpt|rr             # pic_lpt @30020 | pic_round_robin @30030
#   bash cluster.sh check                     # curl /health on every endpoint
#   bash cluster.sh accept lpt|rr [N nchunk total]   # run verify_scatter.py vs a router
#   bash cluster.sh down                      # kill everything this script started
#
# Env overrides (defaults = QS):
#   PYTHON=/usr/bin/python
#   MODEL=/workspace/models/Qwen3.5-35B-A3B
#   DISTPIC_UCX_ENV="UCX_NET_DEVICES=mlx5_1:1,...,mlx5_8:1 UCX_IB_GID_INDEX=7"   # QS 必需
#   PIC_PW=tc_piecewise|off   PIC_MAMBA=384   PIC_MEMFRAC=0.90
#   PIC_SCATTER_TIMEOUT=600   DECODE_GPU=4   BASE_GPU=5
#
# One-time prereq (pic_lpt router needs the rust binding built once):
#   cd <sglang>/sgl-model-gateway/bindings/python && \
#     PATH=$HOME/.cargo/bin:$PATH $PYTHON -m maturin develop --release -i $PYTHON
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/usr/bin/python}"
MODEL="${MODEL:-/workspace/models/Qwen3.5-35B-A3B}"
SEP='<<PIC_SEP>>'
LOG="${PIC_LOG:-$HERE/logs}"; mkdir -p "$LOG"
PIDS="$LOG/pids"                       # one line per proc: "role port pid"
UCX_ENV="${DISTPIC_UCX_ENV-}"

PREFILL_PORTS=(30001 30003 30005 30007)
BOOT_PORTS=(8998 8999 9000 9001)
DECODE_PORT=30002
BASE_PORT=30010
LPT_PORT=30020
RR_PORT=30030
DECODE_GPU="${DECODE_GPU:-4}"
BASE_GPU="${BASE_GPU:-5}"

# --- piecewise CUDA graph flags (match launch_pd.sh; PIC_PW=off for stock eager) ---
if [ "${PIC_PW:-tc_piecewise}" = off ]; then
  PW_FLAGS="--disable-piecewise-cuda-graph"
else
  PW_FLAGS="--cuda-graph-backend-prefill=${PIC_PW:-tc_piecewise} --cuda-graph-max-bs-prefill ${PIC_PW_MAXBS:-16384}"
fi
COMMON="--model-path $MODEL --served-model-name Qwen3.5-35B-A3B --tp 1 \
  --host 0.0.0.0 --chunked-prefill-size -1 $PW_FLAGS --enable-cache-report"
PIC_ENG="--pic-enable --pic-mode transition --pic-separator-str $SEP \
  --max-mamba-cache-size ${PIC_MAMBA:-384} --mem-fraction-static ${PIC_MEMFRAC:-0.90}"

record() { echo "$1 $2 $3" >>"$PIDS"; }   # role port pid

launch_prefill() { # gpu port bootstrap
  CUDA_VISIBLE_DEVICES=$1 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 PYTHONUNBUFFERED=1 \
    PIC_PERF=${PIC_PERF:-0} env $UCX_ENV \
    nohup "$PYTHON" -m sglang.launch_server $COMMON $PIC_ENG \
    --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port "$3" \
    --pic-scatter-timeout-s "${PIC_SCATTER_TIMEOUT:-600}" --port "$2" \
    >"$LOG/prefill-$2.log" 2>&1 & disown
  record prefill "$2" $!
}

launch_decode() {
  CUDA_VISIBLE_DEVICES=$DECODE_GPU SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 PYTHONUNBUFFERED=1 \
    env $UCX_ENV \
    nohup "$PYTHON" -m sglang.launch_server $COMMON \
    --disaggregation-mode decode --disaggregation-transfer-backend nixl --port "$DECODE_PORT" \
    >"$LOG/decode-$DECODE_PORT.log" 2>&1 & disown
  record decode "$DECODE_PORT" $!
}

launch_baseline() { # single-GPU pic transition, no disagg (verify_scatter <single_url>)
  CUDA_VISIBLE_DEVICES=$BASE_GPU SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 PYTHONUNBUFFERED=1 \
    nohup "$PYTHON" -m sglang.launch_server $COMMON $PIC_ENG --port "$BASE_PORT" \
    >"$LOG/baseline-$BASE_PORT.log" 2>&1 & disown
  record baseline "$BASE_PORT" $!
}

launch_router() { # policy port
  local prefill_args=""
  for i in "${!PREFILL_PORTS[@]}"; do
    prefill_args="$prefill_args --prefill http://127.0.0.1:${PREFILL_PORTS[$i]} ${BOOT_PORTS[$i]}"
  done
  PATH="$HOME/.cargo/bin:$PATH" PYTHONUNBUFFERED=1 \
    nohup "$PYTHON" -m sglang_router.launch_router \
    --pd-disaggregation $prefill_args --decode "http://127.0.0.1:$DECODE_PORT" \
    --policy "$1" --host 0.0.0.0 --port "$2" --prometheus-port "$(($2 - 1000))" \
    >"$LOG/router-$1-$2.log" 2>&1 & disown
  record "router:$1" "$2" $!
}

cmd_up() {
  for i in "${!PREFILL_PORTS[@]}"; do
    launch_prefill "$i" "${PREFILL_PORTS[$i]}" "${BOOT_PORTS[$i]}"
  done
  launch_decode
  launch_baseline
  echo "up: 4 prefill GPU0-3 (${PREFILL_PORTS[*]}), decode GPU$DECODE_GPU ($DECODE_PORT), baseline GPU$BASE_GPU ($BASE_PORT)"
  echo "poll $LOG/*.log for 'server is fired up', then: bash cluster.sh router lpt|rr && bash cluster.sh check"
}

cmd_router() {
  case "${1:?router lpt|rr}" in
    lpt) launch_router pic_lpt "$LPT_PORT"; echo "router pic_lpt on :$LPT_PORT" ;;
    rr)  launch_router pic_round_robin "$RR_PORT"; echo "router pic_round_robin on :$RR_PORT" ;;
    *) echo "bad router arg (lpt|rr)"; exit 2 ;;
  esac
}

cmd_check() { # curl /health everywhere (ss is broken on QS; curl is the probe)
  local ok=1
  for p in "${PREFILL_PORTS[@]}" "$DECODE_PORT" "$BASE_PORT" "$LPT_PORT" "$RR_PORT"; do
    if curl -sf -m 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1; then
      echo "  :$p  UP"
    else
      echo "  :$p  down/starting"; ok=0
    fi
  done
  [ "$ok" = 1 ] && echo "all endpoints healthy" || echo "some endpoints not ready (routers only listed if started)"
}

cmd_accept() { # lpt|rr [N nchunk total]
  local which="${1:?accept lpt|rr}"; shift || true
  local N="${1:-5}" NCHUNK="${2:-8}" TOTAL="${3:-32000}"
  local pd
  case "$which" in
    lpt) pd=$LPT_PORT ;;
    rr)  pd=$RR_PORT ;;
    *) echo "bad accept arg (lpt|rr)"; exit 2 ;;
  esac
  # varlen=1 → 4-2 load (8-chunk variable length, fresh-salt all-miss).
  exec "$PYTHON" "$HERE/verify_scatter.py" \
    "http://127.0.0.1:$BASE_PORT" "http://127.0.0.1:$pd" "$N" "$NCHUNK" "$TOTAL" 1
}

cmd_down() {
  [ -f "$PIDS" ] || { echo "no $PIDS; nothing this script started"; return 0; }
  while read -r role port pid; do
    [ -n "${pid:-}" ] || continue
    if kill -0 "$pid" 2>/dev/null; then echo "kill $role :$port pid=$pid"; kill -9 "$pid" 2>/dev/null; fi
  done <"$PIDS"
  rm -f "$PIDS"
  echo "down: killed all tracked procs"
}

case "${1:?usage: cluster.sh up|router|check|accept|down}" in
  up)     cmd_up ;;
  router) shift; cmd_router "$@" ;;
  check)  cmd_check ;;
  accept) shift; cmd_accept "$@" ;;
  down)   cmd_down ;;
  *) echo "unknown cmd: $1"; exit 2 ;;
esac
