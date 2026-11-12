#!/usr/bin/env bash
# Benchmark the attention paths of this image on one GPU, inside the image (a Modal function via modal_bench.py,
# or any container/pod running ghcr.io/wolfgang9999/decision-models-vllm).
#
#   MODEL=baselabs/joey-1 ARMS="T T3 FI F4 BARE" bash runpod_bench.sh
#
# Arms (every arm answers byte-identical requests, so answers can be compared with arm T):
#   T     TRITON_ATTN, FlashAttention fast path off: the reference. Only images up to vllm-be47189f3-de41652 honour
#         VLLM_PREFIX_LM_FA=0; later images ship #65222's file as submitted (no switch), so on SM 8.x/9.x T equals T3.
#   T3    TRITON_ATTN with the FlashAttention prefill path (FA3 on Hopper, FA2 on Ampere/Ada)
#   FI    FLASHINFER, native PrefixLM mask (#46558)
#   F4    FLASH_ATTN with FA4, native PrefixLM mask
#   BARE  plain `vllm serve MODEL --served-model-name NAME`: whatever the image picks by default
# Results: $OUT/summary.jsonl, raw_<arm>.jsonl, logs, report.md (set RESULTS_PORT to also serve them over HTTP).
set -u
MODEL=${MODEL:-baselabs/joey-1}; NAME=${MODEL##*/}
ARMS=${ARMS:-"T T3 FI F4 BARE"}
OUT=${OUT:-/tmp/results/$NAME}; PORT=8100
REPO=${REPO:-https://raw.githubusercontent.com/Wolfgang9999/decision-models-vllm/${BRANCH:-main}}
GRID=${GRID:-"--buckets 500 4000 ref --qs 11 44 --levels 1 4 8 16 32"}
mkdir -p $OUT /tmp/bench && cd /tmp/bench
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $OUT/driver.log; }
# INPUTS_DIR (states.jsonl + payloads.jsonl) overrides the repo's inputs, e.g. private data passed in by modal_bench.py.
FILES="grid_bench.py analyze.py"; [ -z "${INPUTS_DIR:-}" ] && FILES="$FILES inputs/states.jsonl inputs/payloads.jsonl"
for f in $FILES; do
  mkdir -p $(dirname $f); curl -fsSL $REPO/bench/$f -o $f || { log "cannot fetch $f"; exit 1; }
done
INPUTS=${INPUTS_DIR:-inputs}
[ "${RESULTS_PORT:-0}" != 0 ] && (cd $OUT && nohup python3 -m http.server $RESULTS_PORT > /dev/null 2>&1 &)
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | tee $OUT/gpu.txt
python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" | tee -a $OUT/gpu.txt
export HF_HOME=${HF_HOME:-/tmp/hf}
export PYTHONFAULTHANDLER=1  # a native crash in the engine then leaves a stack in serve_<arm>.log
COMMON="--served-model-name $NAME --host 127.0.0.1 --port $PORT --max-num-batched-tokens 65536 --max-num-seqs 64 --gpu-memory-utilization 0.85"
for ARM in $ARMS; do
  case $ARM in
    T)    ENVS="VLLM_PREFIX_LM_FA=0"; ARGS="$COMMON --attention-backend TRITON_ATTN" ;;
    T3)   ENVS=""; ARGS="$COMMON --attention-backend TRITON_ATTN" ;;
    FI)   ENVS=""; ARGS="$COMMON --attention-backend FLASHINFER" ;;
    F4)   ENVS=""; ARGS="$COMMON --attention-backend FLASH_ATTN --attention-config {\"flash_attn_version\":4}" ;;
    BARE) ENVS=""; ARGS="--served-model-name $NAME --host 127.0.0.1 --port $PORT" ;;
    *) log "unknown arm $ARM"; continue ;;
  esac
  log "arm $ARM: starting ($ENVS vllm serve $MODEL $ARGS)"
  env $ENVS vllm serve $MODEL $ARGS > $OUT/serve_$ARM.log 2>&1 &
  SRV=$!
  for t in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $SRV 2>/dev/null || break; sleep 5; done
  if ! curl -sf http://127.0.0.1:$PORT/health > /dev/null; then
    log "arm $ARM: server failed"; tail -30 $OUT/serve_$ARM.log >> $OUT/driver.log; kill $SRV 2>/dev/null; sleep 10; continue
  fi
  grep -hE "using .* attention for the PrefixLM|Using FlashInfer GDN|Using .*backend|prefix caching" $OUT/serve_$ARM.log | head -5 >> $OUT/driver.log
  log "arm $ARM: healthy"
  BENCH_STATES=$INPUTS/states.jsonl BENCH_PAYLOADS=$INPUTS/payloads.jsonl BENCH_MODEL=$NAME \
    python3 grid_bench.py http://127.0.0.1:$PORT $OUT $ARM $GRID 2>&1 | tail -n +1 >> $OUT/driver.log
  log "arm $ARM: done"
  kill $SRV; for t in $(seq 60); do curl -sf http://127.0.0.1:$PORT/health > /dev/null || break; sleep 2; done; sleep 10
done
python3 analyze.py $OUT --ref T --drop > $OUT/report.md 2>&1
log ALL_DONE
