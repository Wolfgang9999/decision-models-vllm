#!/usr/bin/env bash
# Serve a Base Labs decision model (elephant-1, joey-1, mammoth-1) with vLLM.
# Every setting has an env default; extra arguments are passed to `vllm serve`.
set -euo pipefail
export MODEL=${MODEL:-baselabs/elephant-1}
export REVISION=${REVISION:-main}
MODEL_NAME=${MODEL_NAME:-${MODEL##*/}}
export MODEL_DIR=${MODEL_DIR:-/models/${MODEL_NAME}}
cat /opt/decision-build/provenance.json
# Safetensors only; skip GGUF and .bin weights if a repo carries them.
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL"], revision=os.environ["REVISION"],
                  local_dir=os.environ["MODEL_DIR"], ignore_patterns=["*.gguf", "*.bin"])
PY
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --dtype bfloat16 --language-model-only \
  --max-model-len 16384 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --attention-backend TRITON_ATTN --mamba-cache-mode align \
  --no-enable-log-requests "$@"
