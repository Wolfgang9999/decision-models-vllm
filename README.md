# Base Labs decision models on vLLM

Serve [joey-1](https://huggingface.co/baselabs/joey-1) and [elephant-1](https://huggingface.co/baselabs/elephant-1)
with plain `vllm serve` on any recent NVIDIA GPU. The image is `vllm/vllm-openai` (pinned nightly) plus the open pull
requests these models need, and it picks the fastest attention path for the GPU it runs on.

```bash
docker run --gpus all --ipc=host -p 8000:8000 ghcr.io/wolfgang9999/decision-models-vllm \
  baselabs/joey-1 --served-model-name joey-1
```

The entrypoint is the same `vllm serve` as `vllm/vllm-openai`, so the image drops into any setup built for it,
including Helm charts. The server speaks the model cards' `POST /v1/systemone` API (the same as OneJev's server), so
only the URL changes for existing clients.

> **Status:** support for these models is not in a vLLM release yet. This repo pins a vLLM nightly and adds the open
> pull requests listed below as a Python-only overlay. When they are released, upstream `vllm serve` will do the same.

<!-- RESULTS:SUMMARY -->

## Quick start

### Docker

```bash
docker run --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/wolfgang9999/decision-models-vllm baselabs/elephant-1 --served-model-name elephant-1
```

No tuning flags are needed: for these models the image defaults to BF16, a 16,384-token context, prefix caching, the
PrefixLM scheduler, prefill steps of at least 32,768 tokens and the attention backend for your GPU. Any `vllm serve`
option still overrides them (for example `--revision <commit>`, `--gpu-memory-utilization 0.6`,
`--max-num-batched-tokens 65536`, `--attention-backend TRITON_ATTN`).

### Helm and Kubernetes

Use the image wherever a chart runs `vllm/vllm-openai`; the container args are the `vllm serve` arguments:

```yaml
image:
  repository: ghcr.io/wolfgang9999/decision-models-vllm
  tag: latest
args: ["baselabs/joey-1", "--served-model-name", "joey-1", "--port", "8000"]
resources:
  limits:
    nvidia.com/gpu: 1
```

Readiness: `GET /health`. Prometheus metrics: `GET /metrics` (including `vllm:prefix_cache_*`).

### Without Docker

Linux x86_64, Python 3.10-3.13, an NVIDIA driver for CUDA 13 (R580 or newer):

```bash
git clone https://github.com/Wolfgang9999/decision-models-vllm && cd decision-models-vllm
python -m venv .venv && source .venv/bin/activate
./install.sh            # pinned vLLM nightly + the overlay (checked file by file)
vllm serve baselabs/joey-1 --served-model-name joey-1
```

### Request

```python
import httpx
r = httpx.post("http://localhost:8000/v1/systemone", json={
    "model": "joey-1",
    "state": "I was charged twice this month, please refund one of them.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, refunds", "technical": "App faults", "fraud": "Unauthorised use"}},
    },
})
print(r.json()["answers"])
```

## What you get on each GPU

| GPU | Prefix cache (one state prefill shared by all its questions) | Attention for the PrefixLM mask | Status |
|---|---|---|---|
| A100, A30 (SM 8.0) | Triton GDN checkpoints (#65186) | Triton, prefill on FlashAttention-2 (#65222) | measured |
| A10, A40, L4, L40S, RTX 30/40 (SM 8.6/8.9) | same | same | expected, not measured |
| H100, H200 (SM 9.0) | FlashInfer GDN prefill (in vLLM) | FlashInfer native mask (#46558) | expected, not measured |
| B200, GB200 (SM 10.x) | FlashInfer GDN prefill (in vLLM) | FlashInfer native mask | expected, not measured |
| RTX PRO 6000, RTX 5090 (SM 12.x) | FlashInfer GDN prefill (in vLLM) | FlashInfer native mask | measured |

<!-- RESULTS:TABLES -->

## How it works

A decision request is one long **state** (a conversation, documents) and several **questions** about it; the model
returns a calibrated probability for every answer option. Nothing is generated, so the cost is reading the state.

- **One state prefill per request, and per state.** joey-1 and elephant-1 are trained with a block-bidirectional mask:
  the state is one block and each question another, so the state's representation does not depend on the questions.
  vLLM PR #63114 adds the model, a PrefixLM scheduler and `/v1/systemone`; each question becomes a sequence that hits
  the prefix cache for the state. A later request with the same state hits the cache too.
- **Prefix caching for the hybrid layers.** The models mix full attention with GDN linear-attention layers, whose
  recurrent state must be saved mid-prefill to be reused. vLLM does that with FlashInfer's kernel on SM 9.0+; #65186
  makes the Triton kernel do it, which is what SM 8.x needs.
- **Attention.** #63114 applies the mask with vLLM's Triton kernel on every GPU. On SM 8.x, #65222 sends those prefills
  to FlashAttention-2 (one non-causal call per block). From SM 9.0 this repo selects FlashInfer, whose prefill applies
  the mask natively (#46558); measured 5-28% faster than Triton on an RTX PRO 6000 with the same answers.

### Pull requests included

| PR | What | State |
|---|---|---|
| [vllm#63114](https://github.com/vllm-project/vllm/pull/63114) | model, PrefixLM scheduler, `/v1/systemone` | open |
| [vllm#65186](https://github.com/vllm-project/vllm/pull/65186) | Triton GDN prefill checkpoints (prefix cache on SM 8.x) | open |
| [vllm#65222](https://github.com/vllm-project/vllm/pull/65222) | FlashAttention-2 PrefixLM prefill on SM 8.x | open |
| [vllm#46558](https://github.com/vllm-project/vllm/pull/46558) | FlashInfer native mm-prefix attention | merged |

On top of those, the overlay carries what it takes to combine them on current vLLM main:

- #63114 still checks `use_v2_model_runner`, which main removed with Model Runner V1 (#63373); the check is dropped.
- #63114 keeps prefix caching when the Triton GDN backend exports checkpoints (#65186).
- #63114 accepts FlashInfer (and FlashAttention with FA4), flags the model as a PrefixLM so they apply its mask, and
  picks the backend per GPU; vLLM's rule that forces unchunked multimodal input for such models is skipped for this
  text-only model, whose hybrid prefix cache needs chunked scheduling.

`overlay/manifest.json` lists every file and the SHA256 of the pinned vLLM file it replaces; `install_overlay.py`
refuses to apply on any other build.

## Notes and limits

- Text states only (`--language-model-only`); image states need OneJev.
- `confidence` in answers is computed differently from OneJev; the option probabilities agree.
- Check these models through decisions, not chat: joey-1's chat output is noise in every runtime.
- Requests are limited to 16,384 tokens, like the model cards.
- vLLM's FlashAttention-2 build has no SM 12.x code, which is one reason SM 9.0+ uses FlashInfer here.

## Reproducing the numbers

`bench/grid_bench.py` drives a server over a grid of state sizes (about 500, 2,000, 4,000 and 13,200 tokens),
questions per request (1, 11, 44) and concurrency (1 to 64), closed loop, 20 s per window, with a unique nonce per
request so nothing is reused across requests; then the same state asked in three requests (4, 4 and 3 questions).
`bench/analyze.py` builds the tables and compares every answer with OneJev's on byte-identical requests.

## License

Apache-2.0, as vLLM. Files under `overlay/` are modified vLLM sources.
