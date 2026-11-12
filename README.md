# Base Labs decision models on vLLM

Serve [joey-1](https://huggingface.co/baselabs/joey-1) and [elephant-1](https://huggingface.co/baselabs/elephant-1)
with plain `vllm serve` on any recent NVIDIA GPU. The image is `vllm/vllm-openai` (pinned nightly) plus the open pull
requests these models need, and it picks the fastest attention path for the GPU it runs on.

```bash
# image: ghcr.io/wolfgang9999/decision-models-vllm  (vLLM nightly + the pending PRs, drop-in for vllm/vllm-openai)
vllm serve baselabs/joey-1 --served-model-name joey-1
```

The entrypoint is the same `vllm serve` as `vllm/vllm-openai`, so the image drops into any setup built for it,
including Helm charts. The server speaks the model cards' `POST /v1/systemone` API (the same as OneJev's server), so
only the URL changes for existing clients.

> **Status:** support for these models is not in a vLLM release yet. This repo pins a vLLM nightly and adds the open
> pull requests listed below as a Python-only overlay. When they are released, upstream `vllm serve` will do the same.

## Results at a glance

Decisions per second (one decision = one answered question), cold states, against OneJev, the runtime the model cards
install. Higher is better.

| Model, GPU, workload | OneJev | vLLM main + #63114 only | This repo |
|---|---|---|---|
| joey-1, A100, 13.2k-token state, 11 questions | 8.4 | 2.5 | **28.7** (3.4x) |
| joey-1, A100, 4k-token state, 44 questions | 27.1 | 12.0 | **84.0** (3.1x) |
| joey-1, A100, 13.2k state asked in 3 requests | 4.7 | n/a | **26.9** (5.7x) |
| elephant-1, A100, 13.2k state, 11 questions | 5.6 | | **7.4** (1.3x) |
| elephant-1, A100, 13.2k state, 44 questions | 12.6 | | **16.2** (1.3x) |
| elephant-1, A100, 13.2k state asked in 3 requests | 2.2 | | **6.2** (2.8x) |
| joey-1, RTX PRO 6000, 4k state, 44 questions | 53.2 | 133.3 (Triton) | **149.5** (2.8x) |
| joey-1, RTX PRO 6000, 13.2k state, 11 questions, 4 concurrent | 12.1 | 38.4 (Triton) | **49.2** (4.1x) |
| joey-1, H100, 13.2k state, 11 / 44 questions | | 42.8 / 93.0 (Triton) | **92.5 / 183.6** (Triton + FA3) |
| joey-1, B200, 13.2k state, 11 / 44 questions | | 82.0 / 165.0 (Triton) | **155.3 / 275.5** (FlashAttention-4) |

H100 and B200 rows use a public dataset and compare against vLLM's Triton path (OneJev was not run there).
Answers match OneJev to BF16 noise: 0.3-0.9% of decisions change their top option and the mean probability
difference is 0.0023-0.0044 (two BF16 runs of one server already differ by about 0.003).

## Quick start

### Docker

```bash
docker run --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/wolfgang9999/decision-models-vllm baselabs/elephant-1 --served-model-name elephant-1
```

No tuning flags are needed: for these models the image defaults to text-only serving (pass `--limit-mm-per-prompt`
to accept images), BF16, a 16,384-token context, prefix caching, the
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

| GPU | Prefix cache (one state prefill shared by all its questions) | PrefixLM attention picked by default | Status |
|---|---|---|---|
| A100, A30 (SM 8.0) | Triton GDN checkpoints (#65186) | Triton, prefill on FlashAttention-2 (#65222) | measured |
| A10, A40, L4, L40S, RTX 30/40 (SM 8.6/8.9) | same | same | expected, not measured |
| H100, H200 (SM 9.0) | FlashInfer GDN prefill (in vLLM) | Triton, prefill on FlashAttention-3 (this repo) | measured |
| B200, GB200 (SM 10.x) | FlashInfer GDN prefill (in vLLM) | FlashAttention-4, native mask | measured |
| RTX PRO 6000, RTX 5090 (SM 12.x) | FlashInfer GDN prefill (in vLLM) | FlashInfer, native mask (#46558) | measured |

Each default is the fastest path measured on that GPU (tables below); `--attention-backend` overrides it.

## Measurements

All numbers are peak decisions/s over closed-loop concurrency 1-64 (the concurrency of the peak in brackets) unless a
table says otherwise. Every request carries a unique nonce, so no state is reused across requests; only the questions
of one request (or the 3 requests of the split scenario) share it. Inputs are real multilingual customer-support
states (conversation + retrieved documents) with their own 11 questions; 44 questions = the 11 plus 33 copies.
"This repo" rows on A100 were measured on the serving image that this repo grew from (vLLM PR #63114 head + the same
Triton checkpoint and FlashAttention-2 changes); the release build on the pinned nightly is being measured now.

### joey-1 on A100 80GB

| State tokens | Questions | OneJev | vLLM main + #63114 | This repo |
|---|---|---|---|---|
| 500 | 1 | 8.8 (c8) | 75.4 (c64) | **80.2** (c64) |
| 500 | 11 | 54.3 (c8) | 78.7 (c64) | **90.4** (c32) |
| 500 | 44 | 70.3 (c16) | 79.7 (c4) | **94.5** (c16) |
| 2,000 | 11 | 33.5 (c16) | 23.6 (c64) | **84.6** (c64) |
| 2,000 | 44 | 42.3 (c4) | 25.0 (c4) | **103.2** (c4) |
| 4,000 | 11 | 21.1 (c4) | 11.3 (c4) | **59.2** (c64) |
| 4,000 | 44 | 27.1 (c16) | 12.0 (c8) | **84.0** (c8) |
| 13,200 | 1 | 1.6 (c4) | 2.5 (c16) | **3.7** (c16) |
| 13,200 | 11 | 8.4 (c16) | 2.5 (c16) | **28.7** (c16) |
| 13,200 | 44 | 11.5 (c16) | 1.8 (c1) | **59.0** (c16) |

Same state asked in 3 requests (4, 4 and 3 questions), peak decisions/s and median latency of request 1 -> request 3:

| State tokens | OneJev | This repo |
|---|---|---|
| 500 | 28.2 (2.1 s -> 2.1 s) | **80.5** (2.8 s -> 2.7 s) |
| 4,000 | 13.5 (2.2 s -> 2.1 s) | **55.2** (6.5 s -> 2.9 s) |
| 13,200 | 4.7 (13.8 s -> 11.0 s) | **26.9** (18.7 s -> 3.3 s) |

Without this repo's prefix caching fix, vLLM switches the cache off on A100, so every question re-reads the state:
vLLM main falls behind OneJev from 2,000 tokens on.

### elephant-1 on A100 80GB

13.2k-token states, OneJev and this repo driven at the same time on the same GPU pool (decisions/s, median latency):

| Workload | Concurrency | OneJev | This repo |
|---|---|---|---|
| 11 questions | 1 | 3.22 (3.7 s) | **3.74** (3.0 s) |
| 11 questions | 4 | 5.58 (7.9 s) | **7.21** (6.1 s) |
| 11 questions | 8-16 | 5.60 | **7.41** |
| 44 questions | 4 | 12.63 (13.0 s) | **16.09** (10.0 s) |
| 44 questions | 8 | 12.60 | **16.21** |
| 3 requests on one state (4/4/3) | 8 | 2.22 (13.1 s -> 11.0 s) | **6.21** (8.8 s -> 1.7 s) |
| 1 question | 16 | 0.63 | **0.95** |

By state size (this repo, peak decisions/s): 500 tokens 21.6 (11 q) / 22.5 (44 q); 2,000 tokens 20.4 / 25.8;
4,000 tokens 14.7 / 21.0; 13,200 tokens 7.5 / 15.7.

### joey-1 on RTX PRO 6000 Blackwell (SM 12.0)

Same workload as the A100 tables (peak decisions/s, concurrency in brackets). "This repo" picks FlashInfer here.

| State tokens | Questions | OneJev | Triton attention | This repo (FlashInfer) |
|---|---|---|---|---|
| 500 | 11 | 106.1 (c16) | 158.0 (c32) | **160.6** (c32) |
| 2,000 | 11 | 64.8 (c64) | 141.4 (c8) | **150.4** (c8) |
| 2,000 | 44 | 80.3 (c8) | 176.7 (c16) | **186.6** (c8) |
| 4,000 | 11 | 41.1 (c16) | 93.9 (c4) | **104.6** (c32) |
| 4,000 | 44 | 53.2 (c4) | 133.3 (c32) | **149.5** (c32) |
| 13,200 | 11 | 12.1 (c4 only) | 39.3 (c64) | **50.7** (c64) |
| 13,200 | 44 | | 82.7 (c64) | **106.9** (c16) |
| 4,000, same state in 3 requests | 4/4/3 | 24.3 (c4) | 87.8 (c64) | **97.8** (c64) |

Kernel level (one 2,048-token state + question, head size 256): FlashInfer 0.46 ms vs Triton 0.84 ms per state
prefill, outputs equal within 2e-2.

The bare command (`vllm serve MODEL --served-model-name joey-1`, nothing else) matched the tuned flags here at 4
concurrent requests: 49.5 vs 49.2 decisions/s at 13,200 tokens, 104.4 vs 104.4 at 4,000.

### joey-1 on H100 80GB and B200 (Modal)

Public inputs built from [BEV Decision Mix](https://huggingface.co/datasets/avbiswas/bev-decision) (test split; 64
cases of concatenated states with 11 of their own questions, `bench/build_bev_inputs.py`), concurrency 4/16/32. Every
path answers byte-identical requests; agreement is against the Triton path (T).

| GPU | State | Questions | Triton (T) | Triton + FA3 | FlashInfer | FlashAttention-4 | Bare `vllm serve` |
|---|---|---|---|---|---|---|---|
| H100 | 4,000 | 11 | 201.2 | **269.4** | 243.5 | 269.1 | 248.1 |
| H100 | 4,000 | 44 | 331.4 | 421.1 | 398.0 | 408.5 | **444.7** |
| H100 | 13,200 | 11 | 42.8 | **92.5** | 67.9 | 89.1 | 67.5 |
| H100 | 13,200 | 44 | 93.0 | **183.6** | 144.4 | 180.3 | 147.5 |
| H100 | 13,200, 3 requests | 4/4/3 | 38.2 | **67.4** | 51.3 | 63.7 | 54.4 |
| B200 | 4,000 | 11 | 342.9 | n/a | 392.0 | **450.4** | 411.2 |
| B200 | 4,000 | 44 | 541.6 | n/a | 575.3 | 619.6 | **726.6** |
| B200 | 13,200 | 11 | 82.0 | n/a | 97.2 | **155.3** | 97.7 |
| B200 | 13,200 | 44 | 165.0 | n/a | 202.3 | **275.5** | 216.9 |
| B200 | 13,200, 3 requests | 4/4/3 | 58.0 | n/a | 70.6 | **84.6** | 69.6 |

Agreement with Triton: 0.30-0.41% of decisions change their top option, mean probability difference 0.0019, for
every path on both GPUs. The explicit paths ran with `--max-num-seqs 64`; the bare command keeps vLLM's 128, which is
why it leads at 44 questions (44 sequences per request). The bare runs above predate the per-GPU defaults (they picked
FlashInfer everywhere); the image now picks FA3 on H100 and FA4 on B200, the fastest paths here.

### Answer agreement with OneJev

| Setup | Decisions compared | Top option changed | Mean abs probability difference |
|---|---|---|---|
| joey-1, A100, vLLM main + #63114 | 24,967 | 0.54% | 0.0023 |
| joey-1, A100, this repo | 46,522 | 0.51% | 0.0024 |
| joey-1, RTX PRO 6000, Triton | 72,128 | 0.43% | 0.0023 |
| joey-1, RTX PRO 6000, this repo (FlashInfer) | 78,057 | 0.47% | 0.0023 |
| joey-1, RTX PRO 6000, bare `vllm serve MODEL --served-model-name joey-1` | 4,554 | 0.44% | 0.0023 |
| elephant-1, A100, this repo | 704 | 0.7-0.9% | 0.0043 |

### Open measurements

- elephant-1 on the current build; the bare command with the new per-GPU defaults on H100 and B200 (the FA3 and
  FA4 paths themselves were measured with explicit flags).

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
- **Attention.** #63114 applies the mask with vLLM's Triton kernel on every GPU. This repo sends those prefills to
  FlashAttention-2 on SM 8.x (#65222) and FlashAttention-3 on SM 9.x (one non-causal call per block), uses
  FlashAttention-4's native mask on SM 10.x and FlashInfer's (#46558) on SM 11.x-12.x. Measured 1.2-2.2x faster than
  the Triton kernel at long states with the same answers.

### Pull requests included

| PR | What | State |
|---|---|---|
| [vllm#63114](https://github.com/vllm-project/vllm/pull/63114) | model, PrefixLM scheduler, `/v1/systemone` | open |
| [vllm#65186](https://github.com/vllm-project/vllm/pull/65186) | Triton GDN prefill checkpoints (prefix cache on SM 8.x) | open |
| [vllm#65222](https://github.com/vllm-project/vllm/pull/65222) | PrefixLM prefill on FlashAttention-2 (SM 8.x) and FlashAttention-3 (SM 9.x) inside the Triton backend | open |
| [vllm#46558](https://github.com/vllm-project/vllm/pull/46558) | FlashInfer native mm-prefix attention | merged |

`overlay/vllm/v1/attention/backends/triton_attn.py` is #65222's file as submitted. The rest of the overlay is #63114
and #65186 plus the changes it takes to combine them on current vLLM main and serve every GPU well. Those changes edit
#63114's code, so they are proposed there rather than in a separate PR
([comment on #63114](https://github.com/vllm-project/vllm/pull/63114#issuecomment-6825498326)):

- #63114 still checks `use_v2_model_runner`, which main removed with Model Runner V1 (#63373); the check is dropped.
- #63114 keeps prefix caching when the Triton GDN backend exports checkpoints (#65186).
- #63114 accepts FlashInfer and FlashAttention (FA4) besides Triton, flags the model as a PrefixLM so they apply its
  mask, and picks the backend per GPU (Triton on SM 8.x/9.x, FA4 on SM 10.x, FlashInfer on SM 11.x/12.x).
- vLLM's rule that forces unchunked multimodal input for PrefixLM models is skipped for this text-only model, whose
  hybrid prefix cache needs chunked scheduling.
- The model is served text-only unless `--limit-mm-per-prompt` is given, so a plain `vllm serve` does not load or
  profile the vision tower.
- The prefill step budget defaults to at least 32,768 tokens.

`overlay/manifest.json` lists every file and the SHA256 of the pinned vLLM file it replaces; `install_overlay.py`
refuses to apply on any other build.

## Notes and limits

- Text states only (`--language-model-only`); image states need OneJev.
- `confidence` in answers is computed differently from OneJev; the option probabilities agree.
- Check these models through decisions, not chat: joey-1's chat output is noise in every runtime.
- Requests are limited to 16,384 tokens, like the model cards.
- vLLM's FlashAttention-2/-3 builds have no SM 12.x code, so the RTX PRO 6000 class uses FlashInfer.

## Reproducing the numbers

`bench/grid_bench.py` drives a server over a grid of state sizes (about 500, 2,000, 4,000 and 13,200 tokens),
questions per request (1, 11, 44) and concurrency (1 to 64), closed loop, 20 s per window, with a unique nonce per
request so nothing is reused across requests; then the same state asked in three requests (4, 4 and 3 questions).
`bench/analyze.py` builds the tables and compares every answer with OneJev's on byte-identical requests.

## License

Apache-2.0, as vLLM. Files under `overlay/` are modified vLLM sources.
