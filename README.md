# Base Labs decision models on vLLM (A100 bridge)

A Docker image that serves [elephant-1](https://huggingface.co/baselabs/elephant-1) and
[joey-1](https://huggingface.co/baselabs/joey-1) with vLLM, using the same `POST /v1/systemone` API as the
OneJev runtime the model cards install. It batches requests and caches each state's prefix, so on one
A100 it serves more decisions per second than OneJev once several requests are in flight.

> **Status:** the recipe is the one behind our production A100 image, but this repackaged form has not
> been rebuilt and smoke-tested yet. Comparisons against current vLLM main and FlashInfer are in progress.

This is a stopgap. Everything here is on its way into vLLM itself (see [Upstream status](#upstream-status));
once those PRs merge, `vllm serve baselabs/elephant-1` will do the same and this repo can be retired.

## Quick start

```bash
docker build -t decision-vllm .

# elephant-1 (9B)
docker run --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -v ~/.cache/decision-models:/models \
  -e MODEL=baselabs/elephant-1 decision-vllm

# joey-1 (2B); larger prefill steps help a small model
docker run --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -v ~/.cache/decision-models:/models \
  -e MODEL=baselabs/joey-1 -e MAX_NUM_BATCHED_TOKENS=65536 decision-vllm

python decide.py http://localhost:8000 elephant-1
```

The request and response follow the model cards: a `state`, and `questions` of type `noul`, `choice` or
`score`. Only the URL changes when moving from OneJev's server. The in-process `qev.DecisionEngine`
Python API is not available; call the HTTP endpoint instead.

Settings (environment variables):

| Variable | Default | Notes |
|---|---|---|
| `MODEL` | `baselabs/elephant-1` | Hugging Face repo |
| `REVISION` | `main` | Pin a commit for reproducible results |
| `MODEL_NAME` | repo name | The `model` field clients send |
| `PORT` | `8000` | |
| `MAX_NUM_BATCHED_TOKENS` | `32768` | Tokens per prefill step; 65536 measured +4% on joey-1 |
| `MAX_NUM_SEQS` | `64` | |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Lower it if the GPU is shared |
| `VLLM_PREFIX_LM_FA2` | `1` | `0` turns the FlashAttention-2 prefill off |

Extra arguments after the image name go to `vllm serve`.

## What it changes

The image starts from vLLM [PR #63114](https://github.com/vllm-project/vllm/pull/63114), which adds these models
(their block-bidirectional mask, a PrefixLM scheduler and `/v1/systemone`), and adds two A100 fixes on top:

1. **Prefix caching on A100.** #63114 caches a state only when the linear-attention (GDN) prefill can export
   its recurrent state mid-prompt, which needs FlashInfer's kernel on Hopper or newer. The overlay makes
   the Triton kernel write fp32 checkpoints every 64-token chunk, so all questions about one state reuse
   one prefill. Checkpoints are bit-identical to a prefill cut at that point
   (`tests/test_fla_gdn_prefill_checkpoints.py`).
2. **FlashAttention-2 for PrefixLM prefill.** A query in bidirectional range `[start, end]` attends to keys
   `0..end`, which is plain non-causal attention, so pure prefill batches go to FA2 instead of the Triton
   attention kernel. Attention fell from 37% to 18% of GPU time.

`install_pr.py` downloads the pinned #63114 source (checksum-verified), installs it over the matching
nightly wheel, and refuses to apply an overlay file unless the file it replaces has the expected SHA256.
`/opt/decision-build/provenance.json` in the image records every hash.

## Results

One A100 80GB, BF16, cold prefix cache (a unique nonce per state), 64 long-context requests with 11
questions each, OneJev and vLLM run at the same time on the same node pool:

| Concurrent requests | OneJev | This image | |
|---|---:|---:|---:|
| 1 | 3.18 | 3.27 | +3% |
| 4 | 5.64 | 7.08 | +26% |
| 8 | 5.53 | 7.22 | +31% |
| 16 | 5.53 | 7.24 | +31% |

decisions/s, elephant-1. joey-1 on the same image reaches 26.6 decisions/s from 16 concurrent
requests (27.7 with 65536-token steps); OneJev was not benchmarked for joey-1.

Agreement with OneJev on elephant-1: 5 to 6 decisions out of 704 flip (under 1%), mean absolute
probability difference 0.004, largest 0.054. The FA2 path changed 0 of 880 decisions against vLLM's Triton
kernel. joey-1 probabilities match OneJev to about two decimals.

The gain comes from batching and prefix reuse. A single client sending one request at a time sees about
the same speed as OneJev.

## Known differences and limits

- **Tested on A100 only.** Builds default to `TORCH_CUDA_ARCH_LIST=8.0`; pass
  `--build-arg TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9"` for A10, L4 or L40S (untested). On H100, #63114
  already caches prefixes through FlashInfer; this image has not been measured there, and
  `VLLM_PREFIX_LM_FA2=0` restores vLLM's own attention path.
- **Text only.** `--language-model-only` is set; image states are not served.
- **mammoth-1** should load the same way but has not been tried.
- **`confidence`** differs from OneJev (near 0 on vLLM, up to about 0.35 on OneJev) although the
  probabilities agree. Do not port thresholds on that field.
- **Check models through decisions, not chat.** joey-1's `/v1/chat/completions` output is noise in every
  runtime, including plain transformers; that is the model, not this image.
- Requests are limited to 16,384 tokens, like the model cards.

## Upstream status

| PR | Covers | |
|---|---|---|
| [#63114](https://github.com/vllm-project/vllm/pull/63114) | Model, scheduler, `/v1/systemone` | open |
| [#65186](https://github.com/vllm-project/vllm/pull/65186) | Triton GDN prefill checkpoints (fix 1, independent implementation) | open |
| [#65222](https://github.com/vllm-project/vllm/pull/65222) | FA2 PrefixLM prefill on SM 8.x (fix 2) | open |

When all three are in a vLLM release, run that release with the `vllm serve` flags in `serve.sh` and drop
this image (the flags may change while #63114 is in review).

## License

Apache-2.0, as vLLM. The files under `overlay/` are modified vLLM sources.
