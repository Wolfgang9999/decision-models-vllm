# Base Labs decision models (joey-1, elephant-1) on vLLM: the pinned vLLM nightly image plus a Python-only overlay of
# open PRs (see README). Nothing is compiled; the overlay layer is a few hundred KB. The entrypoint stays the base
# image's `vllm serve`, so this is a drop-in replacement for vllm/vllm-openai (Helm charts, `docker run IMAGE MODEL`).
ARG VLLM_COMMIT=be47189f37f6162991366ff6b91539f6e9d1f432
FROM vllm/vllm-openai:nightly-${VLLM_COMMIT}
COPY overlay /opt/decision/overlay
COPY install_overlay.py /opt/decision/install_overlay.py
RUN python3 /opt/decision/install_overlay.py
LABEL org.opencontainers.image.source="https://github.com/Wolfgang9999/decision-models-vllm" \
      org.opencontainers.image.description="vLLM nightly + open PRs to serve Base Labs joey-1 and elephant-1 on any GPU" \
      org.opencontainers.image.licenses="Apache-2.0"
