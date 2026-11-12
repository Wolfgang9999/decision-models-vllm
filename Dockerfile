# Base Labs decision models (joey-1, elephant-1) on vLLM: the pinned vLLM nightly image plus a Python-only overlay of
# open PRs (see README). Nothing is compiled; the overlay layer is a few hundred KB.
ARG VLLM_COMMIT=be47189f37f6162991366ff6b91539f6e9d1f432
FROM vllm/vllm-openai:nightly-${VLLM_COMMIT}
COPY overlay /opt/decision/overlay
COPY install_overlay.py /opt/decision/install_overlay.py
RUN python3 /opt/decision/install_overlay.py
COPY decision-serve /usr/local/bin/decision-serve
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENTRYPOINT ["decision-serve"]
CMD ["baselabs/elephant-1"]
