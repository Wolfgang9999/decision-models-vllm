# Python serving entrypoint for vLLM PR #63114; native kernels from its exact base.
# Two stages: the wheel upgrade replaces torch/CUDA, and layers only add, so the
# superseded copies would stay in the image. The final stage flattens the result.
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34 AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /uvx /usr/local/bin/
USER root
ENV UV_HTTP_TIMEOUT=180 UV_LINK_MODE=copy
ARG VLLM_PR_SHA=af2ee49269522fc99a2ad4bba1dd9b6066ba4c2a
ARG VLLM_NATIVE_SHA=c9af9acc6ac22f6351e214eb0d7d9a7a26dfa870
LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="af2ee49269522fc99a2ad4bba1dd9b6066ba4c2a"
# Upgrade dependencies together with the native wheel, preserving its torch ABI.
RUN uv --version && uv pip install --system --no-cache \
    "https://wheels.vllm.ai/${VLLM_NATIVE_SHA}/vllm-0.33.0rc1.dev387%2Bgc9af9acc6-cp38-abi3-manylinux_2_28_x86_64.whl"
COPY install_pr.py /opt/decision-build/install_pr.py
COPY overlay /opt/decision-build/overlay
RUN python3 /opt/decision-build/install_pr.py
# Patch Ubuntu OpenSSL and Brotli (known CVEs in the base image) without
# changing the CUDA/PyTorch dependency stack.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade --no-install-recommends \
       openssl libssl3t64 libbrotli1 \
    && dpkg --compare-versions "$(dpkg-query -W -f='${Version}' openssl)" ge 3.0.13-0ubuntu3.15 \
    && dpkg --compare-versions "$(dpkg-query -W -f='${Version}' libssl3t64)" ge 3.0.13-0ubuntu3.15 \
    && dpkg --compare-versions "$(dpkg-query -W -f='${Version}' libbrotli1)" ge 1.1.0-2ubuntu0.1 \
    && dpkg-query -W openssl libssl3t64 libbrotli1 \
    && rm -rf /var/lib/apt/lists/*
# Drop build-only files before flattening. The overlay and its tests stay out of the image.
RUN rm -rf /opt/uv/cache /root/.cache /tmp/* /var/tmp/* /var/lib/apt/lists/* /var/cache/apt/* \
           /opt/decision-build/overlay
COPY serve.sh /opt/decision-build/serve.sh

FROM scratch
COPY --from=build / /
# Config of vllm/vllm-openai:v0.32.0 (a flattened image keeps none of it).
ENV PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib \
    NVARCH=x86_64 CUDA_VERSION=13.0.2 NV_CUDA_CUDART_VERSION=13.0.96-1 CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive CRIU_CUDA_PLUGIN_DIR=/usr/local/lib/criu \
    UV_HTTP_TIMEOUT=500 UV_INDEX_STRATEGY=unsafe-best-match UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python UV_CACHE_DIR=/opt/uv/cache UV_COMPILE_BYTECODE=1 \
    UV_OVERRIDE=/etc/uv-overrides.txt VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
    VLLM_USAGE_SOURCE=production-docker-image
# Architectures for runtime JIT compiles. Tested on A100 (8.0) only; pass e.g.
# --build-arg TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9" for A10/L4/L40S.
ARG TORCH_CUDA_ARCH_LIST=8.0
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="af2ee49269522fc99a2ad4bba1dd9b6066ba4c2a"
WORKDIR /vllm-workspace
# Python `vllm serve` frontend; the PR's Rust frontend changes are not used.
ENTRYPOINT ["bash", "/opt/decision-build/serve.sh"]
CMD []
