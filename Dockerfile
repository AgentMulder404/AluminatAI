# AluminatAI GPU monitoring agent
#
# The NVML library (libnvidia-ml.so) is provided by the NVIDIA container
# runtime at run time — it is not baked into this image.
#
# Run with:
#   docker run --rm --runtime=nvidia --pid=host \
#     -e ALUMINATAI_API_KEY=alum_... \
#     ghcr.io/agentmulder404/aluminatai-agent:latest

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# python3-pip brings Python 3.10 (Ubuntu 22.04 default) — satisfies >=3.8
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Dedicated non-root user
RUN groupadd -r aluminatai && \
    useradd -r -g aluminatai -m -d /app aluminatai

WORKDIR /app

# Install the agent from PyPI — no source copy needed
RUN pip3 install --no-cache-dir aluminatiai

# WAL directory for metric buffering during API outages
RUN mkdir -p /app/data/wal && chown -R aluminatai:aluminatai /app

USER aluminatai

HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD aluminatiai --help > /dev/null || exit 1

ENTRYPOINT ["aluminatiai"]
