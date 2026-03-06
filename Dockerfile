# AluminatAI GPU monitoring agent
#
# The NVML library (libnvidia-ml.so) is provided by the NVIDIA container
# runtime at run time — it is not baked into this image.
#
# Run with:
#   docker run --rm --runtime=nvidia --pid=host \
#     -e ALUMINATAI_API_KEY=alum_... \
#     aluminatai/agent:latest
#
# Or via docker-compose:  deploy/docker-compose.yml

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 (Ubuntu 22.04 ships 3.10 by default)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-pip \
        python3.11-venv && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3.11 -m pip install --upgrade pip --quiet

# Dedicated non-root user matching the systemd service
RUN groupadd -r aluminatai && \
    useradd -r -g aluminatai -m -d /app aluminatai

WORKDIR /app

# Install Python dependencies first — better layer caching when code changes
COPY pyproject.toml .
COPY agent/ agent/

RUN python3.11 -m pip install --no-cache-dir -e ".[kubernetes,prometheus]"

# WAL directory for metric buffering during API outages
RUN mkdir -p /app/data/wal && chown -R aluminatai:aluminatai /app

USER aluminatai

# Health: check that the CLI entry point is present
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD aluminatai-agent --help > /dev/null || exit 1

ENTRYPOINT ["aluminatai-agent"]
