FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential python3 python3-dev python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /build
COPY pyproject.toml setup.py MANIFEST.in ./
COPY src ./src
RUN python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
        python3 -m pip wheel . --no-deps --wheel-dir /wheels

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3 python3-venv ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install pytest

COPY --from=builder /wheels /tmp/wheels
RUN python3 -m pip install --no-deps /tmp/wheels/moe_topk-*.whl \
    && rm -rf /tmp/wheels
COPY tests /opt/tests
WORKDIR /tmp
CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]
