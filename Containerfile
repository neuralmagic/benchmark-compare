FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt/benchmark

RUN git clone https://github.com/vllm-project/vllm.git vllm && \
    cd vllm && \
    git checkout benchmark-output && \
    uv venv venv-vllm-src --python 3.12 && \
    . venv-vllm-src/bin/activate && \
    VLLM_USE_PRECOMPILED=1 uv pip install -e . && \
    uv pip install pandas datasets numpy && \
    cd ..

COPY benchmark_1000_in_100_out.sh .
RUN chmod +x benchmark_1000_in_100_out.sh

VOLUME ["/host"]

# warn if HF_TOKEN missing, activate venv, run benchmark, then copy results.json to host
ENTRYPOINT ["bash", "-c", "\
    if [ -z \"$HF_TOKEN\" ]; then \
      echo \"WARNING: HF_TOKEN is not defined\" >&2; \
    fi && \
    source vllm/venv-vllm-src/bin/activate && \
    bash /opt/benchmark/benchmark_1000_in_100_out.sh && \
    if [ -f /opt/benchmark/results.json ]; then \
      cp /opt/benchmark/results.json /host/results.json && \
      echo \"Copied results.json to host\"; \
    else \
      echo \"ERROR: results.json not found after benchmark\" >&2; \
      exit 1; \
    fi\
"]
