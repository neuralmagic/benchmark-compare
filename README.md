## Benchmarking Comparison

## Launch `vllm`

### Install

```bash
uv venv venv-vllm --python 3.12
source venv-vllm/bin/activate
uv pip install vllm==0.8.3
```

### Launch

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
vllm serve $MODEL --disable-log-requests
```

> When inspecting logs, make sure prefix cache hit rate is low!

## Launch `sglang`

### Install

```bash
uv venv venv-sgl --python 3.12
source venv-sgl/bin/activate
uv pip install "sglang[all]==0.4.4.post1" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

### Launch Server

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
python3 -m sglang.launch_server --model-path $MODEL  --host 0.0.0.0 --port 8000 # --enable-mixed-chunk --enable-torch-compile
```

> When inspecting logs, make sure cached-tokens is small!

## Benchmark

### Install
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout benchmark-output
uv venv venv-vllm-src --python 3.12
source venv-vllm-src/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e .
uv pip install pandas datasets
cd ..
```

### Run Benchmark

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct FRAMEWORK=vllm bash ./benchmark_1000_in_100_out.sh
MODEL=meta-llama/Llama-3.1-8B-Instruct FRAMEWORK=sgl bash ./benchmark_1000_in_100_out.sh
python3 convert_to_csv.py --input-path results.json --output-path results.csv
```

### Pull Into Local

```bash
scp rshaw@beaker:~/benchmark-compare/results.csv ~/Desktop/
```

### Running in a Container

Build the container image using the Containerfile located in the root of the directory. This example uses quay registry and podman runtime. Swap these with the container tools of your choice.

Run the container build from the root of this repo.

```
QUAY_ORG=<quay_account_here>
podman build -f Containerfile -t quay.io/$QUAY_ORG/vllm-benchmark:latest
```

- Start the inference engine you are testing as shown above, e.g. SGLang or vLLM. Update the endpoint in the run command below if not using the default HOST:PORT values.

```
MODEL=meta-llama/Llama-3.1-8B-Instruct
podman run --rm -it \
    --network host \
    -e MODEL="${MODEL}" \
    -e FRAMEWORK=vllm \
    -e HF_TOKEN=<INSERT_HF_TOKEN> \
    -e PORT=8000 \
    -e HOST=127.0.0.1 \
    -v "$(pwd)":/host:Z \
    -w /opt/benchmark \
    quay.io/$QUAY_ORG/vllm-benchmark:latest
```

The json benchmark results will be copied to the host machine in the same directory the container was run from.
