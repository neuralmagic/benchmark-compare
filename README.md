## Benchmarking Comparison

## Launch `vllm`

### Install

```bash
uv venv venv-vllm --python 3.12
source venv-vllm/bin/activate
uv pip install vllm==0.8.2
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
python3 -m sglang.launch_server --model-path $MODEL  --host 0.0.0.0 --port 8000 --enable-mixed-chunk
```

> When inspecting logs, make sure cached-tokens is small!

## Benchmark

### Install
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout benchmark-output
uv venv venv-vllm-src --python 3.12
source venv-vllm-src /bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e .
uv pip install pandas datasets
cd ..
```

### Run Benchmark

- `vllm`

```bash
FRAMEWORK=vllm bash ./benchmark_1000_in_100_out.sh
python3 convert_to_csv.py --input-path vllm-results.json --output-path vllm-results.csv
```

- `sgl`
```bash
export MODEL=meta-llama/Llama-3.1-8B-Instruct
FRAMEWORK=sgl bash ./benchmark_1000_in_100_out.sh
python3 convert_to_csv.py --input-path sgl-results.json --output-path sgl-results.csv
```

### Pull Into Local

```bash
scp rshaw@beaker:~/benchmark_compare/sgl-results.csv ~/Desktop/
scp rshaw@beaker:~/benchmark_compare/vllm-results.csv ~/Desktop/
```
