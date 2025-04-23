# Combined end to end inference and benchmark

This runs both the benchmark script and start the inference framework for an end to end automated run.

### Run

All dependant repos and builds are managed by the script. A log directory will contain the inference
framework logs being benchmarked along with the benchmark script logs. The first run will need the `HF_TOKEN`
in the env to download the appropriate tokenizer.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install requests
export HF_TOKEN=<token> # or inline with the binary
python ./benchmark-e2e --port 8000 --model meta-llama/Llama-3.1-8B-Instruct --cuda-device 0
```

The benchmark results are written to `benchmark-compare/results.json`.
