# Combined end to end inference and benchmark

This go binary will run both the benchmark script and start the inference framework for an end to end automated run.

### Run

All dependant repos and builds are managed by the binary. A log directory will contain the inference
framework logs being benchmarked along with the benchmark script logs. The first run will need the `HF_TOKEN`
in the env to download the appropriate tokenizer.

```bash
go build
export HF_TOKEN=<token> # or inline with the binary
./benchmark-go --port 8000 --model meta-llama/Llama-3.1-8B-Instruct --cuda-device 0
```

The benchmark results are written to `benchmark-compare/results.json`.
