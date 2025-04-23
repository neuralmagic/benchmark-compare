#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser(description="Run vLLM & SGLang benchmarks")
    p.add_argument("--port", type=int, default=8080, help="Port for both servers")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model identifier")
    p.add_argument("--cuda-device", default=os.getenv("CUDA_VISIBLE_DEVICES", ""), help="CUDA_VISIBLE_DEVICES override")
    p.add_argument("--async", action="store_true", help="(ignored)")
    return p.parse_args()


def run_cmd(cmd, cwd=None, logfile=None, logger=None):
    if logger:
        logger.info(f"▶ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, stdout=logfile or sys.stdout, stderr=logfile or sys.stderr)
    proc.check_returncode()


def wait_for_server(host, port, logger, timeout_s=120, interval_s=2):
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1)
            if "data" in r.text:
                return
        except Exception:
            pass
        time.sleep(interval_s)
    raise TimeoutError(f"Timeout waiting for server at {url}")


def ensure_uv(logger):
    if shutil.which("uv") is None:
        logger.info("`uv` not found; installing via astral.sh...")
        run_cmd(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], logger=logger)


def global_setup(root_dir, logger):
    to_remove = [
        root_dir / "benchmark-compare",
        root_dir / "venv-vllm",
        root_dir / "venv-vllm-src",
        root_dir / "venv-sgl",
    ]
    for p in to_remove:
        logger.info(f"Removing {p}")
        shutil.rmtree(p, ignore_errors=True)

    ensure_uv(logger)

    # clone benchmark-compare
    run_cmd(["git", "clone",
             "https://github.com/neuralmagic/benchmark-compare.git",
             str(root_dir / "benchmark-compare")],
            logger=logger)
    # clone vllm@benchmark-output
    vllm_dir = root_dir / "benchmark-compare" / "vllm"
    run_cmd(["git", "clone",
             "https://github.com/vllm-project/vllm.git",
             str(vllm_dir)],
            logger=logger)
    run_cmd(["git", "-C", str(vllm_dir), "checkout", "benchmark-output"],
            logger=logger)


class BaseJob:
    def __init__(self, name, cfg, root_dir, logs_dir):
        self.name = name
        self.port = cfg.port
        self.model = cfg.model
        self.cuda_dev = cfg.cuda_device
        self.root_dir = root_dir
        self.logpath = logs_dir / f"{name}.log"
        self.logfile = open(self.logpath, "a")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(self.logfile))
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def run(self):
        raise NotImplementedError


def run_jobs(jobs, logger):
    for job in jobs:
        logger.info(f"▶ Running {job.name}")
        try:
            job.run()
            logger.info(f"✓ {job.name} completed")
        except Exception as e:
            logger.error(f"✗ {job.name} failed: {e}")
            return
        if job.name == "vllm":
            logger.info("Killing vllm serve process group")
            subprocess.run(["pkill", "-f", "vllm serve"], check=False)


class VLLMJob(BaseJob):
    def run(self):
        self.logger.info("=== vllm benchmark start ===")

        # create venv & install vllm via uv
        run_cmd(["uv", "venv", "venv-vllm", "--python", "3.12"],
                cwd=self.root_dir, logfile=self.logfile, logger=self.logger)
        run_cmd(["bash", "-c", "source venv-vllm/bin/activate && uv pip install vllm==0.8.3"],
                cwd=self.root_dir, logfile=self.logfile, logger=self.logger)
        self.logger.info("vllm package installed in venv-vllm")

        # launch vllm serve
        env = os.environ.copy()
        if self.cuda_dev:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_dev
        serve_cmd = ["vllm", "serve", self.model, "--disable-log-requests", "--port", str(self.port)]
        self.logger.info(f"▶ source venv-vllm/bin/activate && {' '.join(serve_cmd)}")
        proc = subprocess.Popen(
            "bash -c 'source venv-vllm/bin/activate && " +
            " ".join(serve_cmd) + "'", cwd=self.root_dir,
            stdout=self.logfile, stderr=self.logfile,
            env=env, preexec_fn=os.setsid, shell=True
        )
        self.logger.info(f"Started vllm serve (pid={proc.pid})")

        # wait for ready
        self.logger.info("Waiting for vllm to load…")
        wait_for_server("localhost", self.port, self.logger)
        self.logger.info(f"vllm inference server ready at http://localhost:{self.port}/v1/models")

        # 4) setup vllm-src & deps via uv with precompiled
        vllm_src = self.root_dir / "benchmark-compare" / "vllm"
        self.logger.info(f"Creating venv-vllm-src in {vllm_src}")
        run_cmd(["uv", "venv", "venv-vllm-src", "--python", "3.12"],
                cwd=vllm_src, logfile=self.logfile, logger=self.logger)
        deps_cmd = (
            "source venv-vllm-src/bin/activate && "
            "export VLLM_USE_PRECOMPILED=1 && "
            "uv pip install -e . && "
            "uv pip install numpy pandas datasets"
        )
        self.logger.info(f"▶ {deps_cmd}")
        run_cmd(["bash", "-c", deps_cmd],
                cwd=vllm_src, logfile=self.logfile, logger=self.logger)
        self.logger.info("vllm-src dependencies installed (precompiled)")

        # run benchmark script
        bench_dir = self.root_dir / "benchmark-compare"
        bench_log = self.root_dir / "logs" / "bench-vllm.log"
        with open(bench_log, "a") as bf:
            self.logger.info(">>> Starting vllm benchmark; output → bench-vllm.log")
            bench_cmd = (
                "source vllm/venv-vllm-src/bin/activate && "
                f"VLLM_USE_PRECOMPILED=1 MODEL={self.model} FRAMEWORK=vllm "
                "bash ./benchmark_1000_in_100_out.sh"
            )
            self.logger.info(f"▶ {bench_cmd}")
            subprocess.run(
                ["bash", "-c", bench_cmd],
                cwd=bench_dir, stdout=bf, stderr=bf, check=True
            )
            self.logger.info("vllm benchmark script completed")

        # 6) tear down
        self.logger.info(f"Stopping vllm server (pid={proc.pid})")
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        self.logger.info("=== vllm benchmark done ===")


class SGLangJob(BaseJob):
    def run(self):
        self.logger.info("=== sglang benchmark start ===")

        # create venv & install sglang via uv
        run_cmd(["uv", "venv", "venv-sgl", "--python", "3.12"],
                cwd=self.root_dir, logfile=self.logfile, logger=self.logger)
        install_cmd = (
            "source venv-sgl/bin/activate && "
            "uv pip install \"sglang[all]==0.4.4.post1\" "
            "--find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python"
        )
        self.logger.info(f"▶ {install_cmd}")
        run_cmd(["bash", "-c", install_cmd],
                cwd=self.root_dir, logfile=self.logfile, logger=self.logger)
        self.logger.info("sglang package installed in venv-sgl")

        # launch sglang serve
        env = os.environ.copy()
        if self.cuda_dev:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_dev
        serve_cmd = ["python3", "-m", "sglang.launch_server",
                     "--model-path", self.model,
                     "--host", "0.0.0.0", "--port", str(self.port)]
        self.logger.info(f"▶ source venv-sgl/bin/activate && {' '.join(serve_cmd)}")
        proc = subprocess.Popen(
            "bash -c 'source venv-sgl/bin/activate && " +
            " ".join(serve_cmd) + "'", cwd=self.root_dir,
            stdout=self.logfile, stderr=self.logfile,
            env=env, preexec_fn=os.setsid, shell=True
        )
        self.logger.info(f"Started sglang serve (pid={proc.pid})")

        # wait for ready
        self.logger.info("Waiting for sglang to load…")
        wait_for_server("localhost", self.port, self.logger)
        self.logger.info(f"sglang inference server ready at http://localhost:{self.port}/v1/models")

        # 4) run benchmark script
        bench_dir = self.root_dir / "benchmark-compare"
        bench_log = self.root_dir / "logs" / "bench-sglang.log"
        with open(bench_log, "a") as bf:
            self.logger.info(">>> Starting sglang benchmark; output → bench-sglang.log")
            bench_cmd = (
                "source vllm/venv-vllm-src/bin/activate && "
                f"VLLM_USE_PRECOMPILED=1 MODEL={self.model} FRAMEWORK=sgl "
                "bash ./benchmark_1000_in_100_out.sh"
            )
            self.logger.info(f"▶ {bench_cmd}")
            subprocess.run(
                ["bash", "-c", bench_cmd],
                cwd=bench_dir, stdout=bf, stderr=bf, check=True
            )
            self.logger.info("sglang benchmark script completed")

        # tear down
        self.logger.info(f"Stopping sglang server (pid={proc.pid})")
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        self.logger.info("=== sglang benchmark done ===")


def main():
    cfg = parse_args()
    root = Path.cwd()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(logging.StreamHandler(sys.stdout))

    main_logger.info(f"Using port: {cfg.port}")
    global_setup(root, main_logger)

    jobs = [VLLMJob("vllm", cfg, root, logs),
            SGLangJob("sglang", cfg, root, logs)]
    run_jobs(jobs, main_logger)

    main_logger.info("✅ Benchmark results are in benchmark-compare/results.json")


if __name__ == "__main__":
    main()
