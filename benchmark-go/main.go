// main.go
package main

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

// Config holds CLI/env settings.
type Config struct {
	Port       int
	Model      string
	CudaDevice string
	Async      bool
}

// BaseJob contains common fields & helpers for each benchmark job.
type BaseJob struct {
	Name       string
	Port       int
	Model      string
	CudaDevice string
	RootDir    string
	LogFile    *os.File
	Logger     *log.Logger
}

// BenchmarkJob is the interface each framework job implements.
type BenchmarkJob interface {
	Name() string
	Run() error
}

func main() {
	rootDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal: cannot get working dir: %v\n", err)
		os.Exit(1)
	}

	// Parse CLI / ENV
	cfg := initConfig()

	mainLogger := log.New(os.Stdout, "[main] ", log.LstdFlags)

	// Ensure logs/ exists
	logsDir := filepath.Join(rootDir, "logs")
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		mainLogger.Fatalf("Cannot create logs directory: %v", err)
	}

	mainLogger.Printf("Using port: %d", cfg.Port)

	// Global setup: cleanup, ensure uv, clone repos
	if err := globalSetup(rootDir, mainLogger); err != nil {
		mainLogger.Fatalf("Setup error: %v", err)
	}

	// Instantiate jobs
	jobs := []BenchmarkJob{
		NewVLLMJob(cfg, rootDir, logsDir),
		NewSGLangJob(cfg, rootDir, logsDir),
	}

	// Run benchmarks (sync or async)
	runJobs(jobs, cfg.Async, mainLogger)

	// Success message
	mainLogger.Println("Benchmark results are in benchmark-compare/results.json")
}

// initConfig parses flags and environment into Config.
func initConfig() Config {
	pflag.Int("port", 8080, "Port for both servers")
	pflag.String("model", "meta-llama/Llama-3.1-8B-Instruct", "Model path or identifier")
	pflag.String("cuda-device", "", "CUDA_VISIBLE_DEVICES override")
	pflag.Bool("async", false, "Run benchmarks in parallel")

	// bind before parse so viper picks up CLI overrides
	viper.BindPFlags(pflag.CommandLine)
	pflag.Parse()

	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer("-", "_"))
	viper.BindEnv("cuda-device", "CUDA_VISIBLE_DEVICES")

	return Config{
		Port:       viper.GetInt("port"),
		Model:      viper.GetString("model"),
		CudaDevice: viper.GetString("cuda-device"),
		Async:      viper.GetBool("async"),
	}
}

// globalSetup does the one‑time cleanup, uv install, and repo clones.
func globalSetup(rootDir string, logger *log.Logger) error {
	toRemove := []string{
		filepath.Join(rootDir, "benchmark-compare"),
		filepath.Join(rootDir, "venv-vllm"),
		filepath.Join(rootDir, "venv-vllm-src"),
		filepath.Join(rootDir, "venv-sgl"),
	}
	for _, p := range toRemove {
		logger.Printf("Removing %s", p)
		os.RemoveAll(p)
	}

	if err := ensureUV(logger); err != nil {
		return err
	}

	if err := cloneRepo(
		"https://github.com/neuralmagic/benchmark-compare.git",
		filepath.Join(rootDir, "benchmark-compare"),
		"",
		logger,
	); err != nil {
		return err
	}

	vllmDir := filepath.Join(rootDir, "benchmark-compare", "vllm")
	return cloneRepo(
		"https://github.com/vllm-project/vllm.git",
		vllmDir,
		"benchmark-output",
		logger,
	)
}

func ensureUV(logger *log.Logger) error {
	if _, err := exec.LookPath("uv"); err != nil {
		logger.Println("`uv` not found; installing via astral.sh...")
		cmd := exec.Command("bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh")
		cmd.Stdout = logger.Writer()
		cmd.Stderr = logger.Writer()
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("uv install failed: %w", err)
		}
	}
	return nil
}

func cloneRepo(url, dest, branch string, logger *log.Logger) error {
	logger.Printf("▶ git clone %s %s", url, dest)
	cmd := exec.Command("git", "clone", url, dest)
	cmd.Stdout = logger.Writer()
	cmd.Stderr = logger.Writer()
	if err := cmd.Run(); err != nil {
		return err
	}
	if branch != "" {
		logger.Printf("▶ git -C %s checkout %s", dest, branch)
		cmd = exec.Command("git", "-C", dest, "checkout", branch)
		cmd.Stdout = logger.Writer()
		cmd.Stderr = logger.Writer()
		cmd.Run()
	}
	return nil
}

// runJobs ensures that after vllm completes we kill its serve group before sglang.
func runJobs(jobs []BenchmarkJob, async bool, logger *log.Logger) {
	if !async {
		for _, job := range jobs {
			logger.Printf("▶ Running %s", job.Name())
			if err := job.Run(); err != nil {
				logger.Printf("✗ %s failed: %v", job.Name(), err)
				return
			}
			logger.Printf("✓ %s completed", job.Name())
			if job.Name() == "vllm" {
				logger.Println("Killing vllm serve process group")
				exec.Command("pkill", "-f", "vllm serve").Run()
			}
		}
	} else {
		var wg sync.WaitGroup
		for _, job := range jobs {
			wg.Add(1)
			go func(j BenchmarkJob) {
				defer wg.Done()
				logger.Printf("▶ %s (async)", j.Name())
				if err := j.Run(); err != nil {
					logger.Printf("✗ %s failed: %v", j.Name(), err)
				} else {
					logger.Printf("✓ %s completed", j.Name())
				}
			}(job)
		}
		wg.Wait()
	}
}

func runCmd(name string, args []string, dir string, logfile *os.File, logger *log.Logger) error {
	logger.Printf("▶ cmd: %s %s", name, strings.Join(args, " "))
	cmd := exec.CommandContext(context.Background(), name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	cmd.Env = os.Environ()
	cmd.Stdout = logfile
	cmd.Stderr = logfile
	return cmd.Run()
}

// --- vLLM Job --------------------------------------------------

type VLLMJob struct{ BaseJob }

func (j *VLLMJob) Name() string { return j.BaseJob.Name }

func (j *VLLMJob) Run() error {
	defer j.LogFile.Close()
	j.Logger.Println("=== vllm benchmark start ===")

	if err := runCmd("uv", []string{"venv", "venv-vllm", "--python", "3.12"},
		j.RootDir, j.LogFile, j.Logger); err != nil {
		return err
	}
	if err := runCmd("bash", []string{"-c", "source venv-vllm/bin/activate && uv pip install vllm==0.8.3"},
		j.RootDir, j.LogFile, j.Logger); err != nil {
		return err
	}

	// launch vllm serve
	cudaPrefix := ""
	if j.CudaDevice != "" {
		cudaPrefix = fmt.Sprintf("CUDA_VISIBLE_DEVICES=%s ", j.CudaDevice)
	}
	serve := fmt.Sprintf("source venv-vllm/bin/activate && %svllm serve \"%s\" --disable-log-requests --port %d",
		cudaPrefix, j.Model, j.Port)
	j.Logger.Printf("▶ %s", serve)
	cmdSrv := exec.Command("bash", "-c", serve)
	cmdSrv.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmdSrv.Dir = j.RootDir
	cmdSrv.Stdout = j.LogFile
	cmdSrv.Stderr = j.LogFile
	if err := cmdSrv.Start(); err != nil {
		return err
	}

	j.Logger.Println("Waiting for vllm to load...")

	// wait until server responds to indicate a ready state
	if err := waitForServer("localhost", j.Port, j.Logger); err != nil {
		return err
	}

	j.Logger.Println("vllm inference server ready; starting benchmark tests")

	// setup benchmark venv in the vllm dir
	vllmDir := filepath.Join(j.RootDir, "benchmark-compare", "vllm")
	if err := runCmd("uv", []string{"venv", "venv-vllm-src", "--python", "3.12"},
		vllmDir, j.LogFile, j.Logger); err != nil {
		return err
	}
	deps := "source venv-vllm-src/bin/activate && export VLLM_USE_PRECOMPILED=1 && uv pip install -e . && uv pip install numpy pandas datasets"
	j.Logger.Printf("▶ %s", deps)
	if err := runCmd("bash", []string{"-c", deps},
		vllmDir, j.LogFile, j.Logger); err != nil {
		return err
	}

	// run benchmark
	benchDir := filepath.Join(j.RootDir, "benchmark-compare")
	bench := fmt.Sprintf(
		"source vllm/venv-vllm-src/bin/activate && VLLM_USE_PRECOMPILED=1 MODEL=%s FRAMEWORK=vllm bash ./benchmark_1000_in_100_out.sh",
		j.Model,
	)

	j.Logger.Println(">>> Starting vllm benchmark script; output logged to logs/bench-vllm.log")

	benchLogPath := filepath.Join(j.RootDir, "logs", "bench-vllm.log")
	benchLogF, err := os.OpenFile(benchLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		j.Logger.Printf("Cannot open bench-vllm.log: %v", err)
		return err
	}
	defer benchLogF.Close()

	cmdBench := exec.Command("bash", "-c", bench)
	cmdBench.Dir = benchDir
	cmdBench.Stdout = benchLogF
	cmdBench.Stderr = benchLogF
	if err := cmdBench.Run(); err != nil {
		return err
	}

	// kill process group
	j.Logger.Printf("Stopping vllm server (pgid %d)", cmdSrv.Process.Pid)
	syscall.Kill(-cmdSrv.Process.Pid, syscall.SIGKILL)
	cmdSrv.Wait()

	j.Logger.Println("=== vllm benchmark done ===")
	return nil
}

// NewVLLMJob builds the vllm BenchmarkJob.
func NewVLLMJob(cfg Config, rootDir, logsDir string) BenchmarkJob {
	logF, err := os.OpenFile(filepath.Join(logsDir, "vllm.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		log.Fatalf("cannot open vllm log: %v", err)
	}
	mw := io.MultiWriter(logF, os.Stdout)
	logger := log.New(mw, "[vllm] ", log.LstdFlags)
	return &VLLMJob{BaseJob{
		Name:       "vllm",
		Port:       cfg.Port,
		Model:      cfg.Model,
		CudaDevice: cfg.CudaDevice,
		RootDir:    rootDir,
		LogFile:    logF,
		Logger:     logger,
	}}
}

// --- SGLang Job --------------------------------------------------

type SGLangJob struct{ BaseJob }

func (j *SGLangJob) Name() string { return j.BaseJob.Name }

func (j *SGLangJob) Run() error {
	defer j.LogFile.Close()
	j.Logger.Println("=== sglang benchmark start ===")

	// 1) create & install sglang venv
	if err := runCmd("uv", []string{"venv", "venv-sgl", "--python", "3.12"},
		j.RootDir, j.LogFile, j.Logger); err != nil {
		return err
	}
	install := "source venv-sgl/bin/activate && uv pip install \"sglang[all]==0.4.4.post1\" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python"
	j.Logger.Printf("▶ %s", install)
	if err := runCmd("bash", []string{"-c", install},
		j.RootDir, j.LogFile, j.Logger); err != nil {
		return err
	}

	// launch sglang instance
	cudaPrefix := ""
	if j.CudaDevice != "" {
		cudaPrefix = fmt.Sprintf("CUDA_VISIBLE_DEVICES=%s ", j.CudaDevice)
	}
	serve := fmt.Sprintf("source venv-sgl/bin/activate && %spython3 -m sglang.launch_server --model-path \"%s\" --host 0.0.0.0 --port %d",
		cudaPrefix, j.Model, j.Port)
	j.Logger.Printf("▶ %s", serve)

	cmdSrv := exec.Command("bash", "-c", serve)
	cmdSrv.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmdSrv.Dir = j.RootDir
	cmdSrv.Stdout = j.LogFile
	cmdSrv.Stderr = j.LogFile
	if err := cmdSrv.Start(); err != nil {
		return err
	}

	j.Logger.Println("Waiting for sglang to load...")

	// 3) wait until server responds
	if err := waitForServer("localhost", j.Port, j.Logger); err != nil {
		return err
	}

	// run benchmark from root of benchmark-compare
	benchDir := filepath.Join(j.RootDir, "benchmark-compare")
	bench := fmt.Sprintf(
		"source vllm/venv-vllm-src/bin/activate && VLLM_USE_PRECOMPILED=1 MODEL=%s FRAMEWORK=sgl bash ./benchmark_1000_in_100_out.sh",
		j.Model,
	)

	j.Logger.Println(">>> Starting sglang benchmark script; output logged to logs/bench-sglang.log")

	benchLogPath := filepath.Join(j.RootDir, "logs", "bench-sglang.log")
	benchLogF, err := os.OpenFile(benchLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		j.Logger.Printf("Cannot open bench-sglang.log: %v", err)
		return err
	}
	defer benchLogF.Close()

	cmdBench := exec.Command("bash", "-c", bench)
	cmdBench.Dir = benchDir
	cmdBench.Stdout = benchLogF
	cmdBench.Stderr = benchLogF
	if err := cmdBench.Run(); err != nil {
		return err
	}

	// kill serve process group
	j.Logger.Printf("Stopping sglang server (pgid %d)", cmdSrv.Process.Pid)
	syscall.Kill(-cmdSrv.Process.Pid, syscall.SIGKILL)
	cmdSrv.Wait()
	j.Logger.Println("=== sglang benchmark done ===")
	return nil
}

// NewSGLangJob builds the sglang BenchmarkJob.
func NewSGLangJob(cfg Config, rootDir, logsDir string) BenchmarkJob {
	logF, err := os.OpenFile(filepath.Join(logsDir, "sglang.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		log.Fatalf("cannot open sglang log: %v", err)
	}
	mw := io.MultiWriter(logF, os.Stdout)
	logger := log.New(mw, "[sglang] ", log.LstdFlags)
	return &SGLangJob{BaseJob{
		Name:       "sglang",
		Port:       cfg.Port,
		Model:      cfg.Model,
		CudaDevice: cfg.CudaDevice,
		RootDir:    rootDir,
		LogFile:    logF,
		Logger:     logger,
	}}
}

// waitForServer polls until the server responds.
func waitForServer(host string, port int, logger *log.Logger) error {
	url := fmt.Sprintf("http://%s:%d/v1/models", host, port)
	timeout := time.After(120 * time.Second)
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-timeout:
			return fmt.Errorf("timeout waiting for server at %s", url)
		case <-tick.C:
			resp, err := http.Get(url)
			if err != nil {
				continue
			}
			body, _ := ioutil.ReadAll(resp.Body)
			resp.Body.Close()
			if strings.Contains(string(body), "data") {
				return nil
			}
		}
	}
}
