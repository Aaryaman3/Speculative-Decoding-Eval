#!/bin/bash
# run_real_experiment.sh — Full end-to-end benchmark on Mac/MLX bare metal
# Run from the "Project 2/" directory.
set -e

N_SAMPLES=10
GPU_TYPE="mac"

BASELINE_URL="http://localhost:8000"
SPEC_URL="http://localhost:8001"
STARTUP_TIMEOUT=300  # seconds to wait for server ready

echo "========================================================"
echo "  EAGLE-3 Speculative Decoding — Mac M4 Bare Metal Run"
echo "========================================================"
echo "  Samples per task : $N_SAMPLES"
echo "  Tasks            : chat  code  summarization"
echo "  Concurrency      : 1 4 8 16 32"
echo "  Trials           : 1"
echo "  Max tokens       : 128"
echo ""

# ---- helpers ----------------------------------------------------------------
wait_for_server() {
  local url="$1/v1/models"
  local log_file="$2"
  local deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
  echo -n "  Waiting for server"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo " ready!"
      return 0
    fi
    echo -n "."
    sleep 5
  done
  echo " TIMEOUT — check $log_file"
  return 1
}

kill_server() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    local pid=$(cat "$pid_file")
    kill "$pid" 2>/dev/null || true
    rm -f "$pid_file"
    sleep 3
  fi
}

# ---- Phase 1: Baseline sweep ------------------------------------------------
echo "=== Phase 1: MLX Baseline Server ==="

nohup python3 -m mlx_lm.server \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --port 8000 \
  > logs/baseline.log 2>&1 &
echo $! > logs/baseline.pid
echo "  Server PID: $(cat logs/baseline.pid)"

wait_for_server "$BASELINE_URL" "logs/baseline.log"

echo ""
echo "  Running baseline sweep..."
bash benchmark/run_sweep.sh mlx_baseline "$GPU_TYPE" "$BASELINE_URL"

echo ""
echo "  Stopping baseline server..."
kill_server logs/baseline.pid

echo ""
echo "=== Phase 2: MLX Speculative Decoding Server ==="

nohup python3 -m mlx_lm.server \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --draft-model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --port 8001 \
  > logs/mlx_spec.log 2>&1 &
echo $! > logs/mlx_spec.pid
echo "  Server PID: $(cat logs/mlx_spec.pid)"

wait_for_server "$SPEC_URL" "logs/mlx_spec.log"

echo ""
echo "  Running speculative decoding sweep..."
bash benchmark/run_sweep.sh mlx_spec "$GPU_TYPE" "$SPEC_URL"

echo ""
echo "  Stopping speculative server..."
kill_server logs/mlx_spec.pid

# ---- Phase 3: Analysis ------------------------------------------------------
echo ""
echo "=== Phase 3: Generating Plots & Tables ==="
python3 benchmark/plot_results.py

echo ""
echo "========================================================"
echo "  Experiment complete!"
echo "  Plots  → results/plots/"
echo "  Tables → results/tables/"
echo "  Raw    → results/raw/"
echo "========================================================"
