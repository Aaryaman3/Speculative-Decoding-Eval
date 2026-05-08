#!/bin/bash
# run_sweep.sh — Runs all (task × concurrency × trial) combinations
#
# Usage:
#   bash benchmark/run_sweep.sh <system> <gpu_type> <server_url>
#
# Examples (Mac):
#   bash benchmark/run_sweep.sh mlx_baseline mac http://localhost:8000
#   bash benchmark/run_sweep.sh mlx_spec     mac http://localhost:8001
#
# Examples (GPU):
#   bash benchmark/run_sweep.sh baseline L4   http://localhost:8000
#   bash benchmark/run_sweep.sh eagle3   L4   http://localhost:8001

set -e
SYSTEM="${1:?Usage: $0 <system> <gpu_type> <server_url>}"
GPU_TYPE="${2:?Usage: $0 <system> <gpu_type> <server_url>}"
SERVER_URL="${3:?Usage: $0 <system> <gpu_type> <server_url>}"

TASKS="chat code summarization"
CONCURRENCIES="1 4 8 16 32"
N_TRIALS=1
PROMPTS_DIR="data/prompts"
N_SAMPLES=10

echo "=== Sweep: system=$SYSTEM  gpu=$GPU_TYPE  server=$SERVER_URL ==="
echo "    Tasks       : $TASKS"
echo "    Concurrency : $CONCURRENCIES"
echo "    Trials      : $N_TRIALS"
echo ""

for trial in $(seq 1 "$N_TRIALS"); do
  for task in $TASKS; do
    PROMPT_FILE="$PROMPTS_DIR/${task}_${N_SAMPLES}.jsonl"
    if [ ! -f "$PROMPT_FILE" ]; then
      echo "ERROR: prompt file missing: $PROMPT_FILE"
      echo "Run: python3 data/prepare_datasets.py --n-samples $N_SAMPLES"
      exit 1
    fi
    for concurrency in $CONCURRENCIES; do
      echo "--- task=$task  concurrency=$concurrency  trial=$trial ---"
      python3 benchmark/load_test.py \
        --system "$SYSTEM" \
        --gpu-type "$GPU_TYPE" \
        --task "$task" \
        --prompts-file "$PROMPT_FILE" \
        --concurrency "$concurrency" \
        --trial "$trial" \
        --server-url "$SERVER_URL" \
        --no-wait
    done
  done
done

echo ""
echo "=== Sweep complete! ==="
echo "Result files:"
ls results/raw/*.jsonl 2>/dev/null | wc -l
echo "Run: python3 benchmark/plot_results.py"
