#!/bin/bash
# Start MLX baseline server (no speculative decoding) on port 8000
MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
PORT=8000

mkdir -p logs

echo "Starting MLX Baseline Server..."
echo "  Model : $MODEL"
echo "  Port  : $PORT"
echo "  Log   : logs/baseline.log"

nohup python3 -m mlx_lm.server \
  --model "$MODEL" \
  --port "$PORT" \
  > logs/baseline.log 2>&1 &

echo $! > logs/baseline.pid
echo "Baseline server started (PID $(cat logs/baseline.pid))"
echo "Monitor: tail -f logs/baseline.log"
echo "Stop:    kill \$(cat logs/baseline.pid)"
