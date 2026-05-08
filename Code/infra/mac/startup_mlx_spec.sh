#!/bin/bash
# Start MLX speculative decoding server on port 8001
# Uses 8B target + 1B draft model (standard speculative decoding)
MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DRAFT_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
PORT=8001

mkdir -p logs

echo "Starting MLX Speculative Decoding Server..."
echo "  Target model : $MODEL"
echo "  Draft model  : $DRAFT_MODEL"
echo "  Port         : $PORT"
echo "  Log          : logs/mlx_spec.log"

nohup python3 -m mlx_lm.server \
  --model "$MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --port "$PORT" \
  > logs/mlx_spec.log 2>&1 &

echo $! > logs/mlx_spec.pid
echo "Speculative server started (PID $(cat logs/mlx_spec.pid))"
echo "Monitor: tail -f logs/mlx_spec.log"
echo "Stop:    kill \$(cat logs/mlx_spec.pid)"
