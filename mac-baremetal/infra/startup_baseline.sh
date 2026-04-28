#!/bin/bash
# GPU: Start vLLM baseline server (greedy, no speculative decoding) on port 8000
MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"
PORT=8000

mkdir -p logs

echo "Starting vLLM Baseline Server..."
echo "  Model : $MODEL"
echo "  Port  : $PORT"

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype bfloat16 \
  --port "$PORT" \
  --max-model-len 4096 \
  > logs/baseline.log 2>&1 &

echo $! > logs/baseline.pid
echo "Baseline server started (PID $(cat logs/baseline.pid))"
echo "Monitor: tail -f logs/baseline.log"
