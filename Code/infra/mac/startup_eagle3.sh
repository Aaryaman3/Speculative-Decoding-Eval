#!/bin/bash
# GPU: Start vLLM EAGLE-3 speculative decoding server on port 8001
MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"
DRAFT_MODEL="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
PORT=8001

mkdir -p logs

echo "Starting vLLM EAGLE-3 Speculative Server..."
echo "  Target model : $MODEL"
echo "  Draft model  : $DRAFT_MODEL"
echo "  Port         : $PORT"

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --speculative-model "$DRAFT_MODEL" \
  --speculative-model-uses-eagle \
  --num-speculative-tokens 5 \
  --dtype bfloat16 \
  --port "$PORT" \
  --max-model-len 4096 \
  > logs/eagle3.log 2>&1 &

echo $! > logs/eagle3.pid
echo "EAGLE-3 server started (PID $(cat logs/eagle3.pid))"
echo "Monitor: tail -f logs/eagle3.log"
