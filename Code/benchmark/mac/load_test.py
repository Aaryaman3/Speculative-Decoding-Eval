"""
load_test.py — Async load generator for MLX/vLLM OpenAI-compatible servers.

Usage (Mac/MLX):
  python3 benchmark/load_test.py \
    --system mlx_baseline --gpu-type mac \
    --task chat --prompts-file data/prompts/chat_50.jsonl \
    --concurrency 1 --trial 1 --server-url http://localhost:8000

Usage (GPU/vLLM):
  python3 benchmark/load_test.py \
    --system eagle3 --gpu-type L4 \
    --task code --prompts-file data/prompts/code_50.jsonl \
    --concurrency 8 --trial 1 --server-url http://localhost:8001
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
MAX_NEW_TOKENS = 128

# Model names as served by mlx_lm.server
SYSTEM_MODEL = {
    "mlx_baseline": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx_spec":     "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "baseline":     "unsloth/Meta-Llama-3.1-8B-Instruct",
    "eagle3":       "unsloth/Meta-Llama-3.1-8B-Instruct",
}


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    server_url: str,
    prompt: str,
    request_id: str,
    semaphore: asyncio.Semaphore,
    model_name: str = "default",
) -> dict:
    """Send one streaming chat-completion request and collect timing metrics."""
    async with semaphore:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "stream": True,
        }
        url = f"{server_url}/v1/chat/completions"

        t_send = time.perf_counter()
        ttft_ms: float | None = None
        output_text = ""
        output_tokens = 0
        prompt_tokens = 0
        error: str | None = None

        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                resp.raise_for_status()
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # First token → record TTFT
                    delta_content = ""
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        delta_content = delta.get("content", "")
                        if delta_content and ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_send) * 1000
                        if delta_content:
                            output_text += delta_content
                            output_tokens += 1

                    # Some servers embed usage in the last chunk
                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens"):
                        output_tokens = usage["completion_tokens"]
                    if usage.get("prompt_tokens"):
                        prompt_tokens = usage["prompt_tokens"]

        except Exception as exc:
            error = str(exc)

        t_done = time.perf_counter()
        total_latency_ms = (t_done - t_send) * 1000

        if ttft_ms is None:
            ttft_ms = total_latency_ms  # fallback: no streaming data received

        # tpot = time per output token (excluding first token latency)
        remaining_tokens = max(output_tokens - 1, 1)
        tpot_ms = max((total_latency_ms - ttft_ms) / remaining_tokens, 0.0)
        tokens_per_sec = output_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0.0

        return {
            "request_id": request_id,
            "error": error,
            "ttft_ms": round(ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "output_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
        }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

async def run_benchmark(
    prompts: list[dict],
    server_url: str,
    system: str,
    gpu_type: str,
    task: str,
    concurrency: int,
    trial: int,
) -> list[dict]:
    model_name = SYSTEM_MODEL.get(system, "default")
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, p in enumerate(prompts):
            request_id = f"{task}_{i+1:03d}_c{concurrency}_t{trial}"
            tasks.append(send_request(session, server_url, p["prompt"], request_id, semaphore, model_name))
        results_raw = await asyncio.gather(*tasks)

    results = []
    for raw in results_raw:
        if raw["error"]:
            print(f"  WARN request {raw['request_id']} failed: {raw['error']}")
            continue
        results.append(
            {
                "request_id": raw["request_id"],
                "task": task,
                "concurrency": concurrency,
                "system": system,
                "gpu_type": gpu_type,
                "trial": trial,
                "ttft_ms": raw["ttft_ms"],
                "tpot_ms": raw["tpot_ms"],
                "total_latency_ms": raw["total_latency_ms"],
                "output_tokens": raw["output_tokens"],
                "tokens_per_sec": raw["tokens_per_sec"],
                "acceptance_rate": None,  # MLX does not expose this; set by post-processing for vLLM
                "gpu_cost_usd": 0.0,      # bare metal — no cloud cost
            }
        )
    return results


def save_results(results: list[dict], system: str, task: str, concurrency: int, trial: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{system}_{task}_c{concurrency}_t{trial}.jsonl"
    out_path = RESULTS_DIR / fname
    with open(out_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    return out_path


def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def wait_for_server(server_url: str, timeout: int = 120) -> bool:
    """Poll /health or /v1/models until the server is ready."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    check_url = f"{server_url}/v1/models"
    print(f"  Waiting for server at {check_url} ...", end="", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(check_url, timeout=3):
                print(" ready!")
                return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(3)
    print(" TIMEOUT")
    return False


def main():
    parser = argparse.ArgumentParser(description="Async load test for LLM servers")
    parser.add_argument("--system", required=True, choices=["baseline", "eagle3", "mlx_baseline", "mlx_spec"])
    parser.add_argument("--gpu-type", required=True, help="e.g. L4, A100, mac")
    parser.add_argument("--task", required=True, choices=["chat", "code", "summarization"])
    parser.add_argument("--prompts-file", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--no-wait", action="store_true", help="Skip server readiness check")
    args = parser.parse_args()

    if not args.prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")

    prompts = load_prompts(args.prompts_file)
    print(f"\nLoaded {len(prompts)} prompts from {args.prompts_file}")
    print(f"Config: system={args.system}  task={args.task}  concurrency={args.concurrency}  trial={args.trial}")
    print(f"Server: {args.server_url}")

    if not args.no_wait:
        if not wait_for_server(args.server_url):
            raise RuntimeError("Server did not become ready in time. Check logs.")

    t0 = time.perf_counter()
    results = asyncio.run(
        run_benchmark(
            prompts=prompts,
            server_url=args.server_url,
            system=args.system,
            gpu_type=args.gpu_type,
            task=args.task,
            concurrency=args.concurrency,
            trial=args.trial,
        )
    )
    elapsed = time.perf_counter() - t0

    out_path = save_results(results, args.system, args.task, args.concurrency, args.trial)

    if results:
        avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
        avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
        avg_lat = sum(r["total_latency_ms"] for r in results) / len(results)
        print(f"\nResults ({len(results)} requests in {elapsed:.1f}s):")
        print(f"  avg tokens/sec  : {avg_tps:.1f}")
        print(f"  avg TTFT        : {avg_ttft:.0f} ms")
        print(f"  avg total lat   : {avg_lat:.0f} ms")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
