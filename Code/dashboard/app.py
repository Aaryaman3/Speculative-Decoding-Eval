"""
Live benchmark dashboard — Streamlit app.

Shows side-by-side comparison of baseline vs EAGLE-3 (or MLX variants).
Can run in two modes:
  1. LIVE MODE   — fires real requests to running servers, shows real-time metrics
  2. RESULTS MODE — loads from results/raw/*.jsonl, shows analysis plots

Usage:
    streamlit run dashboard/app.py
"""

import asyncio
import json
import time
from pathlib import Path

import aiohttp
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EAGLE-3 Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
)

RESULTS_DIR = Path("results/raw")

GPU_HOURLY_RATES = {
    "T4":   0.35,
    "L4":   0.70,
    "A100": 3.67,
    "mac":  0.00,
}

st.sidebar.title("⚡ EAGLE-3 Benchmark")
mode = st.sidebar.radio("Mode", ["Live Demo", "Results Analysis"])

# ── LIVE DEMO MODE ─────────────────────────────────────────────────────────────
if mode == "Live Demo":
    st.title("⚡ Live: Greedy Baseline vs EAGLE-3 Speculative Decoding")
    st.caption(
        "Both servers receive the **same request** simultaneously. "
        "EAGLE-3 uses a lightweight draft model to speculatively generate 5 tokens at once, "
        "then the target model verifies them in a single forward pass — "
        "reducing the number of sequential target-model calls."
    )

    with st.expander("ℹ️ Metric glossary", expanded=False):
        st.markdown(
            "| Metric | Meaning | Who wins? |\n"
            "|--------|---------|------------|\n"
            "| **TTFT** | Time to First Token — how long until the first word appears | Lower is better |\n"
            "| **TPOT** | Time Per Output Token — latency *between* each subsequent token | **Lower is better — this is the key speculative-decoding metric** |\n"
            "| **Tokens/sec** | Total output tokens ÷ total wall time (includes TTFT) | Higher is better |\n"
            "| **Acceptance rate** | % of draft tokens accepted by the target model without regeneration | Higher → more speedup |\n"
            "| **Speedup** | EAGLE-3 TPOT ÷ Baseline TPOT (>1× = EAGLE-3 wins) | Higher is better |"
        )

    st.sidebar.subheader("Server Config")
    baseline_url = st.sidebar.text_input("Baseline URL", "http://localhost:8000")
    eagle3_url   = st.sidebar.text_input("EAGLE-3 URL",  "http://localhost:8001")
    gpu_type     = st.sidebar.selectbox("GPU Type", ["A100", "L4", "T4", "mac"])

    st.sidebar.subheader("Request Config")
    task = st.sidebar.selectbox("Task", ["chat", "code", "summarization"])
    concurrency = st.sidebar.select_slider(
        "Concurrency",
        [1, 4, 8, 16, 32],
        value=1,
        help="Keep at 1 to see the clearest per-request speedup. "
             "Increase to stress-test batching behavior.",
    )
    max_tokens = st.sidebar.slider("Max output tokens", 64, 512, 256)

    DEFAULT_PROMPTS = {
        "chat": "Explain the concept of recursion in simple terms.",
        "code": "Write a Python function that checks if a string is a palindrome.",
        "summarization": (
            "Summarize the following: The Apollo 11 mission was the first crewed lunar landing mission. "
            "Launched on July 16, 1969, it carried astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins. "
            "Armstrong and Aldrin landed on the Moon on July 20, while Collins orbited above. "
            "Armstrong became the first person to walk on the Moon."
        ),
    }
    prompt = st.text_area("Prompt", value=DEFAULT_PROMPTS[task], height=120)

    # ── async helpers ──────────────────────────────────────────────────────────

    async def send_one(session, url, model_name, prompt_text, gpu_type_str):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        t_start = time.perf_counter()
        t_first = None
        text = ""
        chunk_count = 0
        server_tokens = None
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        usage = chunk.get("usage")
                        if usage and "completion_tokens" in usage:
                            server_tokens = usage["completion_tokens"]
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        content = choices[0]["delta"].get("content", "")
                        if content:
                            if t_first is None:
                                t_first = time.perf_counter()
                            text += content
                            chunk_count += 1
                    except Exception:
                        continue
        except Exception as e:
            return {"error": str(e), "ttft_ms": None, "tpot_ms": None,
                    "tokens_per_sec": None, "cost_usd": None, "text": ""}

        t_end = time.perf_counter()
        total = t_end - t_start
        tokens = server_tokens if server_tokens is not None else chunk_count
        ttft = (t_first - t_start) * 1000 if t_first else None
        tpot = ((t_end - t_first) / (tokens - 1) * 1000) if t_first and tokens > 1 else None
        cost = GPU_HOURLY_RATES.get(gpu_type_str, 0) / 3600 * total

        return {
            "ttft_ms": round(ttft, 1) if ttft else None,
            "tpot_ms": round(tpot, 1) if tpot else None,
            "tokens_per_sec": round(tokens / total, 1) if total > 0 else None,
            "output_tokens": tokens,
            "cost_usd": cost,
            "text": text,
            "error": None,
        }

    async def get_acceptance_rate(session, url):
        """Pull acceptance rate from vLLM Prometheus /metrics endpoint."""
        try:
            async with session.get(f"{url}/metrics",
                                   timeout=aiohttp.ClientTimeout(total=5)) as resp:
                for line in (await resp.text()).split("\n"):
                    if "spec_decode_draft_acceptance_rate" in line and not line.startswith("#"):
                        val = float(line.split()[-1])
                        return val if val > 0 else None
        except Exception:
            pass
        return None

    async def run_live_demo(prompt_text, concurrency_n):
        prompts = [prompt_text] * concurrency_n
        connector = aiohttp.TCPConnector(limit=concurrency_n * 2 + 4)
        async with aiohttp.ClientSession(connector=connector) as session:
            baseline_tasks = [send_one(session, baseline_url, "baseline", p, gpu_type) for p in prompts]
            eagle3_tasks   = [send_one(session, eagle3_url,  "eagle3",   p, gpu_type) for p in prompts]
            all_results    = await asyncio.gather(*baseline_tasks, *eagle3_tasks)
            acceptance     = await get_acceptance_rate(session, eagle3_url)

        b_res = list(all_results[:concurrency_n])
        e_res = list(all_results[concurrency_n:])
        return b_res, e_res, acceptance

    def avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    # ── Run button ─────────────────────────────────────────────────────────────

    if st.button("▶  Run Benchmark", type="primary"):
        with st.spinner(f"Firing {concurrency} concurrent request(s) to both servers…"):
            b_results, e_results, acceptance_rate = asyncio.run(
                run_live_demo(prompt, concurrency)
            )

        b_tpot = avg(b_results, "tpot_ms")
        e_tpot = avg(e_results, "tpot_ms")
        b_tps  = avg(b_results, "tokens_per_sec")
        e_tps  = avg(e_results, "tokens_per_sec")

        # ── Headline speedup banner ────────────────────────────────────────────
        if b_tpot and e_tpot and b_tpot > 0:
            tpot_speedup = b_tpot / e_tpot
            tps_speedup  = (e_tps / b_tps) if (b_tps and e_tps) else None

            if tpot_speedup >= 1.0:
                st.success(
                    f"⚡ EAGLE-3 is **{tpot_speedup:.2f}× faster** per token (TPOT)  "
                    + (f"| **{tps_speedup:.2f}× overall throughput**" if tps_speedup else "")
                    + (f"  | Draft acceptance rate: **{acceptance_rate:.0%}**" if acceptance_rate else "")
                )
            else:
                st.warning(
                    f"At concurrency={concurrency}, speculative decoding overhead outweighs gains "
                    f"(TPOT ratio: {tpot_speedup:.2f}×). Try concurrency=1 for peak benefit."
                )

        # ── Side-by-side metrics ───────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔵 Baseline (greedy decoding)")
            b_ttft = avg(b_results, "ttft_ms")
            b_cost = sum(r.get("cost_usd") or 0 for r in b_results)
            st.metric("TPOT — time per output token", f"{b_tpot:.1f} ms" if b_tpot else "N/A")
            st.metric("Tokens / sec",                 f"{b_tps:.1f}"     if b_tps  else "N/A")
            st.metric("TTFT — time to first token",   f"{b_ttft:.0f} ms" if b_ttft else "N/A")
            st.metric("GPU cost (this run)",           f"${b_cost:.5f}")
            st.metric("Acceptance rate",               "N/A (no draft model)")
            if b_results[0].get("text"):
                st.text_area("Sample output", b_results[0]["text"][:500], height=160,
                             key="b_text")

        with col2:
            st.subheader("🔴 EAGLE-3 (speculative decoding, k=5)")
            e_ttft = avg(e_results, "ttft_ms")
            e_cost = sum(r.get("cost_usd") or 0 for r in e_results)

            tpot_delta = f"{b_tpot - e_tpot:+.1f} ms" if (b_tpot and e_tpot) else None
            tps_delta  = f"{e_tps - b_tps:+.1f} tok/s" if (b_tps and e_tps) else None
            ttft_delta = f"{avg(b_results,'ttft_ms') - e_ttft:+.0f} ms" if (b_ttft and e_ttft) else None

            st.metric("TPOT — time per output token",
                      f"{e_tpot:.1f} ms" if e_tpot else "N/A",
                      delta=tpot_delta, delta_color="inverse")
            st.metric("Tokens / sec",
                      f"{e_tps:.1f}" if e_tps else "N/A",
                      delta=tps_delta)
            st.metric("TTFT — time to first token",
                      f"{e_ttft:.0f} ms" if e_ttft else "N/A",
                      delta=ttft_delta, delta_color="inverse")
            st.metric("GPU cost (this run)", f"${e_cost:.5f}")
            st.metric("Draft acceptance rate",
                      f"{acceptance_rate:.1%}" if acceptance_rate else "see /metrics")
            if e_results[0].get("text"):
                st.text_area("Sample output", e_results[0]["text"][:500], height=160,
                             key="e_text")

        errors = [r for r in b_results + e_results if r.get("error")]
        if errors:
            st.error(f"{len(errors)} request(s) failed: {errors[0]['error']}")

        with st.expander("How to read these results"):
            st.markdown(
                "**TPOT is the key metric for speculative decoding.** "
                "It measures how fast tokens stream to the user *after* the first one. "
                "Lower TPOT = faster perceived generation.\n\n"
                "**Why might TTFT be higher for EAGLE-3?** "
                "The draft model needs to set up its KV cache on the first request. "
                "At low concurrency (c=1) after warmup, TTFT converges.\n\n"
                "**Why does speedup shrink at high concurrency?** "
                "At c≥8 the GPU is compute-bound and the overhead of running the draft model "
                "starts to outweigh the verification savings."
            )


# ── RESULTS ANALYSIS MODE ──────────────────────────────────────────────────────
else:
    st.title("Results Analysis — Full Sweep Data")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    @st.cache_data
    def load_results():
        records = []
        for p in RESULTS_DIR.glob("*.jsonl"):
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        if r.get("ttft_ms") is not None:
                            records.append(r)
                    except Exception:
                        continue
        return pd.DataFrame(records)

    if not any(RESULTS_DIR.glob("*.jsonl")):
        st.warning("No results found in results/raw/. Run the benchmark sweep first.")
    else:
        df = load_results()
        st.success(f"Loaded {len(df)} request records from {RESULTS_DIR}")

        col1, col2, col3 = st.columns(3)
        with col1:
            gpu_filter = st.multiselect("GPU Type", df["gpu_type"].unique().tolist(),
                                        default=df["gpu_type"].unique().tolist())
        with col2:
            task_filter = st.multiselect("Task", df["task"].unique().tolist(),
                                         default=df["task"].unique().tolist())
        with col3:
            system_filter = st.multiselect("System", df["system"].unique().tolist(),
                                           default=df["system"].unique().tolist())

        df_filtered = df[
            df["gpu_type"].isin(gpu_filter) &
            df["task"].isin(task_filter) &
            df["system"].isin(system_filter)
        ]

        agg = df_filtered.groupby(["system", "gpu_type", "task", "concurrency"])[
            ["ttft_ms", "tpot_ms", "tokens_per_sec", "gpu_cost_usd"]
        ].mean().reset_index()

        st.subheader("Time to First Token vs Concurrency")
        for gpu in agg["gpu_type"].unique():
            for task in agg["task"].unique():
                subset = agg[(agg["gpu_type"] == gpu) & (agg["task"] == task)]
                if subset.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                for system, grp in subset.groupby("system"):
                    grp = grp.sort_values("concurrency")
                    ax.plot(grp["concurrency"], grp["ttft_ms"], marker="o", label=system, linewidth=2)
                ax.set_title(f"TTFT — {task} on {gpu}")
                ax.set_xlabel("Concurrent Requests")
                ax.set_ylabel("TTFT (ms)")
                ax.set_xticks([1, 4, 8, 16, 32])
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Throughput vs Concurrency")
        for gpu in agg["gpu_type"].unique():
            for task in agg["task"].unique():
                subset = agg[(agg["gpu_type"] == gpu) & (agg["task"] == task)]
                if subset.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                for system, grp in subset.groupby("system"):
                    grp = grp.sort_values("concurrency")
                    ax.plot(grp["concurrency"], grp["tokens_per_sec"], marker="s", label=system, linewidth=2)
                ax.set_title(f"Tokens/sec — {task} on {gpu}")
                ax.set_xlabel("Concurrent Requests")
                ax.set_ylabel("Tokens / Second")
                ax.set_xticks([1, 4, 8, 16, 32])
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        with st.expander("Raw aggregated data"):
            st.dataframe(agg)

        st.subheader("Cost Analysis")
        df_cost = df_filtered[df_filtered["output_tokens"] > 0].copy()
        df_cost["cost_per_token_usd"] = df_cost["gpu_cost_usd"] / df_cost["output_tokens"]
        cost_table = df_cost.groupby(["system", "gpu_type"])["cost_per_token_usd"].mean().reset_index()
        cost_table["cost_per_1k_tokens_usd"] = cost_table["cost_per_token_usd"] * 1000
        st.dataframe(cost_table[["system", "gpu_type", "cost_per_1k_tokens_usd"]].round(6))
